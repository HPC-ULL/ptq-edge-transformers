# Local patches on ONNX Runtime v1.23.2 (pinned, not untouched upstream)

This project builds `onnxruntime_perf_test` from the `v1.23.2` tag of
[microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) (same commit
the cluster uses, `a83fc4d5`), **not** from the HEAD of `main`. Two real bugs
in that version, found and confirmed on this same Jetson AGX Orin, have been
fixed with minimal local patches instead of jumping to another version (which
would break artifact parity with the cluster). Both are **necessary, not
optional**, to reproduce this project on a Jetson.

If you rebuild `onnxruntime_perf_test` from scratch (fresh clone of the repo,
another device, etc.), apply both before compiling.

---

## 1. Static linking + CUDA does not resolve `cudart`

**File:** `cmake/onnxruntime_unittests.cmake`, inside the `else()` block of the
`onnxruntime_BUILD_SHARED_LIB` branch for the `onnxruntime_perf_test` target
(~line 1277).

**Symptom:** when building with `-DBUILD_SHARED_LIBS=OFF` (static build, so the
final binary does not depend on `LD_LIBRARY_PATH`) and CUDA enabled, linking
fails with unresolved symbols: `cudaMemcpy`, `cudaGetErrorString`,
`cudaStreamCreate` (used directly in
`onnxruntime/test/perftest/ort_test_session.cc`).

**Cause:** the `else()` branch (static build) of that CMake section never adds
`target_link_libraries(onnxruntime_perf_test PRIVATE CUDA::cudart)`, unlike the
`onnxruntime_BUILD_SHARED_LIB` branch (dynamic build), which does. An oversight
of that release in the static+CUDA combination, not something we configured
wrongly.

**Fix:** add the same line (`target_link_libraries(...  CUDA::cudart)`) in the
`else()` branch as well, with the same condition (`onnxruntime_USE_CUDA OR
onnxruntime_USE_NV OR onnxruntime_USE_TENSORRT`). Already documented inline in
the patched file itself.

**Origin:** there is no reference upstream commit/PR -- it is our own fix, not
a cherry-pick. It was confirmed that newer ONNX Runtime versions (tested
against the HEAD of `main`, ~v1.31.0-dev) no longer have this problem in that
branch, so it was probably fixed at some intermediate point without an isolated
commit being identified.

---

## 2. `CPUIDInfo::ArmLinuxInit` -- out-of-bounds write with offline cores (Jetson in a reduced power mode)

**File:** `onnxruntime/core/common/cpuid_info.cc`, inside
`CPUIDInfo::ArmLinuxInit()` (~line 208).

**Symptom:** `onnxruntime_perf_test` (with `-e cpu` **or** `-e cuda`,
regardless of the value of `-x`) completes the real benchmark and prints
results that look correct, but the process ends with `free(): invalid pointer`
and `SIGABRT` (exit code 134) -- **only** when the Jetson is in a power mode
with cores administratively switched off (`MODE_30W`: cores 0-7 online, 8-11
off; `MODE_15W`: only 0-3 online). **It does not happen in `MODE_50W`/`MAXN`**
(all 12 cores online), which is where all the project's initial validation was
done -- that is why it was not detected until measuring in lower power modes.
Preceded by the warning `onnxruntime cpuid_info warning: Unknown CPU vendor.
cpuinfo_vendor value: 0`.

**Cause (confirmed by the ONNX Runtime maintainers themselves, issues
[#24092](https://github.com/microsoft/onnxruntime/issues/24092) and
[#28301](https://github.com/microsoft/onnxruntime/issues/28301)):**
`core_uarchs_`/`is_armv8_narrow_ld_` are sized according to the **number of
online cores** (`cpuinfo_get_cores_count()`), but are indexed with
`proc->linux_id` -- the raw processor ID, which is **not contiguous** when some
core is switched off (with 8 online cores out of 12 in total, `linux_id` can be
8, 9, 10, 11). Writing to `core_uarchs_[coreid]` with those indices is an
out-of-bounds write that corrupts the heap; the `SIGABRT` only appears later,
when the object is destroyed, not at the moment of the write -- that is why the
benchmark itself completes and shows valid results before the process dies.

**Confirmed on this device:** 100% reproducible in `MODE_30W` and `MODE_15W`,
with `-e cpu` and with `-e cuda`, independent of `-x` (reproduced even with
`-x 1`). `llama-bench`/`llama.cpp` does **not** suffer from this problem (it
uses its own CPU detection, without `cpuinfo`).

**Fix applied (partial cherry-pick):** only the essential *bounds check*, taken
from the upstream commit
[`d113549`](https://github.com/microsoft/onnxruntime/commit/d113549)
(PR [#28344](https://github.com/microsoft/onnxruntime/pull/28344), merged into
`main` on 2026-06-02, first released in **v1.27.0**):

```cpp
auto coreid = proc->linux_id;
if (coreid < 0 || static_cast<size_t>(coreid) >= core_uarchs_.size()) {
  continue;
}
auto uarch = corep->uarch;
core_uarchs_[coreid] = uarch;
```

The full *diff* of `d113549` touches 3 files; of these, **only this hunk applies
cleanly on v1.23.2** (the other two are cosmetic -- relocating the suppression
of the "Unknown CPU vendor" warning for Wasm builds, and a test assertion --
and do not apply because of unrelated intermediate structural changes). That is
why the "Unknown CPU vendor" warning still appears after the patch: it is
harmless, that part was deliberately not touched, to keep the patch limited
exactly to the real fix.

**Validated after the patch:** real execution (not just `--self-test`) in
`MODE_30W` and `MODE_15W`, `-e cpu` and `-e cuda`, both invoking the binary
directly and through `runner_vit` -- exit code 0 in all 4 cases, without any
`SIGABRT`.

**Impact if not applied:** `jetson_cpu_15w`, `jetson_cpu_30w`, `jetson_gpu_15w`,
`jetson_gpu_30w` (4 of the 6 Jetson entries in `configs/devices.yaml`) are
unusable for the ViT branch -- only `jetson_cpu_50w`/`jetson_gpu_50w` would work
without this patch.
