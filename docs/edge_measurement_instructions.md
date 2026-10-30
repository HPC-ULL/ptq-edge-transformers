# Instructions — Performance and energy measurement on edge devices

This document describes what is needed to measure performance and energy on an edge device, in parallel with the measurements carried out on the Jetson. It lists the essential steps to get started.

## What to install

Only the components that run benchmarks are needed, not those that quantize; quantization is done on the cluster.

- **`onnxruntime_perf_test`**: built from the ONNX Runtime source code.
- **`llama-bench`**: included when building `llama.cpp` normally, with nothing special.

PyTorch, Brevitas and any other quantization package are not needed for the measurements.

## Reference version of the tools

**ONNX Runtime 1.23.2** is used, the same version as on the cluster. Only the `onnxruntime_perf_test` binary has to be built; no Python wheel is needed (the binary does not depend on it at all, and the measurement pipeline does not use it at any point either). The tag is cloned as follows:

```bash
git clone --depth 1 --branch v1.23.2 --recursive --shallow-submodules \
  https://github.com/microsoft/onnxruntime.git
cd onnxruntime && ./build.sh --config Release --build_shared_lib --parallel --skip_tests
```

**If a device cannot run 1.23.2** (constraints specific to its board, CUDA, etc.), the version should not be forced: the closest one that works is used instead, and the **actual installed version must be recorded** in the device's `devices.yaml` entry (`notes` field) and reported. This is not a blocking requirement, but if two devices end up running different ORT versions, any latency/throughput difference between them is contaminated: it could be the hardware, or it could simply be that ORT changed something internally between versions. The version has to be known in order to interpret the results, and it must not remain hidden.

## Reference version of llama.cpp

A specific commit is pinned as the reference, as with ONNX Runtime: the one the cluster uses to quantize and the one already used on the Jetson, **`b95502ba9`**.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && git checkout b95502ba9
cmake -B build && cmake --build build --config Release -j$(nproc)
```

If the clone is made with `--depth 1` and the `checkout` of that commit fails because there is not enough history, run `git fetch origin b95502ba9` first (or clone without `--depth` directly).

**If that commit does not build on a given piece of hardware**, it should not be forced: the version that does work is used, and the **actual version used must be recorded** in the `notes` field of the device's entry in `devices.yaml`, for the same reason as with ONNX Runtime (comparability between devices).

Even if the version is not recorded, it is not entirely lost: **every `llama-bench` run already reports `build_commit`/`build_number` in its own output JSON**. These fields should be checked to be populated when reviewing the results; they are the fallback traceability if something is left unannotated.

## Which models to use, and why there are 4 packages instead of one

A single zip with everything is not distributed: it weighed ~28 GB because it included the 3 LLM models at once, and devices typically do not have room to keep them all on disk simultaneously (zip + extracted can need up to double). There are **4 independent packages**:

- `edge_bundle_code_vit.zip` — code + the 10 ViT artifacts (small).
- `edge_bundle_llm_tinyllama_1_1b.zip`, `edge_bundle_llm_qwen2_5_1_5b.zip`, `edge_bundle_llm_ministral_3_3b.zip` — one per LLM model (~7-11 GB each).

`edge_bundle_code_vit.zip` is installed first (it brings the code and the template `configs/devices.example.yaml`), and then only the LLM model(s) to be measured at each moment. An LLM package can be downloaded, extracted, measured and **deleted** before moving on to the next, without having all three at once.

From each package, **all the artifacts it brings** are used (they already come filtered: there is nothing else to choose).

## What the ViT package contains

`edge_bundle_code_vit.zip` carries the code and the artifacts together: there is no separate artifacts zip and no need to install one on top of another. There are **10** artifacts. **8 are in `artifacts/vit_deployment/`, in the deployment regime with real acceleration** (`output_quant` enabled, which does achieve INT8 integer kernels on CPU), for two models, `deit_tiny` and `mobilevit_s`:

- **2 FP32 references** (one per model), to compute the real speed-up ratio on the device's own hardware.
- **2 base INT8 configurations** (one per model): `mse` on `deit_tiny`, and `percentile` + BatchNorm folding on `mobilevit_s`, both with the best calibration confirmed with data.
- **4 variants**: `granularity=per_channel` and `symmetry=asymmetric` for each model, applied on top of that same calibration. They answer whether the overhead of these two factors depends on the hardware, with real kernels.

The other **2 are fallback references** and go in `artifacts/vit/` (the only ones in that folder): the ablation INT8, without `output_quant`, of `deit_tiny` (`mse`) and of `mobilevit_s` (`percentile`). They run the quantization in emulated form, without integer kernels, and are used to measure on the device's hardware the cost of that emulation against FP32 and against the accelerated INT8 ones.

**What it does not contain:** the rest of the calibration ablation artifacts; they do not have to be measured. **Why there is no `efficientvit_b1`:** the third model is not part of this set — it neither achieves real acceleration in the tests nor keeps usable accuracy with this configuration. It is excluded with a documented reason; it is not an oversight.

**Manifest:** `run_manifest_code_vit.csv`, at the root of the zip. It is measured with `run_batch.py`, in a single pass (the package already includes everything that has to be measured at once). **If on any device (especially a RISC-V or x86 one) the acceleration is not replicated or the binary fails with this configuration, this should be reported**: it is real and relevant data for the study (platform heterogeneity), not an operator error.

## Which code goes in each package

`benchmarking_ondevice/` (with `runner_vit/`, `runner_llm/`, `power_logging/`) and **`scripts/`** (the driver that goes through the manifest, see below) go in all 4 packages; it is stateless code, safe to overwrite the same way each time. Also in all 4 go these **8 additional files** (real dependencies that `runner_vit`/`runner_llm` import, even though they live outside `benchmarking_ondevice/`; they must not be recreated by hand, the ones distributed in the packages are used):

```
configs/__init__.py
configs/common.py
configs/llm_manifest.py
configs/vit_manifest.py
quantization/__init__.py
quantization/llm_llamacpp/__init__.py
quantization/llm_llamacpp/paths.py
quantization/llm_llamacpp/quant_type.py
```

**`configs/devices.example.yaml` (template) goes ONLY in `edge_bundle_code_vit.zip`**, not in the LLM packages, and **no zip includes a `configs/devices.yaml`**: it belongs to each machine, and extracting a new zip on top does not overwrite it. Even so, when an LLM package is installed after `devices.yaml` has already been filled in, **the project directory must never be deleted to extract the new zip**; the zip is extracted *on top of* the existing directory (`unzip edge_bundle_llm_X.zip -d ~/quantization_transformers/`, without a prior `rm -rf`). Deleting the directory and re-extracting from scratch would lose both the already filled-in `devices.yaml` and `results/raw/results_<device_id>.csv` with the measurements accumulated so far (that CSV does not travel in any zip: the device's own runner generates it the first time it measures, and it survives as long as the folder is not deleted).

## The device's own `PowerMeter`

Each device needs its own implementation of the energy meter for AccelPowerCape (equivalent to the one already available for the Jetson). Inside `power_logging/` (it goes in all 4 packages) there is `dummy_powermeter.py`, the template that defines exactly what the implementation must do and what it must return; the implementation is built on top of it.

**Two requirements that every implementation must fulfill without fail, for the results to be comparable:**

1. **Measure the total consumption of the device**, not just a part: if the sensor gives several channels/rails separately, they are all added up.
2. **Subtract the idle consumption** before computing the energy of each measurement: the device is measured for a while at idle, that value is stored, and it is subtracted from the consumption measured during the benchmark. Without this, two configurations would not be comparable with each other just because they take different times to run.

Once the implementation is built and follows the `dummy_powermeter.py` contract, nothing else in the driver needs to be touched: `run_batch.py` (see below) is only told where it is, with `--power-impl`.

## How to fill in the entry in `devices.yaml`

If `configs/devices.yaml` does not exist yet, it is created by copying the template (`cp configs/devices.example.yaml configs/devices.yaml`); without it, the scripts fail with `No device catalog found` and do not use the template on their own. If it is already filled in from a previous package, there is nothing to do. One entry is added per device, following the example (`rpi_cm4`) already included. The optional field `min_uptime_s` is not needed (only the Jetson entries use it). The fields to fill in:

```yaml
- device_id: device_name
  display_name: "Human-readable name of the device"
  architecture: ARM   # or whatever applies
  backend: cpu         # or cuda, if the device has a relevant GPU
  power_mode_w: null   # or the power mode, if the device has one
  ram_gb: <confirm with free -g>
  cpu_cores: <confirm with nproc/lscpu -- PHYSICAL cores, not threads>
  power_measurement: accelpowercape
  added_by: <name>
  notes: ""   # optional -- use this field to note ONNX Runtime/llama.cpp versions different from the reference, see the sections above
```

## How to run the benchmarks: `scripts/run_batch.py`

`runner_vit`/`runner_llm` **do not have a "run everything" mode**: each processes a single artifact per invocation, on purpose. The manifest does not have to be traversed by hand: each package brings a `run_manifest_<package>.csv` (columns `run_id,repeat_id,artifact_path`), and **`scripts/run_batch.py` goes through it** — it runs each row, writes the result, and if it is interrupted halfway (or something fails), it can be relaunched and only repeats what was missing, not everything from the start.

**Performance only** (recommended for the first validation pass, before measuring energy for real):

```bash
python -m scripts.run_batch \
  --manifest run_manifest_llm_qwen2_5_1_5b.csv \
  --branch llm \
  --device-id your_device_id
```

For ViT, it is exactly the same command, changing `--branch` and the manifest:

```bash
python -m scripts.run_batch \
  --manifest run_manifest_code_vit.csv \
  --branch vit \
  --device-id your_device_id
```

(all the options below — `--with-power`, `--power-impl`, cooldown — work the same in both branches, only `--branch`/`--manifest` change)

**Performance + energy**, once the `PowerMeter` is built: `--with-power` is added and `--power-impl` indicates where the implementation is (the name of the Python module of the device's own implementation; it does not need to be registered anywhere beforehand):

```bash
python -m scripts.run_batch \
  --manifest run_manifest_llm_qwen2_5_1_5b.csv \
  --branch llm \
  --device-id your_device_id \
  --with-power --power-impl benchmarking_ondevice.power_logging.accelpowercape_powermeter \
  --repeats 30
```

(the module name is adjusted to whatever the implementation is called — it only has to fulfill the `dummy_powermeter.py` contract)

**Important, only for LLM + energy:** `--repeats 30` must always be used when measuring energy in the LLM branch (`--branch llm --with-power`). The default value of `llama-bench` (5) does not dilute the fixed cost of loading the model enough, and that biases `energy_per_inference` systematically upwards (always in the same direction, never randomly); its relative weight can vary from one model to another (it is not characterized), so it can also affect energy comparisons between models, not just the absolute value of each one. Confirmed with a convergence test (N=5/15/30): with N=30 the change between repetitions is already small (<2%), with N=5 or N=15 it is not. This **does not apply to ViT** (`onnxruntime_perf_test` already uses a much larger number of repetitions by default, without this problem); it only affects LLM, and only when energy is measured (`--with-power`). If only performance is measured, the default value of `llama-bench` is still valid.

**Important:** if a pass was already run without `--with-power` and is then relaunched with `--with-power` over the same manifest, `run_batch.py` detects which rows already have energy data and which do not; it only re-runs those that lack it, it does not repeat those that are already complete. This **does not detect** whether a row already has measured energy but with an insufficient `--repeats` (for example, if it was measured with the default value before this note was read). In that case, the explicit re-run of those rows has to be forced (`--force`, or the equivalent mechanism used to force).

`--dry-run` shows what would be done without running anything real, which is useful to check that the manifest and the arguments are the right ones before launching for real.

## Cooldown between runs (recommended)

`run_batch.py` supports two ways of waiting between runs, depending on what each device needs; the same one used on the Jetson does not have to be used.

**Fixed time**, the simplest, valid for any device:

```bash
python -m scripts.run_batch \
  --manifest run_manifest_llm_qwen2_5_1_5b.csv \
  --branch llm \
  --device-id your_device_id \
  --cooldown-seconds 30
```

**By temperature**, if the hardware allows measuring it: the driver waits until the temperature drops below a threshold, instead of a fixed time, with a safety cap in case it never drops. As with energy, the Jetson implementation does not have to be used: all that is needed is a `get_cpu_temperature() -> float` function in the device's own module, passed with `--temp-cooldown-impl`:

```bash
python -m scripts.run_batch \
  --manifest run_manifest_llm_qwen2_5_1_5b.csv \
  --branch llm \
  --device-id your_device_id \
  --temp-cooldown --temp-cooldown-impl your.own.module \
  --temp-threshold-c 55 --cooldown-max-seconds 120
```

The threshold and the safety cap are adjusted according to what is known about each device; there is no universally correct value. If nothing is specified, there is no wait between runs.

## Order of execution, for each new device

1. **First, `--self-test`** in each runner (`runner_vit` and `runner_llm`), to confirm that the binary built on that device produces an output that the code recognizes. If the ONNX Runtime version is not the reference 1.23.2 (see above), this is what really confirms whether the parser withstands the difference. If `--self-test` fails, the exact installed version should be reported before trying to force it.
2. **Then, a real test run** against a single artifact (directly with `runner_vit`/`runner_llm`, without going through `run_batch.py` yet), checking that the latency/throughput number makes sense (nothing extreme, neither zero nor absurdly high).
3. **When it is validated, `scripts/run_batch.py` is launched** over the complete `run_manifest_<package>.csv`: first without `--with-power` to confirm that performance comes out right in all rows, and then with `--with-power` (as soon as the `PowerMeter` is ready) to complete the energy.

## Where the results end up

Every device that is measured generates its own CSV (`results_<device_id>.csv`); it is normal to have several files, one per device. Both ViT and LLM share the same CSV, which leaves several columns blank, but this way twice as many results files are avoided.

---
