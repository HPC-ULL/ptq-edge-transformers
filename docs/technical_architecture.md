# Technical Architecture of the Project: Code Organization for the Calibration Experiments (ViT + LLM)

This document translates the two theoretical protocols (ViT/Brevitas and LLM/llama.cpp) into a project structure. It contains no code, only the organization and the data flow between modules.

## 1. General design principle

Both pipelines (ViT and LLM) differ in tools but share the same shape: **an offline phase** (calibration + quantization + accuracy evaluation, on the highest-capacity workstation) and **an on-device phase** (latency/energy benchmarking on each edge device). This was already the case in the original paper — the novelty is that the offline phase now has a factor of variation (the calibration) instead of being a fixed configuration.

The architecture must reflect that separation clearly, because they are different execution environments (workstation vs. open catalog of edge devices, Section 10.5) and it is not advisable to mix them in the same module.

## 2. Directory structure

```
project/
├── configs/
│   ├── vit_experiment_manifest.yaml     # list of ALL the ViT configurations (Phases 1-3)
│   ├── llm_experiment_manifest.yaml     # list of ALL the LLM configurations (Phases 1-3)
│   ├── devices.yaml                     # open catalog of edge devices (Section 10.5); local, does not travel in any zip
│   └── devices.example.yaml             # template that does travel in the ViT packages
│
├── data_prep/                           # download/preparation of ImageNet-1K and WikiText-2
│                                         # (train calibration pool + fixed val/test subset)
│
├── quantization/
│   ├── vit_brevitas/                    # ViT calibration + quantization pipeline (offline, cluster)
│   └── llm_llamacpp/                    # imatrix generation + quantize (offline, cluster)
│
├── export/
│   └── vit_to_onnx/                     # Brevitas -> ONNX/QONNX (the LLM .gguf files are already the final artifact)
│
├── artifacts/                           # already quantized models (ONNX/GGUF), generated once and shared
│   │                                     # between whoever quantizes and whoever runs on the edge devices
│   ├── vit/                             # ViT calibration ablation (99 artifacts, without output_quant)
│   └── vit_deployment/                  # ViT deployment set (output_quant; findings_vit.md, 7.6)
│
├── evaluation_offline/
│   ├── vit_accuracy/                    # top1/top5 on ImageNet val — does not depend on the device
│   └── llm_perplexity/                  # perplexity on WikiText-2 — does not depend on the device
│
├── benchmarking_ondevice/
│   ├── runner_vit/                      # invokes onnxruntime_perf_test (official ONNX Runtime tool)
│   ├── runner_llm/                      # invokes llama-bench (already used in the original paper)
│   └── power_logging/                   # common interface + per-hardware adapter (Section 6.4)
│
├── scripts/                             # execution scripts (run_batch.py) and research/diagnostic scripts
│
├── results/
│   ├── raw/                             # one entry per run (see common schema, Section 4)
│   ├── investigations/                  # outputs of investigations and diagnostics, separate from the pipeline
│   ├── characterization/                # characterization of the measurement instrument (written and read by power_logging)
│   ├── backups/                         # backup copies of the recomputation scripts, never in raw/
│   ├── appendix/                        # model of the previous iteration (outside the main study)
│   └── schema.md
│
├── analysis/
│   ├── phase3_selection/                # selection of Phase 3 winners
│   ├── phase4_vit/                      # ViT Phase 4 (inventory, Level 1 and Level 2); does not travel in the zips
│   ├── phase4_llm/                      # LLM Phase 4 (activations from the imatrix, weights, robustness controls, per-tensor intervention and ceiling effect); does not travel in the zips
│   └── figures/
│
├── CLAUDE.md                            # Claude Code context; loaded automatically at startup (repo root)
│
└── docs/
    ├── calibration_protocol_vit.md
    ├── calibration_protocol_llm.md
    ├── technical_architecture.md
    ├── findings_vit.md
    ├── findings_llm.md
    ├── edge_measurement_instructions.md
    └── future_work.md
```

## 3. The experiment manifest as the central piece

Since the design is **one-factor-at-a-time** (not a full factorial), the biggest risk of programming this "by hand" is the duplication of configurations or forgetting some. The way to avoid it:

- Define a **baseline configuration** (Phase 1 of each protocol) as a base dictionary/structure.
- Generate the rest of the configurations **programmatically**, as "baseline + a single override" for each level of each factor (this is literally the definition of OFAT, and the code should express it that way explicitly, not as a hand-written list).
- The resulting manifest (Phases 1+2+3) is a list of complete configurations, each with a unique and traceable `run_id`, which feeds both the offline phase and the on-device one.

This turns the manifest into the **single source of truth**: any later module (quantization, benchmarking, analysis) simply iterates over it, without each module needing to "know" the experimental design.

## 4. Common results schema (long-format)

To be able to cross ViT and LLM results in the final analysis (Phase 4 of both protocols), a single results table with the same structural columns is advisable, even if some fields only apply to one branch:

| Column | Applies to | Notes |
|---|---|---|
| `run_id` | Both | Unique and traceable identifier of the configuration |
| `model_family` | Both | `ViT` / `LLM` |
| `model_name` | Both | e.g. `DeiT-Tiny`, `Qwen2.5-1.5B` |
| `phase` | Both | `1_baseline`, `2_ofat`, `3_combined` |
| `factor_varied` | Both | Which factor was changed with respect to the baseline (or `none` in Phase 1) |
| `factor_level` | Both | Concrete value of the factor |
| `calibration_algorithm` | ViT only | `minmax`, `percentile`, `mse`, `entropy` |
| `granularity` | ViT only | `per_tensor` / `per_channel` |
| `symmetry` | ViT only | `symmetric` / `asymmetric` |
| `calibration_corpus` | LLM only | identifier of the corpus used for the imatrix |
| `quant_scheme` | LLM only | `legacy` / `k_quant` |
| `calib_set_size` | Both | no. of samples/sequences used in calibration |
| `bitwidth` | Both | e.g. `int8`, `int4`, `Q4_K_M` |
| `device` | On-device benchmarking only | one of the devices of the open catalog (Section 6), or `offline` for accuracy/perplexity. For the Jetson, it is not a single value: each backend×power combination is a different `device` entry (e.g. `jetson_cpu_15w`, `jetson_gpu_50w`) — see Section 6.2 |
| `repeat_id` | Both | repetition number (minimum 3, because of the calibration sampling variance) |
| `accuracy_metric_name` / `accuracy_value` | Both | top1/top5 or perplexity, depending on the branch |
| `latency_ms`, `energy_per_inference`, `perf_per_watt`, `model_size_mb` | Both | only populated in the on-device runs (accuracy/perplexity are computed only once offline and repeated in the rows of all devices, they are not recomputed per device) |
| `num_inferences_measured` | Both | internal repetitions the benchmark did during that execution, as reported by the tool itself — needed to compute `energy_per_inference` without warm-up bias (see Section 6.5) |
| `throughput` | ViT only | inferences/second — a single figure per execution, there is no prefill/decode distinction in image classification |
| `throughput_prefill_tps` / `throughput_decode_tps` | LLM only | tokens/second of *prompt processing* (`pp`) and *text generation* (`tg`) separately — `llama-bench` already reports them as independent metrics, do not collapse them into a single column (see Section 6.3) |

This schema is what allows the combination script and the later analysis to operate the same way on both branches without complex conditional logic.

## 5. Separation of responsibilities per module

- **`quantization/`**: only produces quantized artifacts (ONNX/QONNX or GGUF) + the offline accuracy metrics. It knows nothing about edge devices.
- **`benchmarking_ondevice/`**: only consumes already quantized artifacts and produces performance/energy metrics. It does not repeat or recompute accuracy.
- **`analysis/phase4_vit/` and `analysis/phase4_llm/`**: Phase 4 (distribution metrics per layer or per tensor, interventions and controls). They start from the FP32 models, the imatrix files and the already generated artifacts, do not recompute accuracy nor device measurements, and do not travel in the zips.
- **Phase 4 results:** each subpackage writes to `results/investigations/<branch>/phase4/`, and the results CSVs in `results/raw/` are not modified.

## 6. Division of work: centralized quantization, distributed benchmarking

Given that quantization is done only once on a higher-power machine (as in the original paper) and the same artifact is then reused on all the edge devices, no automatic orchestration module is needed. The real division of work is:

- **Whoever quantizes** (`quantization/` + `export/`) deposits the final artifacts in `artifacts/`, together with their corresponding manifest `run_id`. These artifacts **are not regenerated** on each machine — this is important because the calibration has randomness (minimum 3 different samplings, Section 6 of both protocols), and if each person requantized on their own, any difference between devices would be contaminated by "is it the hardware, or was the calibration different?" (the same confounding variable that reviewers' criticism 9 points out).
- **Whoever runs on each device** only needs `benchmarking_ondevice/` (the runners) and the already generated artifacts — manually copies the corresponding `.onnx`/`.gguf`, runs the runner with the path and the device name as parameters, and the result is saved in `results/raw/` with the `device` field filled in (see Section 4). There is no need to automate the deployment: it is enough that both use the **same runner** to avoid subtle methodological differences between what each one measures (number of repetitions, warm-up, etc.).
- **Final combination:** a simple script in `results/` that reads the `raw/` files of both machines/people, validates that each row has a non-empty `device`, and concatenates everything into the final table for `analysis/`. The validation that the device is documented in the open catalog (`devices.yaml`, Section 10.5) has **one reserved exception**: `device == "offline"` (accuracy/perplexity rows, computed once without depending on edge hardware) is always valid without needing an entry in `devices.yaml` — that catalog documents only real physical edge hardware, it is not polluted with a fictitious entry just to pass the validation.

## 6.1 RISC-V contingency

`onnxruntime_perf_test` is not distributed as a precompiled binary for most platforms — it has to be built from source (up to ~40 min per build), which adds real risk on the RISC-V devices, where there were already doubts about software maturity. If building `onnxruntime_perf_test` (or ONNX Runtime in general) gives too many problems on RISC-V:

- **Technical fallback before discarding:** use the native profiling of the ONNX Runtime Python API (`session_options.enable_profiling = True`), which does not require building the C++ binary, at the cost of a somewhat more manual script configuration.
- **Scope contingency (already assumed):** if even that is not viable, the RISC-V devices can be dispensed with for the ViT branch and the study of that branch limited to x86/ARM/Jetson. This somewhat reduces the comparative value of the work (RISC-V is part of the hardware heterogeneity argument), but it is preferable to blocking the rest of the experimental schedule because of a problem with a single type of device — and it can be documented as an explicit limitation in the paper, as was already done with the Odroid Xu4 in the original study ("unable to execute some benchmarks").

**Status.** ViT has been measured on the VisionFive 2 (RISC-V) with the 10-row deployment manifest (`device_id` `visionfive_2`, one repetition), and the kernel probe confirms that the quantized layers use integer kernels (`findings_vit.md`, Section 7.7). For LLM, the subset was fixed before seeing results: TinyLlama with the 4 baseline format rows (Q4_K_M, Q4_0, Q3_K_M and Q2_K), Qwen2.5 if there is time, `--repeats 10`, and Ministral-3-3B left out because of execution time on the VisionFive 2 and the CM4 (TinyLlama, the smallest, already took hours); it is declared as a limitation.

**Jetson-specific note:** when building `onnxruntime_perf_test` on the Jetson, use the CUDA/cuDNN/TensorRT that the **JetPack** installed on the device itself brings, not a generic arm64 CUDA combination — NVIDIA distributes for the Jetson a specific ecosystem (JetPack) with paired versions of CUDA/cuDNN/TensorRT that do not always coincide with what would be installed on a generic ARM. Verify this as part of the same technical spike. When installing CUDA/cuDNN packages via `apt`, confirm the **exact origin of the package** (channel/repository, not just the version number) — the same version number can correspond to different builds depending on the channel (Jetson-specific vs. generic SBSA), with really different behavior despite the identical label.

## 6.2 Jetson: CPU + GPU × 3 power modes, also for ViT

With the change from TFLite to ONNX Runtime, a limitation of the original paper disappears: TFLite did not support GPU on NVIDIA, so CNN/RNN were only measured on the Jetson's CPU. ONNX Runtime does have a CUDA/TensorRT execution provider, so it is now possible (and methodologically desirable, to close that asymmetry) to measure ViT on the Jetson's GPU too, in the same three power modes (15W/30W/50W) already used for LLM in the original paper.

This expands the Jetson from "one device" to **6 `device` entries** (CPU×3 powers + GPU×3 powers). Applying the complete calibration ablation (Phase 2, 8 configurations) to the 6 combinations would multiply the volume of runs a lot, so the same criterion of Section 2.1 of the protocols (accuracy-only factors vs. accuracy+performance) is applied:

- **Phases 1 (baseline) and 3 (final combination):** run on the 6 complete combinations — they are the "headline" results, directly comparable with the original paper's figures. **For ViT, this is all that is measured of the calibration ablation at the edge** (see the note below).
- **Phase 2, accuracy-only factors** (calibration algorithm, calibration set size in ViT; corpus and imatrix in LLM): they do not need to be repeated per device/power combination — accuracy does not depend on the execution backend, it is computed once offline.
- **Phase 2, accuracy+performance factors — applies only to LLM** (block scheme, legacy vs. K-quants): verified on a representative subset of 2 Jetson combinations (e.g. `jetson_cpu_50w` and `jetson_gpu_50w`), enough to detect whether there is a systematic CPU vs. GPU difference without running the 6 complete combinations.

**Note — ViT, granularity/symmetry withdrawn from the edge measurement of the ablation (decision after the first version of this document):** it was confirmed that the ViT calibration ablation in its original regime (without output `Q`, Section 6.4.2/7 of `findings_vit.md`) always runs through fallback — the granularity/symmetry overhead cannot be tested with real hardware in that regime (any measured difference would be Q/DQ emulation overhead, not integer kernels). That question (does the overhead depend on the hardware?) is answered instead in the **deployment set** (separate artifacts, with `output_quant`, real kernels confirmed) — see `findings_vit.md` Section 7.7, the 4 additional artifacts (`granularity=per_channel`/`symmetry=asymmetric` × DeiT-Tiny/MobileViT-S). The calibration ablation's granularity/symmetry is not measured on any edge device for ViT.

## 6.3 Prefill/decode separation in LLM (performance)

Quantization affects the two inference phases of an LLM differently: the **prefill** (prompt processing, parallel over all input tokens, compute-bound) and the **decode** (token-by-token generation, sequential, memory-bandwidth-bound). Quantization typically benefits decode much more than prefill, so collapsing both into a single throughput figure can hide precisely the type of pattern the ablation is looking for.

`llama-bench` already reports these two metrics separately natively (`pp` = prompt processing, `tg` = text generation) — no additional instrumentation is required, only making sure the results collection script does not collapse them into a single column (see common schema, Section 4: `throughput_prefill_tps` / `throughput_decode_tps`).

## 6.4 Common energy measurement interface (per-hardware implementation, shared contract)

The energy sensor differs by device type (the Jetson measures via its integrated INA3221; the rest of the devices via AccelPowerCape/INA219), so it makes no sense to share a single literal implementation — each person maintains their own module (`power_logging/jetson_powermeter.py`, `power_logging/accelpowercape.py`, etc.), specific to the hardware they measure. What must be common, and what really matters for the results to be comparable across devices, is the **input/output contract**:

- **Input:** a function that wraps the execution of the benchmark (`onnxruntime_perf_test` / `llama-bench`) and runs it while measuring.
- **Output:** a homogeneous result, independent of the sensor used underneath, with at least these fields — `energy_joules_total`, `duration_s`, `avg_power_w`, and **`num_inferences_measured`** (the number of internal repetitions the benchmark did during that execution, as reported by the tool itself — needed to compute `energy_per_inference` correctly, see Section 6.5).

A "dummy"/reference interface module is defined in `power_logging/` (a function with the signature and the output schema already fixed, with no real sensor implementation) that each person adapts to their own hardware. As long as the output meets that schema, the rest of the pipeline (`results/raw/`, `analysis/`) does not need to know which sensor produced it.

## 6.4.1 Two mandatory requirements of the contract, not optional per implementation

Unlike the rest of the contract (field names, shape of the output), these two points affect the **validity of the comparison across devices** and must be fulfilled in all implementations (Jetson, AccelPowerCape, any future device), not left to the discretion of whoever builds each module:

- **Total power of the device, not just the active rail/component.** When the sensor exposes separate rails (e.g. the Jetson's INA3221: GPU+SoC on one side, CPU+CV on the other), the measurement must **add up all the rails**, not report only the one that is "active" according to the execution backend (CPU or GPU). Reason: running on the GPU still generates real consumption on the CPU rail (orchestration, kernel launching) — measuring only the active rail would systematically underestimate the GPU cost. This also closes a limitation that the original paper itself already acknowledged without resolving (the per-rail measurements "include contributions from other components", without aggregating them).
- **Mandatory idle consumption subtraction, with the net as the main metric.** Without subtracting idle, the measured energy includes the device's base consumption during the whole duration of the benchmark — and since that duration varies between configurations, a faster configuration would "appear" more energy-efficient just because it takes less time to run, not because it consumes less real power during the computation (the same type of duration confounder already identified in Section 6.5 for the warm-up). Each implementation must measure an idle window before the load and subtract it.

**Output schema extended accordingly:** `energy_joules_total`/`avg_power_w` (Section 6.4) are reported in two variants — `..._raw` (gross, without subtracting idle) and `..._net` (net, idle subtracted). **`..._net` is the mandatory metric for all ablation comparisons**; `..._raw` is kept as additional data, useful for the interpretation of real deployment cost (a device in production consumes that idle continuously), but it must not be used to compare configurations with each other.

**Derived metrics (closed definition).**
- `energy_per_inference = energy_net_j / N`, with `N = num_inferences_measured`. In LLM it is the energy **per repetition** (one execution of pp128 + tg32, 160 tokens), not per token, and it includes the amortized fixed cost of loading and warm-up (a residue of the order of 2-3 % at N=30, `findings_llm.md`). Per token: `energy_per_inference / 160`.
- `perf_per_watt`, always over **net** power/energy and always derived from `energy_net_j`/N, not from `throughput`: in ViT, `N / energy_net_j` (inferences per joule); in LLM, `(n_prompt + n_gen) / energy_per_inference` (net tokens per joule, secondary metric — see the note below).
- **Units and name of `perf_per_watt`:** it is performance per watt, `(work/s)/W`, which is equivalent to work per joule: inferences per joule in ViT and tokens per joule in LLM. The column name is kept for schema compatibility (renaming it would force changing code, CSVs and already distributed packages); in the paper it is defined with its units. In LLM, `energy_per_inference / 160` (J per token) is exactly its inverse, adds no new information and averages prefill and decode tokens, so it is not comparable with the J per generated token that the literature quotes.
- **Correction of the ViT formula (the same type of failure as in LLM, found when reviewing the coherence between branches):** the original formula was `throughput / avg_power_w_net`, which is not exactly `N / energy_net_j` — `throughput` excludes the loading and warm-up from the computation, but the power window (`avg_power_w_net`) does include them, so both formulas diverge (little with large N, but they diverge). It is corrected to `N / energy_net_j`, computed from the same two fields already stored, without measuring again. Recompute offline any ViT `perf_per_watt` already obtained, as was done for the LLM failure.
- **Correction of the LLM formula:** the first version divided the 160 tokens by `energy_net_j`, which accumulates the N repetitions, and returned tokens/J divided by N (a factor of 30 at N=30). The error was detected in the devices' CSVs and corrected in the shared code; the rows already collected are recomputed offline from `energy_net_j` and N. The results combination script must recompute this column from those two fields instead of trusting the stored value.
- **Decided: main metric in LLM = `energy_per_inference` (net joules per request of 128+32 tokens).** `perf_per_watt` (tokens/J) mixes prefill and decode tokens as if they cost the same, when decode is much more expensive per token (same reason why `throughput_prefill_tps`/`throughput_decode_tps` are already reported separately, Section 6.3) — it also depends on the 128/32 ratio chosen, it is not a property of the hardware alone. `energy_per_inference`, already computed, is interpreted directly ("energy cost of one response") and does not weight tokens of different cost. `perf_per_watt` remains as a secondary metric, marked as specific to this workload. It did not require touching code or recomputing anything — it was a decision about what to highlight in the text, not about what to measure.

**Idle subtraction in AccelPowerCape — confirmed and homogeneous with the Jetson.** The colleague measures the idle consumption and subtracts it, like the Jetson implementation (Section 6.4.1, above). **Correction of an earlier version of this note, which said that idle is measured before each execution:** the CSVs show that the implicit idle (`avg_power_w_raw − avg_power_w_net`) is identical, to three decimals, in groups of rows (e.g. 11 rows of the same model on the n150 and on the rk1-02; 20, 11, 7 and 3 rows in the Jetson's 30 W mode) and only changes between groups. That is, it is measured **once per `run_batch` invocation** and shared among all the rows of that batch; the variation between batches (2-4 % on the Jetson) is a session effect that affects the net power. Consequence: rows measured in different batches do not share an idle baseline, and this should be kept in mind when comparing rows from different sessions. Sentence for the Methodology: *"idle energy is measured once at the start of each batch of runs and subtracted from the gross energy of all the runs of that batch; the same procedure is applied on all the study's devices."*

## 6.4.2 Confirmed and corrected finding: the original pipeline does not accelerate; the cause is a specific export configuration, not a limitation of the ecosystem

**History of the finding, important to avoid repeating the interpretation error:** the first verification (profiling on `deit_tiny`, Jetson CPU and GPU) confirmed that the INT8 of the original pipeline is slower than FP32 on both backends (`FALLBACK`, 0 integer kernels) — but the initial conclusion that it was a "structural limitation of the ONNX ecosystem" **was incorrect**. A later investigation (comparing against ONNX Runtime's native quantization tool, `quantize_static`, which does fuse the same model 100%) identified the real cause: **the original pipeline quantizes only the input of each `Conv`/`Linear`, without a `Q` (quantization) at the output** — without that second node, ONNX Runtime does not recognize the complete pattern needed to fuse into `QLinearConv`/`QLinearMatMul`. By adding that output `Q` (`output_quant` parameter of the quantizer), fusion is achieved 100% on CPU (x86 and ARM). The resulting acceleration depends on the number of threads and the power mode: the figures of 1.15-1.66× are at 1 thread, and with the real deployment it can disappear (`findings_vit.md`, Section 7.8).

**Full detail, including the GPU verification (where the fallback does persist, confirmed as a platform limitation) and the decision on which models enter the deployment set with integer kernels: see `findings_vit.md`, Section 7.**

**Summary of the final situation, for whoever only needs the headline:**
- **Calibration ablation (99 artifacts: 63 from Phases 1-2 and 36 from Phase 3):** unchanged, kept in the original regime (without `output_quant`) — the relative comparisons between calibration configurations remain valid within that regime, and answer the question of which calibration strategy best preserves accuracy, regardless of whether there is real acceleration.
- **Deployment set with real integer kernels (CPU only):** DeiT-Tiny and MobileViT-S, with `output_quant` enabled — artifacts and measurements separate from the calibration ablation (`artifacts/vit_deployment/`, do not mix with `artifacts/vit/`).
- **EfficientViT-B1 excluded** from the deployment set: only partial kernel fusion on CPU, and accuracy already collapsed under `output_quant` even before considering the fusion problem — see `findings_vit.md` Section 7.6 for the full reasoning, including the answer already prepared for a reviewer's question about why the fusion problem was not investigated more deeply.
- **GPU (CUDA EP):** the fallback persists even with `output_quant` and with ORT's native tool — confirmed as a real limitation of the CUDA EP (the path to real acceleration on NVIDIA GPU is TensorRT, discarded for the same methodological reason as XNNPACK/QONNX, Section 6.1). The 10 rows of the set were also measured on the Jetson's GPU (N = 5000, three modes), and the INT8 turns out ~2× slower and with ~2× more energy than its FP32; the FP32 on GPU dominates the INT8 on CPU (`findings_vit.md`, Section 7.9).

## 6.4.3 Energy sampling characterization — summary for the Methodology

Verification carried out to anticipate an IoT-J reviewer's question about whether each measurement window contains enough samples to trust the mean (threshold fixed beforehand, before seeing data: ≥30 samples/window; retroactive estimates only accepted with a margin ≥3× that threshold). Both sensors confirmed with a margin well above what is required:

- **Jetson (INA3221):** ~27.5 Hz effective measured in the 3 power modes (configured at 30 Hz; native sensor refresh 1 ms according to `update_interval_ms`, confirming that the bottleneck is in the Python loop, not in the hardware). Minimum windows with the final data (computed as `energy_raw_j / avg_power_w_raw` in the Jetson's results CSVs): **ViT, 31.1 s** (DeiT-Tiny FP32 on GPU at 50 W, N = 5000; on CPU, 32.6 s), about 855 samples at ~27.5 Hz (margin ×28); **LLM, 9.8 s** (TinyLlama Q4_0 without imatrix on GPU at 50 W), about 270 samples (margin ×9). The 18 F16 rows meet the threshold (`meets_threshold=True`). It replaces an earlier minimum of 255 samples, measured with the ablation's ViT rows on GPU (N = 1000), which are no longer part of the study.
- **AccelPowerCape (INA219/pmlib):** ~908 samples/s measured by the colleague in a 10 s idle test (9080 samples; maximum configured frequency of 1200 Hz). **Window duration obtained from their own results:** their LLM CSVs do not record the number of samples, but the duration of each window comes from `energy_raw_j / avg_power_w_raw` (it coincides with `energy_net_j / avg_power_w_net` in the 62 rows, discrepancy < 1 %). The shortest window is 80.3 s on the n150 and 60.2 s on the rk1-02 (both, TinyLlama with `legacy_corpus_none`), i.e. at least ~54,700 samples expected at 908 Hz: a margin of ~1,800× over the threshold of 30. It is a derivation (duration × measured effective frequency), not a direct count of samples, and it no longer depends on any figure from the previous study. **Pending:** repeat the computation with the ViT windows when they are measured, which will be the shortest of the study.

With both sensors well above the required threshold, no additional remeasurement is considered necessary for this reason.

## 6.4.4 Repetitions per row and rows with a median of three (ViT, Jetson CPU)

Each row is normally measured **only once** (`rep1`). For the rows that support the INT8 versus FP32 comparison, **three repetitions were made, each one in a separate `run_batch --force` invocation**. Since idle is measured once per invocation (Section 6.4.3), each repetition is a different session. Repetition 1 is the original batch.

- **Rows with three repetitions:** the FP32 and the INT8 core of DeiT-Tiny and MobileViT-S in the 15, 30 and 50 W modes; the three INT8 of DeiT-Tiny at 50 W; the three INT8 variants of MobileViT-S at 15 W, and the FP32 and the INT8 core of MobileViT-S on the GPU at 15 W.
- **Row with one remeasurement:** `per_tensor_asymmetric` of DeiT-Tiny at 15 W, with a criterion fixed before measuring.
- **Rest of the rows** (variants in the other modes, fallback references): a single measurement.

**Source of the data and role of the repetitions.** The results CSV is the **only source** of the study's figures, with one measurement per row on all devices. The repetitions are a **characterization of repeatability on the Jetson**, a separate diagnostic to quantify the uncertainty and fix the reading rule, and they do not replace any value in the CSV. For the repeated rows, the CSV keeps the **median-latency** execution (a real execution), whose latency coincides with the median of the three; the energy may differ from the per-metric median by ~2 %. It is a small asymmetry with respect to the devices with a single measurement, and it is declared.

**Where everything is.** The `summary_*.csv` (three repetitions, medians, ranges, `cv_pct`, idle of each session and speedup per session) and the `selection_*.csv` (which repetition was kept) are generated by the dispersion script on the Jetson and live in `results/diagnostics/`. **They are a diagnostic artifact of the Jetson: they do not travel in any package and the results combination script does not read them.** They are cited in the paper for the uncertainty and the reading rule. The per-metric medians may come from different executions, so they do not exactly satisfy power × time = energy among themselves.

**Reading.** The speedups are classified with a rule fixed in advance (speedup if min(FP32)/max(INT8) > 1; slower if max(FP32)/min(INT8) < 1; otherwise, no appreciable difference), and the speedup per session is also reported. Variant-versus-core comparisons must be made within the same session, because rep1 is measured in a different batch. The single-measurement rows inherit the variability between invocations, which on the Jetson at 12 threads reaches 16 % in the INT8 of MobileViT-S and 4-11 % in the DeiT-Tiny variants; it is not due to thermal drift (`findings_vit.md`, Sections 7.8.4 and 7.8.5).

## 6.5 Internal repetitions of the benchmark and warm-up dilution

The energy measured by the wrapper covers the **whole execution** of the benchmark call (model loading, warm-up included), while the latency/throughput reported by the benchmark itself normally **excludes the warm-up** from its figures. If `energy_per_inference` is computed by dividing the total energy by the number of "official" (post-warm-up) inferences, the consumption per inference is systematically overestimated. How the relative weight of that fixed cost varies between models of different size is not characterized (loading and warm-up grow with the model, but so does the energy of each repetition) — see `findings_llm.md`, section on the fixed cost of the energy window.

The mitigation is one of configuration, not of additional code: set a number of internal repetitions high enough for the loading and warm-up cost to be diluted. The `num_inferences_measured` field (Section 6.4) makes this traceable and auditable instead of assumed. **Measured results** (`findings_llm.md`, fixed cost section): in LLM with N = 30, the fixed cost in energy is ≤ 2 % on CPU with a warm cache and indistinguishable from zero on GPU (cold it can add 1-9 %); a time-proportional correction was tested with a criterion fixed in advance and is not adopted, because the fixed part spends almost no net power. In ViT with N = 1000 on CPU, the fraction of fixed time is ~1 % (max. 2.5 %); on GPU N = 5000 was used, and only the FP32 rows of DeiT-Tiny (5.5-10.3 %), the fastest ones, remain above 5 % in time.

## 6.6 Capturing the output of the benchmarks

The runners launch the binaries via `subprocess` and capture their output to reformat it into the common schema (Section 4).

**`llama-bench`:** use `--output json` (or `csv`) explicitly, instead of parsing its default text output — it avoids depending on a format designed for human reading, which can change between versions without notice.

**`onnxruntime_perf_test`:** it does not offer an equivalent structured output (confirmed) — its text must be parsed from stdout. **Warning confirmed in practice:** the format varies between versions — the initial documentation/reference of this project assumed fields such as `Total iterations`/`Min/Max/P50/...Latency`, but the real installation used (ONNX Runtime v1.23.2, built on the cluster) produces a different format, without percentiles by default unless the positional `result_file` is omitted (`-s` is only activated if no `result_file` is given, a counterintuitive behavior confirmed in the real `--help`):

```
Total inference requests: 5
Average inference time cost: 24.7294 ms
Number of inferences per second: 40.4307
```

Mandatory fields extracted by the parser (`runner_vit/output_parser.py`): `num_inferences_measured` (from `Total inference requests`), `avg_latency_ms`, `throughput_ips` — explicit failure if any is missing, do not assume the input value (`-r N`) as a substitute. Other fields, best-effort. **The parser must be re-verified against the real output of any new ONNX Runtime build/version** (confirmed that neither the output format nor the set of input flags is stable between versions) — `runner_vit` includes a `--self-test` mode for this verification (see below).

## 6.6.1 Why INT6/INT4 in ViT are not represented with a native data type

The ONNX standard (`QuantizeLinear`/`DequantizeLinear`) does not define native containers below 8 bits — INT6/INT4 are emulated **inside an `int8`/`uint8` container, clipping the range of representable values by means of `clipping`**, not with a 6- or 4-bit data type of their own. Confirmed in the Brevitas exporter code (`StdQCDQCastONNXMixin`, method `validate_8b_bit_width(module.bit_width(), le_then=True)` — the check accepts any bit-width "less than or equal" to 8, all represented in the same 8-bit container). Direct consequence: the on-disk size of an INT6/INT4 artifact is not smaller than that of INT8 (same container), and any hardware acceleration that depends on the container's data type (not on the range of values used inside it) treats INT6/INT4 the same as INT8 — same fusion/fallback behavior documented in Section 6.4.2 and in `findings_vit.md`, Section 7.

**Parameters to set explicitly, not left at default, for reproducibility:**

| Parameter | Fixed value | Reason |
|---|---|---|
| `-o` (optimization level) | `99` (all) — keep the default, but declared explicitly | Realistic deployment performance, consistent with the original paper not disabling optimizations in TFLite either |
| `-S` (seed) | Explicit fixed value (e.g. `42`) | Without this, each execution generates different synthetic input tensors — all configurations must compete on the same data |
| `-z` (denormal as zero) | Enabled, the same in all configurations | Different calibration schemes can generate a different amount of denormal values; without fixing this, that difference creeps in as noise in the latency comparisons |
| `-D` (disable spinning) | Enabled | Thread spinning consumes real energy without useful computation, biasing the energy measurement upwards — more coherent with an energy-efficiency study to measure without that bias |
| `-x` / `-y` (intra/inter-op threads) | `-x` = no. of physical cores of the device (not hyperthreads), `-y 1` | No ambiguity of "auto-select" (value `0`), which would introduce the same software/system confounding variable that reviewers' criticism 9 points out. `-x` is documented as the `cpu_cores` field in `devices.yaml` (Section 10.5), fixed once per device |
| `-e` (execution provider) | Resolved from `backend` in `devices.yaml` (`cpu`→`-e cpu`, `cuda`→`-e cuda`) | Not fixed/hardcoded — needed to cover the Jetson's CPU/GPU combinations (Section 6.2) |
| `-I` (generate input tensor binding) | Enabled, always | **Technical requirement, not a methodological decision** — unlike the previous parameters (which affect what is measured or how it is interpreted), `-I` is mandatory because the `.onnx` files exported by Brevitas do not include an embedded `test_data_set_N/`; without this flag, the binary fails immediately with "no test data for model" on any artifact of the project. Confirmed necessary when verifying the real installation |
| `-m times -r N` | Configurable per invocation (`--repeats`, default 1000), not fixed in `devices.yaml` — reducible on slow devices (Nezha D1, VisionFive 2) if the total experiment time skyrockets | Rule: `N` must be enough for the loading+warm-up time to stay below ~5% of the total measured time (Section 6.5). The value of `-r` **actually requested** in each execution is persisted as the `requested_repeats` column in the results CSV (Section 4) — not just "documented" externally, it is traceable per row |

**`--self-test` mode (`runner_vit`, and a pattern to replicate in `runner_llm`/`power_logging`):** it runs once against a minimal synthetic artifact and confirms that the parser recognizes the real output format of the binary installed in that specific environment — to be run once per new device/environment (especially Jetson and RISC-V, where the compiled ONNX Runtime version may differ from the cluster's) before trusting real measurements there. Limited scope, explicitly communicated in its own output: it validates only the output format, it does not validate that the input parameters (`-x`, `-e`, `-r`, `-o`, etc.) are recognized by that version/build of the binary — an unsupported flag (e.g. `-e cuda` on a binary built without CUDA) fails in real execution, not in `--self-test`.

**The energy wrapper also measures the start-up of the process itself** (Python spawn, binary loading by the OS), not just the execution of the benchmark itself — it is the same source of bias as Section 6.5, slightly broadened. It requires no additional mitigation, it only reinforces that the number of internal repetitions (`-r`) set must be generous.

**Future-scope notes, not part of the current design:** `onnxruntime_perf_test` supports per-operator profiling (`-p profile_file`) and there are alternative execution providers for ARM (`xnnpack`) — see `future_work.md` for the full reasoning of when and how to consider enabling them.

## 6.7 F16 reference in LLM and artifact sizes (Jetson done; colleague's devices pending)

**Gap.** No LLM performance or energy measurement includes the F16: the 31 rows per configuration (11 for TinyLlama, 11 for Qwen2.5 and 9 for Ministral-3-3B) are quantized, and the F16 is only used as a perplexity reference. Without it, the speedup or the energy saving of quantization in LLM, which is part of RQ3a, cannot be expressed. The LLM rows also do not fill in `model_size_mb`.

**Decisions.**
- **F16 on the Jetson:** one row per model and configuration (3 models × 6 configurations = 18 rows), with the same `llama-bench` and the same N as the rest. It is the thoroughly characterized platform and has plenty of memory.
- **F16 on the colleague's devices:** measured where viable. Viability is confirmed by them, without setting a numeric rule: "possible" means that the whole F16 file fits in the available RAM (the `available` column of `free -h`, with a margin of ~1 GB), without using swap or re-reading the file from disk during the measurement (with `mmap`, the failure mode is re-reading the file, not the swap). What does not fit is declared ("F16 of X not measured on Y for memory reasons"), not forced.
- **Distribution:** a small supplementary package per model (the F16 GGUF and a one-row manifest), without regenerating the zips already distributed. Approximate sizes: 2.2 GB (TinyLlama), 3.1 GB (Qwen2.5) and 6.9 GB (Ministral-3-3B).
- **Artifact sizes:** a script, run on the cluster where the files are, that generates a separate CSV (`results/artifact_sizes.csv`) guided by the measurement manifests, so that only the artifacts measured on device enter (in ViT, the 10 of the deployment set; in LLM, the GGUFs of the 31 rows per model and the base F16 ones), not those of the ablation. It is joined to the results by `run_id` in the analysis; the results CSVs are not modified.

**Design of the Jetson's F16 rows** (fixed before measuring).
- **Rows:** one per model (`llm_<model>_..._f16`, with an identifier that does not collide with the existing ones), whose artifact is the cluster's base F16 GGUF (`artifacts/llm_base/<model>_f16.gguf`), without a quantization step or imatrix. In the 6 Jetson configurations (CPU and GPU × 15, 30 and 50 W): 18 rows, with a 3-row manifest per `run_batch` invocation (6 invocations; idle is measured once per invocation, as in the rest).
- **Protocol identical to that of the other LLM rows:** `llama-bench -p 128 -n 32 -r 30`, the same threads per mode on CPU and `-ngl` on GPU, `--with-power` and the same cooldown policy: `--temp-cooldown` without an explicit threshold (the `run_batch` default value applies), which is what is recorded in the Jetson's history for all LLM batches (`python3 -m scripts.run_batch --branch llm --manifest <manifest> --device-id <device> --with-power --repeats 30 --temp-cooldown`, in the order 50, 30 and 15 W and, in each mode, CPU and then GPU). The F16 GGUFs have to be sent to the Jetson (about 12 GB) with `rsync`.
- **Quantized references per format:** the baseline rows without imatrix of Q4_K_M (`phase1_baseline_default_q4km`), Q3_K_M (`phase3_baseline_q3_k`) and Q2_K (`phase3_baseline_q2_k`), and the legacy scheme Q4_0 (`phase2_quant_scheme_legacy_corpus_none_q4km`). The imatrix does not change the format and, within a session, the sibling rows vary ≤ 7 % in prefill and ≤ 3.7 % in energy, so they are not measured separately against F16.
- **Ratios, per model and configuration:** decode speed `tg_Q / tg_F16` and prefill speed `pp_Q / pp_F16` (≥ 1 means that quantization speeds up), energy per request `energy_per_inference_F16 / energy_per_inference_Q` (≥ 1 means saving; it is the main metric, and tokens per joule are secondary), and the size reduction `size_F16 / size_Q`, from the sizes CSV.
- **Reading, fixed before measuring:** since the F16 and the quantized ones are measured in different invocations, a ratio is only declared different from 1 if it falls outside [0.9, 1.1] (the uncertainty between LLM sessions is not characterized; within a session the repeatability is 1-1.4 % CV); if it falls inside, "sin diferencia apreciable" (no appreciable difference). No anchor rows are added.
- **Trade-off with accuracy:** each ratio is presented alongside the perplexity cost (`ΔPPL` against the recorded F16), which gives the relationship between saving and quality of each format.
- **Prior checks (with Code, without measuring):** that `run_batch` and the manifest accept a row whose artifact is the base F16 (unquantized), what value the format columns take, and where the file must be on the Jetson.
- **Limits:** a single session per row; the comparison depends on how optimized the F16 and quantized paths are on each CPU and GPU, which is part of what RQ3a wants to characterize.

**Status.** The sizes CSV is done (`results/artifact_sizes.csv`, 44 rows, no warnings; results in `findings_llm.md` and `findings_vit.md`, Section 7.10). **The 18 Jetson F16 rows are measured:** each `results_jetson_*.csv` went from 41 to 44 rows and the previous 41 remain identical field by field (checked with `check_csv_additions.py` on the Jetson copies, as with the `power_sampling_*.csv`); all with N=30, net energy and sampling above the minimum. The fixed-time fraction of the window of the F16 rows goes from 2.6 % (GPU 15 W) to 22.5 % (CPU 50 W); its effect on the energy ratios is bounded in `findings_llm.md` (section «F16 reference on the Jetson: speedup and energy saving from quantization (RQ3a)»). Analysis of the ratios against F16: done, in that same section. The reference F16 perplexities are in `results/raw/llm_perplexity_f16_reference.csv`. **Pending:** the colleague's devices, depending on whether they confirm RAM viability (without paging); what does not fit is declared.

## 6.8 LLM measurement command, default values and versions

- **Command** (`build_benchmark_command`, in `benchmarking_ondevice/runner_llm/runner.py`): `llama-bench -m <artifact> -o json -t <cpu_cores> -p 128 -n 32 -ngl <0 on CPU | 99 on GPU> -r <repetitions>`. The repetitions are passed with `--repeats 30` in all batches of the study (the code's default value is 5). Flash attention, `-b` and `-ub` are not set: the default values of the pinned version apply, which its JSON output gives as `flash_attn` = −1 (automatic), `n_batch` = 2048 and `n_ubatch` = 512.
- **Cooldown** (default values of `scripts/run_batch.py`): with `--temp-cooldown`, it waits until the temperature drops below 55 °C, with a cap of 120 s and readings every 5 s; without it, a fixed 30 s. On the Jetson `--temp-cooldown` was used with those values in all batches; it is only verified that the wait never activated in the original ViT batch.
- **Versions:** llama.cpp `b95502ba9` (tag `b10516`) unmodified. ONNX Runtime v1.23.2 (commit `a83fc4d5`), the same version in all environments. On the Jetson two local patches are applied, exclusive to that device and documented in `docs/onnxruntime_patches.md`.

## 7. Incremental validation (smoke test)

Before launching the complete manifest (24+ runs per branch), it is advisable to first run only Phase 1 (baseline) of one model on one device per available architecture (x86, one ARM, and the RISC-V if the spike of Section 6.1 confirms it viable), to check that the whole chain (quantization → export → shared artifact → benchmarking → power logging → writing in `results/raw/` with the correct `device`) works end to end before committing computing time on the rest of the manifest. This smoke test is also the natural moment to decide the contingency of Section 6.1.

## 8. Correspondence with the phases of the protocols

| Phase (protocol) | Module(s) involved |
|---|---|
| Phase 1 — Baseline | `configs/` (one entry) → `quantization/` → `artifacts/` → `benchmarking_ondevice/` |
| Phase 2 — OFAT | `configs/` (programmatic generation) → same flow, looping over the manifest |
| Phase 3 — Combined vs. aggressive | Requires `analysis/` to first identify the Phase 2 "winner" before generating the Phase 3 configs — it is the only point where the manifest depends on previous results, it is not 100% static from the start |
| Phase 4 — Descriptive, causal analysis and interventions | `analysis/phase4_vit/` and `analysis/phase4_llm/`, independent of the rest |

---

**Note:** Phase 3 introduces a real sequential dependency in the pipeline (you cannot generate its manifest until you have the Phase 2 results). It is worth deciding explicitly whether this is solved with manual intervention (reviewing Phase 2 results, hand-writing the Phase 3 config) or with an automatic step that selects the "winner" according to a defined rule (e.g. highest accuracy relative to the baseline). The latter is more reproducible but requires fixing the selection criterion beforehand.

## 9. The sequential dependency of Phase 3, in detail

### 9.1 What "combining the winners" assumes

Phase 3 combines the best level of each Phase 2 factor (best algorithm + best granularity + best symmetry + best calibration set size, in the case of ViT) into a single configuration. This **assumes that the factors do not interact with each other** — that the effect of each is independent of the level of the others. It is a reasonable assumption for an OFAT design (which by construction cannot detect interactions; that would require a full factorial), but it must be **explicitly declared as a limitation** in the paper, not implying that the "optimal combination" is mathematically guaranteed.

### 9.2 The risk of selecting a "winner" that is only one because of noise

With only 3 repetitions per configuration (Section 6 of both protocols), a small difference between two levels of a factor may not be a real difference if the standard deviation between repetitions is of the same order. A naive `argmax` over the mean runs the risk of building Phase 3 on a "winner" that is actually sampling noise.

### 9.3 Proposed selection rule

Without going as far as a formal statistical test (unreliable with only 3 repetitions), a simple rule with a safeguard is proposed:

1. For each factor, compute the mean of `accuracy_value` of each level across its repetitions (`repeat_id`).
2. Select the level that **improves on the baseline in the desired direction of the branch's metric** (higher accuracy in ViT; lower perplexity in LLM — do not assume "higher is better" literally, the rule must generalize according to the sense of the metric), **only if that improvement exceeds the combined standard deviation of baseline and candidate** (simplified minimum margin, not a formal test). **Special case:** if the baseline is deterministic (n=1, no standard deviation — as happens in the 6 baseline/`legacy_corpus_none` rows of the LLM branch that do not depend on calibration sampling, Section 10.4), the margin is computed only with the candidate's standard deviation, not of "both".
3. If no level exceeds that margin, **keep the baseline's level** for that factor — "no clear winner" is a valid result and a choice must not be forced.

This rule guarantees that the combined Phase 3 configuration is never "worse by chance" than the baseline in any individual factor, and it is fully auditable: anyone can review the code and verify exactly which criterion was applied.

### 9.4 Location in the architecture

This selection fits as a function inside `analysis/`, which takes as input the Phase 2 table from `results/raw/` (using the `factor_varied`, `factor_level`, `accuracy_value` and `repeat_id` columns of the common schema, Section 4) and returns the combined configuration that feeds the generation of the Phase 3 manifest. By relying on the common schema, the same function serves ViT and LLM without duplicating logic — it is the only point of the project where `analysis/` feeds back into `configs/`, instead of being a final consumer.

## 10. Conventions closed before coding

### 10.1 Concrete skeleton of the manifest

The manifest defines **quantization configurations**, not executions on a device — `device` and `repeat_id` are added later, in the benchmarking phase, not in the manifest. **`run_id` convention:** all rows keep the same fixed shape (`{family}_{model}_{phase}_{factor}_{level}_{bitwidth}`), with no special cases — the baseline rows use the literal placeholder `default` in the `factor`/`level` segment instead of omitting those segments, so that the `run_id` is always parseable with the same number of fields. Real example of two entries (one Phase 1, one Phase 2) per branch:

`configs/vit_experiment_manifest.yaml`
```yaml
- run_id: vit_deit_tiny_phase1_baseline_default_int8
  model_family: ViT
  model_name: deit_tiny
  phase: 1_baseline
  factor_varied: none
  factor_level: none
  calibration_algorithm: minmax
  granularity: per_tensor
  symmetry: symmetric
  calib_set_size: 128
  bitwidth: int8

- run_id: vit_deit_tiny_phase2_algorithm_percentile_int8
  model_family: ViT
  model_name: deit_tiny
  phase: 2_ofat
  factor_varied: calibration_algorithm
  factor_level: percentile
  calibration_algorithm: percentile
  granularity: per_tensor
  symmetry: symmetric
  calib_set_size: 128
  bitwidth: int8
```

`configs/llm_experiment_manifest.yaml`
```yaml
- run_id: llm_tinyllama_1_1b_phase1_baseline_default_q4km
  model_family: LLM
  model_name: tinyllama_1_1b
  phase: 1_baseline
  factor_varied: none
  factor_level: none
  calibration_corpus: none
  quant_scheme: k_quant
  calib_set_size: 0
  bitwidth: Q4_K_M

- run_id: llm_tinyllama_phase2_corpus_wikitext_q4kM
  model_family: LLM
  model_name: tinyllama_1_1b
  phase: 2_ofat
  factor_varied: calibration_corpus
  factor_level: wikitext2
  calibration_corpus: wikitext2
  quant_scheme: k_quant
  calib_set_size: 200
  bitwidth: Q4_K_M
```

All the fields not applicable to a branch (e.g. `granularity` in LLM) simply do not appear in that entry — the common results schema (Section 4) already contemplates them being empty columns, there is no need to force them.

### 10.2 `run_id` convention (manifest) vs. result row identifier

Two different levels, so as not to mix "which quantization configuration it is" with "which specific execution produced this data point":

- **`run_id`** (lives in the manifest, identifies a unique quantization configuration): `{family}_{model}_{phase}_{factor}_{level}_{bitwidth}`
- **`result_id`** (lives in `results/raw/`, identifies a specific row/execution): `{run_id}__{device}__rep{repeat_id}`

A single `run_id` generates a single quantized artifact, but multiple result rows (one per device × repetition). This is consistent with `artifacts/` storing one file per `run_id`, not per `result_id`.

### 10.3 Results storage format

Given the manageable volume (hundreds of rows, not millions), **CSV** instead of parquet/database: it is directly readable, easy to inspect by hand, and simple to merge between two people using git (merge conflicts in a single-table CSV are manageable; in parquet they cannot even be seen). Proposed convention: each person/machine writes to their own file (`results/raw/results_<device_or_person>.csv`), and a single script in `results/` concatenates them into `results/results_combined.csv`, validating that `result_id` is not repeated between files (if it is, someone has re-run something by mistake or there is a device name collision).

### 10.4 Flow for sharing quantized artifacts

Quantization is run on the **university cluster** (the same resource already used in the original paper), which meets the memory/compute requirements and where a proven workflow already exists for sharing models and code among the team. The `.onnx`/`.gguf` files **are not versioned in git** (they are large binaries, bad repository practice); they are deposited in the cluster's own space with the structure `artifacts/vit/{run_id}.onnx` and `artifacts/llm/{run_id}.gguf` — the file name is literally the `run_id`, so no separate mapping/index is needed: whoever runs on an edge device only needs to know which `run_id` is theirs and copy the file with that name from the cluster.

**Exception to "one artifact per `run_id`": configurations with random calibration sampling.** The variance required by Section 6.1 (minimum repetitions due to sampling variance) is variance of **which artifact is generated**, not measurement variance on a fixed artifact — so the repetitions must be different artifacts, not repeated measurements of the same file. Rule:

- **Rows without random sampling** (LLM: `calibration_corpus=none`, i.e. the Phase 1 baseline and the `legacy_corpus_none` row; it does not apply to ViT, where every configuration uses a calibration set): quantized **only once**, artifact `{run_id}.gguf`/`{run_id}.onnx`, without suffix.
- **Rest of the rows** (they depend on a sampled corpus/calibration set): **3 independent artifacts** are generated, one per repetition, with a seed derived deterministically and different per repetition (`seed = 42 + (repeat_id - 1)`, not a fixed global seed). Name: `{run_id}__rep{repeat_id}.gguf`/`.onnx`.
- Volume consequence: in LLM, 6 deterministic rows (2 per model × 3 models: baseline and `legacy_corpus_none`) + 15 stochastic rows × 3 repetitions (5 per model × 3 models) = **51 artifacts**, not 21. In ViT (when Brevitas is implemented), the 24 rows × 3 repetitions = **72 artifacts**, with no possible deterministic exception.
- **These extra artifacts only affect the cluster**, not the edge devices: the corpus sampling variance can only manifest in accuracy/perplexity (an offline metric, computed on the cluster); performance and energy depend on the structure of the artifact (scheme, bit-width), not on which specific corpus calibrated it. **A single artifact per configuration** is sent to the edge (by convention, the one with `repeat_id=1`), never the 3.

**Software environment on the cluster:** the system CUDA/driver is managed via `module load` (responsibility of the cluster, do not touch); everything else (PyTorch, Brevitas, Python dependencies) via conda/venv, responsibility of the project.

**Specific GPU and version chain (confirmed):** the cluster offers a **Tesla V100** (Volta architecture, compute capability 7.0), with driver `525.147.05`, whose CUDA runtime ceiling is **12.0** (`nvidia-smi` → "CUDA Version: 12.0"). This is a real constraint to be fixed explicitly, not just a compatibility detail:

- Starting with PyTorch 2.11, the prebuilt builds for CUDA 12.8/12.9 **stopped including support for Volta**; CUDA 13.x removed it completely. Installing "the latest version" of PyTorch without specifying the CUDA index will, with high probability, give a build without kernels for the V100.
- **Version pinned for `environment.yml`:** **PyTorch 2.5.1 + `cu118` wheel** (`pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118`). Justification of this specific version, not just of the CUDA index:
  - It is the highest version with confirmed availability of the `cu118` wheel as a stable combination documented in the community — later PyTorch versions progressively cut back the availability of `cu118` builds.
  - It sits comfortably within the range that **Brevitas declares to officially support (PyTorch >= 1.12, <= 2.8; later versions "untested")** — that is, the real ceiling for this project is not imposed by the V100's driver, but by Brevitas itself, and both constraints point to the same band of versions.
  - Clearly below the driver ceiling (12.0), with Volta support guaranteed (the cut of V100 support in PyTorch did not arrive until the CUDA 12.8+ builds, far above this band), avoiding depending on minor-version compatibility between CUDA 12.0 (driver) and 12.1+ (runtime), which is not uniform across all libraries (cuBLAS/cuDNN in particular).
  - If a specific problem arises with this combination during implementation, it is the first variable to review before any other — record the reason for the choice in the `environment.yml` itself so as not to have to rediscover it.
- Brevitas must be installed on that same PyTorch build (2.5.1+cu118) — it follows recent versions but does not require the latest.
- If at some point the cluster's GPU is used to accelerate the generation of llama.cpp's `imatrix`, the precompiled binaries have also dropped Volta — it would have to be built from source with `sm_70` in the architecture flag. It is not necessary for the LLM quantization itself (a CPU process).

The complete environment is pinned in an `environment.yml` (or `requirements.txt`) **versioned in the repository**, not installed "by hand" by each person separately — with two people working on the project, this avoids version divergences between whoever quantizes and whoever later replicates/verifies results, and reinforces the same reproducibility argument already used in the original paper.

### 10.5 `devices.yaml` schema

Reference catalog that validates the `device` column of the results (Section 10.3) and prevents the same device from being described differently by each person. The energy measurement method **does vary by device type** (Section 6.4: the Jetson uses its integrated INA3221 sensor via its own module; the rest of the devices via AccelPowerCape) — the `power_measurement` field documents which one corresponds to each entry, even though the output is homogenized by the common contract:

```yaml
- device_id: jetson_cpu_50w
  display_name: "NVIDIA Jetson AGX Orin (CPU, 50W)"
  architecture: ARM
  backend: cpu
  power_mode_w: 50
  ram_gb: 32
  cpu_cores: 12
  power_measurement: jetson_ina3221
  min_uptime_s: 600  # wait after boot before the first row (see the `min_uptime_s` field below)
  added_by: nicolas

- device_id: jetson_gpu_50w
  display_name: "NVIDIA Jetson AGX Orin (GPU, 50W)"
  architecture: ARM
  backend: cuda
  power_mode_w: 50
  ram_gb: 32
  cpu_cores: 12
  n_gpu_layers: 99  # TODO: confirm the real value when deploying on the device (no. of model layers to offload to GPU in llama-bench/-ngl; only applies to backend: cuda entries, see Section 6.6)
  power_measurement: jetson_ina3221
  min_uptime_s: 600  # wait after boot before the first row (see the `min_uptime_s` field below)
  added_by: nicolas

- device_id: rpi_cm4
  display_name: "Raspberry Pi CM4"
  architecture: ARM
  backend: cpu
  power_mode_w: null
  ram_gb: 8
  cpu_cores: 4
  power_measurement: accelpowercape
  added_by: daniel
  notes: ""  # optional, free text
```

**`n_gpu_layers` field (added after building `runner_llm`):** only relevant in `backend: cuda` entries — number of model layers to offload to GPU, resolved into the `-ngl` flag of `llama-bench` (Section 6.6). It is never left at the tool's default (`-1`, "use GPU if there is one") — if a `cuda` entry does not have this field, `runner_llm` fails explicitly instead of assuming a value. It does not apply to `backend: cpu` entries (like `jetson_cpu_50w` above, which does not need it).

**`notes` field (optional, free text, added after the delivery to edge devices):** the loader (`runner_vit/devices.py`, `runner_llm/devices.py`) tolerates any unrecognized field without failing (verified with a real load), so `notes` requires no code change — it is documented here so that it is in the canonical schema, not just mentioned in passing in the measurement instructions. Intended use: record deviations from a device's reference configuration (e.g. an ONNX Runtime version different from the one fixed as reference, Section 6.6) that would otherwise go untraced — relevant because, as that section already points out, two devices with different tool versions contaminate the comparability of their results if it is not recorded.

**`min_uptime_s` field (optional; if missing, 0):** seconds of system uptime that `run_batch.py` waits, since boot, before measuring the first real row of the batch. **Reason:** on the Jetson's CPU a performance step was detected (+13 % to +24 % in decode on Qwen depending on the power mode; +33 % on TinyLlama at 50 W) between 6 and 7.5 minutes after boot or after the power mode change, which stays stable afterwards. It is neither thermal nor due to CPU/GPU/memory frequencies, and its cause is not identified (`findings_llm.md`); measurements made before that point come out systematically slower. The Jetson entries carry `min_uptime_s: 600`, a margin over the 6.2-7.3 min observed. **Why it goes in `devices.yaml` and not as a command-line flag:** it is a property of the device, not of the invocation — it is recorded next to its entry, it is the same for anyone measuring on it and it does not depend on someone remembering to pass it. It replaces the former `--min-uptime-seconds` flag, which no longer exists. A device without the field waits for nothing, so the rest of the fleet does not change; that is a decision about time cost, not a guarantee that the effect does not exist outside the Jetson (it has not been characterized there). A badly written value makes the batch fail, instead of being silently ignored and measuring without the wait.

**Where the real catalog lives:** `configs/devices.yaml` does not travel in any zip; each person maintains their own, created from the template `configs/devices.example.yaml` (schema commented field by field, the two Jetson entries and the `rpi_cm4` example), which only the ViT code zip and the deployment one carry, not the LLM ones. The loaders (`runner_vit/devices.py`, `runner_llm/devices.py`, `run_batch.py`) read only `devices.yaml` and, if it is missing, fail with `No device catalog found` instead of falling back to the template, so that no measurement is labeled with an example entry. A test ensures that none of them reads the template.

When either of the two adds a new device (not necessarily one of the 7 of the original paper), they only need to fill in this template with one more entry, indicating the real measurement method of that device — the field exists precisely to document that variation, not to force a single value.
