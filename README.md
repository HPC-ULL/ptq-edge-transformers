# From Calibration to Deployment: Accuracy and Energy of Post-Training Quantized Transformers on Heterogeneous Edge Devices

Code, configuration, documentation and result files of a study on post-training quantization (PTQ) of transformers for edge devices. The study asks (1) how the calibration strategy determines predictive quality, (2) which distribution properties are associated with that sensitivity, and (3) what real speedup and energy saving are obtained when deploying with general-purpose toolchains on heterogeneous hardware. It is not a hardware benchmarking study.

- **Vision models (Brevitas → ONNX, QCDQ):** DeiT-Tiny (`deit_tiny_patch16_224.fb_in1k`), MobileViT-S (`mobilevit_s.cvnets_in1k`), EfficientViT-B1 (`efficientvit_b1.r224_in1k`).
- **Language models (llama.cpp / GGUF):** TinyLlama-1.1B (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`), Qwen2.5-1.5B (`Qwen/Qwen2.5-1.5B-Instruct`), Ministral-3-3B (`mistralai/Ministral-3-3B-Instruct-2512-BF16`).

Salamandra-2B was evaluated at an early stage and discarded; it is not part of the paper.


## Repository structure

```
analysis/                 phase 3 winner selection, phase 4 analyses (phase4_vit/, phase4_llm/), deployment summaries
benchmarking_ondevice/    runner_vit/ (onnxruntime_perf_test), runner_llm/ (llama-bench), power_logging/ (energy measurement)
configs/                  experiment manifests (generation + YAML), device catalog (`devices.example.yaml` template; `devices.yaml` with the dev/test entry used by the tests)
data_prep/                preparation of ImageNet-1K, WikiText-2 and the Dolly-15k domain corpus
docs/                     protocols, findings, technical architecture, measurement instructions, ONNX Runtime patches, glossary
evaluation_offline/       vit_accuracy/ (top-1/top-5), llm_perplexity/ (WikiText-2 test perplexity)
export/                   Brevitas → ONNX export (vit_to_onnx/)
plots/                    figure scripts (scripts/), shared style, summary CSVs and figure data (out/)
quantization/             vit_brevitas/ (calibration + quantization), llm_llamacpp/ (imatrix + llama-quantize), investigations/
results/                  result files that back the paper and the supplementary material (see below)
scripts/                  run_batch.py (on-device driver), result combination, ratios, McNemar, diagnostics
tests/                    unit tests (pytest)
requirements_edge.txt     minimal dependencies for edge devices
verify_edge_bundle.py, verify_results_integrity.py   consistency checks
```

Documentation (in `docs/`): [`calibration_protocol_vit.md`](docs/calibration_protocol_vit.md), [`calibration_protocol_llm.md`](docs/calibration_protocol_llm.md), [`findings_vit.md`](docs/findings_vit.md), [`findings_llm.md`](docs/findings_llm.md), [`technical_architecture.md`](docs/technical_architecture.md), [`edge_measurement_instructions.md`](docs/edge_measurement_instructions.md), [`future_work.md`](docs/future_work.md), [`onnxruntime_patches.md`](docs/onnxruntime_patches.md) and [`GLOSSARY.md`](docs/GLOSSARY.md). The documents are English translations of the original Spanish working documents. Spanish strings that are data values (labels written into result files) are explained in the glossary. `docs/` is the record of the study; the hypothesis numbering of the two protocols differs from the paper's, see the note at the top of each protocol.

Other documentation: [`quantization/vit_brevitas/investigations/brevitas_issue_klminimizerthreshold_parity_bug.md`](quantization/vit_brevitas/investigations/brevitas_issue_klminimizerthreshold_parity_bug.md). There was no other partial README in subdirectories.

## Requirements and versions

| Component | Version |
|---|---|
| Python | 3.10.12 |
| timm | 1.0.28 |
| PyTorch | 2.5.1 (CUDA 11.8, `cu118`) |
| Brevitas | 0.13.0 |
| ONNX opset | 13 |
| datasets | 5.0.1 |
| huggingface_hub | 0.36.2 |
| ONNX Runtime | 1.23.2 (`a83fc4d5`), plus two local patches on the Jetson |
| llama.cpp | `b95502ba9` (tag `b10516`), unmodified |
| Jetson | JetPack 6.2.2 / CUDA 12.6 |

The two local ONNX Runtime patches (static linking + CUDA, and an out-of-bounds write with offline cores) are documented in [`docs/onnxruntime_patches.md`](docs/onnxruntime_patches.md); both are needed to reproduce the on-device measurements on a Jetson. Environment setup and the pinned result `requirements-lock.txt` are described below. Edge devices only need `requirements_edge.txt` (PyYAML).

### Pinned dependencies

`requirements-lock.txt` is the unmodified output of `pip freeze` of the cluster environment (Python 3.10.12) in which the models were calibrated, quantized and evaluated. It lists **all** the packages of that environment, not a minimal list. Install with:

```bash
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

The edge devices only run the Python runners; the measurements depend on `onnxruntime_perf_test` and llama.cpp (`llama-bench`), built as described in this README and in [`docs/onnxruntime_patches.md`](docs/onnxruntime_patches.md).

### Creating the Python environment (without the lock file)

```bash
conda create -n ptq_calib_env python=3.10.12 -y && conda activate ptq_calib_env
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install brevitas timm onnx onnxruntime onnxoptimizer datasets huggingface_hub scipy
```

(`onnxoptimizer` is required by the Brevitas ONNX exporter; `scipy` by the ViT phase 4 correlations; `datasets` and `huggingface_hub` by `data_prep/`.) The unpinned versions of the last line are what `requirements-lock.txt` fixes.

### Building llama.cpp

**CPU only** (used on the cluster, where quantization, imatrix and perplexity need no GPU, and on the N150, the RK1, the CM4 and the VisionFive 2). Release build with `GGML_NATIVE=ON`:

```bash
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp && git checkout b95502ba9
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --config Release -j"$(nproc)"
export LLAMA_CPP_DIR=$PWD
```

**Jetson AGX Orin** (JetPack 6.2.2, CUDA 12.6, `nvcc` V12.6.85), Release build with `GGML_NATIVE=ON`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FORCE_MMQ=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF \
  -DGGML_OPENMP=ON -DGGML_LLAMAFILE=ON \
  -DGGML_CPU_REPACK=ON -DGGML_CPU_KLEIDIAI=OFF
cmake --build build --config Release -j"$(nproc)"
```

If the system compiler is too old (gcc < 9 failed to link `std::filesystem`), install a modern one and pass it to CMake:

```bash
conda install -y -c conda-forge "gxx_linux-64>=11" "gcc_linux-64>=11" cmake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(find "$CONDA_PREFIX/bin" -maxdepth 1 -name '*-gcc' | head -n1)" \
  -DCMAKE_CXX_COMPILER="$(find "$CONDA_PREFIX/bin" -maxdepth 1 -name '*-g++' | head -n1)"
```

`convert_hf_to_gguf.py` needs torch to load the Hugging Face weights; install its requirements without letting pip replace the pinned torch:

```bash
printf 'torch==2.5.1+cu118\ntorchvision==0.20.1+cu118\n' > torch_constraint.txt
python -m pip install -r requirements/requirements-convert_hf_to_gguf.txt --constraint torch_constraint.txt   # or requirements.txt, whichever this checkout has
```

### Building `onnxruntime_perf_test` (ONNX Runtime v1.23.2, CPU only)

Clone the tag that matches the installed `onnxruntime` wheel. The two local patches of [`docs/onnxruntime_patches.md`](docs/onnxruntime_patches.md) are **required on the Jetson**: the first to build in static mode with CUDA, the second to run in the power modes with cores switched off (15 and 30 W). They are not required on the other platforms, although applying them does not change the result. Apply them before building:

```bash
git clone --depth 1 --branch v1.23.2 --recursive --shallow-submodules https://github.com/microsoft/onnxruntime.git
cd onnxruntime
# (apply the patches of docs/onnxruntime_patches.md here)
./build.sh --config Release --build_dir build --parallel --compile_no_warning_as_error \
  --cmake_extra_defines "CMAKE_CXX_FLAGS=-include cstdint" --update --build
find build -maxdepth 3 -name onnxruntime_perf_test -type f      # the binary
export ONNXRUNTIME_DIR=$PWD
```

`-include cstdint` is needed with recent GCC (GCC >= 13 fails with "'uint32_t' does not name a type" in `onnxruntime/core/common/semver.h`). If the build runs out of memory while linking, use `--parallel <N>` (e.g. 4) instead of a bare `--parallel`. If you built with the conda-forge compiler above, run the build as `CC=<gcc> CXX=<g++> ./build.sh ...`. The edge-device build (`--build_shared_lib --parallel --skip_tests`) is in [`docs/edge_measurement_instructions.md`](docs/edge_measurement_instructions.md).

Environment variables read by the code: `DATASETS_ROOT` (default `~/datasets`), `LLAMA_CPP_DIR`, `ONNXRUNTIME_DIR`.

## Reproducing each phase

Design: phase 1 baseline; phase 2 one factor at a time (OFAT, 3 repetitions); phase 3 winning combination against an aggressive extreme (INT6 and INT4 for ViT; Q3_K_M and Q2_K for LLM); phase 4 distribution analysis. See the protocols. Run everything from the repository root. Heavy steps (quantization, accuracy evaluation, phase 4) were run on a GPU cluster; on-device steps on the edge devices.

### 1. Data preparation

```bash
python -m data_prep prepare-all        # also: imagenet-calibration, imagenet-eval, wikitext2, domain-corpus, info (prepare-all covers all three datasets)
```

### 2. Manifests, calibration and quantization

```bash
python -m configs vit                  # configs/vit_experiment_manifest.yaml (phases 1-2); also: llm
python -m quantization.vit_brevitas --all            # ViT: 24 rows x 3 repetitions (--dry-run, --force available)
python -m quantization.llm_llamacpp --all            # LLM: imatrix generation + llama-quantize
```

### 3. Accuracy / perplexity evaluation and phase 3 selection

```bash
python -m evaluation_offline.vit_accuracy --all
python -m evaluation_offline.llm_perplexity --all
python -m analysis.phase3_selection                  # writes results/phase3_selection_report.txt
python -m configs vit-phase3                         # also: llm-phase3
python -m quantization.vit_brevitas --all --phase3
python -m quantization.llm_llamacpp --all --phase3
python -m evaluation_offline.vit_accuracy --all --phase3
```

McNemar tests (ViT): per-image predictions first, then the tests.

```bash
python quantization/vit_brevitas/investigations/per_image_predictions.py
python scripts/mcnemar_vit.py --list
python scripts/mcnemar_vit.py --run
```

### 4. Distribution analysis (phase 4)

Every module below has a CLI (`argparse`, run with `python -m`); `--dry-run` checks inputs and prints the plan where available. `analysis/phase4_llm/summary.py` is a library module (per-model summaries and bootstrap, no CLI) used by `weights --summarize`.

ViT (`analysis/phase4_vit/`):

```bash
python -m analysis.phase4_vit.inventory
python -m analysis.phase4_vit.level1 --dry-run
python -m analysis.phase4_vit.level1
python -m analysis.phase4_vit.level2 --models deit_tiny --smoke
python -m analysis.phase4_vit.level2
python -m analysis.phase4_vit.stability --dry-run
python -m analysis.phase4_vit.stability
python -m analysis.phase4_vit.attribution --models mobilevit --dry-run
python -m analysis.phase4_vit.attribution --models mobilevit
python -m analysis.phase4_vit.attribution --models efficientvit_b1
python -m analysis.phase4_vit.replication --dry-run
python -m analysis.phase4_vit.replication
python -m analysis.phase4_vit.replication_deit --dry-run
python -m analysis.phase4_vit.replication_deit
python -m analysis.phase4_vit.replication_random --dry-run
python -m analysis.phase4_vit.replication_random
python -m analysis.phase4_vit.calibration_attribution --dry-run
python -m analysis.phase4_vit.calibration_attribution --control-rep1-only
python -m analysis.phase4_vit.calibration_attribution
```

LLM (`analysis/phase4_llm/`):

```bash
python3 -m analysis.phase4_llm.sets
python3 -m analysis.phase4_llm.activations
python -m analysis.phase4_llm.weights --dry-run
python -m analysis.phase4_llm.weights
python3 -m analysis.phase4_llm.weights --summarize
python3 -m analysis.phase4_llm.robustness
python -m analysis.phase4_llm.intervention --model tinyllama_1_1b --dry-run
python -m analysis.phase4_llm.intervention --model tinyllama_1_1b --controls
python -m analysis.phase4_llm.intervention --model tinyllama_1_1b --rep 2
python -m analysis.phase4_llm.intervention --model tinyllama_1_1b --extra
python3 -m analysis.phase4_llm.intervention --analyze
python -m analysis.phase4_llm.ceiling --dry-run
python -m analysis.phase4_llm.ceiling --controls
python -m analysis.phase4_llm.ceiling --format Q2_K --rep 1 2 3
python -m analysis.phase4_llm.ceiling --format Q3_K_M --rep 1 2 3
python3 -m analysis.phase4_llm.ceiling --analyze
python -m analysis.phase4_llm.f16_reference --dry-run
python -m analysis.phase4_llm.f16_reference --model tinyllama_1_1b
python -m analysis.phase4_llm.f16_reference --csv
```

Options common to several modules: `--models` (ViT: `deit_tiny`, `mobilevit`, `efficientvit_b1`; LLM: `tinyllama_1_1b`, `qwen2_5_1_5b`, `ministral_3_3b`), `--out-dir`, `--threads` (LLM evaluations). Each script's docstring lists its inputs and outputs. `intervention` and `f16_reference` take `--model` one model at a time (repeat the command for each model); `--rep` selects the calibration repetition.

### 5. Deployment (on-device latency and energy)

Build the ViT deployment set and the edge packages (cluster):

```bash
python -m quantization.vit_brevitas.deployment --dry-run
python -m quantization.vit_brevitas.deployment
python verify_edge_bundle.py
```

On each edge device (see [`docs/edge_measurement_instructions.md`](docs/edge_measurement_instructions.md); create `configs/devices.yaml` from `configs/devices.example.yaml` first, check power mode and uptime):

```bash
python -m benchmarking_ondevice.runner_vit --self-test --device-id <device_id>
python -m scripts.run_batch --branch vit --manifest run_manifest_code_vit.csv --device-id <device_id> --with-power --temp-cooldown
python -m scripts.run_batch --branch llm --manifest <run_manifest_llm_model.csv> --device-id <device_id> --with-power --repeats 30 --temp-cooldown
```

(`--repeats 30` is required for LLM energy measurements; the code default is 5. The power-meter implementation is selected with `--power-impl`.) Combine the per-device CSVs and build the summaries:

```bash
python scripts/combine_edge_results.py
python3 -m analysis.vit_deployment_summary
python3 -m analysis.llm_deployment_summary
python3 scripts/f16_ratios.py
python3 scripts/gpu_vs_cpu.py                        # no arguments; run from the repository root
python scripts/profile_scaling.py analyze results/investigations/vit/thread_scaling/jetson_cpu_50w.json
```

Figures: `python plots/scripts/fig_vit_phase3.py`, `fig_vit_phase4_recovery.py`, `fig_vit_deployment.py`, `fig_llm_deployment.py`, `fig_calibration_concept.py` (needs `plots/out/calibration_concept_data.npz`, written by `capture_calibration_concept.py` on the cluster), and `fig_pipeline.tex` (see its header for the `pdflatex` command).

Tests: `python -m pytest tests` (907 pass and 20 are skipped without torch/onnx/brevitas or without the result files not included here).

## Included results

| Path | Content |
|---|---|
| `results/raw/` | `vit_accuracy_offline.csv`, `vit_accuracy_fp32_reference.csv`, `llm_perplexity_offline.csv`, `llm_perplexity_f16_reference.csv`, and the 10 per-device `results_<device>.csv` (inputs of `combine_edge_results.py`, `f16_ratios.py` and `gpu_vs_cpu.py`) |
| `results/artifact_sizes.csv` | on-disk sizes of the measured artifacts |
| `results/combined/edge_results_combined.csv` | per-device performance and energy rows (one measurement per row) |
| `plots/out/vit_deployment_summary.csv`, `plots/out/llm_deployment_summary.csv` | deployment summaries |
| `results/investigations/vit/` | `phase4/`, `mcnemar/`, `deployment_gate/`, `output_quant_gate/`, `fusion_probes/` (JSON/CSV only), `thread_scaling/` |
| `results/investigations/llm/` | `phase4/`, `f16_ratios/`, `gpu_vs_cpu/` |
| `results/diagnostics/` | `dispersion_*`, `thermal_series_mobilevit_50w_session1`/`session2`, `n_sweep_llm/`, `n_sweep_llm_cold3/` |

Absolute paths in result CSV/JSON files were replaced by paths relative to the repository root (only the path prefix changed). Logs, model files and figures are not included. `scripts/combine_edge_results.py`, `scripts/f16_ratios.py` and `scripts/gpu_vs_cpu.py` re-run from this repository;

## Data not included and how to obtain it

- **Quantized artifacts** (`.onnx`, `.gguf`, imatrix `.dat`): regenerable with the commands above. Not versioned.
- **ImageNet-1K:** `ILSVRC/imagenet-1k`, revision `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`. Access is gated; accept the provider's terms on Hugging Face and authenticate (`hf auth login`). The evaluation subset is the first two images of each class of the validation split (2000 images); the calibration bank is 10,000 images of the train split.
- **WikiText-2:** `Salesforce/wikitext`, config `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3` (train for the imatrix corpus, test for perplexity). **Dolly-15k:** `databricks/databricks-dolly-15k`, revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`.
- **Checkpoints (Hugging Face):** the timm weights named above (downloaded by timm) and these LLM repositories, downloaded by `quantization/llm_llamacpp`:
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` @ `fe8a4ea1ffedaf415f4da2f062534de366a451e6`
  - `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
  - `mistralai/Ministral-3-3B-Instruct-2512-BF16` @ `b6d637bef2393152b3da2b2fde72eecdee30557e`

  These are the revisions used in the study. The code does not pin them (only the ImageNet revision is pinned in `data_prep/imagenet.py`).

## Tables and figures of the paper and the result files they come from

Table and figure numbers refer to the submitted version of the paper and of the supplementary material.

**Paper**

| Table / figure | Content | Files |
|---|---|---|
| Table I | Models and FP32/F16 references | `results/raw/vit_accuracy_fp32_reference.csv`, `results/raw/llm_perplexity_f16_reference.csv` |
| Tables II and III | Factors, hypotheses | `docs/calibration_protocol_vit.md`, `docs/calibration_protocol_llm.md` |
| Table IV | Devices | `configs/devices.example.yaml` and Table S7 |
| Table V and Fig. 2 | Vision ablation | `results/raw/vit_accuracy_offline.csv`; significance marks: `results/investigations/vit/mcnemar/mcnemar_vit.csv` |
| Table VI | Language ablation | `results/raw/llm_perplexity_offline.csv`, `results/raw/llm_perplexity_f16_reference.csv` |
| Table VII | Ceiling effect | `results/investigations/llm/phase4/ceiling/analysis.csv` |
| Table VIII | Deployment accuracy | `results/investigations/vit/deployment_gate/`, `results/investigations/vit/output_quant_gate/` |
| Table IX | Sizes | `results/artifact_sizes.csv` |
| Fig. 3 | Targeted interventions | `results/investigations/vit/phase4/level2/` (`evaluations.csv` and `replication/`) |
| Fig. 4 | Vision deployment | `plots/out/vit_deployment_summary.csv`, built from `results/combined/edge_results_combined.csv` |
| Fig. 5 | Language deployment | `plots/out/llm_deployment_summary.csv`, `results/investigations/llm/f16_ratios/` |

**Supplementary material**

| Item | Content | Files |
|---|---|---|
| S1-S5 | Complete ablation | `results/raw/vit_accuracy_offline.csv`, `results/raw/vit_accuracy_fp32_reference.csv`, `results/raw/llm_perplexity_offline.csv`, `results/raw/llm_perplexity_f16_reference.csv`; Table S5 also `results/investigations/llm/phase4/ceiling/analysis.csv` |
| S6 | McNemar | `results/investigations/vit/mcnemar/mcnemar_vit.csv`, computed from `results/investigations/vit/per_image_predictions/` |
| S7-S8 | Measurement settings and windows | `plots/out/vit_deployment_summary.csv`, `plots/out/llm_deployment_summary.csv` |
| Section S2 text | Sessions, thermal diagnosis, fixed cost | `results/diagnostics/` (`dispersion_*`, `thermal_series_mobilevit_50w_*`, `n_sweep_llm/`, `n_sweep_llm_cold3/`) |
| S9 | Protocols and findings | `docs/` |
| S10-S11 | Phase 4 level 1 | `results/investigations/vit/phase4/level1/` (`correlations.csv`, `layers_*.csv`) |
| S12 | Level 2 | `results/investigations/vit/phase4/level2/evaluations.csv` |
| S13 | Attribution | `results/investigations/vit/phase4/level2/attribution/summary.csv` |
| S14-S15 | Replication | `results/investigations/vit/phase4/level2/replication/` (`summary.csv`, `random_summary.csv`, `deit_summary.csv`) |
| S16 | Calibration attribution | `results/investigations/vit/phase4/level2/calibration_attribution/summary.csv` |
| S17 | LLM activations | `results/investigations/llm/phase4/activations/summary.csv`, `results/investigations/llm/phase4/robustness.csv` |
| S18 | LLM weights | `results/investigations/llm/phase4/weights/` (`summary.csv`, `head.csv`), `results/investigations/llm/phase4/robustness.csv` |
| S19-S20 | Per-tensor intervention | `results/investigations/llm/phase4/intervention/` (`analysis.csv`, `extension.csv`, `extension_sets.csv`, `sets.json`) |
| S21-S22 | Deployment gates | `results/investigations/vit/deployment_gate/`, `results/investigations/vit/output_quant_gate/` |
| S23 | Fusion probes | `results/investigations/vit/fusion_probes/` |
| S24-S26, S28 | ViT deployment | `plots/out/vit_deployment_summary.csv` |
| S27 | Thread scaling | `results/investigations/vit/thread_scaling/jetson_cpu_50w.json`, `results/investigations/vit/thread_scaling/analysis/profile_scaling_analyze.txt` |
| S29-S36 | LLM deployment | `plots/out/llm_deployment_summary.csv`, `results/investigations/llm/f16_ratios/`, `results/investigations/llm/gpu_vs_cpu/` |
| S37-S41 | Reproducibility | `requirements-lock.txt`, `README.md`, `docs/onnxruntime_patches.md` |

**Scripts that generate each output** (established from the code):

| Output | Generated by | Result files it reads |
|---|---|---|
| ViT bit-width figure (phase 3) | `plots/scripts/fig_vit_phase3.py` | `results/raw/vit_accuracy_offline.csv`, `results/raw/vit_accuracy_fp32_reference.csv` |
| ViT phase 4 recovery figure | `plots/scripts/fig_vit_phase4_recovery.py` | `results/investigations/vit/phase4/level2/evaluations.csv`, `.../level2/replication/evaluations.csv`, `results/raw/vit_accuracy_fp32_reference.csv` |
| ViT deployment figure (CPU) | `plots/scripts/fig_vit_deployment.py` | `plots/out/vit_deployment_summary.csv` |
| LLM deployment heatmap | `plots/scripts/fig_llm_deployment.py` | `plots/out/llm_deployment_summary.csv` |
| Calibration concept figure | `plots/scripts/fig_calibration_concept.py` | `plots/out/calibration_concept_data.npz` (not included) |
| Pipeline figure | `plots/scripts/fig_pipeline.tex` | none |
| ViT deployment summary | `analysis/vit_deployment_summary.py` | `results/combined/edge_results_combined.csv`, `results/diagnostics/dispersion_*/rep{1,2,3}.csv` |
| LLM deployment summary | `analysis/llm_deployment_summary.py` | `results/combined/edge_results_combined.csv`, `results/artifact_sizes.csv` |
| F16 ratios (LLM, Jetson) | `scripts/f16_ratios.py` | `results/raw/results_jetson_*.csv`, `results/artifact_sizes.csv`, perplexity CSVs, `results/investigations/llm/phase4/ceiling/`; output `results/investigations/llm/f16_ratios/ratios.csv` |
| GPU vs CPU ratios (LLM) | `scripts/gpu_vs_cpu.py` | `results/raw/results_jetson_*.csv`; output `results/investigations/llm/gpu_vs_cpu/ratios.csv` |
| McNemar tests (ViT) | `scripts/mcnemar_vit.py` | `results/investigations/vit/per_image_predictions/`; output `results/investigations/vit/mcnemar/` |
| Phase 4 ViT (levels 1-2, attribution, replication) | `analysis/phase4_vit/` | `results/investigations/vit/phase4/` |
| Phase 4 LLM (activations, weights, robustness, intervention, ceiling) | `analysis/phase4_llm/` | `results/investigations/llm/phase4/` |
| Thread scaling (Jetson) | `scripts/profile_scaling.py analyze` | `results/investigations/vit/thread_scaling/jetson_cpu_50w.json`; output `.../thread_scaling/analysis/profile_scaling_analyze.txt` |
| Dispersion / repeatability diagnostics | `scripts/dispersion_rows.py`, `scripts/thermal_series.py`, `scripts/n_sweep.py` | `results/diagnostics/` |

## Notes

- Numbers in the translated documents use the English decimal point and thousands comma; values are unchanged from the Spanish originals.
- `configs/devices.example.yaml` is the device-catalog template. The measurement parameters of the study are in Table IV of the paper and Table S7 of the supplementary material. `configs/devices.yaml` only holds the development/test entry (`cluster_dev_test`) that part of the test suite uses.
- The format of the imatrix files is interpreted according to `tools/imatrix/imatrix.cpp` of llama.cpp at commit `b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9`.

## Citation

If you use this code or these results, please cite:

N. Hernández, D. Suárez, P. Toledo, V. Blanco, and F. Almeida, "From
Calibration to Deployment: Accuracy and Energy of Post-Training Quantized
Transformers on Heterogeneous Edge Devices," submitted to *IEEE Internet of
Things Journal*, 2026.

```bibtex
@article{hernandez2026calibration,
  author  = {Hern{\'a}ndez, Nicol{\'a}s and Su{\'a}rez, Daniel and Toledo, Pedro
             and Blanco, Vicente and Almeida, Francisco},
  title   = {From Calibration to Deployment: Accuracy and Energy of
             Post-Training Quantized Transformers on Heterogeneous Edge Devices},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {Submitted}
}
```

This reference will be updated when the paper is published. The same
information is available in `CITATION.cff`.

## License

The code and the result files in this repository are released under the MIT
License (see `LICENSE`). The bounds check described in
`docs/onnxruntime_patches.md` is taken from ONNX Runtime commit `d113549`,
also released under the MIT License. The pretrained checkpoints and the
datasets are not included and are subject to their own terms.

## Contact

Nicolás Hernández (nhernang@ull.es), Department of Computer Engineering and
Systems, University of La Laguna. Questions about the code can also be raised
as issues in this repository.
