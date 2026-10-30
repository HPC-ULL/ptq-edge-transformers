# Ideas for Future Work / Deeper Analysis

Notes on possible extensions or additional analyses, conditional on what the real results of the ablation show — they are not part of the pipeline to be built now. They are reviewed when Phase 2 data is at hand, not before. Cross-reference: Section 6.6 of `technical_architecture.md` (mentions `-p` and `xnnpack` as available technical options).

## Per-operator profiling (`onnxruntime_perf_test -p`) as evidence for criticism 9

**Motivation:** several reviewer criticisms (7, 7.5, 8) pointed out that the original paper described phenomena without explaining them at the mechanistic level. A `-p` trace showing, operator by operator, where the extra cost of a configuration is concentrated (e.g. per-channel granularity) turns an aggregate observation ("this config is 8% slower") into verifiable evidence ("operator X takes Y µs longer").

**Two possible variants, with different value:**

- **Jetson CPU vs. GPU (same device, two backends):** a more controlled experiment (it isolates a single variable), and it answers "why does this happen?" at the mechanistic level within a single piece of hardware. Cheap to do.
- **Two different devices (e.g. x86 vs. ARM or RISC-V):** noisier (more variables change at once), but it answers reviewers' **criticism 9** directly (each platform's software stack as a possible confounding variable) and reinforces the thesis already present in the original paper's conclusions ("there is no universal optimal configuration, it depends on the hardware") with mechanistic evidence, not just aggregate evidence.

**Recommended order, if there is time for both:** Jetson CPU vs. GPU first (it builds intuition about the mechanism in a controlled environment), and with that basis what appears when crossing different devices (which is the one that really answers the reviewers' criticism) is interpreted with more confidence.

**Conditions for it to carry real weight in the paper, not just illustrative value:**
1. Use it to explain a finding that already appeared in the aggregate Phase 2 data — not the other way around (searching with `-p` first and building the narrative afterwards is harder to defend methodologically).
2. Pay attention only to **performance** findings, not accuracy findings — `-p` profiles execution times, it does not explain accuracy loss (that depends on calibration, not on how the graph is executed). By design, only the "granularity" and "symmetry" factors are marked as affecting performance (Section 2.1 of the ViT protocol) — that is where to look first.
3. If it is a single device/comparison, present it explicitly as a qualitative illustration, not as generalizable evidence; to claim that a pattern "holds", it must be reproduced in at least 2 cases.

## `xnnpack` execution provider for ARM devices

To be considered only if the standard ONNX Runtime `cpu` EP shows, on the project's ARM devices (RPi CM4, Rock 960, Odroid XU4, VisionFive 2), performance that looks like a bottleneck or little representative of a real optimized deployment on that architecture. It is not part of the current design.

## Microscaling (MX) formats as a possible mitigation of the EfficientViT-B1 collapse

**Context:** already available in Brevitas (active support, an industry standard — OCP — promoted by AMD/Xilinx among others). It is not a generic speculative idea — it connects directly with the diagnosis already closed in `findings_vit.md`.

**Why it could be relevant, not just "yet another format":** the diagnosis identified the **activations** as the dominant cause of the accuracy collapse in EfficientViT-B1, with the mechanism of noise accumulated by a per-tensor scale that is too coarse, without renormalization between blocks. The granularity test already carried out (`per_channel`) **only applied to weights** (Section 2.1 of the ViT protocol) — a finer granularity specifically on activations was never tried. MX assigns one scale per small block (not per whole tensor), applicable to both weights and activations — it is the first candidate technique that attacks exactly the dimension left unexplored in the diagnosis, not a generic alternative.

**Why it is not pursued now:** the ViT pipeline is at an advanced stage (99 ablation artifacts already built, causal diagnosis already closed and documented). Adopting MX would not be replacing one level of an existing factor — it changes the nature of the `bitwidth` variable (block floating point with a shared exponent, not integers with a per-tensor/per-channel scale), which would affect the results schema and comparability with the rest of the already completed ablation.


**If it is taken up in the future:** the most informative experiment would not be to replace a whole model, but to repeat specifically scenario 4 of the EfficientViT-B1 diagnosis (quantized activations, float weights) using MX instead of uniform INT8 for the activations — it would isolate whether block granularity by itself recovers accuracy, confirming or ruling out the diagnosed mechanism with a more targeted intervention than those already tried.

**Update — investigated and blocked (do not pursue for now):** the maturity of MX in the installed Brevitas version (0.13.0) was checked. Quantizing is trivial — `MXInt8Weight`/`MXInt8Act`/`MXFloat8e4m3Weight`/`Act` are direct presets, with no composition conflict, and they cover both weights and activations (better coverage than `percentile`/`entropy`, which only apply to activations in this project). However, there is a **real blocker confirmed in the code, not inferred from documentation**: both Brevitas ONNX exporters (QCDQ and QONNX) explicitly reject modules with "groupwise" quantization (`assert not module.is_groupwise`), and MX is inherently groupwise (`group_size=32`, not optional). This means that **an MX-quantized model cannot be exported to ONNX** with this Brevitas version — it would break the convention of the rest of the pipeline (ONNX → onnxruntime → accuracy evaluation, the same artifact later reusable for on-device benchmarking), leaving as the only alternative an evaluation exclusively in PyTorch that would lose the possibility of measuring performance/energy at the edge — precisely half of the value that motivated the proposed substitution.

An additional reason, independent of the export blocker: the activation scale in MX is computed dynamically for each inference (`RuntimeDynamicGroupStatsScaling`), it is not calibrated once from the calibration set like the rest of the levels of the `calibration_algorithm` factor — it is a structurally different PTQ paradigm, not one more comparable option within the same axis of the current ablation.

**Decision:** Brevitas is not updated to unblock this — the project has already exhaustively validated the behavior of the current version (including the finding of the MinMax bug on activations), and changing version mid-project would risk subtly invalidating those already confirmed results, in exchange for a functionality that is not guaranteed. Review in the future whether a later Brevitas version supports ONNX export of groupwise modules, without needing to act on it now.

---

*Add here any other similar idea that comes up during implementation or when reviewing results, so it is not lost without needing to touch the pipeline specification documents.*

## Phase 4 in the deployment regime (`output_quant` and folded BN)

**Context:** ViT Phase 4 (`findings_vit.md`, Sections 9 and 10) was done in the ablation regime, without output quantization, which is the one that is not deployed. In the deployment regime (real integer kernels), MobileViT-S with its best configuration drops from 68 % to 63 % and EfficientViT-B1 collapses even with `percentile`, which is what left it out of the deployment set.

**Why it would be valuable:** it would tie RQ2 to RQ3. Two concrete questions: (1) are the two depthwise layers of MobileViT-S still responsible for the accuracy cost with `output_quant`?; (2) which layers explain the EfficientViT-B1 collapse in that regime? The second could reopen its exclusion, since `findings_vit.md` (Section 7.6) records that `per_channel` with BN folding was not tried on that model.

**Why it is not done now:** it is not a low-complexity test. It requires new builds (`output_quant` active and BN folded as identity) and deciding how the output quantization of the neighboring layers behaves when a layer is left in float, and the deployment set already has its measured and documented closure.

**If it is taken up:** minimal design, only MobileViT-S with the deployment core (`percentile` + `per_channel` + `asymmetric` + BN folding): a K = 0 control that reproduces the recorded accuracy, a targeted curve with the same two layers and a random control with K = 2, with the same «clearly» criterion as Level 2. Then EfficientViT-B1 if the first result justifies it.

**Coherence note:** the MX formats section of this document describes the EfficientViT-B1 mechanism as noise accumulated by a per-tensor scale. Phase 4 qualifies it (the damage is concentrated at the input of a few depthwise layers, and the recovery depends on the calibration sample), so it should be reviewed if that idea is taken up.

## LLM Phase 4: per-tensor imatrix intervention

**Context:** LLM Phase 4 is only descriptive (`calibration_protocol_llm.md`, Section 3.0.1), because with 3 models no inference can be made across them. Within a model, the tensor is a unit of analysis just like the layer in ViT, and the Level 2 design could be transferred.

**Experiment:** in TinyLlama and Qwen2.5, which are the models where the imatrix helps, apply the imatrix only to the tensors with the highest `ρ`, only to those with the lowest `ρ`, or to a random set of the same size, and measure the perplexity. It would give the LLM branch the same causal step as ViT.

**Feasibility (checked on the cluster, `b95502ba9`).** `llama-quantize` has `--include-weights` and `--exclude-weights` (not combinable) and `--tensor-type`. Tested with TinyLlama, Q3_K_M and the imatrix of repetition 1 (`results/investigations/llm/phase4/intervention_probe/compare.log`):
- quantization is deterministic (two runs without imatrix, byte-for-byte identical);
- the full imatrix changes the 154 tensors that have an entry in the imatrix (`token_embd` and `output.weight` do not have one);
- `--include-weights blk.1.attn_q.weight` changes only that tensor, and it is identical to its version with the full imatrix, so the tensors are quantized independently;
- `--include-weights attn_q` changes all 22 `attn_q` (substring match), and the full names with `.weight` do not collide;
- the flag can be repeated;
- passing the 154 explicit names reproduces the full imatrix exactly (the control equivalent to K = 0 in ViT).

**Why it is not done now:** the design has to be decided (size of the sets, random control, criterion) and it would cost several quantizations and perplexity evaluations per model. It will be designed after the Phase 4 robustness controls.

**If it is taken up:** first verify feasibility with a single tensor and with a hand-modified imatrix file, before designing the full experiment.

## Phase 4: unexplained aspects and experiments that could address them

Phase 4 is closed: all pre-registered questions have an answer. Some aspects remain unexplained, of two kinds: those that depend on comparing more models (with 2 or 3 per branch they cannot be answered) and finer mechanisms within a model (possible, but they open new questions instead of closing an RQ).

| Unexplained aspect | Possible experiment | Approximate cost | Why not now |
|---|---|---|---|
| Why `R` identifies the responsible layers in MobileViT-S and EfficientViT-B1 and not in DeiT-Tiny; why `ρ` does so in TinyLlama and not in Qwen2.5 | Repeat the design on more models of each family | Large (quantization, evaluation and full Phase 4 per model) | It is a question across models; with 2-3 per branch it has no answer |
| Whether `ρ` identifies where the imatrix acts in a third model | Per-tensor intervention on Ministral-3-3B at Q2_K, where the imatrix does help (`G` ≈ 0.31) | ~22 evaluations of ~70 min (plus the extension of 20 random ones), 1-2 days of cluster | It adds one case, it does not change the conclusion (`ρ` is not a universal predictor) |
| Which tensors concentrate the imatrix benefit in TinyLlama | Per-block or per-tensor contribution (imatrix on a single block or tensor at a time) | 22 to 154 evaluations of ~33 min | It details a mechanism; it does not answer any RQ |
| Whether the damage in the two depthwise layers of MobileViT-S comes from outlier channels or from isolated values | Per-channel quantization or per-channel clipping only at the input of those two layers | Few evaluations, but new implementation in Brevitas and the export | Fine mechanism; the practical result is already there (`percentile` at those two inputs) |
| The 50 % of the EfficientViT-B1 gap that 8 layers do not recover | Targeted curve with K = 16 and 32 | About 8 evaluations | The superiority over chance is already there; widening K does not change the conclusion |
| Why Ministral responds less to the imatrix than the other two, even at 2 bits | More models, or the per-tensor intervention above | See above | Across models, same as the first row |
| Phase 4 in the deployment regime | See the section «Phase 4 in the deployment regime» | New builds | See that section |
