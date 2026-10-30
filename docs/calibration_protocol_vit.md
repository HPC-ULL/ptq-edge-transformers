> **Note on hypothesis numbering.** This protocol numbers the hypotheses differently from the paper; the numbering below is kept unchanged because it is the record of the study. Correspondence for the vision models: protocol H1 = paper H4, protocol H2 = paper H1 and the effect of the algorithm at INT8, protocol H3 = paper H2, protocol H4 = paper H3.

# Experimental Protocol: Ablation of PTQ Calibration Strategies on Vision Transformers

## 1. General theoretical objective

The original study treated PTQ quantization as a black box: pretrained model (input) → quantized model (output), without analyzing what happens inside the calibration process. The reviewers pointed this out as the central shortcoming (criticisms 8, 8.5, 8.6).

This experiment seeks to answer a deeper question, which connects two criticisms at once (8.x and 7.x):

> **What determines whether a calibration strategy recovers more or less accuracy under the same bit-width, and does that sensitivity depend on the model's architectural design?**

The framing hypothesis (general H0) is:

> The final quality of a PTQ model does not depend only on the target bit-width, but on the interaction between (a) the calibration strategy employed and (b) the intrinsic statistical properties of the model's weights/activations (distribution, presence of outliers, per-channel "smoothness").

This makes it possible, in the same experiment, to produce both the calibration ablation (criticism 8.x) and the causal sensitivity analysis (criticism 7.x), instead of treating them as two independent add-ons.

## 1.1 Final models

| # | Model | Role in the design |
|---|---|---|
| 1 | **DeiT-Tiny** | "Pure" transformer, without efficiency tricks — greater sensitivity was expected than in the conv-attention hybrids; the real results show the opposite (see H4, Section 5) |
| 2 | **MobileViT-S** | Conv+attention hybrid, designed for edge — a direct contrast with the pure ViT |
| 3 | **EfficientViT-B1** | New-generation efficient attention design — represents the state of the art in lightweight ViTs |

Confirmed after the technical conversion spike (Brevitas → ONNX) and viable on the study's edge devices — concrete variants already fixed, nothing pending confirmation.

## 1.2 Model identifiers and size

| Model | timm identifier (timm 1.0.28) | Parameters | Input |
|---|---|---|---|
| DeiT-Tiny | `deit_tiny_patch16_224.fb_in1k` | 5,717,416 | 224 |
| MobileViT-S | `mobilevit_s.cvnets_in1k` | 5,578,632 | 256 |
| EfficientViT-B1 | `efficientvit_b1.r224_in1k` | 9,102,024 | 224 |

There are several weight variants per architecture in timm; these identifiers fix the one used. Parameters as reported by the pipeline.

## 2. Independent variables (factors to manipulate)

To keep the design manageable with 3 ViT models, 4 main factors are proposed. Not all of them are crossed in a full factorial (see Section 4).

| Factor | Proposed levels | Justification | What it measures (see Section 2.1) |
|---|---|---|---|
| **Calibration algorithm** | MinMax (baseline), Percentile (99.9%), MSE, ~~Entropy/KL-divergence~~ | Covers from the simplest uniform scheme to the most outlier-robust one | Accuracy only |
| **Granularity** | Per-tensor vs. Per-channel | Per-channel usually helps more the more heterogeneous the weight distribution is across channels/attention heads | Accuracy + performance/energy |
| **Symmetry** (activations only) | Symmetric vs. Asymmetric | Especially affects post-GELU/Softmax activations, which are not centered at 0. **It only applies to the activations:** the weights are always symmetric (`quantize.py`: `symmetry` only reaches the activation quantizer; `quantizer_config.py` documents it) | Accuracy + performance/energy |
| **Calibration set size** | 32, 128, 512 samples | Allows seeing whether there is a saturation point (diminishing returns) and whether it varies per model | Accuracy only |

**Post-hoc exclusion of the "Entropy/KL-divergence" level:** originally designed as part of the "calibration algorithm" factor, it was excluded from the final ablation after an exhaustive diagnosis (all 3 models, including DeiT-Tiny) that confirmed **instability in the reference implementation** (`KLMinimizerThreshold`, Brevitas 0.13.0) — a parity bug makes the execution fail for most values of `num_bins` (only odd values do not fail), and among the odd values that do run, the resulting threshold varies substantially (14.89%-34.22% of the maximum activation magnitude, with no convergence) — instability of the method in this version of the library, not of the project's own pipeline nor of the model's activation distribution. Reported as an issue in the Brevitas repository. The count of valid configurations for the final report is adjusted accordingly (see Section 4).

**Note for writing the paper (entropy algorithm):** it is not a method of our own nor exclusive to Brevitas — it is the classic INT8 calibration method popularized by TensorRT (Migacz, 2017). When describing this factor in the Methodology, cite the origin of the method, not just the tool used to apply it — find the correct formal reference at the time of writing (the original TensorRT talk has no DOI, it may be advisable to cite it via some quantization survey that references it). The implementation details in Brevitas (internal classes, mixins used) are reproducibility information, not scientific methodology — they go in the code repository, not in the text of the paper.

**Factor already covered by the previous design (not repeated here):** bit-width (kept as a cross-cutting dimension: the full ablation is run at least at INT8, and a reduced version at two aggressive points, INT6 and INT4, to see whether calibration matters more the more aggressive the quantization is — hypothesis H2 below. Two points, not one, because with a single aggressive point a real trend cannot be distinguished from an isolated value — same criterion applied in the LLM protocol).

## 2.0.1 Which operators are quantized (and which are not)

Distinguish two things that are not the same: **which layers have quantized weights by design** (fixed, independent of the export regime) and **which operators appear running in floating point in the profiling** (depends on whether the exported graph includes an output `Q` — Section 6.4.2/7 of `technical_architecture.md`/`findings_vit.md`).

**What is quantized (weights), in the three models:** all `Linear`/`Conv2d` layers — including the `qkv` projections, the attention output projection, the MLP, and the patch embedding. This does not change between export regimes.

**What always remains in floating point, in both regimes:** `LayerNorm`, `Softmax`, `GELU`, the attention `MatMul`s (`Q·Kᵀ` and `attention·V`), and the residual additions — these operators are never quantized in this pipeline, with or without an output `Q`.

**What does change between regimes** (see `findings_vit.md`, Section 7): in the calibration ablation regime (without output `Q`, 99 artifacts), the profiling shows that **even the `Gemm`/`Conv` with quantized weights run in the `float_compute` category** — the graph quantizes the weights but never gets to fuse into an integer kernel, so in practice all the computation runs in float (emulation). In the deployment set (with `output_quant`, DeiT-Tiny/MobileViT-S), those same `Gemm`/`Conv` do fuse into real integer kernels.

This has two implications for the writing:

- **It is not directly comparable** with works such as PTQ4ViT and similar, which also quantize the attention matmuls — declare this difference in scope explicitly when comparing accuracy/performance figures with that literature.
- **It is consistent with** (does not "reinforce", a stronger claim than the data allow) the LayerNorm hypothesis of the diagnosis (`findings_vit.md`, Section 3): that LayerNorm operates in floating point is a condition compatible with the hypothesis that it acts by renormalizing the quantization noise, not evidence in its favor by itself — the hypothesis remains pending direct confirmation in Phase 4.

**Note — two export regimes, do not confuse them:** the calibration ablation (99 artifacts, Phases 1-3, this section) uses the regime **without** output `Q` — valid for comparing calibration strategies with each other, but without real acceleration on any backend tested. There is a **separate deployment set**, with `output_quant` enabled, generated only for the 2 models where real acceleration and non-collapsed accuracy were confirmed (DeiT-Tiny, MobileViT-S) — artifacts and results in a separate directory (`artifacts/vit_deployment/`), never mixed with the 99 of the ablation. Full detail on the why and on the third, excluded model: `findings_vit.md`, Section 7.

## 2.1 Which factors affect accuracy vs. performance/energy

Not all the ablation factors have the same impact on the four metrics of Section 3. It is important to distinguish this before programming, so as not to measure too much or assume too little:

- **Accuracy only (expected performance identical):** calibration algorithm and calibration set size. Both only determine *which value* of scale/zero-point is used; once fixed, the quantized model executes the same arithmetic operation regardless of how that value was computed. The hardware does not distinguish the origin of the number.
- **Accuracy + performance/energy (to be verified empirically, not assumed):** granularity and symmetry. Per-channel requires applying a different scale per channel when dequantizing (extra computational cost if the device's kernel has no optimized path for it); asymmetric quantization adds an offset (zero-point) operation in each arithmetic operation. Support for these paths varies a lot between ARM, x86 and especially RISC-V.

**Practical implication — corrected after the `output_quant` finding (`findings_vit.md`, Section 7):** for "algorithm" and "calibration set size", the expected flat performance is confirmed without the need for further checks (both only change *which value* is computed, not the arithmetic operation). For "granularity" and "symmetry" in **ViT**, the question of whether the overhead depends on the hardware **cannot be tested in the calibration ablation regime**, which always runs through fallback (Section 6.4.2 of the architecture) — any difference measured there would reflect Q/DQ emulation overhead, not real integer kernels, and would not answer the real question of criticism 9. This question is answered instead in the **deployment set** (artifacts with `output_quant`, real kernels confirmed) — 4 additional artifacts, `granularity=per_channel`/`symmetry=asymmetric` × DeiT-Tiny/MobileViT-S (`findings_vit.md`, Section 7.7). Granularity/symmetry **are not measured** on any edge device within the Phase 2 calibration ablation.

## 3. Dependent variables (metrics)

Standard metrics for an edge deployment study, justified in themselves (accuracy, inference cost, energy cost, memory footprint are the four usual dimensions when evaluating the viability of quantization for edge, not a replica of those of any previous work):
- Accuracy (top-1 / top-5, ImageNet-1K **validation split**)
- Latency / throughput (inferences per second)
- Energy (performance per watt)
- Model size

**New metrics for the causal analysis (link with criticism 7.x):**
- Weight distribution per layer (range, kurtosis, outlier/interquartile-range ratio)
- Dynamic range of activations per layer (computed offline with hooks on the FP32 model)
- Per-channel "smoothness" (variance of the weight distribution across the channels of one layer)

These metrics are computed **per layer, not once per model** (see the redesigned design in Section 3.0.1 — with only 3 models per branch, a single metric per model does not give enough statistical power for correlation) and serve to correlate with the ablation results: the question is whether the layers with "harder" distributions (outliers, high kurtosis) show greater **baseline sensitivity to quantization** — measured as local error under the baseline configuration (Section 3.0.1, Level 1; it is not measured against each model's best calibration, by design, so as not to mix the distribution signal with the mitigation effect of a specific calibration).

## 3.0.1 Design of Phase 4 (redesigned after review: with 3 models per branch, the "per model" correlation cannot be tested)

Fixed **before coding or running anything**, so as not to adjust the design after seeing results. The criteria also go in the header of the scripts.

**Deviations from the previous version of this design, decided before seeing any Phase 4 data:**
1. **The Level 1 images come from the calibration bank, not from the evaluation subset.** If the layers are ranked with the evaluation images and the accuracy is then measured on those same images, there is selection on the evaluation set. The Level 2 interventions are still evaluated on the 2000 evaluation images.
2. **DeiT-Tiny in Level 2 is measured at INT6, not INT4.** The INT4 baseline is at the floor (0.17 %, and not even `mse` applied to all layers goes beyond ~34 %): leaving ≤ 8 layers in float with more than 40 still at INT4 would give a flat curve at ~0 %, with no information. At INT6 the baseline gives 57.80 % versus 73.50 % for FP32 (mean of 3 ablation repetitions), a measurable and gradual gap of 15.7 pp. The reason for the original design (the gap at INT8 is within the noise) is also solved by INT6.

**Level 1 — Per-layer analysis within each model (correlation, cheap).** The distribution metrics are computed **per quantized layer** (`Linear`/`Conv2d`), not once per model: dozens of points per model, controlled by architecture, which does allow a meaningful correlation.
- **Activation outlier ratio** (of activations, not weights; consistent with the mechanism diagnosed in `findings_vit.md`, Section 3, hypothesis 6): `R = max_c(absmax_c) / (median_c(absmax_c) + 1e-8)`, with the standard median (the mean of the two central values if the number of channels is even) and `absmax_c` over the layer's input, taken over 512 images of the calibration bank and all tokens or spatial positions. Channel: the last one in `Linear` and `dim=1` in `Conv2d`. The 512 images are chosen with fixed seed 0 among those of the bank that do not belong to any calibration sample of repetitions 1-3 (seeds 42, 43 and 44; sizes 32, 128 and 512 are nested within each seed, so at most 1536 of the 10,000 are excluded). It sorts the layers from highest to lowest.
- **Exclusions and ties:** layers with fewer than 8 input channels (`patch_embed` and the stems) are left out of the ranking and the correlation (the median of a few channels is unstable) and remain quantized throughout the study. If a layer has median < 1e-6, it is listed without ranking it. Ties: execution order. Kurtosis and range are recorded as secondary, exploratory metrics.
- **Local quantization error:** relative L2 error per image, `‖y_q − y‖ / ‖y‖`, with the mean over the same 512 images, with the layer quantized in the **baseline configuration** (MinMax, per-tensor, symmetric, INT8, scales of repetition 1) and applied to its FP32 input. A single FP32 forward pass with hooks, without evaluating accuracy per layer. **Explicit decision: the baseline is used, not each model's best calibration.** Level 1 seeks to correlate the layer's distribution with its raw sensitivity to quantization; using the best calibration would mix that signal with the effect of a specific calibration.
- **Prior per-image grouping check:** the local error of the first and the last image of a batch of 16, computed alone and inside the batch, must coincide within 1 % in all layers (MobileViT-S rearranges the batch as `(B·4, N, C)`, and there may be similar rearrangements in `LiteMLA`). If it fails, no correlation is written.
- **Interpretation (fixed before seeing data):** with per-tensor MinMax, a layer's quantization step is set by its absolute maximum, so a positive correlation between `R` and the local error is partly mechanical. Level 1 is descriptive; the causal evidence is Level 2.
- **Ranking stability (added after seeing Level 1 and before seeing any Level 2 data):** the computation of `R` is repeated with a second set of 512 bank images (seed 1), disjoint from the first and from the calibration samples of seeds 42-44, and the Spearman between the two rankings and how many of the top 8 coincide are reported, per model. It is relevant in DeiT-Tiny, where the range of `R` is narrow (1.4-3.0) and the head of the ranking is almost a tie. Level 2 always uses the ranking of the main set; this is only reported, as a limit.
- **Correlation:** Spearman per model, always with n and ρ. The primary one is the global one with `R`, with Holm across the 3 models; the breakdown by type is descriptive. The layers within a model are not independent observations (adjacent ones and those of the same type resemble each other), so no more significance is claimed than that analysis allows.
- **Layer type (fixed after the inventory and before seeing Level 1).** Criterion: functional membership of the layer.
  - **attn:** the layers of an attention module, i.e. the qkv projection and the output one (`Linear` or 1×1 `Conv2d`) and, in `LiteMLA` of EfficientViT-B1, also the multiscale aggregations (`aggreg`, 5×5 depthwise and grouped 1×1), because they are part of the attention mechanism and not of a separate convolutional branch.
  - **conv:** the `Conv2d` outside an attention module (MobileNetV2 blocks, MBConv, stems, `final_conv`).
  - **mlp:** the `Linear` of the MLP blocks (`mlp.fc1`, `mlp.fc2`).
  - **other:** the classifier heads of the three models (`head`, `head.fc`, `head.classifier.*`), for consistency across models; `head.fc` of MobileViT-S falls in mlp only because it has "fc" in its name.
  - **Count** (rankable ones in parentheses when not all are): DeiT-Tiny, conv 1 (0), attn 24, mlp 24, other 1, 49 rankable; MobileViT-S, conv 35 (34), attn 18, mlp 18, other 1, 71 rankable; EfficientViT-B1, conv 46 (45), attn 28, mlp 0, other 2, 75 rankable.
  - **Consequences:** EfficientViT-B1 has no mlp (its FFN is the `local_module`, MBConv, which is conv), so per-type comparisons across models are limited to conv and attn. Types with fewer than 10 rankable layers are not correlated (they are listed). Sensitivity analysis, alongside the main one: in EfficientViT-B1, the per-type breakdown with the 14 `aggreg` classified as conv (attn = 14, conv = 60).

**Level 2 — Targeted interventions (causality, more expensive but bounded).** A fine-grained extension of what was done at a coarse scale in the EfficientViT-B1 diagnosis ("weights only" and "activations only" scenarios).
1. **Configuration:** MobileViT-S and EfficientViT-B1 at INT8 baseline; DeiT-Tiny at INT6 baseline. Only repetition 1 (calibration seed 42), with the gap measured as FP32 minus the accuracy of that repetition's K = 0; it is declared that it is a single calibration sample. The 90 % threshold is recomputed with that K = 0 (with the ablation means it would be ≥ 71.9 % in DeiT-Tiny).
2. **Targeted:** sort the **rankable** layers by `R` (those excluded from the ranking in Level 1 are not eligible) and leave the top-K in floating point, with K = 1, 2, 4, 8, evaluating on the 2000 images. **Stop** at the first of: K = 8, or ≥ 90 % of the gap recovered.
3. **Layer in float:** implemented with a copy in `investigations/`, without touching `quantize.py` nor `quantization/`, which travels in the zip. Evaluation path identical to that of the ablation artifacts: `build_quantized_model(..., output_quant=False)` with the baseline configuration and seed 42, `export_to_onnx` with the same QCDQ parameters and ONNX Runtime on CPU over the same 2000 images. **K = 0 control:** it must reproduce exactly the recorded accuracy of repetition 1; if it does not coincide, it stops.
4. **Random control (essential for the causal claim):** 3 fixed seeds (1, 2 and 3), one permutation per seed **of the rankable layers, the same set from which the targeted one chooses** (otherwise chance could leave in float a layer that the targeted one cannot choose and the curves would not be comparable), with K as a prefix of the permutation (nested, like the targeted one). If the targeted one stops before K = 8, the control reaches the same K. Without this control, a recovery could be due to there being fewer quantized layers and not to those layers being the responsible ones. The per-image top-1 predictions of each evaluation are stored.
5. **Secondary targeted curve by local error (exploratory; added after seeing Level 1 and before seeing any Level 2 data).** A second targeted curve that sorts the rankable layers by local error (from highest to lowest), with the same K, evaluated up to the targeted-by-`R` curve's K_final and against the same random controls. It answers whether the local error, a cheap indicator, predicts the causal impact better than `R`, and avoids being left without an answer if the curve by `R` is not distinguishable from chance (in DeiT-Tiny, `R` is not associated with the local error and the `mlp.fc2` with the highest error occupy positions 11-30 by `R`). The criterion of point 6 and the main conclusions apply only to the curve by `R`; the results of this one are presented as exploratory.
6. **"Clearly" criterion, fixed now:** the targeted curve beats the random one if, at K_final and in at least ⌈n/2⌉ of the n evaluated values of K (not counting K = 0, and including K_final), its accuracy is strictly greater than the maximum of the 3 random seeds (n = 1 requires 1; n = 2 requires 1; n = 3 requires 2; n = 4 requires 2). If not, "not distinguishable from chance" is reported. If with K = 8 neither the targeted nor the random one recovers 10 % of the gap, "no recovery with K ≤ 8" is reported, without concluding whether the degradation is diffuse.
7. **Implementation details, fixed before running.** Reference FP32: top-1 measured in PyTorch (`fp32_jpeg_vs_original.json`, `stored_q95`: 73.50 / 79.35 / 81.20), while the rest is evaluated in ONNX Runtime (the FP32 export coincides with a difference of 6e-5). Recovery: `(acc_K − acc_0) / (FP32 − acc_0)` on top-1. Random permutation: the rankable layers in execution order, shuffled with the corresponding seed. Secondary curve: sorts by descending mean local error, and ties are resolved by execution order. The ONNX files are discarded after evaluating them and their sha256 is stored. Repetition 1 references for the K = 0 control (baseline top-1 and the 90 % and 10 % thresholds): MobileViT-S INT8, 2.95 (≥ 71.71 and ≥ 10.59); EfficientViT-B1 INT8, 0.20 (≥ 73.10 and ≥ 8.30); DeiT-Tiny INT6, 58.80 (≥ 72.03 and ≥ 60.27). If the K = 0 control coincides in accuracy but there are differences in some per-image prediction, it is reported without stopping.
8. **Weights-or-activations attribution experiment (exploratory; added after seeing the Level 2 results, which fix the layers and the K, and before seeing any data from this experiment).** Level 2 leaves a whole layer in float, which eliminates at once the quantization of its weights and of its input, so it does not separate which one causes the damage. Evaluated, with the same path and the same 2000 images, are the layers that the targeted curve left in float at the K where it ends: **MobileViT-S with K = 2** (`stages.0.0.conv2_kxk` and `stages.1.0.conv2_kxk`) and **EfficientViT-B1 with K = 8** (the first 8 by `R`), with (a) **only the weights of those layers in float** (their input quantizer remains active) and (b) **only their input quantizer in float** (the weights remain quantized). There are 4 evaluations. Let `Rec_full` be the recovery of the gap with the whole layer in float (that of the targeted curve at that K: 93.9 % in MobileViT-S and 49.6 % in EfficientViT-B1) and `Rec_a` and `Rec_b` those of (a) and (b), all with the formula `(acc − acc_0) / (FP32 − acc_0)` on top-1. **Criterion fixed now:** «activations» if `Rec_b ≥ 0.75·Rec_full` and `Rec_a ≤ 0.25·Rec_full`; «weights» if `Rec_a ≥ 0.75·Rec_full` and `Rec_b ≤ 0.25·Rec_full`; in any other case, «mixed or interaction». If the export fails with a single quantizer disabled, it is reported before changing anything. **Interpretation:** `R` measures activations; if the damage were in the weights, `R` would identify the right layers for a reason different from the one its definition suggests, and the reading of RQ2 changes. It only applies to the two indicated configurations, with a single calibration sample (repetition 1). **Result:** «activations» in both models (`findings_vit.md`, Section 10.1).
9. **Replication with other calibration samples (added after seeing the results of Level 2 and of the attribution experiment, and before seeing any data from the replication).** Level 2 was measured only with repetition 1 (calibration seed 42). It is repeated with repetitions 2 and 3 (seeds 43 and 44) for **MobileViT-S with K = 2 and EfficientViT-B1 with K = 8**: per repetition, the K = 0 control and the targeted curve at that K, with the same layers (the first ones by `R` of the main ranking, which comes from FP32 and does not depend on the calibration sample). There are 8 evaluations. The K = 0 of each repetition must reproduce exactly the recorded top-1 and top-5 of that repetition; if not, it stops. The recovery is computed with the K = 0 of each repetition: `(acc − acc_0,k) / (FP32 − acc_0,k)`. **Criterion fixed now:** the recovery **replicates** if in both repetitions it is ≥ 0.75 times that of repetition 1, i.e. ≥ 70.4 % in MobileViT-S (reference 93.9 %) and ≥ 37.2 % in EfficientViT-B1 (reference 49.6 %); if not, it does not replicate. The random control and DeiT-Tiny (where `R` did not identify the sensitive layers) are not replicated, so this replication tests the recovery of the targeted curve and not the superiority over chance. It is a robustness check against the calibration sample, with three samples in total. **Result:** it replicates in MobileViT-S and not in EfficientViT-B1 (`findings_vit.md`, Section 10.2).
10. **EfficientViT-B1 random control in repetitions 2 and 3 (added after seeing the replication result and before seeing any data from this control).** The replication of point 9 did not test superiority over chance, and in EfficientViT-B1 the "clearly better" verdict of Level 2 rested on a single calibration sample. For each repetition (calibration seeds 43 and 44), the 3 random selections are evaluated with K = 8, which are the same layers as in repetition 1 (permutations of seeds 1, 2 and 3 of the rankable layers), built with the calibration of that repetition: 6 evaluations. **Criterion fixed now:** the superiority over chance holds with three samples if, in both repetitions, the top-1 of the targeted one with K = 8 (34.25 % in repetition 2 and 29.15 % in 3) is strictly greater than the maximum of the 3 random seeds of that repetition; if it fails in either, it does not hold. The per-seed values and their recovery are also reported. Only K = 8 is evaluated (the final K of the replication), unlike the multi-K criterion of Level 2. **Result:** the superiority holds in both repetitions (`findings_vit.md`, Section 10.2).
11. **Attribution of `percentile` to the two MobileViT-S layers: sufficiency and necessity (exploratory; added after seeing the results of Level 2, the attribution and the replication, and before seeing any data from this test).** It links the calibration effect (H2: `percentile` improves MobileViT-S) with the location of the damage (Level 2: two 3×3 depthwise convolutions, `stages.0.0.conv2_kxk` and `stages.1.0.conv2_kxk`, whose quantized input causes the collapse). **Design:** MobileViT-S, INT8, repetitions 1, 2 and 3 (calibration seeds 42, 43 and 44). For a set S of layers, the layers of S use the ablation's `percentile` quantization configuration (the same quantizers and parameters as the recorded `percentile` artifact; if that configuration also affects the weights, it is applied to the whole layer) and the rest use the baseline (MinMax). Three artifacts per repetition: **(control)** S = all layers, which must reproduce exactly the recorded top-1 and top-5 of the `percentile` artifact of that repetition (if not, it stops); **(sufficiency)** S = only those two layers; **(necessity)** S = all but those two. The K = 0 (S = none) is the recorded baseline. There are 9 evaluations. Let `Rec_perc,k` be the recovery of the recorded `percentile` artifact of repetition k, with `Rec = (acc − acc_0,k) / (FP32 − acc_0,k)` on top-1, and `Rec_suf,k` and `Rec_nec,k` those of the other two variants. **Criterion fixed now:** the two layers are **sufficient** if `Rec_suf,k ≥ 0.75·Rec_perc,k` in the three repetitions; they are **necessary** if `Rec_nec,k ≤ 0.25·Rec_perc,k` in all three. Interpretation: sufficient and necessary, the effect of `percentile` in MobileViT-S goes through the calibration of those two layers; sufficient but not necessary, they are enough but not the only route; necessary but not sufficient, they are needed but not enough; neither, the relationship between both results is not supported. It does not include a control with two random layers: this is justified by Level 2, where no random selection with K = 2 went above 3.05 %, and it is declared as a limit. **Clarification, resolved before running:** `percentile` only changes the input quantizer (activations): `Int8ActPerTensorFloatPercentile` (per-tensor, symmetric, scale by the 99.9th percentile) versus the baseline's `Int8ActPerTensorFloatMinMax`; the weights use the same class (`Int8WeightPerTensorFloat`) in both. Therefore, applying `percentile` to a layer amounts to changing the calibration of its input, and the script checks on startup that the weight classes coincide. Sufficiency and necessity thus refer to the calibration of the input activations of those two layers. **Result:** sufficient and necessary in the three repetitions (`findings_vit.md`, Section 10.3).
12. **Replication of DeiT-Tiny INT6 with repetitions 2 and 3 (added after seeing all the previous results and before seeing any data from this replication).** The negative result of DeiT-Tiny («`R` does not identify the sensitive layers») rests on a single calibration sample. For repetitions 2 and 3 (seeds 43 and 44), with DeiT-Tiny INT6 baseline and K = 8, the following are evaluated: the K = 0 control, which must reproduce exactly the recorded top-1 and top-5 of the INT6 baseline of that repetition (if not, it stops); the curve targeted by `R` (the first 8 of the main ranking); the 3 random selections of repetition 1 (permutations of seeds 1, 2 and 3 of the rankable layers); and, as an exploratory one, the local-error curve with K = 8 (the 8 layers with the highest local error). There are 12 evaluations. The recovery is computed with the K = 0 of each repetition. **Criterion fixed now:** the negative result **replicates** if, in both repetitions, the top-1 of the one targeted by `R` with K = 8 is not strictly greater than the maximum of the 3 random seeds of that repetition; if it exceeds it in either, the absence of effect depends on the sample and does not replicate. The local-error curve is compared with the same controls and reported, but it does not enter the criterion. Only K = 8, as in points 9 and 10. **Result:** the negative result replicates in both repetitions (`findings_vit.md`, Section 10.2).

Bounded cost: **4 targeted evaluations + 12 random = 16 per model, plus the K = 0 control and, at most, 4 of the secondary curve (21 per model in total)**, and not one for each layer of the model.

**Results:** `findings_vit.md`, Sections 9 (Level 1) and 10 (Level 2).

**Order and checkpoints:** (0) inventory of layers with their classification, which is reviewed; (1) full Level 1, and the ranking and the correlations are reviewed before continuing; (2) K = 0; (3) targeted; (4) random. Moving from step 1 to 2 requires approval. **The Level 1 review serves to detect implementation errors, not to decide whether Level 2 is done:** Level 2 is run whatever the Level 1 correlation is, and it only stops if the ranking is degenerate (fewer than 12 rankable layers, or more than half of them with the same `R`) or if the K = 0 control fails.

**For LLM (Ministral-3-3B versus TinyLlama/Qwen2.5):** the per-layer outlier magnitude comparison across models is presented as a **descriptive characterization**, not as causal evidence: it is still a comparison between different models, not within the same model, so it does not isolate the effect of the distribution from other architectural differences.

## 3.1 Data separation: calibration vs. evaluation

The calibration set (Section 2, 32/128/512 samples) is drawn from the ImageNet-1K **training split**, never from the validation one. It is the standard practice to avoid *data leakage*: if the calibration set overlapped with the images used to report accuracy, the model would indirectly "see" that data before being evaluated on it, biasing the results in favor of the quantization — a problem a journal reviewer would detect and object to.

The performance benchmarks (`onnxruntime_perf_test`) do not use real data in any case — they measure latency/throughput with synthetic input tensors of the right shape (same image size), without ImageNet needing to be present on the edge devices at any moment.

The evaluation subset used for accuracy must be generated once and reused unchanged across all the ablation configurations — never re-sampled between them, unlike the calibration set.

**Reproducibility of the evaluation subset — resolved:** dataset revision pinned in the code (`ILSVRC/imagenet-1k`, revision `49e2ee26`). Reference sentence for the Methodology: *"ILSVRC/imagenet-1k at revision 49e2ee26; subset: the first two images of each class of the validation split in reading order, 2000 in total."*

**Known limit, declare if a reviewer asks for full robustness:** pinning the dataset revision does not pin the exact streaming order, which also depends on how the `datasets` library orders the shards internally (stable in practice, sorted by name, but not guaranteed across library versions). A possible complement later: publish the list of the exact image identifiers already used (option B, discarded for now because it requires reconstructing IDs from the local manifest, which only stores re-encoded names, not the original ones) — non-blocking, the artifacts already evaluated remain valid with the revision already pinned.

**Preprocessing (identical in calibration and evaluation).** `timm.data.resolve_data_config` + `create_transform`, with each model's own configuration:

| Model | Input | crop_pct | Resize | Crop | Interpolation | Mean | Std |
|---|---|---|---|---|---|---|---|
| DeiT-Tiny | 224 | 0.90 | 248 | center 224 | bicubic (antialias) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |
| MobileViT-S | 256 | 0.90 | 284 | center 256 | bicubic (antialias) | (0, 0, 0) | (1, 1, 1) |
| EfficientViT-B1 | 224 | 0.95 | 235 | center 224 | bicubic (antialias) | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) |

MobileViT-S uses a 256 input and does not normalize with the ImageNet mean and standard deviation; do not assume 224 and that normalization for the three models.

**Datasets.** Evaluation: 2000 images from the validation split (2 per class), stored as JPEG with quality 95. Calibration: bank of 10,000 images from the train split, stored as JPEG with quality 90. Dataset revision: 49e2ee26. **JPEG recompression — measured.** Comparing the FP32 of the three models on the same 2000 images in their stored version (q95) and in the original without recompression: the prediction changes in 29-63 images out of 2000 (1.5-3.2%), but the net accuracy moves only 0.25-0.35pp, in a different direction depending on the model (DeiT-Tiny and MobileViT-S go up with the original, EfficientViT-B1 goes down) — no systematic bias of direction. Small effect, declare with one sentence; the effect of recompressing the calibration set (q90) has not been measured.

**Bias of the evaluation subset against the full validation — measured, and it is systematic.** The FP32 of the three models on the 50,000 images of the full validation (72.16% / 78.31% / 79.26% top-1, consistent with each model's published figures to within a few tenths) is **consistently 1.0-1.9pp lower** (1.34 / 1.04 / 1.94; source `results/raw/vit_accuracy_fp32_reference.csv`) than on the fixed subset of 2000 images used throughout this ablation — all three models, in the same direction, not sampling noise around zero. The selection method (first two images of each class in streaming order, not random) produces a subset that is systematically easier than the full validation.

**Implication, important for the writing:** the *relative* comparisons within the ablation (which calibration wins, how much it compensates for an aggressive bit-width) remain valid — all configurations are measured on the same subset. What cannot be sustained is treating the *absolute* accuracy figures of this study (72.9% DeiT-Tiny baseline, 68.38% for `percentile` in MobileViT-S, etc.) as directly comparable to the official full-validation figures of each model, or to those of other works that do evaluate on the full split — declare this bias (known direction: +1-2pp) explicitly in the Methodology, do not let the reader assume direct comparability.

## 4. Experimental design by phases

A full factorial (4 algorithms × 2 granularities × 2 symmetries × 3 sizes × 3 models × 3 bit-widths) would give 432 configurations — unfeasible. A staged design is proposed:

**Phase 1 — Replication baseline**
Configuration: simplest uniform scheme (MinMax, per-tensor, symmetric, 128 samples, INT8) — not a replica of the full INT8 mode of TFLite, which by default uses per-channel weights and asymmetric activations, a scheme different from this study's baseline. It serves as an anchor to measure relative improvements within the ablation itself, not for direct comparison with the original study (whose manuscript, moreover, is not citable as evidence — unpublished).

**Phase 2 — One-factor-at-a-time ablation**
Starting from the Phase 1 baseline, a single factor is varied at a time, keeping the rest fixed:
- 4 algorithms (with granularity/symmetry/size fixed at the baseline) — **3 valid after the exclusion of entropy** (see Section 2)
- 2 granularities
- 2 symmetries
- 3 calibration set sizes

Total per factor: 4 + 2 + 2 + 3 − 3 (baseline counted only once, not four times) = 8 configurations per model (including the baseline), × 3 models = 24 runs at INT8 (72 artifacts with the 3 repetitions). **Valid total for the final report, after excluding entropy: 7 configurations per model × 3 models = 21 runs (63 artifacts)** — the 9 entropy configurations/artifacts (3 models × 3 repetitions) were generated and evaluated, but are excluded from the analysis and the writing because of the confirmed instability of the method (Section 2), documented as a separate finding, not as one more results row.

**Phase 3 — Optimal combined configuration vs. aggressive extreme**
With the Phase 2 "winners" per model, a combined configuration ("best calibration") is built and compared against the Phase 1 baseline at **two** aggressive bit-widths, **INT6 and INT4**, not just one — with a single point a real trend cannot be distinguished from an isolated value (same reasoning applied in the LLM protocol for Q3_K/Q2_K). This gives a result with a strong narrative hook ("a good calibration compensates up to X fewer bits") and allows H2 to be contrasted directly. Bounded additional cost: 3 models × 2 configurations (best calibration vs. baseline) × 2 aggressive bit-widths = 12 runs, not a complete repetition of Phase 2.

**Phase 4 — Causal analysis**
Correlate the results of Phases 2-3 with the distribution metrics computed in Section 3, per model. This is where criticism 7/7.5 is explicitly answered: not just "which calibration works best" but "why it works best for this specific model".

## 5. Specific hypotheses to be tested

- **H1:** Per-channel calibration reduces the accuracy loss compared with per-tensor, and the benefit is proportional to the heterogeneity of the weight distribution across channels/heads (measured in Phase 4). **Actual status: refuted in the ablation regime (complete Phase 2)** — `granularity=per_channel` did not exceed the improvement margin over the baseline in any of the 3 models (DeiT-Tiny, MobileViT-S, EfficientViT-B1). Report as a negative finding with the mechanism explained — but the mechanism (per-channel acts on weights, while the identified problem is in activations, Section 3 of the diagnosis) only has real explanatory power in MobileViT-S/EfficientViT-B1, which do have severe degradation to recover. In DeiT-Tiny the absence of effect is trivial: at INT8 there is hardly any degradation to correct in the first place, so there is no visible margin for per-channel to show any benefit. Do not present the same mechanism as equally applicable to the three models. **In the deployment regime** (with `output_quant` and folded BN), `per_channel` does improve MobileViT-S (+10.9 pp, n=3) and does not improve DeiT-Tiny (`findings_vit.md`, Section 7.6.4); proposed mechanism not verified (the folding moves the BN scale into the weights).
- **H2:** Outlier-robust algorithms (percentile, MSE) beat MinMax, with the difference widening as the bit-width drops (INT8 → INT6 → INT4). *(Entropy excluded from the test of this hypothesis — see Section 2.)* **Actual status (Phase 3): confirmed only in DeiT-Tiny; in the two sensitive models the observed trend is the opposite.** In DeiT-Tiny, the gap between baseline and combined configuration grows with the aggressive bit-width, as H2 predicts. In MobileViT-S and EfficientViT-B1 the opposite happens: calibration matters **more** at INT8 (where it recovers ~66pp in MobileViT-S, ~8pp in EfficientViT-B1) and its effect **decays** as the bit-width drops, until it disappears at INT4 in both cases. *(Note: the configurations compared at INT8 and at INT6/INT4 are not exactly equivalent — INT8 isolates `percentile`, INT6/INT4 combines `percentile`+`asymmetric` — but the magnitude of the effect is so large that the decreasing trend is clear despite that difference.)* It is an interesting pattern in its own right — in the models with indirect evidence of severe activation outliers, calibration can only compensate up to a certain point of aggressiveness, not indefinitely — and must be described that way, not forced to fit the direction anticipated by H2.
- **H3:** The calibration set size shows diminishing returns beyond a certain threshold, and that threshold is lower in models with "smoother" activations (fewer outliers). **Actual status:** no clear effect detected in any of the 3 models — the differences between 32/128/512 samples fall within the statistical resolution margin of the evaluation subset (Section 6), with no identifiable saturation threshold. It neither confirms nor refutes with the current statistical power; pending contrast with a paired test (McNemar, Section 6) before ruling it out entirely.
- **H4:** The optimal calibration configuration is architecture-dependent: the "pure" ViT (DeiT-Tiny) is expected to be more sensitive to the choice of calibration than the conv-attention hybrid (MobileViT-S) — consistent with the architecture-dependent sensitivity documented in the ViT quantization literature (to be cited in the Methodology with published sources). **Actual status: the observed direction is the inverse of the expected one** — MobileViT-S is substantially more sensitive to the choice of calibration than DeiT-Tiny (collapse to 1.88% vs. partial recovery with `percentile` up to 68.38%, versus a variation of less than one point between algorithms in DeiT-Tiny at INT8). The hypothesis of architectural dependence is confirmed; the specific anticipated direction is not. The full causal explanation is left for Phase 4.

## 6. Statistical considerations

- Sampling of the calibration set (especially at small sizes, 32 samples) introduces variance: each Phase 2 and 3 configuration must be repeated with **at least 3 different random samplings** of the calibration set, reporting mean ± standard deviation.
- **Statistical test — corrected:** with only 3 repetitions, a Wilcoxon signed-rank test has a minimum achievable p-value of 0.25 (it cannot reach conventional significance under any real effect with n=3) — **do not use Wilcoxon on the repetitions**. Instead: **image-level paired McNemar test**, on the same 2000 images of the evaluation subset, comparing hits/misses of each configuration image by image — it has real power with this sample size, and it fits with all configurations already being evaluated on the same fixed subset (Section 3.1).
- **Note on paired comparisons vs. absolute resolution:** Section 3 of the diagnosis gives ±1 percentage point as the resolution of the *absolute* accuracy of the 2000-image subset — but **the comparisons between configurations are paired** (same images evaluated in both), which substantially reduces the effective noise compared with that absolute resolution figure. Do not apply the ±1pt directly to rule out small differences between configurations (e.g. the 0.5-point margin that picks `mse` as the winner in DeiT-Tiny) without first running McNemar on that specific comparison — otherwise, the resolution criterion itself would invalidate selections already used in Phase 3.

### 6.1 McNemar plan, fixed before running it

**Status.** The per-image predictions are generated and validated (129 artifacts, the accuracy of each CSV identical to the recorded one). The DeiT-Tiny deployment artifacts were evaluated with the one-node GELU (`_gelu1`, the ones measured on device), with predictions identical image by image to those of the loose GELU; the contrasts use those. The tests have been run (26 comparisons × 3 repetitions); results and interpretation in `findings_vit.md`, Section 8.

**Scope:** only the ViT branch. McNemar compares per-image paired hits and misses; perplexity does not produce a hit per sample, so in LLM it does not apply and the criterion remains the mean-separation rule.

**Requirement:** the evaluation scripts only store the aggregate accuracy. A separate script is needed that re-evaluates the already existing artifacts (without requantizing) and stores, per image, the label and the top-1 and top-5 predictions. Estimated cost: 1-4 min per artifact.

**Comparisons** (each one, configuration A against B, on the same 2000 images, per model):

| Hypothesis | Comparison | Regime | No. per model |
|---|---|---|---|
| H1 (granularity) | `per_channel` vs. baseline, INT8 | ablation | 1 |
| H1, deployment regime | MobileViT-S: `per_channel`+`symmetric` vs. `per_tensor`+`symmetric`, and `per_channel`+`asymmetric` vs. `per_tensor`+`asymmetric`; DeiT-Tiny: `per_channel` vs. `mse` core | deployment (`output_quant`) | 2 / 1 |
| H2 (algorithm), INT8 | `percentile` vs. baseline; `mse` vs. baseline | ablation | 2 |
| H2 (algorithm), aggressive bit-width | combined vs. baseline, INT6 and INT4 | ablation | 2 |
| H3 (calibration set size) | 32 vs. 128; 512 vs. 128 | ablation | 2 |
| RQ3b (transfer) | deployment core vs. baseline with `output_quant` (DeiT-Tiny: `mse` vs. `minmax`; MobileViT-S: core vs. `minmax`+BN) | deployment | 1 |

H4 is not tested with McNemar: it compares the sensitivity across models (an interaction), not two configurations of the same model. It is presented with effect sizes. The claim of H2 that the gap grows as the bit-width drops is also descriptive: McNemar tests each comparison, not the trend across the three.

**Repetitions:** one test per repetition, pairing repetition k of A with k of B (same calibration set seed). A difference is claimed only if it is significant in all 3 repetitions; the three p-values are reported.

**Multiple comparisons:** Holm correction within each hypothesis (the family is the hypothesis), α = 0.05.

**Relation to the decisions already taken:** McNemar is confirmatory. The winning configurations of Phase 3 and of the deployment set were chosen with the mean-separation rule and are not changed afterwards; if McNemar does not confirm one, it is declared, without rewriting the selection.

## 7. Explicit mapping to the reviewers' criticisms

| Criticism | How it is addressed in this protocol |
|---|---|
| 8 — PTQ as a black box | The whole protocol opens the calibration process to analysis |
| 8.5 — Only basic uniform quantization | Phase 2, "calibration algorithm" factor |
| 8.6 — No hyperparameter ablation | Complete Phase 2 (4 factors) |
| 7 / 7.5 — No causal analysis of sensitivity | Phase 4 (correlation with weight/activation distribution) |

## 8. Scope with respect to LLMs (consistency note with the general design)

This protocol applies **only to the 3 ViT models** (see the previous discussion). For LLMs the llama.cpp/GGUF pipeline is kept, varying only the calibration corpus of the `imatrix` as a bounded ablation, documented separately and explicitly justified in the Methodology as a scope decision (an already mature ecosystem, characterized in the existing literature).

---

*Phases 1-3 complete for the 3 models, accuracy and performance/energy. Concrete findings (real figures, not just hypotheses) in `findings_vit.md`, separate from this document for the same reason as in LLM: this one describes the design, it does not change with each new batch of results.*
