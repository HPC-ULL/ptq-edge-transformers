# ViT Findings: Accuracy Collapse and Real Speedup under PTQ

*Document of concrete results of the ViT branch — same role as `findings_llm.md` for the LLM branch: it is updated as the investigation progresses, not all of its content goes into the paper, but it remains as a reference and a prepared answer to a reviewer's questions. It covers two axes: why the accuracy collapses in MobileViT-S/EfficientViT-B1 (Sections 1-6), and why quantization does not speed up inference as originally configured, and what is needed for it to do so (Section 7).*

## 1. Motivation

During the validation of the PTQ quantization pipeline for the ViT branch, two of the three evaluated models (MobileViT-S, EfficientViT-B1) showed a severe accuracy collapse under the baseline configuration (INT8, per-tensor, MinMax calibration) — top-1 accuracy far below the model's own floating-point reference (MobileViT-S: 1.88% INT8 versus 79.35% FP32; EfficientViT-B1: 0.13% INT8 versus 81.20% FP32 — both FP32 figures measured on the same 2000-image evaluation subset used throughout the ablation). The third model (DeiT-Tiny, 73.50% FP32 versus ~72.9-73.4% INT8) did not show this problem. Instead of discarding the models or the toolchain, a systematic investigation by elimination of hypotheses was carried out to identify the causal mechanism — precisely the kind of analysis that the reviewers' criticisms (7, 7.5, 8) pointed out as absent in the original version of the manuscript.

## 2. Hypotheses evaluated, in order of investigation

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | Excessive or poorly calibrated activation clipping (that discarded valid values of the distribution) | Check of activation ranges vs. the chosen scale | Ruled out — *note: the problem turned out to be the opposite (see hypothesis 6 and the `percentile` finding, below): MinMax does not clip enough, letting activation outliers inflate the scale; more aggressive clipping (`percentile`) is precisely what recovers accuracy. Do not confuse "ruled out" here with "clipping itself is harmful" — it is the opposite.* |
| 2 | Execution bug (Brevitas vs. manual fake-quant) | Cross-comparison of implementation | Ruled out |
| 3 | Insufficient weight granularity (per-tensor vs. per-channel) | Execution of the `granularity=per_channel` row | No effect in either of the two models (ablation regime, without `output_quant`; with `output_quant` and folded BN there is an effect in MobileViT-S, Section 7.6.4) |
| 4 | Calibration algorithm (MinMax vs. MSE vs. Percentile) | Execution of the `calibration_algorithm=mse` and `calibration_algorithm=percentile` rows | See update after the full ablation (note below the table) |
| 5 | Linear attention block (`LiteMLA`) as the isolated origin of the failure (EfficientViT-B1 only) | Exclude `LiteMLA` from quantization, leave it in float | No improvement — the collapse already occurs before reaching that block |
| 6 | Weight vs. activation quantization as the dominant cause (EfficientViT-B1) | "INT8 weights only, activations in float" scenario | **Activations identified as the dominant cause**: substantial recovery (0.50%→55.00% top-1) when leaving activations unquantized. *Note: this isolated test was measured on a subset of only 200 images, different from the 2000-image subset used in the rest of the ablation — valid as a qualitative confirmation of the mechanism (activations as the dominant cause), but the absolute figures of this row are not directly comparable with the rest of the values in this document, measured on the 2000-image subset.* |
| 7 | Artifact of the manual layer-substitution methodology (`return_quant_tensor=False`, without sharing scale between layers) | Attempt at manual substitution with `return_quant_tensor=True` | **Structurally unfeasible**: it produces direct errors (`ndim` not available in `QuantTensor`; `"Scaling factors are different"` when combining tensors of different scales in multi-branch operations) — it confirms that the conservative choice was not avoidable, not that it was suboptimal |

**Methodological note on hypothesis 7:** Brevitas' automatic pipeline (`quantize()`) includes specific machinery to align scales between quantized tensors in multi-branch operations (residuals, concatenations) — the same mechanism whose absence causes the error of hypothesis 7 in manual substitution. The automatic pipeline was tried as an alternative, but both models fail during symbolic tracing (FX) for reasons unrelated to each other and unrelated to quantization: MobileViT-S because of data-dependent control flow in its patch-folding logic; EfficientViT-B1 because `LiteMLA` uses `torch.autocast` with an argument that requires evaluation at trace time, incompatible with symbolic tensors (the `timm` authors themselves already mark this module as non-traceable via `register_notrace_module`). These tracing limitations are independent of the choice of quantization pipeline and fall outside the scope of this study.

**Update of hypothesis 4 after the full ablation (the 7 valid configurations of each model, not just the ad-hoc tests of the initial diagnosis):** the calibration algorithm **does matter in both models, but `percentile` turns out to be the dominant lever, not `mse`** — a nuance that the initial ad-hoc tests did not capture, since `percentile` had not been systematically tested at that time.

- **MobileViT-S:** `percentile` recovers **68.38% top-1** (std 0.58%, not attributable to noise) — far above the 24.63% obtained with `mse` (std 5.49% between repetitions, verified with the 3 individual values — 25.95%/29.35%/18.6% — as genuine sampling dispersion, not a pipeline anomaly). The baseline without aggressive clipping barely reaches 1.88%.
- **EfficientViT-B1:** `percentile` (8.15%) and `symmetry=asymmetric` (1.08%, corrected figure — see note below) show a real improvement over the baseline (0.13%), although **insufficient for practical use** — the original claim of "immune to all calibration levers" is not exact: some levers do help partially, none solves the problem completely.

This reinforces, rather than contradicts, the mechanism already identified (activations with outliers as the dominant cause, Section 3): `percentile` aggressively clips the extreme values of the distribution, while `mse` optimizes reconstruction error in a less drastic way — if the underlying problem is activation outliers, it is coherent that aggressive clipping works better than a more conservative scale search, especially in MobileViT-S.

**Additional note — confirmation by implementation correction (literal min/max vs. hidden 99.999 percentile):** during implementation it was detected that the quantizer initially used for `symmetry=asymmetric` (Brevitas' `ShiftedUint8ActPerTensorFloat`) did not apply literal min/max, but a very lax percentile interval (0.001%-99.999%) — the same type of discrepancy already corrected earlier for the symmetric case. When corrected to a real min/max, the `symmetry=asymmetric` figure changed substantially in the two sensitive models and practically nothing in the non-sensitive model:

| Model | With hidden 99.999 percentile | With literal min/max | Change |
|---|---|---|---|
| DeiT-Tiny | 73.32% | 73.05% | Within repetition noise |
| MobileViT-S | 56.92% | 21.05% | Drop of ~2.7× |
| EfficientViT-B1 | 5.37% | 1.08% | Drop of ~5× |

This is an additional confirmation, not deliberately sought, of the mechanism already identified: the only difference between both quantizers is whether the most extreme outlier of the distribution is included literally in the scale (percentile 100, real min/max) or minimally clipped (percentile 99.999). That this almost imperceptible clipping moves accuracy by between 2.7× and 5× specifically in the two models already identified as sensitive to activation outliers — and does not affect DeiT-Tiny at all — is consistent with, and reinforces, the outlier-sensitivity story already established in this document. *The absence of effect in DeiT-Tiny is consistent with LayerNorm renormalizing in each block, but this is a plausible explanation, not a confirmed one — it has not yet been measured whether DeiT-Tiny lacks severe activation outliers in the first place (pending Phase 4); the absence of effect could be due to that, to LayerNorm, or to both.* The Phase 3 winner did not change with this correction in any model (Section 4).

## 3. Conclusion per model

**MobileViT-S:** the accuracy collapse corresponds to genuine PTQ quantization noise in the **activations**, not in the weights — per-channel weight granularity (hypothesis 3) had no effect in the ablation regime (with `output_quant` and folded BN it does, Section 7.6.4), explicitly ruling out depthwise weight sensitivity under per-tensor as the explanation (the classic remedy for that problem, per-channel weights, did not help in that regime). It responds substantially to a calibration that aggressively clips activation outliers (`percentile`, 68.38% top-1), more than to a scale search by error minimization (`mse`, 24.63%) — consistent with activation outliers as the main cause, not with a structural limit of the architecture.

**EfficientViT-B1:** the collapse is likewise genuine quantization noise (neither an execution bug nor a clipping one), and none of the mitigation levers tested solves it at a practical-use level — although `percentile` (8.15%) and asymmetric symmetry (1.08%, corrected figure) do show real partial improvements over the baseline (0.13%), ruling out the stronger claim of "immune to any intervention". The dominant cause identified is the quantization of **activations** (not of weights). The explanation most consistent with the results, by elimination of alternatives, is the accumulation of per-tensor quantization noise across the network's numerous sequential boundaries, without an active renormalization mechanism between blocks — unlike DeiT-Tiny, where LayerNorm renormalizes the signal in each block. *Important note: MobileViT-S also uses LayerNorm in its transformer blocks, so the presence/absence of LayerNorm alone does not cleanly distinguish the two sensitive models from the non-sensitive one — a more specific and plausible mechanism is the affine (non-renormalizing) nature of the BatchNorm folded into the convolutions at inference time, present in both sensitive models. This is a hypothesis pending direct confirmation in Phase 4 (correlation with the weight/activation distribution metrics), not an already established fact — do not present it as such in the writing.* Unlike MobileViT-S, not even the aggressive clipping of outliers is enough to bring accuracy to a functional level, which suggests a more severe or extended accumulation of noise in this architecture.

**Note on statistical resolution:** the evaluation subset (2000 images) has an approximate standard error of ±1 percentage point around accuracies of 70-80% — small differences between configurations (e.g. 72.9% vs. 73.4% in MinMax vs. MSE of DeiT-Tiny, or 73.32% vs. 73.05% in the symmetry correction) are within that margin and should not be interpreted as real effects by themselves. It does not affect any of the conclusions of this document, all based on differences of several percentage points or more — but it is advisable to declare the subset size and its resolution in the Methodology, so as not to invite a reviewer to over-interpret minor differences in the full ablation tables.

## 4. Implication for the sensitivity analysis (Phase 4 of the protocol)

This finding is an extreme case, and therefore especially informative, of the phenomenon that motivates the Phase 4 causal analysis: sensitivity to quantization depends on structural properties of the model (presence/absence of renormalization between blocks, activation distribution), not only on the bit-width or the quality of the calibration. The MobileViT-S/EfficientViT-B1 contrast is coherent with the ViT quantization literature documenting architecture-dependent sensitivity (Section 6). What is identified with certainty by our own diagnosis is that **activations** are the dominant cause of the collapse in both models (hypothesis 6, systematic elimination of alternatives) — the specific structural mechanism that explains *why* these two models in particular are sensitive (renormalization, outlier distribution) remains a hypothesis to be confirmed in Phase 4, not an already established fact.

**Note on the original study's rejected manuscript:** that work is not published and cannot be cited as evidence in the paper — any comparison with its results (e.g. the ConvNeXt/MobileNet contrast) must be reformulated relying on published literature, or explicitly presented as the team's internal motivation, never as citable support. The reviewers' comments on that manuscript seemed internally inconsistent about ConvNeXt's sensitivity at first sight (criticism 7 describes it with high loss, the strengths assessment describes it as insensitive) — clarified for the team's internal use: both were reading different dimensions of the manuscript itself, sensitive in accuracy under INT8 but insensitive in throughput/efficiency, not a real contradiction of the reviewers. This clarification is for internally understanding the reviewers' comment, not for citing as evidence in the new paper.

## 4.1 Extension to aggressive bit-widths (Phase 3): H2 confirmed in DeiT-Tiny, inverse trend in the sensitive models

The Phase 3 results (INT6, INT4, comparing the MinMax baseline against the winning combined configuration from Phase 2) provide additional findings that reinforce and qualify the above — **the direction of H2 (the gap widens as the bit-width drops) is only confirmed in DeiT-Tiny; in MobileViT-S and EfficientViT-B1 the opposite happens**, see Section 5, H2, of the protocol:

**H2 is confirmed in a pronounced way in DeiT-Tiny.** At INT8, the difference between MinMax and MSE was barely half a point (72.9% vs. 73.4%). At INT4, the same choice of algorithm is the difference between a practically collapsed model (baseline: 0.17% top-1, versus the model's own 73.50% FP32) and a partially functional one (combined: 34.33% top-1) — and already at INT6 the gap is substantial (57.80% baseline vs. 70.32% combined). It directly confirms that the importance of the calibration algorithm widens the more aggressive the bit-width is, with DeiT-Tiny **not showing the same severe degradation as the other two models** (not necessarily because it lacks severe activation outliers — that has not yet been measured, pending Phase 4; what is an observed fact is that its INT8 baseline does not collapse, unlike MobileViT-S/EfficientViT-B1). *Precision on the "combined configuration" per model: DeiT-Tiny's consists only of changing the calibration algorithm to `mse` (granularity, symmetry and calibration set size remain at their baseline value, none of those factors exceeded the improvement margin in the Phase 2 selection) — so "same choice of algorithm" is exact in this specific case. Not so in MobileViT-S/EfficientViT-B1, where the combined configuration wins in two factors at once (algorithm and symmetry), see below.*

**In MobileViT-S, the trend is the opposite of H2: calibration matters more the less aggressive the bit-width is, not the other way around.** At INT8, `percentile` alone recovers the model from 1.88% to 68.38% top-1 — a gap of ~66 points. At INT6, the combined configuration (`percentile`+`symmetry=asymmetric`, Phase 2 winner) recovers from 0.12% to only 22.07% top-1 — a gap of ~22 points, notably smaller. At INT4, the recovery **disappears completely**: combined drops to 0.05%, indistinguishable from the baseline (0.07%) — a gap of zero. *(Note: the INT8 and INT6/INT4 configurations are not identical — INT8 isolates `percentile`, INT6/INT4 combines `percentile`+`asymmetric` — but the magnitude of the drop, from 66 to 22 to 0 points, is too large to be explained only by that difference in configuration; the decreasing trend is clear.)* It is not a progressive degradation symmetric to that of DeiT-Tiny — it is the inverse pattern: calibration can only compensate up to a certain point of numerical aggressiveness, with a discrete threshold between INT6 and INT4 below which no strategy tested helps anymore. Relevant for the study's deployment recommendations: for this architecture, careful calibration extends viability up to INT6, but no further — and its relative benefit is greatest precisely at the least aggressive bit-width (INT8), not at the most aggressive, contrary to what H2 predicted.

**In EfficientViT-B1, the same inverse trend is observed even more markedly.** At INT8, `percentile` recovers ~8 points over the baseline (0.13%→8.15%). At INT6 and INT4, the combined configuration (`percentile`+`symmetry=asymmetric`) provides no measurable improvement: baseline and combined give practically identical results both at INT6 (0.10% both) and at INT4 (0.12% both) — the gap disappears entirely, it is not just reduced. This extends the finding already established in Phase 2 (no mitigation lever solves the problem at a practical level) — here it is confirmed that neither does it at any aggressive bit-width tested, consistent with the cause (noise accumulation in activations without active renormalization between blocks, a hypothesis pending confirmation in Phase 4) not depending on the numerical precision point evaluated.

**Joint interpretation:** in the two models with indirect evidence of severe activation outliers (the substantial improvement with `percentile`, an algorithm designed specifically to clip them, and the strong impact of the symmetry correction — not a direct measurement of the distribution, which is left for Phase 4), calibration has a marked effect at INT8 and decays as the bit-width drops, until it disappears — the exact opposite pattern to the one expected by H2, and also opposite to the one confirmed in DeiT-Tiny. It is a finding with value of its own, not a failure of the hypothesis to be hidden: it suggests that, below a certain threshold of numerical precision, quantization noise comes to be dominated by something that no scale calibration strategy can compensate anymore — probably related to the numerical resolution of the integer format itself, not to where the scale is placed.

## 5. Note for the final writing

When incorporating this into the paper, preferably present Section 2 as a table or decision diagram (hypothesis → test → result), not just as narrative text — it communicates the rigor of the elimination process more directly than prose. Avoid reporting EfficientViT-B1's accuracy number (0.13%) without the full context of this diagnosis: in isolation, it is indistinguishable from a pipeline error; accompanied by the elimination trail, it is rigorous evidence of a real mechanism.

## 6. Anchoring in the literature and precise scope of the claim

**Do not claim that these models "cannot be quantized"** — it is too broad a claim, easy to object to by citing works that do achieve it with specialized techniques. The correct and defensible claim is more precise:

> **Uniform/naive** PTQ quantization applied through a general-purpose toolchain (Brevitas) collapses the accuracy of these models, consistent with the ViT-specific PTQ literature, which documents this same problem and has therefore developed specialized techniques outside the scope of this experimental study.

This is supported by already existing literature, it is not an isolated finding of this project:

- **PTQ4ViT** (Yuan et al., 2022) — establishes the seminal result: applying standard PTQ "directly" to ViT produces more than 1 point of accuracy drop even at 8-bit, and attributes the problem to the post-Softmax/GELU activation distributions being very far from a Gaussian, which a generic uniform quantizer does not handle well — the same underlying mechanism (activations, not weights) identified in this diagnosis (Section 2, hypothesis 6).
- **NoisyQuant** (Liu et al., CVPR 2023) and **AdaLog** — propose specific techniques (noise injection, non-uniform logarithmic quantizers) precisely to mitigate this same activation distribution problem.
- **DopQ-ViT** (paper title: DopQ-ViT) — paraphrase, not a verbatim quote: it attributes the degradation to the existing quantization paradigm not fitting well the power-law-like distribution of post-Softmax activations (verify the exact wording against the English original when writing, and add the full reference to the paper). It is additional evidence that the problem documented in the literature is in the **activations**, coherent with the dominant cause already identified in this diagnosis — but the specific mechanism it proposes (power-law-like distribution) is different from the one proposed here (per-tensor noise accumulation without renormalization); do not force an equivalence between both mechanisms when citing it, present them as convergent evidence of "the problem is in activations", not as the same causal mechanism.

**How to use this in the writing:** cite PTQ4ViT at a minimum (seminal reference of the problem), and optionally 1-2 more of the above, at the same point where the EfficientViT-B1/MobileViT-S diagnosis is presented — it turns the finding from an "unexpected result that had to be investigated" into an "empirical confirmation, with its own causal mechanism, of a problem already known and actively researched in the ViT quantization literature". This also directly preempts the most predictable reviewer objection ("why did they not compare with ViT-specific PTQ techniques?") with an already incorporated answer: that is not the objective of this study (which evaluates standard deployment toolchains, not proposing a new quantization method), and the diagnosis itself already identifies why that type of technique would be needed.

*Note: the above references were located through a quick bibliographic search; verify the full citation (venue, year, exact DOI/arXiv id) before including them in the paper's final bibliography.*

## 7. Real INT8 speedup: QDQ kernels, output_quant, and the deployment set

*It complements the previous sections (causes of the accuracy collapse) with the dimension of performance/real speedup — the result of a later investigation into why quantization, as originally configured, does not speed up inference.*

### 7.1 Starting point



A reviewer could point out that INT6/INT4 are exported in INT8 containers (Section 6.6.1 of the architecture) and therefore do not accelerate. When checking with per-operator profiling (`onnxruntime_perf_test -p`), it was discovered that **not even INT8 uses integer kernels** in the pipeline as originally configured — a finding broader than what motivated the check.

### 7.2 Initial diagnosis (DeiT-Tiny, cluster x86 CPU, 1 thread)

| Variant | Integer kernels | Latency | vs. FP32 |
|---|---|---|---|
| FP32 | — | 20.3 ms | — |
| Pipeline INT8 (Brevitas, only input quantized) | 0% (`FALLBACK`) | 24.9–25.7 ms | ~1.25× slower |
| ORT native INT8 (`quantize_static`) | 100% | 18.2 ms | 1.11× faster |
| ORT native without output Q | 10% | 22.9 ms | 1.13× slower |
| Brevitas without `Clip` | 0% | 24.8 ms | like Brevitas |
| Brevitas + `output_quant` | 100% of quantized layers* | 17.2 ms | 1.18× faster |

*Only the attention `MatMul`s remain in float, which the pipeline does not quantize by design (Section 2.0.1 of the ViT protocol).

**Confirmed root cause:** the original pipeline quantizes only the *input* of each `Conv`/`Linear`, without a `Q` at the output — without that second node, ONNX Runtime does not recognize the complete pattern needed to fuse into `QLinearConv`/`QLinearMatMul`. Ruled out as causes: the `Clip` node (narrow-range clipping), the weight's `Transpose`, the integer bias (`Int32Bias`).

**It is not a general limitation of ONNX Runtime** — the native quantization tool itself (`quantize_static`) fuses the same model 100%. The limitation is specific to the pipeline's export configuration (absence of output `Q`), not to the ecosystem.

### 7.3 Accuracy cost of `output_quant` (2000 official images)

**Criterion note — all the figures in this table are from a single repetition (`rep1`), not means over 3 repetitions like the rest of this document and of the calibration ablation** (where, e.g., DeiT-Tiny baseline is 72.9% on average, versus the 73.20% of `rep1` that appears here) — pending confirmation with ≥3 repetitions before citing in the paper (Section 7.7). Do not mix these `rep1` figures with the means of the rest of the document when writing; use a single criterion (means) in the final version.

| Row | Current regime (rep1) | `output_quant` (rep1) | `output_quant` + BN folding (rep1) |
|---|---|---|---|
| DeiT-Tiny baseline | 73.20% | 72.10% (−1.1) | — (no BN) |
| MobileViT-S baseline | 2.95% | 0.45% | 0.60% |
| EfficientViT-B1 baseline | 0.20% | 0.25% | 0.10% |
| MobileViT-S percentile | 68.90% | 40.85% (−28) | 49.00% (−20) |
| EfficientViT-B1 percentile | 8.25% | 5.00% | 0.25% (collapse) |

FP32 references (2000 images): DeiT-Tiny 73.50%, MobileViT-S 79.35%, EfficientViT-B1 81.20%.

**BN folding:** hypothesis tested (quantizing the output before BN, with unnormalized channels, caused additional loss) — partially confirmed: it recovers 8 of 28 points in MobileViT-S, but **sinks EfficientViT-B1** (the BN scale passes to the per-tensor quantized weights, aggravating the already known problem of activation sensitivity). It was not investigated further — the same criterion of diminishing returns already applied at other points of the project.

### 7.4 Verification on the Jetson (ARM) — CPU

`output_quant` also fuses completely on ARM, with **more speedup than on x86**:

| Model | FP32 | INT8 `output_quant` | Relative to FP32 | Integer kernels |
|---|---|---|---|---|
| DeiT-Tiny (CPU, 1 thread) | 137.5 ms | 83.0 ms | 1.66× faster | 50/50 layers |
| MobileViT-S (CPU, 1 thread) | 277.6 ms | 197.0 ms | 1.41× faster | 72/72 layers |

Conditions: `uptime > 600s` (the already known boot step mitigation), stable samples. **Declared limit — the speedup figures of this section (1.66× and 1.41×) are at 1 thread and do not represent the real deployment.** With the threads of each power mode the speedup changes a lot and can disappear (Section 7.8); DeiT-Tiny's, in addition, predate the GELU fix (Section 7.8.1). Measured at 1 thread; the real study measures with 4/8/12 threads depending on the power mode — the speedup with several threads is not verified, pending confirmation in the real deployment measurement.

### 7.4.1 BatchNorm folding in MobileViT-S: accuracy improvement confirmed, and an unanticipated fusion divergence between x86 and ARM

**BN folding (a mathematically exact identity in floating-point arithmetic — not under quantization, where the transformed graph does quantize differently from the original and that is why the accuracy changes, see below) — accuracy improvement confirmed with 3 repetitions**, not noise from a single measurement:

| MobileViT-S, `percentile`, n=3 | Top-1 | Top-5 |
|---|---|---|
| `output_quant` unfolded | 38.35 ± 2.95% | 64.58 ± 2.41% |
| `output_quant` + BN folding | 47.48 ± 2.09% | 74.93 ± 1.56% |

Improvement of +9.13pp (top-1), far above the combined margin of standard deviations (5.04pp) — acceptance criterion fixed before seeing the data, widely exceeded.

**Additional optimization attempt (removing altogether the BN nodes already folded to identity, instead of keeping them): it revealed a real, unanticipated divergence between CPU architectures.**

| MobileViT-S, CPU 1 thread | FP32 | `output_quant` (BN as identity) | `output_quant` + BN removed |
|---|---|---|---|
| Jetson (ARM) | 276.8 ms | 195.5 ms · 1.41× · 72/72 integer | 179.7 ms · **1.54×** · 72/72 integer |
| Cluster (x86) | 57.1 ms | 49.1 ms · 1.16× · 72/72 integer | 74.5 ms · **1.31× slower** · 43/72 integer |

**Correction of an earlier assumption:** it had been assumed that the fusion decisions of the CPU EP were equivalent between x86 and ARM (they coincided in all the cases tested up to this point) — this result refutes it. When the BNs are removed, the dequantized output of each `Conv` comes to feed two nodes directly (`Sigmoid` and `Mul` of the `SiLU` activation) instead of just one (previously mediated by the BN-identity); the x86 optimizer does not fuse that double-consumer pattern, whereas the ARM one does — hypothesis of the mechanism, not verified beyond the observation. It is in itself a finding of platform heterogeneity (the same type of argument already used for criticism 9 with GPU/CUDA vs. CPU), but **it is not pursued further** — the same criterion of diminishing returns already applied at other points of the project.

**Decision: the "BN removed" variant is not adopted for the deployment set**, despite its better figure on the Jetson (1.54× vs. 1.41×). Reason: this project's accuracy is always measured offline on the cluster (x86) — with this variant, that measurement would correspond to a partially emulated execution (43/72 kernels), whereas the real deployment on the Jetson would run entirely (72/72), with different numerical results between both executions (already confirmed: 49.00% vs. 48.40% top-1, a difference attributable exactly to those 29 layers). Closing that gap would require measuring accuracy directly on the Jetson — new infrastructure that the rest of the project does not use, for a marginal benefit (0.6pp of accuracy, 0.13x of additional speed). The final deployment set uses **`output_quant` + BN folding kept as identity** (complete and consistent fusion on both architectures, 1.41× on Jetson) — see Section 7.6.

### 7.4.2 INT6/INT4 fusion on Jetson (ARM): fully confirmed, with no speed gain over INT8

Verified with the same profiling probe, on the 8 configurations (2 models × 2 bit-widths × baseline/combined), with BN folded-as-identity (the same artifact used for the accuracy measurement of Section 7.6.2):

| Jetson CPU (ARM), 1 thread (DeiT-Tiny figures predate the GELU fix, Section 7.8.1) | Integer kernels / layers | vs. FP32 | INT8 `output_quant` reference |
|---|---|---|---|
| DeiT-Tiny baseline INT6 / INT4 | 50/50 · 50/50 | 1.64× · 1.64× faster | 1.66× |
| DeiT-Tiny combined INT6 / INT4 | 50/50 · 50/50 | 1.64× · 1.64× | |
| MobileViT-S baseline INT6 / INT4 | 72/72 · 72/72 | 1.39× · 1.40× faster | 1.41× |
| MobileViT-S combined INT6 / INT4 | 72/72 · 72/72 | 1.35× · 1.36× | |

**Complete fusion in the 8 configurations** — the same kernel pattern as INT8 (DeiT-Tiny: 1 `QLinearConv` + 49 `QGemm`; MobileViT-S: 35 `QLinearConv` + 37 `QGemm`). The `Clip` nodes (needed to emulate a bit-width below 8 inside the 8-bit container, Section 6.6.1 of the architecture) do not prevent fusion on ARM, with a marginal cost (1.4-1.8% of the time). Number of Q/DQ nodes identical to INT8 (112 in DeiT-Tiny, 159 in MobileViT-S) — it confirms that it is the same execution mechanism, only with the range of values clipped inside the same container.

**No speed gain over INT8** — the speedup figures are practically identical (and in MobileViT-S combined, slightly worse: 1.35-1.36× versus 1.41×, probably because of the cost of the zero-points in the asymmetric variant — a minor detail, just the opposite of x86, not investigated further). It confirms with measured data, not just with the theoretical argument of the shared container, that going below 8 bits in this pipeline is a purely accuracy decision, never a speed one — the basis of the decision in Section 7.6.2.

### 7.5 Verification on Jetson — GPU (CUDA EP)

**There is no speedup, in any case tested:**

| DeiT-Tiny, Jetson GPU (50W) | Latency | vs. FP32 | Integer kernels |
|---|---|---|---|
| FP32 | 7.4 ms | — | — |
| ORT native INT8 (`quantize_static`) | 10.1 ms | 1.36× slower | 0 (`FALLBACK`) |
| Brevitas INT8 + `output_quant` | 11.9 ms | 1.61× slower | 0 (`FALLBACK`) |

**Confirmed that it is a limitation of the CUDA EP in general, not of the pipeline:** not even ORT's native tool achieves fusion on GPU — the same model that does fuse 100% on CPU. 520 Q/DQ nodes executed explicitly (~38% of the time), ~10% of the time falls to CPU because of `DequantizeLinear` without a CUDA kernel (with the associated host↔GPU copies — up to 50 `MemcpyFromHost` in the Brevitas graph versus 2 in the native one, explaining the small additional difference Brevitas vs. native).

**The real path to accelerated INT8 on NVIDIA GPU is TensorRT** — discarded with the same criterion already applied to XNNPACK/QONNX (contamination of the comparison by layer fusion and auto-tuning unrelated to the quantization itself). It is not investigated further — it is a known platform limitation documented in the ecosystem, not an unknown to be resolved.

### 7.6 Decision: deployment set with real integer kernels

**Included — CPU only:**
- **DeiT-Tiny**: `mse` (its real Phase 2/3 winner, also confirmed under `output_quant` with 3 repetitions: 73.35 ± 0.20% top-1 vs. 72.23 ± 0.28% for baseline). Minimal accuracy cost (−0.08pp, within the dispersion) — very close to FP32 (73.50%). Fusion and latency also confirmed with `mse` (not just with baseline): complete in both environments, identical to baseline (1.66× on Jetson, 1.18× on cluster) — `mse` only changes scales, not the graph. Those figures are at 1 thread and predate the one-node GELU fix (1.75× on Jetson at 1 thread after the fix); with the threads of each power mode, see Section 7.8.
- **MobileViT-S**: deployment reference = `percentile` + `granularity=per_channel` + `symmetry=asymmetric` + BN folding (kept as identity, not removed — Section 7.4.1), with `output_quant`. Complete fusion (72/72). Accuracy: 63.33 ± 0.60% top-1 (n=3) versus 79.35% FP32 — the most accurate and most stable configuration of the 4 granularity×symmetry combinations tested on top of `percentile`. It stays ~5 pp below the best non-accelerated configuration tested (68.38%, `percentile` per-tensor symmetric without `output_quant`); that difference is not a clean cost of acceleration, because both configurations differ in four things at once (`output_quant`, BN folding, granularity and symmetry). In the deployment set only 2 of the other 3 combinations are measured, those that differ from the core in a single factor: `per_channel`/`symmetric` (58.35 ± 4.01%, symmetry variant) and `per_tensor`/`asymmetric` (52.13 ± 1.03%, granularity variant). The fourth, `per_tensor`/`symmetric` (47.48 ± 2.09%), differs from the core in two factors, was measured in accuracy and fusion, but is not part of the deployment set. Decision and caveats in Section 7.6.4. Speedup with the core: 1.37× on Jetson (ARM), 1.18× on cluster (x86), 1 thread.

**Excluded — EfficientViT-B1**, for two independent reasons, either of which would suffice by itself:
1. **Only partial fusion** (42 of 74 `Conv` layers to `QLinearConv`; 32 remain in float despite having an output `Q` confirmed by graph inspection — exact cause not diagnosed; they coincide in number with its `hardswish` layers, correlation not verified). Result: 1.75× slower than FP32, not even with `output_quant`.
2. **Collapsed accuracy** under `output_quant` with the configurations tested (0.25% baseline, 5% percentile; BN folding with per-tensor weights sinks it even further). **`per_channel` + BN folding was not tested on this model**: in MobileViT-S that combination recovers accuracy (Section 7.6.4) and the mechanism that would explain it (the BN scale passes to per-tensor quantized weights) is the one proposed for EfficientViT-B1's sinking, so this reason could change. Consistent with the activation-sensitivity mechanism already diagnosed (7 hypotheses evaluated).

**Answer already prepared, in case a reviewer asks "why did you not investigate the EfficientViT-B1 fusion problem more deeply?":** the exclusion stands on reason 1 alone: with 42 of 74 layers fused the model is 1.75× slower than FP32, so there is no real speedup to deploy, regardless of its accuracy. Reason 2 reinforces the decision with the configurations tested, but `per_channel` + BN folding was not tested and it must not be presented as closed. If the model were to be reopened, the order would be: first the cause of the 32 unfused `Conv` (cheap, dumping the optimized graph) and only if it is resolved, that accuracy combination.

**GPU:** the 10 rows of the set were also measured on the Jetson's GPU (Section 7.9). Rule fixed before measuring: the FP32, the INT8 core and the fallback reference of each model and mode are reported, and the rest remains in the CSV; the selection does not depend on the values.

### 7.6.1 Narrative direction for the paper: connect the two regimes, do not leave them as loose findings

A reviewer would immediately ask: *"the conclusions about calibration are obtained in a regime that is not deployed (without `output_quant`) — do they hold in the one that is deployed?"*. Answer in two parts, both to be included, not alternatives:

1. **Show that the conclusion transfers.** Main evidence, with n=3 (not `rep1`): in **DeiT-Tiny**, `mse` remains clearly superior to baseline under `output_quant` (73.35 ± 0.20% vs. 72.23 ± 0.28%, Section 7.6) — the same hierarchy as in the non-accelerated regime (73.43% vs. 72.90%), with a margin that exceeds the dispersion between repetitions, although the McNemar test does not confirm it: the sign coincides in the 3 repetitions (+0.8 to +1.5 pp), but only 1 of 3 is significant after Holm (Section 8). In **MobileViT-S** (with BN folding), `percentile` remains far superior to baseline in the 4 granularity×symmetry combinations tested (63.33 ± 0.60% in the best, `per_channel`+`asymmetric`, versus 0.42 ± 0.20% for baseline, both n=3, Section 7.6.4) — the same pattern, and McNemar confirms the difference in the 3 repetitions (Section 8). The hierarchy between calibration strategies is kept in both models even though the absolute figures change.
2. **Turn the gap into a contribution of its own.** The configuration with which quantization tools and a good part of the literature report accuracy (`output_quant` not always enabled) is not necessarily the same one that achieves real acceleration in a general-purpose runtime — making it deployable costs accuracy, and that cost depends strongly on the architecture **and on the chosen calibration, not just on the bit-width** (Section 7.6.2: DeiT-Tiny combined at INT6 loses only 1.5pp, whereas baseline at the same bit-width loses ~20pp). Added to the fact that the CUDA EP does not accelerate even with ORT's native tool, it is a genuine finding about the gap between simulated quantization and real acceleration, measured on heterogeneous hardware — it fits directly with IoT-J's interest and with criticism 9.

**Update after the measurement on the Jetson (Section 7.8):** pillar (b) must be formulated with nuance: the speedup that INT8 achieves with a general-purpose runtime depends on the power mode and can disappear, and the energy saving is what holds in all the measured cases.

With this, the paper rests on three pillars, not two: (a) sensitivity to calibration with causal diagnosis (Sections 1-6), (b) the gap between simulated accuracy and real speedup in standard toolchains (Section 7), and (c) energy efficiency measured on heterogeneous hardware (performance/energy on edge devices, once that measurement is complete).

### 7.6.2 INT6/INT4 with `output_quant`: no performance benefit, severe accuracy cost — they stay out of the deployment set

Confirmed (24 runs, official 2000-image subset, n=3 in the stochastic configurations):

| Configuration | Calibration | Top-1 `output_quant` | Top-5 | Previous regime (without `output_quant`/BN) |
|---|---|---|---|---|
| DeiT-Tiny baseline INT6 | minmax | 37.63 ± 0.24% | 61.45 ± 0.77% | 57.80 ± 1.44% |
| DeiT-Tiny baseline INT4 | minmax | 0.05 ± 0.00% | 0.62 ± 0.03% | 0.17 ± 0.16% |
| DeiT-Tiny combined INT6 | mse | **68.78 ± 0.21%** | 88.67 ± 0.35% | 70.32 ± 0.15% |
| DeiT-Tiny combined INT4 | mse | 14.00 ± 1.28% | 32.87 ± 2.25% | 34.33 ± 0.90% |
| MobileViT-S baseline INT6 | minmax | 0.12 ± 0.10% | 0.33 ± 0.08% | 0.12 ± 0.03% |
| MobileViT-S baseline INT4 | minmax | 0.10 ± 0.00% | 0.50 ± 0.00% | 0.07 ± 0.06% |
| MobileViT-S combined INT6 | percentile | 0.15 ± 0.13% | 0.57 ± 0.06% | 22.07 ± 0.46% |
| MobileViT-S combined INT4 | percentile | 0.10 ± 0.00% | 0.50 ± 0.00% | 0.05 ± 0.09% |

**Only DeiT-Tiny combined INT6 keeps useful accuracy** (68.78%, −1.5pp versus the previous regime, far above the ±0.2pp noise). The rest collapses: DeiT-Tiny INT4 drops severely in both configurations; MobileViT-S collapses to chance level in the 4 combinations, including `combined INT6`, which in the previous regime did work partially (22.07%→0.15%).

**The cost of `output_quant` depends strongly on the calibration, not just on the bit-width** — in DeiT-Tiny INT8, with `mse` the cost is minimal (−0.08pp, n=3, Section 7.6) versus −0.67pp with baseline (n=3). At INT6, the same asymmetry is repeated in a much more marked way: the combined one (`mse`) loses only 1.5pp versus the previous regime, whereas the baseline at the same bit-width loses ~20pp. It is not "more aggressive = penalizes more" uniformly — a good calibration keeps absorbing a large part of the cost of `output_quant` even at aggressive bit-widths; it only collapses when in addition the bit-width is too low for any calibration to compensate (INT4, and in MobileViT-S, even INT6).

**Decision: the deployment set with `output_quant` is limited to INT8** — INT6/INT4 contribute nothing when deployed: they do not accelerate more than INT8 (fusion confirmed in full but without speed gain, Section 7.4.2) and here it is confirmed that in addition they lose accuracy severely in most configurations. There is no reason, of performance or accuracy, to consider them in the real deployment.

**Declared caveats:**
- The "previous regime" column mixes two changes in MobileViT-S (without `output_quant` and without BN folding at the same time) — in DeiT-Tiny (no BN) the difference is only `output_quant`, a clean comparison.
- The "combined" configuration was selected with data from the regime without `output_quant` — in theory it might not be the optimal one under this regime. It is not pursued, given that the decision to exclude INT6/INT4 from deployment does not depend on which calibration is used (the performance argument alone already justifies it).

### 7.6.3 Why the deployment set does not measure INT6/INT4 on more devices — answer already prepared for a reviewer

**INT6/INT4 stay out of the deployment set entirely** (not only for accuracy, Section 7.6.2 — their performance is not measured on more devices either; the fusion-equivalence argument with INT8 that was used to justify it does not hold on x86, see below):

**Logical argument, refuted on x86:** INT6/INT4 share with INT8 the same 8-bit container, the same operator types (`QLinearConv`/`QGemm`) and only differ in the range clipped by the `Clip` node (a constant attribute, Section 6.6.1 of the architecture), so it could be expected that any backend would fuse INT6/INT4 if it fuses INT8. The cluster data contradict it: on x86, with BN folded-as-identity and repetition 1 (`results/investigations/vit/fusion_probes/int8_kernel_probe_phase3_{deit_tiny,mobilevit}.json`), (i) DeiT-Tiny INT6/INT4 (baseline and combined) runs 49 integer `QGemm` but leaves the embedding `Conv` in float (49/50, versus the 50/50 of INT8 on ARM), and (ii) MobileViT-S baseline INT6/INT4 leaves the 35 `Conv` in float (37/72 integer kernels; 1.38× slower than FP32), whereas MobileViT-S combined INT6/INT4 does fuse all 72 (35 `QLinearConv` + 37 `QGemm`; 1.16× faster than FP32). Fusion on x86 therefore depends on something more than the bit-width, and a fused INT8 does not guarantee a fused INT6/INT4. Which trait of the artifact determines it is not verified.

**Confirmed empirically only on ARM:** on Jetson, the speedups of INT8/INT6/INT4 were practically identical (DeiT-Tiny: 1.66×/1.64×/1.64×; MobileViT-S: 1.41×/1.39×/1.40×, Section 7.4.2), with complete fusion in the 8 configurations. On x86 it does not hold (see above).

**Consequence for a new device:** having confirmed that INT8 fuses on one device is not enough to take INT6/INT4 as good there; they would have to be probed separately. This point reinforces, not weakens, the decision to leave them out of the deployment set: their exclusion rests on accuracy (Section 7.6.2) and on the fact that on ARM they do not accelerate more than INT8, not on a fusion-equivalence argument between bit-widths.

**Final scope of the deployment set, without INT6/INT4:**

| Artifact | Model | Purpose |
|---|---|---|
| FP32 | DeiT-Tiny | speedup reference |
| INT8 (`mse`) | DeiT-Tiny | deployment core |
| INT8 `per_channel` (on `mse`) | DeiT-Tiny | criticism 9 |
| INT8 `asymmetric` (on `mse`) | DeiT-Tiny | criticism 9 |
| FP32 | MobileViT-S | speedup reference |
| INT8 (`percentile`+BN) `per_channel`+`asymmetric` | MobileViT-S | deployment reference |
| INT8 (`percentile`+BN) `per_channel`+`symmetric` | MobileViT-S | criticism 9 (symmetry variant) |
| INT8 (`percentile`+BN) `per_tensor`+`asymmetric` | MobileViT-S | criticism 9 (granularity variant) |
| INT8 (`mse`) without `output_quant` (ablation) | DeiT-Tiny | fallback reference |
| INT8 (`percentile`) without `output_quant` (ablation) | MobileViT-S | fallback reference |

10 artifacts: 8 INT8 with `output_quant` and 2 ablation references without `output_quant` (fallback). The 2 references go in their original path (`artifacts/vit/`) and are the only ones in that folder that travel to the device; they allow measuring on heterogeneous hardware the cost of the fallback against FP32 and against the real speedup. DeiT-Tiny's (`mse`) is the same calibration as the deployment base, without `output_quant`; MobileViT-S's (the ablation's `percentile`, per-tensor and symmetric) differs from the deployment core in four things: `output_quant`, BN folding, granularity and symmetry. It documents the cost of the fallback against FP32, but does not allow a strict model-by-model comparison with the core; in DeiT-Tiny, on the other hand, the only difference is `output_quant`. Each variant is applied on top of its model's already chosen calibration, so that against the core artifact only that factor changes.

### 7.6.4 Complete deployment set: fusion, latency and accuracy (criticism 9) and MobileViT-S reference

**Fusion and latency against FP32** (CPU, 1 thread; dispersion between rounds of 0.1 to 1 ms). **The DeiT-Tiny rows of this table predate the one-node GELU fix and are superseded by those of Section 7.8.1** (at 1 thread: 1.75×/1.74×/1.75× on Jetson and 1.36-1.37× on x86); the accuracy does not change. The MobileViT-S ones remain valid, and all are at 1 thread, not at the production threads (Section 7.8). Probe verdict `MIXED` in all cases, only because the attention `MatMul`s remain in float by design (Section 2.0.1 of the ViT protocol). DeiT-Tiny fuses as 1 `QLinearConv` + 49 `QGemm`; MobileViT-S, as 35 `QLinearConv` + 37 `QGemm`.

| Artifact | Fused layers | Jetson ARM | Cluster x86 |
|---|---|---|---|
| DeiT-Tiny core (`mse`) | 50/50 | 1.66× (83.05 vs. 137.69 ms) | 1.18× |
| DeiT-Tiny `per_channel` | 50/50 | 1.64× (84.06 ms) | 1.16× |
| DeiT-Tiny `asymmetric` | 50/50 | 1.65× (83.27 ms) | 1.24× |
| MobileViT-S core (`per_channel`+`asymmetric`) | 72/72 | 1.37× (202.34 vs. 277.21 ms) | 1.18× |
| MobileViT-S `per_channel`+`symmetric` (symmetry var.) | 72/72 | 1.41× (196.46 ms) | 1.15× |
| MobileViT-S `per_tensor`+`asymmetric` (granularity var.) | 72/72 | 1.37× (201.90 ms) | 1.18× |

**Reading for criticism 9 (does the granularity/symmetry overhead depend on the hardware?):**
- `per_channel`: no speed cost in either environment (identical to the corresponding `per_tensor` variant).
- `asymmetric` does depend on the hardware, independently of the granularity. On ARM it costs ~3% (201.90-202.34 vs. 196.26-196.46 ms) and nothing in DeiT-Tiny; on x86 it is the fastest of the three, with fewer Q/DQ nodes (159 vs. 183 in MobileViT-S; 112 vs. 136 in DeiT-Tiny). Proposed explanation, not verified: x86 needs to convert signed activations to unsigned. It would be the same x86/ARM divergence of Section 7.4.1, now with real integer kernels.
- **Limit:** the "x86" of this table is the cluster's CPU, not an edge device. Claiming the hardware dependence in the paper requires the data from the n150 and the rest of the devices of the real measurement; this table is a preliminary diagnosis.

**Accuracy** (2000 images, n=3, mean ± sample standard deviation):

| Configuration | Top-1 | Top-5 | Repetitions (top-1) |
|---|---|---|---|
| DeiT-Tiny core (`mse`) | 73.35 ± 0.20 | 91.03 ± 0.13 | 73.15 / 73.55 / 73.35 |
| DeiT-Tiny `per_channel` | 73.42 ± 0.12 | 90.92 ± 0.12 | 73.55 / 73.35 / 73.35 |
| DeiT-Tiny `asymmetric` | 73.38 ± 0.24 | 91.17 ± 0.03 | 73.30 / 73.20 / 73.65 |
| MobileViT-S core (`per_channel`+`asymmetric`) | 63.33 ± 0.60 | 91.17 ± 0.10 | 62.65 / 63.75 / 63.60 |
| MobileViT-S `per_channel`+`symmetric` (former core) | 58.35 ± 4.01 | 85.10 ± 1.66 | 62.80 / 57.25 / 55.00 |
| MobileViT-S `per_tensor`+`asymmetric` | 52.13 ± 1.03 | 80.63 ± 0.30 | 50.95 / 52.60 / 52.85 |
| MobileViT-S `per_tensor`+`symmetric` (calibration baseline) | 47.48 ± 2.09 | 74.93 ± 1.56 | 49.00 / 48.35 / 45.10 |
| MobileViT-S baseline (`minmax`) + BN | 0.42 ± 0.20 | 1.42 ± 0.33 | 0.60 / 0.45 / 0.20 |

The DeiT-Tiny core and the three MobileViT-S configurations already measured before (`per_tensor`/`symmetric`, `per_channel`/`symmetric` and `per_tensor`/`asymmetric`) reproduce exactly the previous figures repetition by repetition, which confirms that the deployment manifest builds the same models. MobileViT-S's current core (`per_channel`/`asymmetric`) is a new configuration and has no previous figures to compare with.

**Pre-registered rule** (difference > std_a + std_b), applied twice over the 4 granularity×symmetry combinations of MobileViT-S. DeiT-Tiny: neither `per_channel` nor `asymmetric` separates from `mse` (all at 73.4 ± 0.2); `mse` is kept. MobileViT-S: first `per_channel`+`symmetric` beat `per_tensor`+`symmetric` (+10.87 pp, margin 6.10); then, with `per_channel`+`symmetric` as the reference, `per_channel`+`asymmetric` beats it too (+4.98 pp, margin 4.61 = 4.01+0.60).

**Decision: MobileViT-S's deployment reference is `percentile` + `per_channel` + `asymmetric` + BN folding** (63.33 ± 0.60%), the best and most stable of the 4 granularity×symmetry combinations tested on top of `percentile`. The original `percentile` core (per_tensor, symmetric) was chosen with data from the regime without `output_quant`, and under `output_quant` it stops being the best configuration; the same rule and the same criterion as in DeiT-Tiny are applied (each model with its best confirmed configuration). The difference with the best non-accelerated configuration tested (68.38%) drops from ~21 pp to ~5 pp; it is not a clean cost of acceleration, because both differ in `output_quant`, BN folding, granularity and symmetry at once — word it as "difference between the best deployable configuration and the best non-accelerated one tested". On the devices, the core and the 2 combinations that differ from it in a single factor are measured (`per_channel`+`symmetric` = symmetry variant; `per_tensor`+`asymmetric` = granularity variant), enough for criticism 9. `per_tensor`+`symmetric` differs in two factors and is not measured on devices.

**Declared caveats:**
- The selection reuses the same 3 repetitions as the comparison, twice in a row (first `per_channel` vs. `per_tensor`, then `asymmetric` vs. `symmetric` on the winner): somewhat more optimism than a single comparison, given that the 4 combinations of the 2×2 factorial were explored before fixing the core — small compared with the margins obtained, but declare the procedure.
- It is the complete granularity×symmetry factorial on `percentile` (4 of 4 combinations measured), not an open-ended search of configurations.
- A single thread, and without data from the n150.

**Effect on H1.** In the ablation (without `output_quant`), `per_channel` had no effect in any model (hypothesis 3, Section 2). With `output_quant` and folded BN it recovers ~11 pp in MobileViT-S and does not change DeiT-Tiny (no BN; at INT8 there is hardly any degradation to recover). Plausible mechanism, not verified: the folding moves the BN scale into the weights and makes their per-channel ranges heterogeneous, something that per-channel quantization corrects and per-tensor does not; it fits with what was noted in 7.4.1 about EfficientViT-B1's sinking with the folding. The per-channel weight metrics of Phase 4 could check it. H1 remains refuted in the ablation regime and supported, for MobileViT-S, in the deployment one.

### 7.7 Status of the verifications

**Open**
- **Contrast the speedup with the colleague's devices.** It has a single measurement per row and there is no thread sweep. Its figures inherit the variability measured on the Jetson (up to ~16 % in MobileViT-S's INT8 at 12 threads): do not interpret differences smaller than ~15 % in INT8 with many threads until its noise is quantified on each device.
- **Unverified mechanisms:** the placement of the thread that runs the serial part and the frequency governor (Section 7.8.5), the possible power cap of the 15 W mode and the origin of MobileViT-S's `Transpose`s. Declare as hypotheses, not as findings.

**Resolved**
- **DeiT-Tiny reference figure** (INT8 baseline + `output_quant`): 72.23 ± 0.28 % (n = 3), consistent with repetition 1 (72.10 %); the comparison with INT6/INT4 of Section 7.6.2 uses means of 3 repetitions.
- **DeiT-Tiny fusion with `mse`:** complete in both environments (50/50 layers, 1 `QLinearConv` + 49 `QGemm`) and latency identical to the baseline's; `mse` only changes scales, not the graph.
- **Fallback references:** without output `Q`, the fusion probe gives `FALLBACK` with 0 integer kernels, and they are slower than their FP32 in both environments: DeiT-Tiny 1.24× and MobileViT-S 1.26× on the cluster; 1.12× and 1.13× on the Jetson.
- **Speedup with the real threads:** resolved on the Jetson (Section 7.8); the speedup disappears in several modes.
- **Thermal hypothesis of the variability between invocations:** verified and ruled out (Section 7.8.5).
- **`--dry-run` of the deployment package:** verified on the final zip: 10 rows (8 deployment ones and the 2 fallback references), `devices.example.yaml` included and `devices.yaml` not.
- **Additional artifacts:** the 4 (`per_channel` and `asymmetric` with `output_quant`, in DeiT-Tiny and MobileViT-S) are generated and verified (Section 7.6.4). They answer criticism 9 (whether the granularity and symmetry overhead depends on the hardware) with real integer kernels, the only possible route, because the measurement of the ablation's granularity and symmetry at the edge was withdrawn (always fallback there; Section 2.1 of the ViT protocol).
- **DeiT-Tiny enters with `mse`:** 73.35 ± 0.20 % (n = 3) versus 72.23 ± 0.28 % for the baseline, a difference of means of 1.12 pp against a combined margin of 0.48 pp, so the pre-registered selection criterion is met. **The McNemar test (Section 8) does not confirm it:** the sign coincides in the 3 repetitions (+0.8 to +1.5 pp), but only 1 of 3 is significant after Holm, and in the ablation (INT8 without `output_quant`) none. The selection is kept, because McNemar is confirmatory and does not rewrite selections. The cost of `output_quant` with `mse` is minimal (−0.08 pp) versus −0.67 pp with the baseline, and the figure ends up very close to FP32 (73.35 % versus 73.50 %).
- **`output_quant` as an explicit parameter:** it is an explicit parameter of `build_quantized_model` and of the whole chain, disabled by default; the deployment builder passes it from each manifest row and the ablation CLI does not pass it (there are tests that fix it). None of the 99 ablation artifacts (63 from Phases 1-2 and 36 from 3) has a `Q` after `Conv`/`Gemm`.

**Notes that are kept**
- **The fallback → integer kernels gap is larger on ARM than on x86** (at 1 thread): DeiT-Tiny 1.86× and MobileViT-S 1.60× on the Jetson, versus ~1.5× on the cluster; it is coherent with ARM starting from more per-core speedup margin (1.66× and 1.41× versus 1.18× and 1.15×). It reinforces the platform heterogeneity argument of Section 7.6.1.
- **Fusion on RISC-V (VisionFive 2, 4 threads; diagnostic probe `probe_int8_kernels`):** the quantized layers also run with integer kernels, as on x86 and ARM: 50/50 in DeiT-Tiny (49 `QGemm` and 1 `QLinearConv`; the 24 attention MatMuls remain in float, as on the other platforms) and 72/72 in MobileViT-S. The probe used the `per_channel_symmetric` variant of the two models (not the core of either), 4 threads and 3 rounds of 10 repetitions, so its timings are not campaign ones: the RISC-V ones come from `run_batch` with the deployment manifest (10 rows, device `visionfive_2`).
- **Fusion on the CM4 (Raspberry Pi CM4, ARM Cortex-A72 without `asimddp`, 4 threads; diagnostic probe `probe_int8_kernels`, output in `tmp_coworker_data/cm4-probe/int8-probe/`):** the quantized layers run with integer kernels, as on the Jetson (ARM with `asimddp`), x86 and RISC-V: 50/50 in DeiT-Tiny (49 `QGemm` and 1 `QLinearConv`; the 24 attention MatMuls remain in float, as on the other platforms) and 72/72 in MobileViT-S (37 `QGemm` and 35 `QLinearConv`). It coincides with the other platforms: fusion does not depend on `asimddp`. The probe used the same INT8 artifacts of the deployment regime (with `output_quant`) as on the VisionFive 2, with 4 threads, 3 rounds and 3 profiling repetitions, so its timings are not campaign ones: the CM4 ones come from the measurement campaign with the deployment manifest (device `kube01`).
- **Row labeling (when combining results):** the on-device runner leaves `phase` and `factor_varied` empty in the CSV (it only writes `run_id`); they are recovered when combining, by `run_id`. The 8 deployment rows carry `_deploy_` in the `run_id` and inherit `phase = deployment`. The 2 fallback references inherit their labels from the ablation (`phase2` / `calibration_algorithm` / `mse` or `percentile`): one has to filter explicitly by those two combinations to separate «deployment» from «fallback reference», filtering by `phase` is not enough.

### 7.8 Speedup with several threads and per power mode: what was measured on the Jetson

**Summary.** The 1-thread figures (1.66×, 1.41×, 1.75×) do not hold with the production threads. With the real power modes, the deployment INT8 **clearly accelerates only in the 15 W mode** (4 threads), stays at parity or below FP32 in the others, and **consumes less energy than FP32 in all the measured cases**. The speedup depends on the power mode, and the energy saving is the advantage that holds.

#### 7.8.1 One-node GELU fix (DeiT-Tiny)

With several threads DeiT-Tiny's INT8 came out slower than FP32 (0.81×). The profile showed 12 loose `Erf` nodes: in the INT8 graph, ORT duplicates the `DequantizeLinear` that feeds the GELU and does not fuse it into a single `Gelu`, as it does in FP32. Replacing that structure with a one-node GELU in the export **changes neither the arithmetic nor the accuracy** (difference exactly 0.0 in the 9 repetitions of the deployment set: 73.35 ± 0.20 / 73.42 ± 0.12 / 73.38 ± 0.24 % top-1) and the fusion is kept in full (50/50 layers, 12 `Gelu`, no `Erf`) on the cluster and on the Jetson. At 1 thread, the speedup goes from 1.66× to 1.75× on the Jetson (1.19× to 1.36× on x86). It is applied only to DeiT-Tiny. MobileViT-S uses SiLU (`QuickGelu` in FP32), with other bottlenecks (`Transpose`, `Q`/`DQ`), and is left as is. **It does not change any individual prediction either:** comparing image by image the predictions of the INT8 with the loose GELU and with the one-node GELU, there are 0 differences in top-1 and top-5 in the 12 combinations (the three deployment INT8 and the `minmax`, 3 repetitions each), so the McNemar tests are equally valid for the deployed artifacts.

#### 7.8.2 Scaling with threads (probe, 50 W mode, variable threads)

Profiling probe, medians of 3 rounds with 200 repetitions. The 4- and 8-thread points are noisy (FP32 rounds from 46 to 86 ms in DeiT-Tiny) and should not be read in detail.

| Threads | DeiT-Tiny: INT8 with one-node GELU versus FP32 | MobileViT-S: INT8 versus FP32 |
|---|---|---|
| 1 | 1.75× | 1.37× |
| 4 | 0.95× (noisy) | 1.13× (noisy) |
| 8 | 1.10-1.20× | 1.07× |
| 12 | 0.90-1.18× | 0.93× |

FP32 scales much more with threads than INT8 (from 1 to 12 threads: DeiT-Tiny 3.26× versus 2.12× after the GELU fix, in the 5-round profile of 7.8.3; MobileViT-S 3.8× versus 2.6×, in the probe). The probe prior to the fix, with DeiT-Tiny's INT8 without the one-node GELU, gave 3.5× versus 1.7×. In the profile, DeiT-Tiny's INT8/FP32 speedup goes from 1.72× with 1 thread to 1.12× with 12; `run_batch` gives 1.19× at 12 threads (medians of 3 sessions).

**The probe with N threads on 12 active cores does not reproduce the power modes.** MobileViT-S's FP32 with 8 threads gives 92 ms in the probe and 68-71 ms in the 30 W mode (8 active cores); DeiT-Tiny's, 41-46 ms versus 33.8 ms. Each mode changes the threads, the active cores and the frequency at once, so its result is not a thread curve. Two different claims, which must not be joined in one curve: (i) the speedup depends on the power mode (measured in 7.8.4); (ii) the scaling of INT8 is limited by operators that do not scale (mechanism, 7.8.3).

#### 7.8.3 Why INT8 scales worse (per-operator profile, 50 W mode, median of 5 rounds)

The compute operators scale (`QGemm` 2.8×, `QLinearConv` 4.6×, `Gelu` 4.2×, attention `MatMul` 3-4×), but others do not:

| Operator (1→12 threads) | DeiT-Tiny INT8 | MobileViT-S INT8 |
|---|---|---|
| `DequantizeLinear` | 2.59 → 2.97 ms (0.87×) | 10.76 → 11.41 ms (0.94×) |
| `QuantizeLinear` | 4.51 → 3.71 ms (1.22×) | 20.42 → 8.07 ms (2.53×) |
| `Transpose` | 1.51 → 2.67 ms (0.57×) | 21.51 → 18.23 ms (1.18×) |
| Profile total | 80.8 → 38.0 ms (2.12×) | 208.3 → 89.9 ms (2.32×) |

Fraction of time in operators that scale less than 2× and Amdahl's law prediction (maximum speedup with 12 threads if only those were serial):

| | At 1 thread | At 12 threads | Amdahl max. | Measured |
|---|---|---|---|---|
| DeiT-Tiny FP32 | 8 % | 28 % | 6.22× | 3.26× |
| DeiT-Tiny INT8 | 23 % | 48 % | 3.36× | 2.12× |
| MobileViT-S FP32 | 7 % | 19 % | 6.98× | 3.53× |
| MobileViT-S INT8 | 28 % | 55 % | 2.94× | 2.32× |

INT8 has almost three times as much time in operators that do not scale as FP32, and that alone already takes away roughly half of the scaling margin. Amdahl does not explain everything: the parts that do scale do not reach 12× either (between 2.7× and 4.6×), as in FP32.

**Cause, with its degree of certainty:**
- **`DequantizeLinear` is not split across threads in any node** (0 % of nodes in parallel in both models; direct reading of ORT's scheduling statistics). High certainty. It is behavior of ORT itself, which we do not control; it is only reduced with fewer DQ nodes.
- **`QuantizeLinear` and `Transpose` are split only in part** (26-50 % and 23-67 % of the nodes) and have a significant fixed cost per node (the "fixed cost + cost per byte" fit attributes 53-63 % of the time to the fixed cost). Medium certainty: the fit is approximate and in FP32 it gives an absurd result for lack of variety of sizes, which is discarded.
- **It does not look like bandwidth saturation** (medium certainty): these nodes move 4-10 GB/s effective, computed as input and output bytes divided by the node's time, not as real DRAM traffic. The nominal bandwidth of the AGX Orin is 204.8 GB/s (256-bit LPDDR5 at 6400 MT/s; datasheet figure, not verified here), a theoretical ceiling that the CPU does not reach, so 4-10 GB/s is 2-5 % of the ceiling and not a percentage of what the CPU can achieve.
- **Passes per tensor:** almost all these nodes are isolated (DeiT-Tiny: 123 single and 14 consecutive pairs; MobileViT-S: 131 and 73 pairs).
- **Why MobileViT-S loses more than DeiT-Tiny with many threads (after the GELU fix):** it has more time that does not scale (55 % versus 48 %): 72 `Q`, 87 `DQ` and 118 `Transpose` (versus the 48 `Transpose` of its FP32), with 21.5 ms of `Transpose` at 1 thread versus 4.4 ms in FP32. That those `Transpose` come from the NCHW/NHWC format conversion around `QLinearConv` is an unverified hypothesis.

**Limitations of the profile:** the enabled profile inflates the latencies, so its speedups (1.12× and 0.88× with 12 threads) are not those of the deployment; the figures for the paper come from 7.8.4. All diagnostics are from the Jetson in the 50 W mode.

#### 7.8.4 Deployment results per power mode (`run_batch`, Jetson CPU, 1000 repetitions)

**Source of the figures.** The figures of this section come from the **results CSV**, with one measurement per row, as on the rest of the devices. The repetitions are a **characterization of repeatability on the Jetson**, a separate diagnostic that serves to quantify the uncertainty and fix the reading rule, and it does not replace any value in the CSV. In the rows that were repeated, the CSV keeps the median-latency execution (a real execution), so its latency coincides exactly with the median of the three; the energy and the energy ratios differ from the medians by ≤ 0.02. It is a small asymmetry with respect to the devices with a single measurement, and it is declared.

**Design of the characterization.** Idle is measured once per `run_batch` invocation, so each repetition is a different session (Section 6.4.4 of the architecture). The FP32 and the INT8 core of both models in the three modes, the three INT8 of DeiT-Tiny at 50 W and the three variants of MobileViT-S at 15 W have 3 repetitions. Repetition 1 is the original batch; in DeiT-Tiny its FP32 and its INT8 come from different sessions (the INT8 was remeasured with `--force`), so its per-session speedup is not homogeneous and only in repetitions 2 and 3 is the same invocation guaranteed.

Core of each model (DeiT-Tiny: `mse`, per-tensor, symmetric; MobileViT-S: `percentile`, per-channel, asymmetric). CSV values; the last column is the range of the per-session speedup in the repeatability characterization:

| Model and mode (threads) | Latency FP32 → INT8 (ms) | Speedup | Net power FP32 → INT8 (W) | Energy per inference FP32 → INT8 (mJ) | Energy: INT8 consumes | Speedup range over 3 sessions |
|---|---|---|---|---|---|---|
| DeiT-Tiny, 15 W (4) | 74.2 → 48.6 | 1.53× | 1.16 → 1.14 | 86.8 → 56.0 | 1.55× less | 1.40-1.53 |
| DeiT-Tiny, 30 W (8) | 33.8 → 33.1 | 1.02× | 4.33 → 2.67 | 148.9 → 89.7 | 1.66× less | 1.02-1.05 |
| DeiT-Tiny, 50 W (12) | 38.1 → 32.1 | 1.19× | 3.77 → 2.72 | 145.0 → 88.7 | 1.64× less | 1.19-1.21 |
| MobileViT-S, 15 W (4) | 143.2 → 129.3 | 1.11× | 1.17 → 0.81 | 168.5 → 105.8 | 1.59× less | 1.09-1.14 |
| MobileViT-S, 30 W (8) | 69.4 → 78.7 | 0.88× | 4.13 → 2.91 | 288.6 → 231.6 | 1.25× less | 0.88-0.90 |
| MobileViT-S, 50 W (12) | 75.1 → 82.0 | 0.92× | 3.54 → 2.86 | 267.4 → 237.2 | 1.13× less | 0.90-0.96 |

The DeiT-Tiny range at 15 W includes repetition 1 (1.40×, mixed sessions); with only repetitions 2 and 3, 1.53× and 1.53×.

**Reading rule, pre-registered before seeing the data.** Speedup if min(FP32)/max(INT8) > 1; slower if max(FP32)/min(INT8) < 1; no appreciable difference in the rest. This rule uses the ranges of the repeatability characterization, and therefore **only applies to the Jetson rows that have repetitions**; on the devices with a single measurement per row, the ratio is compared with the uncertainty threshold of that characterization. Classification: DeiT-Tiny accelerates at 15 W and at 50 W; MobileViT-S accelerates at 15 W and is slower at 30 W and at 50 W. Two borderline cases pass the rule by a margin of 0.2 % and 0.4 % (DeiT-Tiny at 30 W; MobileViT-S at 50 W): word them as "parity" and "no appreciable speedup", citing the rule and the median. Per session, DeiT-Tiny at 30 W is faster than FP32 in all 3 (1.016-1.052) and MobileViT-S at 50 W slower in all 3 (0.896-0.956).

**What can be claimed.**
- **Energy saving:** INT8 consumes less energy per inference than FP32 in the 6 cases, from 1.13× to 1.66× less, with a net power difference of 0.3-1.6 W far above the idle variation between sessions (0.08-0.12 W). It is the most solid conclusion.
- **Speedup dependent on the power mode:** DeiT-Tiny 1.53× / 1.02× / 1.19× and MobileViT-S 1.11× / 0.88× / 0.92× at 15 / 30 / 50 W. The pattern is not monotonic with threads because each mode also changes the frequency and the active cores.
- **Fallback references** (one measurement per mode): slower than FP32 in the 6 combinations (0.65-0.81×) and with more energy per inference (0.79-0.85×), the same sign as on the cluster and at 1 thread.

**What cannot be claimed.**
- **Ranking between variants at 12 threads:** in DeiT-Tiny at 50 W the three variants differ by up to 9 % with overlapping ranges (32.1 / 33.4 / 34.8 ms medians); indistinguishable. For criticism 9, the 1-thread probe (Section 7.6.4) and MobileViT-S serve.
- **Energy ranking between MobileViT-S variants at 15 W:** with nearly equal latencies, the energy varies by 10 % (105.8 / 106.9 / 116.8 mJ); not interpretable.
- **Variability between invocations, without a verified explanation.** MobileViT-S's INT8 rose from one repetition to the next (50 W: 78.7 / 82.0 / 83.4 ms; 30 W: 76.5 / 78.7 / 78.9; 15 W: 126.7 / 129.9 / 129.3), and DeiT-Tiny's FP32 at 15 W varied by 7 % between sessions (68.9 versus 74.4 and 74.2 ms) with the same power. A specific experiment (Section 7.8.5) showed that it is not a drift: at 50 W and 12 threads that INT8 oscillates between 78.6 and 91.1 ms (16 %) over 20 invocations in 3 sessions, with no trend. **The thermal hypothesis was refuted**; the cause is provisionally attributed, as an unverified hypothesis, to where the thread that runs the serial part lands (7.8.5).
- **The single-measurement rows** (variants, fallback references, MobileViT-S without repetitions) inherit the variability between invocations, which on the Jetson at 12 threads reaches 16 % in MobileViT-S's INT8 and 4-11 % in the DeiT-Tiny variants. With 3 sessions the observed range does not bound the real variability (at 15 W and 30 W, with 4 and 8 threads, the range of 3 sessions was 1-3 %, but it has not been checked with long series). Do not interpret differences between single-measurement rows smaller than ~15 % in INT8 configurations with many threads.
- **The 15 W may be power-limited:** four heavy rows have an almost identical gross power (3.19-3.20 W: the two FP32 and the two fallback references), which suggests a cap of the mode. If INT8 does not reach it, part of its advantage at 15 W would be frequency and not computation. Unverified hypothesis.

**A discarded anomaly.** The `per_tensor_asymmetric` row of DeiT-Tiny at 15 W came out with a net power of 1.583 W versus ~1.05 W for its siblings, with the same latency. With the criterion fixed before remeasuring it (≤ 1.27 W is a one-off episode), the remeasurement gave 1.028 W and it was replaced. The energy of the original measurement (78 mJ) was an outlier.

**Consequence for the paper (RQ3a).** The speedup that quantized INT8 achieves with a general-purpose runtime depends on the power mode (threads, cores and frequency) and can disappear; the energy saving is what holds. The mechanism identified (quantization operators that ORT runs in a single thread) is a limitation of the runtime, not of the quantization pipeline. The 1.66× figure at 1 thread is not representative of the deployment.

#### 7.8.5 Variability between invocations and ruling out the thermal hypothesis (MobileViT-S, 50 W)

**Why it was done.** MobileViT-S's INT8 core rose between the three repetitions (78.7 → 82.0 → 83.4 ms). The hypothesis was thermal. The temperatures of those sessions had not been recorded, so a new experiment was designed, with the criterion fixed before measuring.

**Design.** Same configuration (CPU, 50 W, 12 threads, `--with-power`, `--temp-cooldown`), on a fresh boot with uptime > 600 s, with `tegrastats` at 1 s during the whole session: (i) one invocation with the full 10-row manifest; (ii) a series of 5 identical invocations with the 2 MobileViT-S rows (FP32 and INT8 core); (iii) a cold control after returning to T0 + 1 °C (control not valid if it was not reached in 60 min). Classification, by precedence: not reproducible (range ≤ 2 %); inconclusive (rise < 3 % or FP32 equals or exceeds it); thermal refuted (the rise counts but is not accompanied by +2 °C nor −2 % of mean frequency); inconclusive because of invalid control; thermal accepted (accompanied and the control returns to < 2 %); and a separate `persistent` field for the control.

**Result.** Verdict: **thermal refuted.** INT8 rise between invocation 1 and 5: +15.0 % (FP32, +6.3 %); mean temperature of the INT8 row: +1.2 °C; mean frequency: −0.4 %. The `persistent` field came out "yes" (cold control at +6.3 %), but **it does not hold**: the control is a single invocation and the series oscillates with a CV of ~5-6 % (79.3 / 86.6 / 80.9 / 86.6 / 91.1 ms), above the criterion's 2 % threshold. The thermal refutation does not depend on that threshold: the latency varies by 15 % while the temperature varies ~1 °C. Arm of the 10-row batch: INT8 9.7 % slower than invocation 1 and FP32 7.3 % slower; inconclusive, within the range of the series itself (79.25-91.10 ms), so the batch composition can be neither identified nor ruled out.

**What is concluded.** There is no systematic drift. There is **variability between invocations** of up to 16 % (78.6-91.1 ms over 20 invocations in 3 sessions), of which the original 78.7 → 82.0 → 83.4 is one sample. The unmet sibling rule at 50 W (`per_channel_symmetric`, −5.1 %) is explained by this variability, not by an anomalous row. What is robust holds: in 14 invocations paired with their FP32, MobileViT-S's INT8 is slower than FP32 (1.03-1.15×).

**Exploratory data (not part of the criterion).**
- Mean CPU utilization × latency is almost constant (3243-3455 in the 7 INT8 invocations, ±3 %; 4156-4286 in the FP32 ones, ±2 %): INT8 does not compute more slowly when it comes out slow, it has more waiting time. The correlation of −1.0 with n = 7 is an algebraic consequence of that constant product and is not independent evidence.
- In each invocation one core is near 100 % (or between 59 and 93 %) and the rest between 25 and 40 %, and that core changes between invocations.
- The three 4-core clusters have their own frequencies that change between invocations (744-1497 MHz), with no visible relation to latency; in the slowest one (91.1 ms) the busy core was at 1497 MHz.

**Hypothesis, unverified.** The INT8 latency varies according to where the thread that runs the serial part (`DequantizeLinear`, Section 7.8.3) lands, and INT8 is very sensitive because ~55 % of its time does not scale. Two caveats: (i) it does not explain why DeiT-Tiny's INT8, with 48 % of time that does not scale, was stable in 3 sessions at 50 W (31.5-32.2 ms), although 3 sessions are not comparable to 20 invocations; (ii) "it is not the frequency" rests on the mean frequency of all cores, which includes idle cores, and on a single run; if the serial thread migrated to a cluster whose frequency has not risen, the effect would be one of frequency and that mean would not see it. Any affinity test should also fix the frequency. It remains as future work, without changing any figure of this document.

**Scope.** The experiment reproduces consecutive invocations on the same boot, not the state of the original batch (another day, another boot, another idle).

### 7.9 ViT on the Jetson's GPU (CUDA EP)

**Measurement.** The 10 rows of the deployment set in the three modes (15, 30 and 50 W), with `--repeats 5000`. With N = 1000 the window would have been ~6-10 s and the fixed cost (CUDA context, warm-up) would weigh too much. The data are coherent (`energy_per_inference = energy_net_j / N`, `perf_per_watt = N / energy_net_j`, no empty fields). The fixed-time fraction, `(window − N·latency)/window`, is ≤ 3 % in all INT8 rows and in MobileViT-S's FP32 (4.3 % at most); only DeiT-Tiny's FP32, the fastest ones, exceed 5 %: 5.5 % (15 W), 7.9 % (30 W) and 10.3 % (50 W). Their energies may be somewhat overestimated; what was measured in LLM (the fixed cost spends little energy) is not directly transferable, because the CUDA initialization can consume power. It does not change any conclusion below.

Latency (ms) and energy per inference (mJ) from each backend's CSV. The INT8 core is `mse` in DeiT-Tiny and `percentile` per-channel asymmetric in MobileViT-S; the GPU INT8 runs in emulation:

| Model and mode | CPU FP32 | CPU INT8 | GPU FP32 | GPU INT8 (emulated) |
|---|---|---|---|---|
| DeiT-Tiny 15 W | 74.2 / 86.8 | 48.6 / 56.0 | 11.7 / 35.6 | 23.3 / 65.3 |
| DeiT-Tiny 30 W | 33.8 / 148.9 | 33.1 / 89.7 | 7.3 / 35.0 | 14.4 / 63.3 |
| DeiT-Tiny 50 W | 38.1 / 145.0 | 32.1 / 88.7 | 5.6 / 42.2 | 10.9 / 76.7 |
| MobileViT-S 15 W | 143.2 / 168.5 | 129.3 / 105.8 | 23.4 / 85.4 | 50.3 / 170.3 |
| MobileViT-S 30 W | 69.4 / 288.6 | 78.7 / 231.6 | 13.3 / 69.1 | 29.6 / 142.4 |
| MobileViT-S 50 W | 75.1 / 267.4 | 82.0 / 237.2 | 6.7 / 87.5 | 15.5 / 178.5 |

**Results.**
- **FP32 on GPU dominates INT8 on CPU** in the 6 cases: from 4.2× to 12.2× faster and from 1.24× to 3.4× less energy. The narrowest margin is MobileViT-S's energy at 15 W (1.24×), which was confirmed with 3 repetitions: with the criterion fixed before measuring, the maximum of the 3 FP32-on-GPU energies (85.4 mJ) is lower than the minimum of the 3 INT8-on-CPU ones (104.1 mJ), an 18 % margin.
- **INT8 on GPU is ~2× slower and spends ~2× more energy than its FP32** (latency 1.9-2.3×, energy 1.8-2.1×, in both models and the three modes). It confirms, with N = 5000, three modes and energy, the emulation of Section 7.5.
- **`output_quant` worsens the emulation on GPU:** the deployment artifacts are 1.3-1.5× slower than the fallback reference (DeiT-Tiny at 15 W: 23.3 versus 18.0 ms; MobileViT-S: 50.3 versus 33.8 ms), because they add Q/DQ nodes.
- **The variants** `per_channel` and `asymmetric` stay within ~6 % of each other (a single measurement): they cannot be ranked.

**Repeatability on GPU (MobileViT-S, 15 W, 3 invocations of 5000 repetitions).** FP32: latency 23.42 / 23.43 / 23.47 ms (CV 0.1 %), energy 84.4 / 85.4 / 82.5 mJ (CV 1.8 %). INT8 core: 50.25 / 50.29 / 50.09 ms (CV 0.2 %), 170.3 / 178.9 / 174.6 mJ (CV 2.5 %). The latency on GPU is much more repeatable than that of INT8 on CPU with 12 threads (up to 16 %). Idle varied 0.12 W between invocations (1.994 / 1.954 / 2.074 W). The CSV keeps the median-latency execution (FP32: the second; INT8: the first, which is not the median-power one, with a 2.5 % difference in energy).

**Limits.** GPU repeatability was only measured in this combination; the rest are single measurements. The conclusion is for a device with a GPU; on devices without one, the relevant comparison remains INT8 versus FP32 on CPU (Section 7.8). **Consequence for the paper (RQ3a):** on the Jetson, the best configuration for these ViTs in latency and energy is FP32 on GPU, and INT8 on CPU only saves energy compared with FP32 on CPU.

### 7.10 On-disk size of the deployment set

Source: `results/artifact_sizes.csv` (`stat` of the ONNX files that the device loads, in MiB; Section 6.7 of the architecture). DeiT-Tiny: FP32 21.9 MiB and INT8 5.9-6.0 MiB (3.7×). MobileViT-S: FP32 21.4 MiB and INT8 5.8-5.9 MiB (3.6-3.7×). The two fallback references (without `output_quant`) weigh 5.8 MiB, the same as the deployment INT8. **The size is reduced about 3.7× even though INT8 does not accelerate** (Section 7.8.4: in MobileViT-S it is slower than FP32 at 30 and 50 W), so the memory saving does not depend on there being acceleration.

### 7.11 Performance and energy of the deployment set on all devices

**Source and selection.** `results/combined/edge_results_combined.csv`, one measurement per row: the 8 deployment rows (`_deploy_` in `run_id`) and the 2 fallback references of each device, 100 rows in total. In the combined CSV the `phase`, `factor_varied` and `calibration_algorithm` columns are empty in all the ViT rows, so the references are selected by `run_id` (`vit_deit_tiny_phase2_calibration_algorithm_mse_int8` and `vit_mobilevit_phase2_calibration_algorithm_percentile_int8`), checking that it coincides with `role = referencia en fallback` (the data value of the `role` column, kept in Spanish; see `docs/GLOSSARY.md`). For the sessions rule the repetitions of `results/diagnostics/dispersion_*/rep{1,2,3}.csv` are used, as in 7.8.4. Script: `analysis/vit_deployment_summary.py`; aggregated values: `plots/out/vit_deployment_summary.csv`.

**Prior checks (all passed).** In the 100 rows, `num_inferences_measured == requested_repeats` (1000 on CPU and 5000 on GPU) and the net energy is valid (> 0). There is no duplicate row per device and configuration, and none is missing. `perf_per_watt = N / energy_net_j` and `energy_per_inference = energy_net_j / N` in all, and the net power coincides with `energy_net_j / window`. In the Jetson repetitions, N = 1000 and the energy is valid in all. In the rows with 3 sessions, the CSV latency is one of the three.

**Measurement window** (`energy_raw_j / avg_power_w_raw`, in s), with the minimum and the maximum per device:

| Device | Minimum window (row) | Maximum window |
|---|---|---|
| Jetson CPU 15 W | 49.3 (DeiT-Tiny, INT8 core) | 177.3 |
| Jetson CPU 30 W | 33.5 (DeiT-Tiny, INT8 core) | 106.1 |
| Jetson CPU 50 W | 32.6 (DeiT-Tiny, INT8 core) | 99.1 |
| Jetson GPU 15 W | 61.9 (DeiT-Tiny, FP32) | 253.9 |
| Jetson GPU 30 W | 39.6 (DeiT-Tiny, FP32) | 149.7 |
| Jetson GPU 50 W | 31.1 (DeiT-Tiny, FP32) | 79.4 |
| n150 | 19.7 (DeiT-Tiny, INT8 core) | 78.1 |
| rk1-02 | 28.0 (DeiT-Tiny, INT8 core) | 98.9 |
| visionfive_2 | 760.7 (DeiT-Tiny, FP32) | 3093.6 |
| kube01 (CM4) | 140.1 (DeiT-Tiny, symmetry var.) | 437.2 |

**Conventions.** Ratios FP32/X (ratios 1 and 2) or core/X (ratio 3), so > 1 favors X. Each metrics cell is mean latency per inference (ms) / net energy per inference (mJ) / mean net power (W). INT8 core: DeiT-Tiny `mse` per-tensor symmetric; MobileViT-S `percentile` per-channel asymmetric. Granularity variant: DeiT-Tiny per-channel symmetric and MobileViT-S per-tensor asymmetric. Symmetry variant: DeiT-Tiny per-tensor asymmetric and MobileViT-S per-channel symmetric.

**Classification.** On the Jetson's CPU, when the two rows of the ratio have 3 sessions, the sessions rule of 7.8.4 is applied (min(ref)/max(X) > 1 → favors X; max(ref)/min(X) < 1 → against; otherwise, no appreciable difference), and the table gives the two bounds. This is the case of FP32 versus the core in the three modes and, in ratio 3, of DeiT-Tiny at 50 W and MobileViT-S at 15 W. In the rest, which has a single measurement, the ±15 % band is applied (|ratio − 1| ≤ 0.15 → no appreciable difference), **fixed after seeing the Jetson's variability** (7.7, 7.8.4). Labels: «acelera» and «más lento» for latency; for energy, the same rule with «menos energía» and «más energía» (the label strings are data values written in the summary CSV and are kept in Spanish in the tables below; the English correspondence is in `docs/GLOSSARY.md`: acelera = speeds up, más lento = slower, menos energía = less energy, más energía = more energy, sin diferencia apreciable = no appreciable difference). **Three points are not pre-registered:** applying the two rules to energy, applying the sessions rule to ratio 3, and taking the symmetric band on the ratio instead of on its inverse. With the band on the inverse (< 0.870), two cases would go from «sin diferencia apreciable» to «más energía»: the energy of MobileViT-S's fallback on the Jetson's CPU at 50 W (0.854) and on the visionfive_2 (0.850).

#### 7.11.1 DeiT-Tiny

| Device and mode (threads) | FP32 | INT8 core | Fallback | Core: speedup / energy | Core: classification | Core: session bounds (lat. / energy) | Fallback: speedup / energy | Fallback: classification |
|---|---|---|---|---|---|---|---|---|
| Jetson CPU 15 W (4) | 74.2 / 86.8 / 1.16 | 48.6 / 56.0 / 1.14 | 94.5 / 110.4 / 1.16 | 1.53× / 1.55× | acelera / menos energía | 1.396–1.533 / 1.437–1.650 | 0.79× / 0.79× | más lento / más energía |
| Jetson CPU 30 W (8) | 33.8 / 148.9 / 4.33 | 33.1 / 89.7 / 2.67 | 52.1 / 185.1 / 3.52 | 1.02× / 1.66× | acelera / menos energía | 1.002–1.067 / 1.525–1.665 | 0.65× / 0.80× | más lento / más energía |
| Jetson CPU 50 W (12) | 38.1 / 145.0 / 3.77 | 32.1 / 88.7 / 2.72 | 51.6 / 173.9 / 3.34 | 1.19× / 1.64× | acelera / menos energía | 1.180–1.213 / 1.619–1.708 | 0.74× / 0.83× | más lento / más energía |
| n150 | 24.5 / 393.1 / 15.91 | 19.5 / 249.7 / 12.67 | 32.0 / 469.9 / 14.60 | 1.26× / 1.57× | acelera / menos energía | — | 0.77× / 0.84× | más lento / más energía |
| rk1-02 | 37.2 / 299.4 / 7.98 | 27.5 / 182.6 / 6.52 | 47.5 / 360.2 / 7.51 | 1.35× / 1.64× | acelera / menos energía | — | 0.78× / 0.83× | más lento / más energía |
| visionfive_2 (4) | 759.0 / 1616.7 / 2.13 | 1133.3 / 2408.2 / 2.12 | 937.5 / 1804.0 / 1.92 | 0.67× / 0.67× | más lento / más energía | — | 0.81× / 0.90× | más lento / sin diferencia apreciable |
| kube01, CM4 (4) | 187.6 / 426.6 / 2.26 | 139.8 / 317.8 / 2.25 | 206.8 / 469.8 / 2.26 | 1.34× / 1.34× | acelera / menos energía | — | 0.91× / 0.91× | sin diferencia apreciable / sin diferencia apreciable |

#### 7.11.2 MobileViT-S

| Device and mode (threads) | FP32 | INT8 core | Fallback | Core: speedup / energy | Core: classification | Core: session bounds (lat. / energy) | Fallback: speedup / energy | Fallback: classification |
|---|---|---|---|---|---|---|---|---|
| Jetson CPU 15 W (4) | 143.2 / 168.5 / 1.17 | 129.3 / 105.8 / 0.81 | 176.4 / 206.4 / 1.16 | 1.11× / 1.59× | acelera / menos energía | 1.082–1.135 / 1.372–1.618 | 0.81× / 0.82× | más lento / más energía |
| Jetson CPU 30 W (8) | 69.4 / 288.6 / 4.13 | 78.7 / 231.6 / 2.91 | 105.6 / 340.7 / 3.21 | 0.88× / 1.25× | más lento / menos energía | 0.863–0.930 / 1.149–1.306 | 0.66× / 0.85× | más lento / más energía |
| Jetson CPU 50 W (12) | 75.1 / 267.4 / 3.54 | 82.0 / 237.2 / 2.86 | 98.6 / 313.2 / 3.16 | 0.92× / 1.13× | más lento / menos energía | 0.896–0.996 / 1.088–1.164 | 0.76× / 0.85× | más lento / sin diferencia apreciable |
| n150 | 61.2 / 920.9 / 14.99 | 52.2 / 729.7 / 13.87 | 77.8 / 1123.4 / 14.39 | 1.17× / 1.26× | acelera / menos energía | — | 0.79× / 0.82× | más lento / más energía |
| rk1-02 | 76.7 / 603.4 / 7.83 | 75.7 / 505.5 / 6.60 | 98.4 / 678.7 / 6.86 | 1.01× / 1.19× | sin diferencia apreciable / menos energía | — | 0.78× / 0.89× | más lento / sin diferencia apreciable |
| visionfive_2 (4) | 2553.2 / 5344.0 / 2.09 | 2355.9 / 4943.3 / 2.09 | 3088.6 / 6284.3 / 2.03 | 1.08× / 1.08× | sin diferencia apreciable / sin diferencia apreciable | — | 0.83× / 0.85× | más lento / sin diferencia apreciable |
| kube01, CM4 (4) | 406.4 / 892.7 / 2.19 | 402.9 / 860.0 / 2.12 | 435.7 / 1011.2 / 2.31 | 1.01× / 1.04× | sin diferencia apreciable / sin diferencia apreciable | — | 0.93× / 0.88× | sin diferencia apreciable / sin diferencia apreciable |

The threads of the n150 and of the rk1-02 are not recorded locally (their `devices.yaml` does not travel); the 4 of the visionfive_2 come from the `devices.yaml` of the RISC-V probe (`tmp_coworker_data/riscv-probe/configs/`), and the 4 of the CM4, from `tmp_coworker_data/cm4-probe/methodology.txt`.

**Note on two borderline classifications (wording agreed for the paper).** With the literal sessions rule, DeiT-Tiny at 30 W comes out «acelera» (minimum bound 1.002, a 0.2 % margin) and MobileViT-S at 50 W comes out «más lento» (maximum bound 0.996, a 0.4 % margin). Both margins are much smaller than the variability between sessions measured on the Jetson (up to ~16 % in MobileViT-S's INT8 at 12 threads; Section 7.8.5), so in the paper they are worded as "parity" (DeiT-Tiny, 30 W) and "no appreciable speedup" (MobileViT-S, 50 W), citing the rule and the median, as in Section 7.8.4. The tables keep the literal classification of the rule.

#### 7.11.3 Cost of the variants against the core (ratio 3)

Core/variant ratio: < 1 means that the variant is slower or spends more energy than the core.

| Model | Device and mode | Core | Granularity var. | Symmetry var. | Granularity: lat. / energy | Classification | Symmetry: lat. / energy | Classification |
|---|---|---|---|---|---|---|---|---|
| DeiT-Tiny | Jetson CPU 15 W (4) | 48.6 / 56.0 / 1.14 | 49.5 / 57.6 / 1.14 | 49.4 / 51.7 / 1.03 | 0.98× / 0.97× | sin diferencia apreciable / sin diferencia apreciable | 0.98× / 1.08× | sin diferencia apreciable / sin diferencia apreciable |
| DeiT-Tiny | Jetson CPU 30 W (8) | 33.1 / 89.7 / 2.67 | 33.9 / 94.4 / 2.72 | 33.1 / 91.3 / 2.72 | 0.97× / 0.95× | sin diferencia apreciable / sin diferencia apreciable | 1.00× / 0.98× | sin diferencia apreciable / sin diferencia apreciable |
| DeiT-Tiny | Jetson CPU 50 W (12) | 32.1 / 88.7 / 2.72 | 33.4 / 91.5 / 2.67 | 34.8 / 90.7 / 2.57 | 0.96× / 0.97× | sin diferencia apreciable / sin diferencia apreciable (sessions: 0.911–1.013 / 0.936–1.013) | 0.92× / 0.98× | sin diferencia apreciable / más energía (sessions: 0.894–1.014 / 0.944–0.996) |
| DeiT-Tiny | n150 | 19.5 / 249.7 / 12.67 | 19.8 / 257.1 / 12.73 | 19.5 / 247.7 / 12.56 | 0.98× / 0.97× | sin diferencia apreciable / sin diferencia apreciable | 1.00× / 1.01× | sin diferencia apreciable / sin diferencia apreciable |
| DeiT-Tiny | rk1-02 | 27.5 / 182.6 / 6.52 | 28.1 / 188.0 / 6.51 | 28.0 / 182.5 / 6.40 | 0.98× / 0.97× | sin diferencia apreciable / sin diferencia apreciable | 0.98× / 1.00× | sin diferencia apreciable / sin diferencia apreciable |
| DeiT-Tiny | visionfive_2 (4) | 1133.3 / 2408.2 / 2.12 | 1142.7 / 2413.1 / 2.10 | 1133.7 / 2469.7 / 2.17 | 0.99× / 1.00× | sin diferencia apreciable / sin diferencia apreciable | 1.00× / 0.98× | sin diferencia apreciable / sin diferencia apreciable |
| DeiT-Tiny | kube01, CM4 (4) | 139.8 / 317.8 / 2.25 | 141.6 / 319.1 / 2.22 | 138.8 / 317.8 / 2.27 | 0.99× / 1.00× | sin diferencia apreciable / sin diferencia apreciable | 1.01× / 1.00× | sin diferencia apreciable / sin diferencia apreciable |
| MobileViT-S | Jetson CPU 15 W (4) | 129.3 / 105.8 / 0.81 | 128.3 / 106.9 / 0.83 | 125.9 / 116.8 / 0.92 | 1.01× / 0.99× | sin diferencia apreciable / sin diferencia apreciable (sessions: 0.980–1.012 / 0.973–1.170) | 1.03× / 0.91× | sin diferencia apreciable / sin diferencia apreciable (sessions: 0.996–1.040 / 0.830–1.108) |
| MobileViT-S | Jetson CPU 30 W (8) | 78.7 / 231.6 / 2.91 | 77.2 / 247.4 / 3.18 | 76.5 / 254.8 / 3.29 | 1.02× / 0.94× | sin diferencia apreciable / sin diferencia apreciable | 1.03× / 0.91× | sin diferencia apreciable / sin diferencia apreciable |
| MobileViT-S | Jetson CPU 50 W (12) | 82.0 / 237.2 / 2.86 | 78.5 / 239.7 / 3.03 | 77.8 / 239.7 / 3.04 | 1.05× / 0.99× | sin diferencia apreciable / sin diferencia apreciable | 1.05× / 0.99× | sin diferencia apreciable / sin diferencia apreciable |
| MobileViT-S | n150 | 52.2 / 729.7 / 13.87 | 52.0 / 725.2 / 13.88 | 53.3 / 739.7 / 13.79 | 1.00× / 1.01× | sin diferencia apreciable / sin diferencia apreciable | 0.98× / 0.99× | sin diferencia apreciable / sin diferencia apreciable |
| MobileViT-S | rk1-02 | 75.7 / 505.5 / 6.60 | 74.5 / 498.7 / 6.63 | 76.9 / 510.1 / 6.56 | 1.02× / 1.01× | sin diferencia apreciable / sin diferencia apreciable | 0.98× / 0.99× | sin diferencia apreciable / sin diferencia apreciable |
| MobileViT-S | visionfive_2 (4) | 2355.9 / 4943.3 / 2.09 | 2359.0 / 4952.7 / 2.10 | 2396.1 / 5003.7 / 2.08 | 1.00× / 1.00× | sin diferencia apreciable / sin diferencia apreciable | 0.98× / 0.99× | sin diferencia apreciable / sin diferencia apreciable |
| MobileViT-S | kube01, CM4 (4) | 402.9 / 860.0 / 2.12 | 399.4 / 854.2 / 2.13 | 408.4 / 865.9 / 2.11 | 1.01× / 1.01× | sin diferencia apreciable / sin diferencia apreciable | 0.99× / 0.99× | sin diferencia apreciable / sin diferencia apreciable |

#### 7.11.4 Jetson GPU (emulated regime, separate)

On the GPU, INT8 runs in emulation (Sections 7.5 and 7.9), so these rows do not measure integer kernels and are not compared with the CPU ones. They all have a single measurement and are classified with the band. MobileViT-S at 15 W has 3 sessions (7.9), and the sessions rule gives the same classification.

| Model and mode | FP32 | INT8 core | Fallback | Core: speedup / energy | Fallback: speedup / energy | Granularity var.: lat. / energy | Symmetry var.: lat. / energy |
|---|---|---|---|---|---|---|---|
| DeiT-Tiny 15 W | 11.7 / 35.6 / 2.88 | 23.3 / 65.3 / 2.76 | 18.0 / 50.4 / 2.76 | 0.50× / 0.55× | 0.65× / 0.71× | 0.94× / 0.94× | 0.99× / 0.99× |
| DeiT-Tiny 30 W | 7.3 / 35.0 / 4.43 | 14.4 / 63.3 / 4.34 | 11.2 / 49.3 / 4.34 | 0.51× / 0.55× | 0.65× / 0.71× | 0.96× / 0.96× | 1.00× / 1.00× |
| DeiT-Tiny 50 W | 5.6 / 42.2 / 6.80 | 10.9 / 76.7 / 6.90 | 7.6 / 58.6 / 7.49 | 0.51× / 0.55× | 0.73× / 0.72× | 1.02× / 1.01× | 1.02× / 1.01× |
| MobileViT-S 15 W | 23.4 / 85.4 / 3.53 | 50.3 / 170.3 / 3.36 | 33.8 / 121.5 / 3.56 | 0.47× / 0.50× | 0.69× / 0.70× | 1.02× / 0.99× | 1.00× / 0.96× |
| MobileViT-S 30 W | 13.3 / 69.1 / 5.10 | 29.6 / 142.4 / 4.76 | 20.0 / 103.9 / 5.14 | 0.45× / 0.49× | 0.66× / 0.67× | 1.03× / 1.03× | 1.01× / 0.95× |
| MobileViT-S 50 W | 6.7 / 87.5 / 12.48 | 15.5 / 178.5 / 11.24 | 10.6 / 125.5 / 11.60 | 0.43× / 0.49× | 0.63× / 0.70× | 1.01× / 1.01× | 1.00× / 1.00× |

Classification: the core and the fallback come out «más lento» and «más energía» in the 6 cases, and the variants come out «sin diferencia apreciable» in latency and energy in all 12.

#### 7.11.5 Readings

**What the tables show (measured):**
- **Energy of the INT8 core versus FP32 (CPU):** it comes out «menos energía» in 11 of 14 model and device combinations, from 1.13× to 1.66×. The three exceptions: on the visionfive_2, DeiT-Tiny «más energía» (0.67×) and MobileViT-S «sin diferencia apreciable» (1.08×); on the CM4, MobileViT-S «sin diferencia apreciable» (1.04×).
- **Latency of the INT8 core versus FP32 (CPU):** DeiT-Tiny «acelera» on the Jetson (the three modes), the n150 (1.26×), the rk1-02 (1.35×) and the CM4 (1.34×), and is «más lento» on the visionfive_2 (0.67×). MobileViT-S «acelera» on the Jetson at 15 W (1.11×) and on the n150 (1.17×, 2.3 points from the edge of the band), comes out «sin diferencia apreciable» on the rk1-02 (1.01×), the visionfive_2 (1.08×) and the CM4 (1.01×), and is «más lento» on the Jetson at 30 W and at 50 W. The borderline cases of the Jetson are the same as in 7.8.4: DeiT-Tiny at 30 W (lower bound 1.002) and MobileViT-S at 50 W (upper bound 0.996).
- **Fallback references (CPU):** «más lento» than FP32 in 12 of the 14 combinations (0.65-0.83×); the two of the CM4 come out «sin diferencia apreciable» (0.91× in DeiT-Tiny and 0.93× in MobileViT-S). In energy, «más energía» in 8 and «sin diferencia apreciable» in 6 (0.85-0.91×). Several energy ratios are within 0.02 of the edge of the band (0.83-0.85).
- **The core versus the fallback changes order on the visionfive_2 with DeiT-Tiny:** the fallback (937.5 ms) is faster than the core with integer kernels (1133.3 ms). On the other CPU devices (the CM4 included), and in MobileViT-S on the visionfive_2, the core is faster than the fallback.
- **Net power:** on the Jetson, the n150 and the rk1-02, the INT8 core has lower net power than FP32 in both models. On the visionfive_2 and on the CM4 the net power is almost the same in all configurations (1.92-2.17 W and 2.11-2.31 W), so there the energy follows the latency.
- **Cost of the variants (ratio 3):** on CPU, all variants come out «sin diferencia apreciable» against their core in latency and in energy, on all devices and modes (latency 0.92-1.05×, energy 0.91-1.08×). The only exception is given by the sessions rule: DeiT-Tiny's symmetry variant at 50 W comes out «más energía» (bounds 0.944-0.996, ~2 %). **With the resolution of these rules, the granularity and symmetry overhead shows no dependence on the device.** Differences below the band are not interpreted. For example, MobileViT-S's symmetric variant runs 3-5 % faster than the core on the Jetson and 1-2 % slower on the n150, the rk1-02, the visionfive_2 and the CM4, and that is not interpretable.
- **GPU (emulated):** the INT8 core is 0.43-0.51× in latency and 0.49-0.55× in energy against FP32. The fallback falls between the two (0.63-0.73× and 0.67-0.72×), so with `output_quant` the emulation is slower than without it, as in 7.9.

**Hypotheses (unchecked):**
- **Dot-product and SIMD instructions of each CPU (presence checked, mechanism not).** The `/proc/cpuinfo` outputs are in `tmp_coworker_data/cpu-flags/`. The Jetson (`asimddp`), the rk1-02 (`asimddp`) and the n150 (`avx_vnni`) have integer dot product. The CM4 (Cortex-A72, ARMv8.0; `fp asimd evtstrm crc32 cpuid`) and the visionfive_2 (`rv64imafdc`) do not. **The presence of dot product does not by itself separate the devices:** the CM4 does not have it and DeiT-Tiny «acelera» on it (1.34×, with 1.34× less energy), as on the rk1-02 (1.35×). **Observation after the data, not pre-registered:** the presence of SIMD instructions is consistent with the DeiT-Tiny pattern. The four CPUs on which the INT8 core «acelera» have SIMD (`asimd` on the Jetson, the rk1-02 and the CM4; `avx2` on the n150). The only one on which it does not accelerate, the visionfive_2 (`rv64imafdc`, without the `v` extension), has no SIMD. In MobileViT-S the pattern is less clear: it comes out «sin diferencia apreciable» in latency on two CPUs with SIMD (the rk1-02, 1.01×, and the CM4, 1.01×), «más lento» on the Jetson at 30 W and «no appreciable speedup» at 50 W; it only «acelera» on the Jetson at 15 W and on the n150. All devices use the same ONNX Runtime version (v1.23.2; on the CM4 it is recorded in `methodology.txt`). **Unchecked:** which `QGemm`/`QLinearConv` kernel path ONNX Runtime uses on each CPU. It is formulated as "consistent with", not as a demonstrated cause.
- **Why MobileViT-S accelerates less than DeiT-Tiny** on the n150 and the rk1-02 (1.17× and 1.01× versus 1.26× and 1.35×). It would be coherent with its larger fraction of Q/DQ/`Transpose` operators measured on the Jetson (7.8.3), but that profile has not been measured on these devices.

**Limits.** There is a single measurement per row except in the FP32 and the core of the Jetson's CPU (the three modes), the DeiT-Tiny variants at 50 W and the MobileViT-S ones at 15 W on that CPU, and the FP32 and the core of MobileViT-S on the GPU at 15 W. The ±15 % band was fixed after seeing the Jetson's variability; the noise between invocations of the n150, the rk1-02, the visionfive_2 and the CM4 has not been quantified. The shortest window is 19.7 s (n150).

## 8. McNemar paired tests (RQ1 and RQ3b, ViT branch)

**Design** (fixed before running, `calibration_protocol_vit.md`, Section 6.1). 26 comparisons × 3 repetitions = 78 tests. Exact McNemar on hit or miss in top-1, pairing repetition *k* of A with *k* of B (same calibration seed); Holm within each hypothesis and repetition (H1: 6 tests, H2: 12, H3: 6, RQ3b: 2); α = 0.05. **A difference is claimed only if it is significant after Holm in the 3 repetitions.** Input: per-image predictions of 129 artifacts (the accuracy of each CSV per image coincides with the recorded one); the DeiT-Tiny deployment artifacts are those with the one-node GELU. Script `scripts/mcnemar_vit.py` (standard library, 10 tests); results in `results/investigations/vit/mcnemar/` (`mcnemar_vit.csv` with 78 rows, which includes the unadjusted p, and the summary of 26). Δ = A − B in percentage points, repetitions 1 / 2 / 3.

| Hyp. | Model | A versus B | Δ pp | Signif. | p Holm |
|---|---|---|---|---|---|
| H1 | DeiT-Tiny | per_channel / baseline (ablation) | −0.35 / +0.65 / +0.40 | 0/3 | 0.953 / 0.308 / 1 |
| H1 | MobileViT-S | per_channel / baseline (ablation) | −0.80 / −0.50 / 0.00 | 1/3 (varies) | 0.0453 / 0.166 / 1 |
| H1 | EfficientViT-B1 | per_channel / baseline (ablation) | +0.05 / +0.25 / +0.05 | 0/3 | 1 / 0.188 / 1 |
| H1 | MobileViT-S | depl. per_channel / per_tensor, symmetric | +13.80 / +8.90 / +9.90 | **3/3** | 7.85e-30 / 1.12e-11 / 8.84e-14 |
| H1 | MobileViT-S | depl. per_channel / per_tensor, asymmetric | +11.70 / +11.15 / +10.75 | **3/3** | 1.62e-19 / 7.26e-19 / 5.26e-17 |
| H1 | DeiT-Tiny | depl. per_channel / `mse` core | +0.40 / −0.20 / 0.00 | 0/3 | 0.906 / 0.665 / 1 |
| H2 | DeiT-Tiny | percentile / baseline INT8 | −2.25 / −1.65 / −1.75 | 2/3 (varies) | 0.00464 / 0.0884 / 0.0453 |
| H2 | DeiT-Tiny | mse / baseline INT8 | +0.40 / +0.65 / +0.55 | 0/3 | 1 / 0.728 / 1 |
| H2 | DeiT-Tiny | combined / baseline INT6 | +11.65 / +11.90 / +14.00 | **3/3** | 1.39e-33 / 1.39e-30 / 2.58e-43 |
| H2 | DeiT-Tiny | combined / baseline INT4 | +34.85 / +33.20 / +34.45 | **3/3** | 5.86e-207 / 4.79e-197 / 3.4e-196 |
| H2 | MobileViT-S | percentile / baseline INT8 | +65.95 / +66.95 / +66.60 | **3/3** | <1e-300 (all 3) |
| H2 | MobileViT-S | mse / baseline INT8 | +23.00 / +27.80 / +17.45 | **3/3** | 1.08e-122 / 3.55e-154 / 1.76e-91 |
| H2 | MobileViT-S | combined / baseline INT6 | +22.10 / +22.35 / +21.40 | **3/3** | 1.96e-130 / 3.16e-130 / 1.69e-124 |
| H2 | MobileViT-S | combined / baseline INT4 | +0.15 / −0.10 / −0.10 | 0/3 | 1 / 1 / 1 |
| H2 | EfficientViT-B1 | percentile / baseline INT8 | +8.05 / +8.15 / +7.85 | **3/3** | 2.24e-46 / 1.37e-48 / 8.76e-47 |
| H2 | EfficientViT-B1 | mse / baseline INT8 | +0.45 / +0.40 / +0.10 | 0/3 | 0.176 / 0.193 / 1 |
| H2 | EfficientViT-B1 | combined / baseline INT6 | 0.00 / 0.00 / 0.00 | 0/3 | 1 / 1 / 1 |
| H2 | EfficientViT-B1 | combined / baseline INT4 | −0.05 / 0.00 / +0.05 | 0/3 | 1 / 1 / 1 |
| H3 | DeiT-Tiny | 32 / 128 | −0.10 / +0.20 / −0.20 | 0/3 | 1 / 1 / 1 |
| H3 | DeiT-Tiny | 512 / 128 | −0.45 / −0.05 / −0.05 | 0/3 | 1 / 1 / 1 |
| H3 | MobileViT-S | 32 / 128 | +0.90 / +0.35 / +2.45 | 1/3 (varies) | 0.148 / 1 / 1.58e-08 |
| H3 | MobileViT-S | 512 / 128 | −0.40 / +0.50 / +0.30 | 0/3 | 1 / 0.729 / 0.547 |
| H3 | EfficientViT-B1 | 32 / 128 | −0.05 / −0.05 / +0.05 | 0/3 | 1 / 1 / 1 |
| H3 | EfficientViT-B1 | 512 / 128 | −0.10 / +0.10 / 0.00 | 0/3 | 1 / 1 / 1 |
| RQ3b | DeiT-Tiny | `mse` / `minmax` (deployment) | +1.05 / +1.50 / +0.80 | 1/3 (varies) | 0.0572 / 0.00683 / 0.163 |
| RQ3b | MobileViT-S | core / `minmax`+BN (deployment) | +62.05 / +63.30 / +63.40 | **3/3** | <1e-300 (all 3) |

**Summary: 9 comparisons are claimed (3/3), 13 are not significant in any repetition and 4 change between repetitions (none is claimed).** Code reported 10 claimed; the count of its own table is 9.

**What they confirm.**
- **H2, calibration matters in the sensitive models:** MobileViT-S with `percentile` (+66 pp), `mse` (+17 to +28 pp, with a lot of variation between repetitions) and the combined one at INT6 (+22 pp); EfficientViT-B1 with `percentile` (+8 pp). In DeiT-Tiny the gap grows as the bit-width drops: no detectable effect at INT8 with `mse` (+0.4 to +0.65 pp), +12 to +14 pp at INT6 and +33 to +35 pp at INT4. In MobileViT-S INT4 and in EfficientViT-B1 INT6/INT4 there is no difference, because both configurations collapse. The trend across bit-widths remains descriptive (one test per point, not of the trend).
- **H1:** `per_channel` has no effect in the ablation in any model (no comparison 3/3), which confirms that it is refuted in that regime. In the deployment regime it improves MobileViT-S by +9 to +14 pp, with symmetry and with asymmetry (3/3), and does not improve DeiT-Tiny.
- **RQ3b:** in MobileViT-S, the core beats the baseline in the 3 repetitions, with the caveat that they differ in three factors at once (algorithm, granularity and symmetry), in addition to `output_quant` and BN folding. In DeiT-Tiny the sign coincides in the 3 repetitions, but only one is significant after Holm, so it is not claimed.

**What they do not confirm, and it is declared without rewriting any selection** (the protocol sets that McNemar is confirmatory):
- **The advantage of `mse` in DeiT-Tiny** (+0.40 / +0.65 / +0.55 pp at INT8; 0/3), which grounded its choice as the Phase 2 and 3 winner with a margin of means of 0.5 pp. In deployment it is not claimed either (1/3). The defensible claim is that, in DeiT-Tiny, calibration has no detectable effect at INT8 and does have one at INT6 and INT4.
- **H3, the calibration set size:** no claimable effect. The only signal is MobileViT-S, 32 versus 128, in repetition 3 (+2.45 pp, p Holm 1.58e-08), which does not appear in the other two (+0.90 and +0.35). It is compatible with sampling variance, but the cause has not been checked.

**How to read "not claimed".** It means not detected with this power, not a null effect. With 2000 images per repetition, differences of ~1 pp or smaller are barely detected, and requiring significance in the 3 repetitions reduces the power even further. Two comparisons have a negative sign in some significant repetition and are not claimed: `per_channel` versus baseline in MobileViT-S (ablation; the baseline is collapsed at ~2 %, so it has no interpretation) and `percentile` versus baseline in DeiT-Tiny (−1.65 to −2.25 pp, significant in 2 of 3).

**Scope.**
- With a fixed calibration sample, McNemar tests whether the difference between A and B exceeds the sampling noise of the 2000 evaluation images. The variability between calibration samples is not modeled: it is only required that the difference be significant in all three. (Code's note said the opposite.)
- The 2000-image subset is easier than the full validation (+1.0 to +1.9 pp: 73.50 / 79.35 / 81.20 % versus 72.16 / 78.31 / 79.26 % on the 50,000; source `results/raw/vit_accuracy_fp32_reference.csv`, Section 4 of the protocol): the relative comparisons remain valid, the absolute figures are not comparable with the official ones. In the paper, the accuracies are given on the 2000 images (the same for all the quantized models) and those of the 50,000 only as context in the Methodology.
- Holm with one family per hypothesis (H1 joins ablation and deployment, and H2 joins INT8 and INT6/INT4) is the most conservative reading. The CSV keeps the unadjusted p, in case a reviewer asks about sensitivity to the family.
- In H3, the samples of 32, 128 and 512 are nested (prefixes of the same permutation), so the two contrasts of each model are not independent. H4 is not tested with McNemar.

## 9. Phase 4, Level 1: per-layer analysis (outlier ratio versus local error)

**Design** (fixed before running, `calibration_protocol_vit.md`, Section 3.0.1). Script `analysis/phase4_vit/level1.py`; outputs in `results/investigations/vit/phase4/level1/` (`layers_<model>.csv`, `correlations.csv`, `checks.csv`, `images.csv` and `run.log`). One FP32 forward with hooks over 512 calibration-bank images (seed 0, disjoint from the calibration samples of seeds 42-44), on CUDA; the INT8 baseline (MinMax, per-tensor, symmetric, 128 calibration images, seed 42) is built and calibrated on CPU. Per layer: `R = max_c(absmax_c) / (median_c(absmax_c) + 1e-8)` over the layer's input, and the local error `‖y_q − y‖ / ‖y‖` per image (mean over the 512), applying the quantized layer to its FP32 input. Kurtosis and range are recorded without correlating.

**Checks:** the per-image grouping, the 512 images in each layer, the absence of non-finite errors and the non-degenerate ranking pass in the three models. Only the 3-channel stems (`patch_embed.proj`, `stem.conv`, `stem.in_conv.conv`) are left out of the ranking, with 49 of 50, 71 of 72 and 75 of 76 layers ranked. No median of `absmax_c` drops below 1e-6 (the smallest is 0.30, in EfficientViT-B1) and the kurtosis gives no NaN.

**Primary correlation** (global Spearman between `R` and the local error, Holm over the 3 models, α = 0.05):

| Model | n | ρ | p | p (Holm) | Significant |
|---|---|---|---|---|---|
| DeiT-Tiny | 49 | 0.06 | 0.68 | 0.68 | no |
| MobileViT-S | 71 | 0.53 | 1.9·10⁻⁶ | 5.8·10⁻⁶ | yes |
| EfficientViT-B1 | 75 | 0.29 | 0.013 | 0.026 | yes |

**By type (descriptive; types with fewer than 10 ranked layers are only listed):**

| Model | conv | attn | mlp |
|---|---|---|---|
| DeiT-Tiny | — (1 layer, not ranked) | −0.02 (n = 24) | −0.15 (n = 24) |
| MobileViT-S | 0.74 (n = 34) | 0.74 (n = 18) | 0.42 (n = 18) |
| EfficientViT-B1 | 0.18 (n = 45) | 0.71 (n = 28) | — (has none) |
| EfficientViT-B1, with the `aggreg` as conv (sensitivity) | 0.23 (n = 59) | 0.91 (n = 14) | — |

**Reading.**
- **There is an association between `R` and the local error in the two models that collapse at INT8** (MobileViT-S, ρ = 0.53; EfficientViT-B1, ρ = 0.29) **and not in DeiT-Tiny** (ρ = 0.06). By type, the association is stronger in attention and in convolution (MobileViT-S) and in attention (EfficientViT-B1).
- **DeiT-Tiny:** the range of `R` is very narrow (1.4-3.0), so the head of the ranking is almost a tie. The highest local errors are in the `mlp.fc2` (0.064-0.099), which occupy positions 11-30 by `R`, hence the null correlation.
- **EfficientViT-B1:** the layer with the highest local error is `head.classifier.0` (0.234), at position 5 by `R`, and it enters the top 8.
- **Ranges of the local error at INT8** (plausible, not validated against any reference): DeiT-Tiny 0.017 / 0.030 / 0.099 (minimum / median / maximum), MobileViT-S 0.011 / 0.033 / 0.125, EfficientViT-B1 0.003 / 0.034 / 0.234.
- **The classification of the `aggreg` is not neutral:** moving them from attn to conv raises the attn correlation from 0.71 to 0.91 (n from 28 to 14) and that of conv from 0.18 to 0.23. It is descriptive; the `aggreg` are the attention layers with the weakest association.

**Cautions.** The p-values are optimistic: the layers of one model are not independent, and Holm only corrects between models. With per-tensor MinMax, a layer's quantization step is set by its absolute maximum, so a positive correlation between `R` and the local error is partly mechanical. Level 1 is descriptive; the causal evidence is Level 2.

**Candidates for targeted Level 2** (the top 8 by `R`, full ranking in `layers_<model>.csv`):

| # | DeiT-Tiny | MobileViT-S | EfficientViT-B1 |
|---|---|---|---|
| 1 | blocks.1.mlp.fc2 | stages.0.0.conv2_kxk | stages.1.blocks.1.main.depth_conv |
| 2 | blocks.5.mlp.fc1 | stages.1.0.conv2_kxk | stages.0.blocks.0.main.depth_conv |
| 3 | blocks.0.mlp.fc1 | stages.1.1.conv2_kxk | stages.1.blocks.0.main.depth_conv |
| 4 | blocks.2.attn.qkv | stages.3.0.conv2_kxk | stages.2.blocks.1 … aggreg.0.1 |
| 5 | blocks.10.mlp.fc1 | stages.1.2.conv2_kxk | head.classifier.0 |
| 6 | blocks.8.mlp.fc1 | stages.4.0.conv3_1x1 | stages.1.blocks.2.main.depth_conv |
| 7 | blocks.6.mlp.fc1 | stages.4.1.transformer.2.mlp.fc2 | stages.3.blocks.2.local_module.point_conv |
| 8 | blocks.9.mlp.fc1 | stages.4.0.conv2_kxk | stages.0.blocks.1.main.depth_conv |

**Ranking stability** (pre-registered after this result, measured afterwards; only reported, and Level 2 used the main ranking). Second set of 512 bank images (seed 1):

| Model | Spearman of `R` between sets | Coinciding top-8 |
|---|---|---|
| DeiT-Tiny | 0.957 | 6/8 |
| MobileViT-S | 0.935 | 5/8 (the first 5, in the same order) |
| EfficientViT-B1 | 0.959 | 7/8 (the first 5, in the same order) |

## 10. Phase 4, Level 2: targeted interventions and controls

**Design** (fixed before running, `calibration_protocol_vit.md`, Section 3.0.1). The K layers with the highest `R` are left in floating point (K = 1, 2, 4, 8, stopping when 90 % of the gap is recovered) and the accuracy is measured on the 2000 evaluation images, against a control with the same K layers chosen at random (3 seeds, from the rankable layers). Configuration: MobileViT-S and EfficientViT-B1 at INT8 baseline, DeiT-Tiny at **INT6** baseline (deviation from the original design, which said INT4); only repetition 1 (calibration seed 42). Scripts `analysis/phase4_vit/level2.py` and `stability.py`; outputs in `results/investigations/vit/phase4/level2/` (`evaluations.csv`, `predictions/`, `summary.csv`, `run.log`).

**Checks.** 53 evaluations (DeiT-Tiny 21, EfficientViT-B1 21 and MobileViT-S 11, which stopped at K = 2), of 144 to 261 s each, with its per-image prediction file and no pair of ONNX files with the same sha256. The export that mixes float `nn.Linear` or `nn.Conv2d` layers with quantized layers worked. The K = 0 control reproduces exactly the recorded top-1 and top-5 (58.8 / 81.55; 2.95 / 7.0; 0.20 / 0.90) and has 0 prediction differences image by image in the three models. Reference FP32 in PyTorch (73.50 / 79.35 / 81.20).

**Primary result: targeted by `R` versus the maximum of the 3 random seeds** (top-1 %; in parentheses, the recovery of the gap):

| Model | Gap | K | Targeted by `R` | Random maximum | Verdict |
|---|---|---|---|---|---|
| MobileViT-S INT8 | 76.40 pp | 1 | 60.10 (74.8 %) | 3.05 | |
| | | 2 | **74.70 (93.9 %)**, stops at 90 % | 3.00 | **clearly better** (2 of 2) |
| EfficientViT-B1 INT8 | 81.00 pp | 1 | 0.15 | 0.25 | |
| | | 2 | 1.85 | 0.85 | |
| | | 4 | 21.45 (26.2 %) | 1.35 | |
| | | 8 | 40.35 (49.6 %) | 15.75 | **clearly better** (3 of 4, including K = 8) |
| DeiT-Tiny INT6 | 14.70 pp | 1 | 58.90 | 60.95 | |
| | | 2 | 59.20 | 61.45 | |
| | | 4 | 59.65 | 61.85 | |
| | | 8 | 61.90 (21.1 %) | 63.75 | **not distinguishable from chance** (0 of 4) |

The clause "no recovery with K ≤ 8" does not apply in any model. In EfficientViT-B1, the advantage at K = 2 (1.85 versus 0.85) is at the accuracy floor and does not weigh: the criterion (⌈4/2⌉ = 2 wins, including K_final) is met anyway with K = 4 and K = 8 (21.45 versus 1.35; 40.35 versus 15.75).

**Secondary curve by local error (exploratory):**

| Model | Top-1 per K | Against the controls |
|---|---|---|
| MobileViT-S | 5.00 / 5.05 (K = 1, 2) | beats chance, but far below the curve by `R` (74.7 at K = 2) |
| EfficientViT-B1 | 0.20 / 0.25 / 0.35 / 14.10 | does not beat chance at any K |
| DeiT-Tiny | 58.90 / 58.20 / 60.80 / 65.70 | beats chance only at K = 8 (1 of 4, would not meet the criterion); with its 8 `mlp.fc2` in float it recovers 47 % |

**Reading.**
- **`R` identifies the responsible layers in the two models that collapse, and not in DeiT-Tiny**, coherently with Level 1 (ρ of 0.53, 0.29 and 0.06). With only 3 models it is a descriptive coincidence.
- **MobileViT-S:** two layers, `stages.0.0.conv2_kxk` and `stages.1.0.conv2_kxk` (3×3 depthwise convolutions), explain 94 % of the gap: from 2.95 % to 74.70 % with only those two in float (FP32: 79.35 %). The random control with K = 2 does not go above 3.05 %. **The recovery replicates with two other calibration samples** (91-94 %, Section 10.2).
- **EfficientViT-B1:** 5 of the 8 layers of the targeted curve are 3×3 depthwise. The only random seed that recovers anything (15.75 % top-1 at K = 8, 19 % of the gap) also leaves four 3×3 depthwise and one 5×5 depthwise in float. It is compatible with the damage being concentrated in the depthwise layers, but that hypothesis was not pre-registered and has not been tested. At K = 8 only 50 % is recovered, so the damage is not exhausted in those 8 layers. **The recovery varies with the calibration sample (50 %, 42 % and 36 %) and does not meet the replication criterion (Section 10.2)**; **the superiority over chance holds with the three samples** (Section 10.2).
- **DeiT-Tiny INT6:** the one targeted by `R` stays below the random maximum at the 4 values of K, and recovers 21 % at K = 8 (the best control, 34 %). `R` does not identify the sensitive layers; the local error points better to the `mlp.fc2` (exploratory, 1 of 4 K). **The negative result replicates with two other calibration samples (Section 10.2).**
- **For the earlier diagnosis (Section 3):** Level 2 locates the damage in a few layers, but leaving a layer in float eliminates at once the quantization of its weights and of its input, so by itself it does not separate which of the two causes it. Section 3 (hypothesis 6) pointed to the activations, and with the depthwise layers per-tensor quantization of the weights was an alternative explanation. **The attribution experiment of Section 10.1 resolves it.**

**Limits.** Level 2 was measured with a single calibration sample (repetition 1); the replication of Section 10.2 covers the recovery of the targeted curve, EfficientViT-B1's random control with K = 8 and DeiT-Tiny's negative result with K = 8, with two other samples. 2000 evaluation images. Only 3 random seeds: in EfficientViT-B1, the control's maximum at K = 8 (15.75 %) comes from a single seed, and MobileViT-S's control only reaches K = 2. The verdicts are comparisons of values, as the pre-registration sets, without a statistical test (the per-image predictions are stored in case McNemar were wanted, which was not pre-registered). In the two models that collapse, the recoveries are measured over a baseline close to zero.

### 10.1 Attribution: weights or activations? (exploratory)

**Design** (pre-registered in `calibration_protocol_vit.md`, Section 3.0.1, point 8, after seeing the Level 2 results and before seeing these data). In the layers that the targeted curve left in float at its final K (MobileViT-S with K = 2 and EfficientViT-B1 with K = 8), it evaluates (a) only the weights in float, with the input quantizer active, and (b) only the input quantizer in float, with the weights quantized. Same construction, calibration (repetition 1) and 2000 evaluation images as Level 2. Before each evaluation, the log checks that in the chosen layers only the intended quantizer is switched off (2 layers in MobileViT-S and 8 in EfficientViT-B1) and that the rest (70 and 68) is fully quantized. The QCDQ export worked with a single quantizer disabled. Outputs in `results/investigations/vit/phase4/level2/attribution/`, 213-258 s per evaluation. `Rec = (acc − acc_0) / (FP32 − acc_0)` on top-1.

| Model (K) | K = 0 | Whole layer in float | (a) weights in float | (b) input in float | `Rec_a / Rec_full` | `Rec_b / Rec_full` | Classification |
|---|---|---|---|---|---|---|---|
| MobileViT-S (2) | 2.95 | 74.70 (Rec_full 93.9 %) | 2.80 (Rec_a −0.2 %) | 74.60 (Rec_b 93.8 %) | −0.00 | 1.00 | **activations** |
| EfficientViT-B1 (8) | 0.20 | 40.35 (Rec_full 49.6 %) | 0.15 (Rec_a −0.1 %) | 39.50 (Rec_b 48.5 %) | −0.00 | 0.98 | **activations** |

**Reading.** Leaving only the weights of those layers in float recovers nothing, and leaving only their input quantizer in float recovers practically the same as the whole layer. It meets the criterion by a wide margin (`Rec_b ≥ 0.75·Rec_full` and `Rec_a ≤ 0.25·Rec_full`) in both models. It rules out the alternative explanation of per-tensor depthwise weights for these layers and is coherent with hypothesis 6 of Section 3. In EfficientViT-B1, (b) is 0.85 pp below the whole layer (17 images out of 2000), within the sampling noise; it is not interpreted as a contribution of the weights.

**What cannot be claimed.**
- That the mechanism is outlier channels. `R` suggests it (it measures how much one channel dominates the per-tensor scale), but the experiment attributes the damage to the quantization of the input, not to a specific mechanism, and it has not been tested directly.
- That it holds for other calibrations: it is with the baseline (MinMax, per-tensor, symmetric). It is coherent with `percentile` recovering MobileViT-S, but it has not been proven that it acts through these layers.
- That it explains all the damage in EfficientViT-B1: with K = 8 only 50 % is recovered.

**Limits.** A single calibration sample (repetition 1), only these two configurations, no statistical test (the criterion is a comparison of values) and exploratory. The per-image predictions are stored. The Phase 4 tests pass on the cluster, and the check of the quantizers' state before each evaluation further backs the results.

### 10.2 Replication with other calibration samples

**Design** (pre-registered in `calibration_protocol_vit.md`, Section 3.0.1, point 9, before seeing these data). For repetitions 2 and 3 (calibration seeds 43 and 44), in MobileViT-S with K = 2 and in EfficientViT-B1 with K = 8: the K = 0 control and the targeted curve with the same layers (the ranking comes from FP32 and does not depend on the calibration sample). The recovery is computed with the K = 0 of each repetition. Criterion: it replicates if in both repetitions it is ≥ 0.75 times that of repetition 1 (≥ 70.4 % in MobileViT-S and ≥ 37.2 % in EfficientViT-B1). Outputs in `results/investigations/vit/phase4/level2/replication/`.

**K = 0 controls** (top-1 / top-5, identical to what was recorded, with 0 different predictions per image): MobileViT-S, 1.55 / 4.70 (repetition 2) and 1.15 / 3.75 (repetition 3); EfficientViT-B1, 0.10 / 0.55 and 0.10 / 0.70.

| Model (K) | Repetition 1 | Repetition 2 | Repetition 3 | Threshold | Verdict |
|---|---|---|---|---|---|
| MobileViT-S (2) | 74.70 → 93.9 % | 74.55 → 93.8 % (×1.00) | 72.60 → 91.4 % (×0.97) | 70.4 % | **replicates** |
| EfficientViT-B1 (8) | 40.35 → 49.6 % | 34.25 → 42.1 % (×0.85) | 29.15 → 35.8 % (×0.72) | 37.2 % | **does not replicate** |

Each cell gives the top-1 and, after the arrow, the recovery of the gap; in parentheses, the ratio with repetition 1.

**Reading.**
- **MobileViT-S:** leaving the two depthwise layers in float recovers between 91 and 94 % of the gap with the three calibration samples.
- **EfficientViT-B1:** repetition 3 ends at 35.8 % against a threshold of 37.2 %, so the verdict is "does not replicate". The recovery with the 8 layers varies between 36 and 50 % depending on the sample: it depends on the calibration more than the criterion admitted, and repetition 1 was the most favorable. That it is substantial in all three is a descriptive reading, not pre-registered.
- **Superiority over chance:** this replication does not test it. In EfficientViT-B1 it was tested separately (random control, below) and DeiT-Tiny's negative result was also replicated separately (at the end of this section); in MobileViT-S the random control is not replicated (declared limit).

**EfficientViT-B1 random control with K = 8** (pre-registered in point 10 of the protocol, before seeing these data). The 3 random selections of repetition 1 (same layers, permutations of seeds 1, 2 and 3), built with the calibration of each repetition: 6 new evaluations, reusing the K = 0 and the targeted one from the replication. Criterion: the targeted one strictly exceeds the maximum of the 3 seeds in both repetitions.

| Repetition (calibration seed) | K = 0 | Targeted | Random s1 | s2 | s3 | Random maximum | Targeted > maximum |
|---|---|---|---|---|---|---|---|
| 1 (42), from Level 2 | 0.20 | 40.35 (49.6 %) | 0.15 | 15.75 | 1.25 | 15.75 | yes |
| 2 (43) | 0.10 | 34.25 (42.1 %) | 0.15 (0.1 %) | 13.80 (16.9 %) | 0.40 (0.4 %) | 13.80 | **yes** |
| 3 (44) | 0.10 | 29.15 (35.8 %) | 0.10 (0.0 %) | 9.75 (11.9 %) | 0.45 (0.4 %) | 9.75 | **yes** |

**The superiority over chance holds with the three calibration samples**, with a margin of 19 to 25 top-1 points over the best random control, even though the magnitude of the recovery varies with the sample. Descriptive observation, not pre-registered: seed 2 is the only random one that recovers anything in the three repetitions (12-17 %), and it shares 3 of its 8 layers with the targeted one, the three depthwise layers of stages 0 and 1; it is coherent with the damage being concentrated in those layers.

**Replication of DeiT-Tiny INT6's negative result with K = 8** (pre-registered in point 12 of the protocol, before seeing these data). Repetitions 2 and 3 (seeds 43 and 44): K = 0 control, targeted by `R` (the first 8 of the main ranking), the 3 random selections of repetition 1 and, as an exploratory one, the local-error curve. 12 evaluations, with 8 layers in float in each and the same lists as in repetition 1. The K = 0 controls reproduce exactly what was recorded, with 0 differences per image (top-1 / top-5: 58.45 / 81.35 in repetition 2 and 56.15 / 80.10 in 3). Criterion: the negative result replicates if, in both repetitions, the targeted one does not strictly exceed the maximum of the 3 random seeds.

| Rep (seed) | K = 0 | Targeted by `R` | Random s1 | s2 | s3 | Random maximum | Local error (exploratory) |
|---|---|---|---|---|---|---|---|
| 1 (42), from Level 2 | 58.80 | 61.90 (21.1 %) | 61.95 (21.4 %) | 59.90 (7.5 %) | 63.75 (33.7 %) | 63.75 | 65.70 (46.9 %) |
| 2 (43) | 58.45 | 61.25 (18.6 %) | 60.70 (15.0 %) | 60.45 (13.3 %) | 63.10 (30.9 %) | 63.10 | 64.35 (39.2 %) |
| 3 (44) | 56.15 | 61.45 (30.5 %) | 60.75 (26.5 %) | 59.30 (18.2 %) | 63.15 (40.3 %) | 63.15 | 64.90 (50.4 %) |

**The negative result replicates:** the one targeted by `R` stays below the random maximum in both repetitions (61.25 versus 63.10 and 61.45 versus 63.15). With three calibration samples, `R` does not identify DeiT-Tiny's sensitive layers. The targeted one stays within the range of the random selections in the three repetitions (above 1 of 3 seeds in repetition 1 and 2 of 3 in repetitions 2 and 3).

**Descriptive observations, not pre-registered.**
- **The order of the three random selections is the same in the three repetitions** (s2 < s1 < s3), and s3 is always the best (31-40 % of the gap). There is, therefore, an effect of which layers are left in float that is stable across calibration samples, and `R` does not capture it. **Composition of s3** (the same 8 layers in the three repetitions, read from `level2/evaluations.csv` and `level2/replication/evaluations.csv`, with Level 1 positions out of 49 layers): `blocks.0.mlp.fc2` (position 6 by local error), `blocks.4.mlp.fc2` (10), `blocks.2.mlp.fc2` (12), `blocks.11.attn.proj` (14), `blocks.4.attn.proj` (20), `blocks.6.attn.proj` (22), `blocks.5.mlp.fc1` (36) and `head` (47). The 8 with the highest local error are all `mlp.fc2` (blocks 11, 10, 6, 8, 7, 0, 9 and 5), and only one, `blocks.0.mlp.fc2`, is in s3; s1 and s2 also share one, so the overlap does not distinguish s3. It is the random selection with the most `mlp.fc2` (3 versus 2), which goes in the direction of the `mlp.fc2` mattering, but it also contains 3 `attn.proj` and the effect cannot be separated: with 3 selections and differences of 1 to 3 pp there is no basis to attribute it. A weak, descriptive and non-pre-registered clue.

| Selection | Mean top-1 (3 repetitions) | `mlp.fc2` it contains | Overlap with the 8 of highest local error |
|---|---|---|---|
| Local error | 65.0 | 8 | 8 |
| s3 | 63.3 | 3 | 1 |
| Targeted by `R` | 61.5 | 1 | 0 |
| s1 | 61.1 | 2 | 1 |
| s2 | 59.9 | 2 | 1 |
- **The local-error curve exceeds the random maximum with K = 8 in the three repetitions** (from 1.25 to 1.95 pp). It is exploratory, with only K = 8 (the Level 2 criterion, with several K, was not met: 1 of 4), and with a margin of the same magnitude as the sampling error of the accuracy over 2000 images (~1 pp); it is not presented as a result.

**Limits.** Three calibration samples, 3 random seeds, only K = 8, 2000 images and no statistical test (the criteria are comparisons of values and ratios).

### 10.3 Sufficiency and necessity of `percentile` in the two layers (exploratory)

**Design** (pre-registered in `calibration_protocol_vit.md`, Section 3.0.1, point 11, after seeing Level 2, the attribution and the replication, and before seeing these data). MobileViT-S, INT8, repetitions 1-3 (calibration seeds 42, 43 and 44). `percentile` only changes the input quantizer (`Int8ActPerTensorFloatPercentile`, 99.9th percentile, versus `Int8ActPerTensorFloatMinMax`); the weights use the same class (`Int8WeightPerTensorFloat`) in all layers. The `percentile` calibration is applied in the layers of a set S and the baseline one in the rest. Three artifacts per repetition: control with S = all layers, sufficiency with S = `stages.0.0.conv2_kxk` and `stages.1.0.conv2_kxk`, and necessity with S = all but those two. 9 evaluations of 224 to 284 s. Outputs in `results/investigations/vit/phase4/level2/calibration_attribution/` (with `layer_configs/`, which confirms per layer the configuration used: 72/0, 2/70 and 70/2 layers with `percentile`/`minmax`).

**Controls.** With S = all layers, the recorded `percentile` artifact of each repetition is reproduced exactly (top-1 / top-5: 68.90 / 89.05; 68.50 / 88.85; 67.75 / 87.80), with 0 prediction differences per image.

| Rep (seed) | K = 0 | `percentile` in all → `Rec_perc` | Only the 2 layers → `Rec_suf` | All but the 2 → `Rec_nec` | `Rec_suf / Rec_perc` | `Rec_nec / Rec_perc` |
|---|---|---|---|---|---|---|
| 1 (42) | 2.95 | 68.90 → 86.3 % | 64.10 → 80.0 % | 4.00 → 1.4 % | 0.93 | 0.02 |
| 2 (43) | 1.55 | 68.50 → 86.1 % | 63.15 → 79.2 % | 3.45 → 2.4 % | 0.92 | 0.03 |
| 3 (44) | 1.15 | 67.75 → 85.2 % | 60.85 → 76.3 % | 3.85 → 3.5 % | 0.90 | 0.04 |

Top-1 in %; the recovery `Rec = (acc − acc_0,k) / (FP32 − acc_0,k)` is computed with each repetition's baseline.

**Result.** The two layers are **sufficient** (`Rec_suf ≥ 0.75·Rec_perc` in the three repetitions; ratios from 0.90 to 0.93) and **necessary** (`Rec_nec ≤ 0.25·Rec_perc` in the three; ratios from 0.02 to 0.04). **The effect of `percentile` in MobileViT-S goes through the calibration of the input of those two layers.**

**Reading.**
- Calibrating only those two inputs with `percentile` recovers 76-80 % of the gap, versus 85-86 % with `percentile` in all 72 layers: they are enough, with 7-10 % of the improvement left outside them. Conversely, the other 70 layers with `percentile` recover only 1.4-3.5 % if those two use MinMax.
- It links the calibration effect (H2) with the location of the damage (Level 2) and with its attribution to the activations (Section 10.1). It is coherent with the damage coming from extreme values that set the per-tensor scale at those inputs, and with `R` (which measures how much one channel dominates that scale) identifying the layers.
- **Practical implication, limited to this regime** (baseline without `output_quant`): calibrating only 2 of the 72 layers with `percentile` recovers almost everything that doing it in all of them recovers.

**What cannot be claimed.**
- **The exact mechanism.** Clipping the top 0.1 % fixes the scale, but this does not distinguish outlier channels from isolated outlier values.
- **That any other two layers would have worked equally:** there is no control with two random layers (declared in the protocol). It is indirectly supported by Level 2, where no random selection with K = 2 went above 3.05 %, and by the necessity test, where 70 layers with `percentile` recover almost nothing.
- That it holds for EfficientViT-B1, DeiT-Tiny, other calibrations or the deployment regime (`output_quant` and folded BN): it has not been tested.

**Limits.** Only MobileViT-S INT8, three calibration samples, 2000 evaluation images, no statistical test (the criterion is a comparison of ratios) and exploratory.

### 10.4 Synthesis of Phase 4 for RQ1 and RQ2 (ViT branch)

- **Level 1 (association):** `R` is associated with the per-layer local error in MobileViT-S (ρ = 0.53) and EfficientViT-B1 (ρ = 0.29), and not in DeiT-Tiny (ρ = 0.06). Descriptive, with optimistic p-values and a partly mechanical correlation.
- **Level 2 (causality):** in the two models that collapse at INT8, leaving in float the layers with the highest `R` clearly beats chance with repetition 1. In MobileViT-S, 2 3×3 depthwise convolutions are enough to recover 91-94 % of the gap, and it replicates with the three calibration samples. In EfficientViT-B1 the recovery with K = 8 is substantial but depends on the sample (36-50 %) and does not meet the replication criterion, whereas the superiority over chance holds with the three samples. In DeiT-Tiny INT6, `R` does not identify the sensitive layers, and the result replicates with three samples.
- **Attribution:** in the identified layers, the damage comes from the quantization of their input (activations), not of their weights.
- **Link with calibration (RQ1):** in MobileViT-S, `percentile`'s improvement goes through the calibration of the input of those two layers, which are sufficient and necessary with the three samples (Section 10.3).
- **Scope:** 3 models, MinMax per-tensor baseline, and three calibration samples for the recovery of the targeted curve (MobileViT-S and EfficientViT-B1), EfficientViT-B1's random control and DeiT-Tiny's negative result (the last two with K = 8). In LLM, Phase 4 is only descriptive and is not done.
