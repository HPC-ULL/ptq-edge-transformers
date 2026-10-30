# Findings of the LLM Branch (PTQ Calibration Ablation)

*Document of concrete results, separate from `calibration_protocol_llm.md` (which describes the theoretical design and does not change with each new batch of data). This one does — it is updated as results collection progresses. It plays the same role that `findings_vit.md` plays for the ViT branch.*

## H1 — Confirmation with Phase 3 data

In the 2 models with a combined configuration different from the baseline measured so far (TinyLlama, Qwen2.5), the perplexity improvement of the combined configuration over the baseline without imatrix grows clearly with the aggressiveness of the bit-width. Absolute values (mean PPL ± std where applicable; the baseline is deterministic, n=1):

| Model | Bit-width | Baseline (no imatrix) | Combined (with imatrix) | Improvement |
|---|---|---|---|---|
| TinyLlama | Q3_K_M | 9.2084 (n=1) | 8.9584 ± 0.0030 | 0.25 points |
| TinyLlama | Q2_K | 12.3096 (n=1) | 10.6121 ± 0.0030 | 1.70 points |
| Qwen2.5 | Q3_K_M | 12.4471 (n=1) | 11.4239 ± 0.0179 | 1.02 points |
| Qwen2.5 | Q2_K | 26.1245 (n=1) | 14.1671 ± 0.0114 | 11.96 points |

Qwen2.5 at Q2_K is the most marked case: 11.96 points of improvement over a baseline perplexity of 26.1245 — the improvement from calibration amounts to almost half of the baseline's own perplexity (11.96/26.1245 ≈ 46%).

**Precision note:** the real Phase 2 winner in both models was the **WikiText-2** corpus (generic), not the domain corpus (Dolly-15k) — so this confirmation is about "having a good imatrix vs. having none", not specifically about "domain corpus vs. generic" (that particular comparison, within H1, showed no difference exceeding the selection margin between WikiText-2 and Dolly-15k in any model — see `phase3_selection_report.txt`; no formal significance test has been applied to this comparison, so "no difference exceeding the margin" is the correct claim, not "no significant difference"). **Confounded with corpus size:** at the default level, Dolly-15k contributes ~2× the tokens of WikiText-2 (LLM protocol, Section 3.1) — the comparison was not at equal data, and Dolly had more and still did not win. With H2 showing little sensitivity to size, the effect is probably small, but it has not been isolated.

**Ministral-3-3B provides no evidence for H1:** confirmed — no factor exceeded the improvement margin in its Phase 2 (`calibration_corpus` keeps baseline, see H4 below), and its real Phase 3 manifest (verified, not just anticipated) only generates the 2 baseline rows, with no `combined` row — the same deduplication case as the model from the previous iteration. The confirmation of H1 rests on 2 of the 3 main models (TinyLlama, Qwen2.5), not on all 3 — declare this explicitly in the writing, do not generalize to "the 3 models confirm H1".

## H2 — Calibration corpus size

**Ministral-3-3B: no clear effect.** Comparing `calib_set_size=50` (9.0532±0.0078), the default value of 200 (9.0529±0.0076) and `calib_set_size=1000` (9.0480±0.0022) — the three means are within a range of 0.005 PPL of each other, with no clear difference nor identifiable saturation threshold.

**TinyLlama and Qwen2.5: small monotonic trend, not exceeding the selection margin in either.** Real values (mean PPL ± std, n=3):

| Model | size=50 | size=200 (default) | size=1000 |
|---|---|---|---|
| TinyLlama | 8.6510 ± 0.0052 | 8.6470 ± 0.0015 | 8.6449 ± 0.0015 |
| Qwen2.5 | 10.5375 ± 0.0164 | 10.5322 ± 0.0122 | 10.5237 ± 0.0066 |

In both models the perplexity decreases monotonically as the corpus size increases — a direction consistent with H2 — but the magnitude is small and, according to `phase3_selection_report.txt`, no level exceeded the improvement margin over the baseline in the real Phase 2 selection (`calib_set_size` keeps baseline in both models). **Actual status: qualitative trend consistent with H2 in 2 of 3 models, but not confirmed with the already established margin criterion; no detectable effect in Ministral-3-3B.** There is no identifiable saturation threshold in any model with the current data.

## H3 — Interpretation note (avoids a contradictory reading)

In the 3 current main models (TinyLlama, Qwen2.5, Ministral-3-3B), H3 is confirmed (`legacy + imatrix` improves over `legacy` alone, with a clear margin) — but **`k_quant` still beats `legacy + imatrix` in absolute terms**, and that is why `quant_scheme` keeps the baseline (`k_quant`) in the combined Phase 3 configuration, in the 3 models. Confirmed with real Ministral-3-3B data: `legacy_corpus_wikitext2` (9.1754 PPL) clearly improves over `legacy_corpus_none` (9.3189 PPL), the same pattern as the other two. **Actual status: confirmed in the 3 models, with the nuance that `k_quant` still wins in absolute terms.**

It is not a contradiction: H3 answers "does the imatrix help legacy?" (yes), not "does legacy+imatrix beat k_quant?" (no) — they are different comparisons. When writing, present both results together so that they do not seem discordant at first sight.

## H4 — Relevant precedent and confirmation in one main model

In an earlier iteration of the protocol, with a different European model evaluated as the third LLM before prioritizing Ministral-3-3B, that model was observed to be the only one of the three where no calibration corpus (neither WikiText-2 nor Dolly-15k) beat the baseline without imatrix — different sensitivity between families, exactly the phenomenon that motivates H4. Complete results of that evaluation kept in `results/appendix/`.

**The same pattern is repeated in Ministral-3-3B (Phase 1-2, real results):** the baseline without imatrix (9.0307 PPL, n=1) has *better* perplexity than the 4 variants with imatrix. Complete figures, with standard deviation (n=3 repetitions each) and the separation from the baseline expressed as a multiple of each variant's own standard deviation (not of the highest one in the table, so as not to mix them):

| Configuration | Mean PPL | Std | Separation vs. baseline | Separation / own std |
|---|---|---|---|---|
| baseline (no imatrix) | 9.0307 | n/a (n=1) | — | — |
| calib_set_size=1000 (wikitext2) | 9.0480 | 0.0022 | 0.0173 | ~7.9× |
| calibration_corpus=wikitext2 | 9.0529 | 0.0076 | 0.0222 | ~2.9× |
| calib_set_size=50 (wikitext2) | 9.0532 | 0.0078 | 0.0225 | ~2.9× |
| calibration_corpus=domain_corpus | 9.0666 | 0.0205 | 0.0359 | ~1.75× |

The 4 variants stay above the baseline consistently, with the separation exceeding their own standard deviation in all 4 (between ~1.75× and ~7.9×) — it is not noise, it is a systematic pattern. **Nuance of tone:** in absolute terms, the difference is small (0.02–0.04 PPL on a base of ~9, i.e. 0.2–0.4%) — what is defensible is that **the imatrix does not help** in this model, not that it makes it worse in any relevant practical sense.

**Correct framing for the writing, important not to confuse:** within the set of 3 current main models, this is **1 of 3** models showing the pattern (Ministral-3-3B), versus TinyLlama/Qwen2.5, where the imatrix does help clearly — not "2 of 3", which is only reached by also counting the appendix model of the previous iteration. The finding of the appendix model is complementary and consistent evidence, citable as such in an appendix if needed, but it must not be added to the evidence of the main set when describing H4 in the body of the paper.

**Phase 3 confirmed (see also H1 above):** `calibration_corpus` has no winner in Ministral-3-3B's selection — confirmed with the real Phase 3 manifest, not just anticipated. The same deduplication case already seen with the model of the previous iteration.

On reaching Phase 4, this pattern gives a directed objective: compare the distribution properties of Ministral-3-3B against TinyLlama/Qwen2.5, looking for what distinguishes it structurally — with the appendix model as an additional contrast point, not as part of the main set.

## Corrected methodology note — "performance step" effect on the Jetson

During the performance/energy measurement on the Jetson, a systematic effect was detected and confirmed: performance (tokens/second in decode) rises abruptly (+13 % to +24 % on Qwen depending on the power mode; +33 % on TinyLlama at 50 W) between minute 6 and 7.5 after the device boots or the power mode changes, and stays stable afterwards. Thermal throttling and CPU/GPU/memory frequency scaling were ruled out as the cause (confirmed with `jetson_clocks`) — exact root cause not identified, but the effect itself is confirmed with multiple repetitions in the 3 CPU power modes.

**Mitigation adopted:** a 600 s uptime wait before measuring the first real row of each batch, configured per device with the `min_uptime_s` field of `devices.yaml` (the Jetson entries carry `min_uptime_s: 600`; Section 10.5 of the architecture), which replaces the former `--min-uptime-seconds` flag. Devices without the field do not wait: it has not been characterized whether the step appears outside the Jetson. Any LLM performance/energy data on the Jetson measured before adopting this mitigation must be treated with caution (see also the note in `results/appendix/README_salamandra_2b.md`, the same problem).

**Also confirmed and corrected:** `llama-bench`'s default `-r 5` systematically biased `energy_per_inference` through insufficient dilution of the fixed model-loading cost — `N=30` fixed as the valid minimum after a real convergence test.

## Fixed cost of the energy measurement window in LLM (loading + warm-up)

**What `energy_net_j` covers.** The energy is integrated over the whole `llama-bench` invocation, not just over the N timed repetitions: it includes the model loading and the warm-up executions. `energy_per_inference = energy_net_j / N` is therefore the energy **per repetition** (one execution of pp128 + tg32, 160 tokens) plus the amortized fixed cost, not the energy per token (that would be `energy_per_inference / 160`). The fixed cost **only affects energy**: the throughput and latency reported by `llama-bench` are computed over the timed repetitions.

**Estimate in time from the CSVs.** The fraction of the window that is not timed repetitions, `(window − N·t_rep)/window` with `t_rep = 128/pp + 32/tg`, is 2-17 % depending on the model and the Jetson configuration, and follows neither the power mode nor the model size (Qwen2.5 goes from 2.4 s on CPU 30 W to 7.0 s on CPU 50 W). It is stable within each session and different between sessions. It is a fraction of **time**; the energy one turned out much smaller.

**N sweep (Jetson, 50 W).** Q4_K_M of the three models on CPU and GPU, with N = 10, 30 and 90, one discarded warm-up and a cold case with N = 30 (page cache flushed). `E(N) = C + N·M` is fitted: `M` is the marginal energy per repetition and `C` the fixed cost in energy. A correction that distributed the window's energy in proportion to the compute time was tested, `E_corr = E_net·(N·t_rep/window)/N`, with the criterion fixed before measuring: median of |E_corr − M|/M ≤ 3 % and no combination above 6 %.

- **The correction is not adopted.** E_corr < M in the 6 combinations (median 5.73 %, maximum 12.06 %): the criterion fails, and because of a systematic deviation of the same sign, not because of noise. The cause is that the fixed part spends almost no net power (CPU: 0.8-2.9 W versus 7.3-7.9 W of the measured part; GPU: from −7 to 2.9 W versus 14-16 W). The fixed cost in energy, `C / E_net(30)`, is 0.7-1.9 % on CPU (the ratio with respect to the time fraction, 0.11-0.39) and on GPU from −4.6 to 1.5 %; a negative cost is not physical, so there it is measurement noise.
- **The uncorrected value is already close to M** (with N = 30 warm, `E_net(30)/30` versus M): CPU +0.7 %, +1.6 % and +1.9 % (TinyLlama, Qwen2.5, Ministral); GPU −4.4 %, −1.2 % and +1.5 %. Applying the criterion to the uncorrected value, the median is 1.56 % and the maximum 4.36 % (versus the least-squares M). **It is a later computation that was not fixed in advance**: it is a result, not a validation.
- **Consistent** with the ~2 % difference between N = 15 and N = 30 measured earlier with TinyLlama Q2_K on CPU at 15 W.

**Cold case.** The cache flush worked on TinyLlama (from 100 % to 0 %); in the other five combinations the cache was already at 0 %, because the models measured earlier had evicted them. The fixed cost cold is 6-14 s (versus 1-3 s warm) and its time fraction 17-35 %. The energy at N = 30 cold is between 1.1 % and 9.1 % higher than warm (CPU: 4.9 / 1.1 / 1.6 %; GPU: 9.1 / 3.7 / 3.7 %). `E_corr` cold comes out between −14 % and −32 % against M, meaningless physically for the same cause.

**What is concluded and how to declare it.**
- The fixed cost in energy is small with N = 30: ≤ 2 % on CPU with a warm cache and indistinguishable from zero on GPU, where the noise of a single measurement (~1-5 %) is larger than the bias one wanted to measure. **No correction is applied**, neither to the CSVs nor in the analysis.
- **Inference, not measured:** the official rows each load a different `.gguf` after a reboot per mode, so they were probably measured cold or with the cache in an intermediate state. If so, the relevant bias is that of warm plus the cold increment, i.e. of the order of 3-6 %, not ≤ 2 %. Declare the uncertainty of the LLM `energy_per_inference` according to what was measured in the repeatability characterization below, and do not interpret smaller differences between rows.
- What cannot be said is that a correction was validated. What was verified is that its absence introduces a small bias.
- **Cold repeatability, measured** (Qwen2.5 Q4_K_M, 50 W, N = 30, cache flushed before each invocation, 3 invocations per backend): `energy_per_inference` of 9.576 / 9.376 / 9.317 J on CPU (range 2.7 %, CV 1.4 %) and of 8.488 / 8.352 / 8.337 J on GPU (range 1.8 %, CV 1.0 %). Fixed-time fraction: 20-21 % on CPU and 34 % on GPU. `t_rep` stable (1.194-1.239 s on CPU; 0.573-0.574 s on GPU). The first invocation is the highest in both cases (+2.1 % and +1.7 % over the mean of the other two), but with n = 3 it is not distinguishable from noise. The implicit idle varied 2.03-2.15 W (CPU) and 2.15-2.23 W (GPU), and grows between consecutive invocations on GPU. Cold, the dispersion between invocations is the same as warm: cold adds fixed time, not variability.
- **Uncertainty to declare.** Between rows measured under the same conditions, **±3 %** (repeatability, one combination); the maximum observed between sibling rows is 3.7 % (Qwen2.5 with imatrix `wikitext2`, CPU 15 W, `plots/out/llm_deployment_summary.csv`). In absolute value, the cold energy can end up above that of a warm cache: +0.8 to +3.6 % on Qwen on CPU (mean +2.0 %, with the sweep's warm value of 9.24 J) and up to +9.1 % in the earlier sweep (TinyLlama on GPU). If all the official rows were measured cold, that bias is common and cancels when comparing them with each other, but not when comparing them with warm values.

## Unquantized reference (F16) and gap of each quantization

**Source:** `results/raw/llm_perplexity_f16_reference.csv` (3 rows in the official schema, `run_id` `llm_<model>_reference_f16`, `device` `offline`). Remeasured on the cluster with the same binary and `-t 8`: they coincide exactly with the values in this table, including the standard error given by `llama-perplexity` (0.0540, 0.0712 and 0.0569), which is the sampling error of the evaluation text, not a variation between executions. Details (command, duration, sha256 of the F16 and logs) in `results/investigations/llm/f16_reference/`. Each evaluation took 20, 24 and 50 min.

Perplexity of F16 on the WikiText-2 test split (`llama-perplexity`, `n_ctx = 512`, all chunks), and its relation to the Q4_K_M baseline without imatrix and to the WikiText-2 imatrix (200 lines, default):

| Model | PPL F16 | PPL Q4_K_M baseline | Gap against F16 | PPL Q4_K_M + imatrix (`wikitext2`) | Fraction of the gap closed by the imatrix |
|---|---|---|---|---|---|
| TinyLlama | 8.5113 ± 0.0540 | 8.7739 | +0.263 (+3.1 %) | 8.6470 | 48 % |
| Qwen2.5 | 10.1692 ± 0.0712 | 10.7936 | +0.624 (+6.1 %) | 10.5322 | 42 % |
| Ministral-3-3B | 8.8760 ± 0.0569 | 9.0307 | +0.155 (+1.7 %) | 9.0529 | −14 % |

- **Consistency:** F16 lies below all the measured quantizations, and in the three models the order Q4_K_M baseline < Q3_K_M < Q2_K is kept.
- **Meaning of the ±:** the one of the F16 reference is the standard error reported by `llama-perplexity` over the test text (sampling uncertainty of the text), not a deviation between calibration repetitions like those in the ablation tables (of the order of 0.002-0.02). Do not present them as the same quantity. The comparisons between configurations are paired (same text), so the uncertainty of their differences is smaller than that of each absolute value; it has not been quantified.
- **Reading for H4, descriptive:** Ministral-3-3B is the model with the smallest gap of Q4_K_M against F16 (1.7 % versus 3.1 % and 6.1 %), and even so the imatrix closes none of that residual gap (it ends up slightly worse), whereas in TinyLlama and Qwen2.5 it closes about half. That is, the fact that the imatrix does not help in Ministral is not explained only by its baseline already being close to F16. This does not identify the mechanism; it is one more piece of data for the Phase 4 analysis.

## Phase 4 (descriptive), part A: activation outlier ratio (ρ)

**Design** (fixed before computing, `calibration_protocol_llm.md`, Section 3.0.1). From the 9 existing imatrix files (K-quant row, WikiText-2, 200 lines, repetitions 1-3 of TinyLlama, Qwen2.5 and Ministral-3-3B). The files are GGUF v3 and store, per tensor, `in_sum2` (the sum of the squared input activation per channel) and `counts`; `m_c = sqrt(in_sum2_c / counts)`. Per tensor, `ρ = max_c(m_c) / (median_c(m_c) + 1e-8)`. Four types per block (qkv, attention output, gate/up and down), counting only once the groups that share an input. Value of each type: median across blocks of the mean `ρ` of the 3 repetitions; 95 % interval by bootstrap over the blocks (10,000 resamples, seed 0), which describes each model and is not a test between models. Criterion, fixed before seeing data: **consistent with the imatrix not helping Ministral if its median ρ is lower than that of TinyLlama and that of Qwen2.5 in at least 3 of the 4 types.** Outputs in `results/investigations/llm/phase4/activations/` (`checks.csv`, `tensors.csv`, `summary.csv`, `criterion.csv`, `run.log`); scripts in `analysis/phase4_llm/`. llama.cpp at `b95502ba9` (confirmed on the cluster with `git rev-parse`).

**Checks.** The 9 files have the expected structure: GGUF v3 (`general.type = imatrix`), the same 7 tensors per block and no other entry (no vision tensors), the protocol's dimensions, a single value of `counts` per file, finite and non-negative values, and identical vectors in q = k = v and in gate = up. In repetition 3 of the three models, `counts` is `chunk_count × 512 + 506`; unexplained, and with no effect on `m_c`, which is divided by the real `counts`.

**Primary result** (median of `ρ` across blocks; in brackets, the 95 % interval):

| Type | TinyLlama | Qwen2.5 | Ministral-3-3B | Ministral lower than both? | Does Ministral's CI overlap with TinyLlama's / Qwen2.5's? |
|---|---|---|---|---|---|
| qkv | 16.45 [14.7-18.4] | 20.70 [18.3-24.5] | 16.76 [15.2-19.3] | no | yes / yes |
| attention output | 6.28 [5.9-8.3] | 8.91 [7.7-12.9] | 7.53 [6.6-8.7] | no | yes / yes |
| gate/up | 5.44 [4.6-5.9] | 16.38 [12.2-20.0] | 10.89 [10.0-11.7] | no | no / no |
| down | 9.87 [8.0-13.7] | 11.84 [10.2-27.9] | 12.49 [9.8-16.9] | no | yes / yes |

**The criterion is not met (0 of 4): Ministral's difference with the imatrix is not explained by this metric.** The dispersion between repetitions is small (range of 0.08 to 0.68 in `ρ`, versus medians of 5 to 21).

**Reading.**
- In qkv, attention output and gate/up, Ministral's median lies between that of TinyLlama and that of Qwen2.5, and in down it is the highest, with overlapping intervals. Only in gate/up do Ministral's intervals not overlap with either of the other two.
- **TinyLlama has the lowest median `ρ` in the 4 types and, together with Qwen2.5, is one of the two models that benefit from the imatrix** (it closes 48 % and 42 % of the Q4_K_M gap against F16, versus −14 % in Ministral). The order of `ρ` between models does not follow that of the imatrix benefit. With the dimension equalized (controls, below), in qkv the lowest becomes Ministral.
- The most extreme maxima are in down and in the first blocks of the three models (TinyLlama, 1101 in block 2; Qwen2.5, 2767 in block 1; Ministral, 2571 in block 2), so they do not distinguish any of them.

**Secondary metrics** (descriptive, outside the criterion): the complete table of the four types and the robustness controls are in the subsection «Robustness controls of part A».

**Limits.** Three models, no inference between them. `ρ` is based on per-channel RMS and is not comparable with ViT's `R`. The activations are those seen by llama.cpp in F16 over ~21,000 to 23,000 WikiText-2 tokens. **The input dimensions differ between models and the maximum grows with the number of channels:** Ministral has the largest in the four types, which inflates its `ρ` and goes against a criterion that requires it to be lower. The control with the dimension equalized (below) reduces it by 8-18 % and does not change the verdict. Part B (weights) is described in the next section.

## Phase 4 (descriptive), part B: weight distribution (ω and kurtosis)

**Design** (fixed before computing, `calibration_protocol_llm.md`, Section 3.0.1). On the F16 GGUFs of the three models (`artifacts/llm_base/`), with `gguf-py`, by chunks of rows and float64 accumulators; no GPU. Per weight tensor (q, k, v, attention output, gate, up and down, per block): `ω = max_c(absmax_c) / (median_c(absmax_c) + 1e-8)`, with `absmax_c` the absolute maximum of the column of input channel `c`, and the tensor's excess kurtosis as a secondary one. **No criterion or directional prediction:** the imatrix weights the error by the activation energy and does not act on the distribution of the weights. Outputs in `results/investigations/llm/phase4/weights/`; scripts in `analysis/phase4_llm/`.

**Checks.** The dimensions of the metadata coincide with those of the protocol; in Ministral-3-3B the attention output has input 4096 and the architecture is `mistral3`; in Qwen2.5 `key_length` is missing and `embedding_length / head_count` is used; all the matrices are F16, with no infinite values or NaN, and there are no tensors outside the blocks except the expected ones. 535 matrices computed (155 + 197 + 183, at 7 per block plus the head).

**ω per tensor** (median across blocks; in brackets, the 95 % interval by bootstrap over the blocks, which describes each model and is not a test):

| Tensor | TinyLlama | Qwen2.5 | Ministral-3-3B | Ministral's position | Is Ministral's interval separate from the others'? |
|---|---|---|---|---|---|
| attn_q | 3.57 [3.27-4.06] | 2.24 [2.03-2.80] | 2.08 [1.95-2.34] | the lowest | from TinyLlama yes; from Qwen2.5 no |
| attn_k | 2.82 [2.63-3.18] | 2.49 [2.34-2.77] | 2.45 [2.21-2.58] | the lowest | from TinyLlama yes (barely); from Qwen2.5 no |
| attn_v | 2.16 [1.90-2.31] | 2.14 [1.94-2.60] | 1.89 [1.77-2.09] | the lowest | from neither |
| attn_output | 7.09 [6.39-8.07] | 4.81 [3.83-5.40] | 8.85 [7.39-9.43] | the highest | from Qwen2.5 yes; from TinyLlama no |
| ffn_gate | 5.23 [4.79-5.51] | 3.68 [3.32-4.19] | 3.22 [2.92-3.49] | the lowest | from TinyLlama yes; from Qwen2.5 no |
| ffn_up | 4.79 [4.06-5.27] | 3.82 [3.41-4.35] | 2.41 [2.18-2.62] | the lowest | from both |
| ffn_down | 9.58 [9.07-10.29] | 7.13 [6.52-7.69] | 9.39 [8.39-9.99] | intermediate | from Qwen2.5 yes; from TinyLlama no |

**Excess kurtosis** (median across blocks): attn_q 1.89 / 2.26 / 1.02; attn_k 1.74 / 1.73 / 1.24; attn_v 0.48 / 0.73 / 0.36; attn_output 1.21 / 2.04 / 1.44; ffn_gate 0.68 / 1.05 / 0.63; ffn_up 0.24 / 0.57 / 0.18; ffn_down 0.89 / 1.02 / 0.62 (TinyLlama / Qwen2.5 / Ministral). Ministral has the lowest in 6 of the 7 tensors; only in attn_output does it fall between the other two.

**Output head** (separate, outside the summaries): TinyLlama `output.weight`, ω 4.19 and kurtosis 1.67; Qwen2.5 `token_embd.weight`, 2.55 and 0.32; Ministral-3-3B `token_embd.weight`, 5.33 (the highest) and 0.42.

**Reading (descriptive).** Ministral generally has weights with lighter tails: lower ω and kurtosis in most tensors, with clear separation from the other two only in `ffn_up`. In `attn_output` it goes the opposite way (the highest ω). There was no directional prediction for the weights and the protocol does not allow attributing the imatrix effect to this metric, so it is not presented as an explanation. An absence of clear differences would be worded as «the weight distribution does not distinguish Ministral in these metrics», without claiming that the weights do not matter.

**Limits.** Three models, no inference between them. `ω` is not adjusted for the tensor's shape: the number of rows over which `absmax_c` is taken and that of columns differ between models and types (e.g. attn_k has 256, 256 and 1024 rows). The effect has not been measured; a larger number of columns in Ministral (3072 versus 2048 and 1536) would tend to raise `ω`, the opposite of what was observed.

### Robustness controls of part A (added after seeing parts A and B, and before computing them)

**Design** (pre-registered in `calibration_protocol_llm.md`, Section 3.0.1). Ministral has the largest input dimension in the four types and the maximum grows with the number of channels, which inflates its `ρ` and goes against the criterion. Three controls, local and read-only on the 9 imatrix files; the primary verdict does not change. Code in `analysis/phase4_llm/robustness.py`; results in `results/investigations/llm/phase4/robustness.csv` (72 rows) and `robustness.log`.

**Control 1: complete secondary metrics** (TinyLlama / Qwen2.5 / Ministral; same 3-of-4-types criterion, applied to each metric separately).

| Metric | qkv | Attention output | Gate/up | Down | Types where Ministral is lower | Verdict |
|---|---|---|---|---|---|---|
| `p99/p50` | 6.83 / 6.29 / **4.41** | 3.13 / 3.81 / 3.40 | 2.33 / 3.03 / 2.70 | 2.65 / 3.33 / **2.51** | 2 of 4 | not met |
| Fraction > 10 × median | 0.0045 / 0.0052 / **0.0010** | 0 / 0 / 0 | 0 / 0.0020 / 0.0003 | 0.00009 / 0.00019 / 0.00016 | 1 of 4 | not met |

In the fraction > 10 × median there are ties at 0 in the attention output, and a tie does not count as "lower".

**Control 2: dimension-adjusted `ρ`.** Subsampling of channels, without replacement, to the smallest dimension of the three models in each type (1536 in qkv, attention output and gate/up; 5632 in down), 1000 times with seed 0; the rest of the aggregation is that of the primary one. Where the dimension is already the reference one (Qwen2.5 in the first three types, TinyLlama in down), the value coincides with the primary one, as it should.

| Type | TinyLlama | Qwen2.5 | Ministral-3-3B | Ministral lower than both? |
|---|---|---|---|---|
| qkv | 15.65 [14.2-17.3] | 20.70 [18.3-24.5] | **13.68** [13.0-14.7] | yes |
| attention output | 5.97 [5.8-8.1] | 8.91 [7.7-12.9] | 6.91 [6.1-7.8] | no |
| gate/up | 5.00 [4.2-5.6] | 16.38 [12.2-20.0] | 9.19 [8.7-9.8] | no |
| down | 9.87 [8.0-13.7] | 11.18 [9.2-23.7] | 11.02 [8.9-14.0] | no |

**Ministral is lower in 1 of 4 types. Fixed wording: the negative result is robust to the dimension** (stored in the Spanish original as «el resultado negativo es robusto a la dimensión», see `docs/GLOSSARY.md`). The adjustment lowers Ministral's `ρ` by between 8 % and 18 % (TinyLlama's, between 5 % and 8 %), so the bias was real and of that magnitude, but it is not enough to meet the criterion.

**Control 3: Gaussian reference for `ω`** (no verdict). `ω` of 20 N(0,1) matrices of the same shape, float32, seed 0; in parentheses, `ω / ω_gauss`.

| Tensor | TinyLlama | Qwen2.5 | Ministral-3-3B |
|---|---|---|---|
| attn_q | 3.57 (2.42) | 2.24 (1.50) | 2.08 (1.40) |
| attn_k | 2.82 (1.77) | 2.49 (1.57) | 2.45 (1.59) |
| attn_v | 2.16 (1.36) | 2.14 (1.34) | 1.89 (1.22) |
| attn_output | 7.09 (4.80) | 4.81 (3.22) | 8.85 (5.81) |
| ffn_gate | 5.23 (3.60) | 3.68 (2.60) | 3.22 (2.25) |
| ffn_up | 4.79 (3.30) | 3.82 (2.70) | 2.41 (1.68) |
| ffn_down | 9.58 (6.15) | 7.13 (4.47) | 9.39 (6.11) |
| head | 4.19 (3.08) | 2.55 (1.95) | 5.33 (4.00) |

`ω_gauss` goes from 1.30 to 1.60 depending on the shape (deviation of 0.04 to 0.11 among the 20 matrices), so the shape effect is small compared with the observed differences. When dividing by the reference, the order between models for each tensor does not change except in `attn_k`, where Ministral ends up slightly above Qwen2.5 (1.59 versus 1.57). In all tensors `ω` exceeds that of a Gaussian of the same shape; the control only rules out the effect of the shape with light tails and says nothing more about the distribution.

**Descriptive observation, later.** The qkv type is the only one in which the robust measures coincide: Ministral has fewer outliers than the other two with dimension-adjusted `ρ` (13.68 versus 15.65 and 20.70), with `p99/p50` (4.41 versus 6.83 and 6.29) and with the fraction of outlier channels (0.0010 versus 0.0045 and 0.0052). It is 1 of 4 types and the pre-registered criterion requires 3, so it does not change the verdict.

**Limits.** Three models, no inference between them; the intervals describe each model. The controls were added after seeing parts A and B.

### Per-tensor imatrix intervention (Q3_K_M)

**Design** (fixed before running, `calibration_protocol_llm.md`, Section 3.0.2). In TinyLlama and Qwen2.5, with Q3_K_M and the WikiText-2 imatrix files of repetitions 1-3, the imatrix is applied only to a set S of tensors with `--include-weights` (each tensor is quantized independently, checked in the feasibility test). Sets, with the same composition per type (77 tensors in TinyLlama and 98 in Qwen2.5): H, the half of the blocks with the highest `ρ` in each type; L, the other half; R1-R3, three random selections; and E, the blocks farthest from the center (depth control, added after seeing the composition of H). `B(S) = (PPL_none − PPL_S) / (PPL_none − PPL_full,k)` is the fraction of the imatrix benefit that S recovers, with the perplexities measured in this test. Criterion, fixed before measuring: «`ρ` identifies the benefit» if in the three repetitions `B(H)` exceeds `B(L)` and the maximum of `B(R1..R3)`; secondary criterion: in addition, `B(H) > B(E)`. Results in `results/investigations/llm/phase4/intervention/`.

**Controls (all passed; the script stops if any of 1, 2, 4 and 5 fails).** Full values in `controls_<model>.json` and `evals/*.json`.

| Control | TinyLlama | Qwen2.5 | Result |
|---|---|---|---|
| 1. Without imatrix versus recorded | 9.2084 / 9.2084 | 12.4471 / 12.4471 | difference 0 |
| 1. Full, repetitions 1 / 2 / 3, versus recorded | 8.9554 / 8.9613 / 8.9585, identical | 11.4282 / 11.4393 / 11.4043, identical | difference 0 in the 3 |
| 2. All tensors passed explicitly versus the full imatrix | 154 tensors, 0 differences in the data | 196 tensors, 0 differences | identical data (the file's sha256, only as a record, also coincides) |
| 3. Binary and node | `~/llama.cpp/build/bin/llama-perplexity`, on CPU | idem | the same binary (checkout at `b95502ba9`, confirmed with `git rev-parse`) |
| 4. The same GGUF evaluated twice | 8.9554 / 8.9554 | 11.4282 / 11.4282 | difference 0 |
| 5. Threads | 8, in `llama-quantize` and in `-t`, in the 22 evaluations | idem | fixed; control 1 confirms they do not change the result |

| Model | Rep | B(H) | B(L) | B(R1) | B(R2) | B(R3) | Random maximum | B(E) | B(H) + B(L) |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama | 1 | 0.869 | 0.087 | 0.367 | 0.483 | 0.846 | 0.846 | 0.724 | 0.956 |
| TinyLlama | 2 | 0.877 | 0.095 | 0.408 | 0.471 | 0.853 | 0.853 | 0.732 | 0.972 |
| TinyLlama | 3 | 0.879 | 0.090 | 0.397 | 0.459 | 0.866 | 0.866 | 0.725 | 0.970 |
| Qwen2.5 | 1 | 0.656 | 0.353 | 0.593 | 0.712 | 0.623 | 0.712 | 0.681 | 1.010 |
| Qwen2.5 | 2 | 0.657 | 0.375 | 0.567 | 0.717 | 0.617 | 0.717 | 0.657 | 1.033 |
| Qwen2.5 | 3 | 0.674 | 0.347 | 0.583 | 0.695 | 0.603 | 0.695 | 0.662 | 1.021 |

**Result with the pre-registered criterion.**
- **TinyLlama: «`ρ` identifies the benefit», and also beyond depth.** `B(H)` exceeds `B(L)`, the random maximum and `B(E)` in the three repetitions. The margin over the best random one (R3) is narrow: 0.023, 0.024 and 0.013 in `B`, and 0.003 to 0.006 in perplexity (H 8.9886 / 8.9916 / 8.9887 versus R3 8.9944 / 8.9977 / 8.9919).
- **Qwen2.5: «not distinguishable from chance».** R2 exceeds H in the three repetitions, so the main criterion is not met and the secondary one does not apply. `B(E)` is on a par with `B(H)`, as expected because they share 77 of the 98 tensors.

**What to keep in mind when reading it.**
- **The criterion discriminates little.** With only 3 random sets, if `ρ` were not informative, H would be one more set and would exceed the 3 with probability 1/4 by chance. The three repetitions use the same sets and only the imatrix calibration sample changes, so they are not independent evidence about the choice of the random sets. The TinyLlama result is suggestive, not established. The extension with 20 random sets (below) contrasts it.
- **The benefit seems concentrated in few tensors (inference, not tested).** In TinyLlama, `B` does not grow with the overlap with H: R3 shares 44 of 77 tensors with H and recovers 0.846 (97 % of `B(H)`), while R1 (37 tensors in common) recovers 0.367 and R2 (32) 0.483. With a nearly additive benefit, this is compatible with a few tensors of H concentrating most of the effect, and R3 containing them by chance. With 20 random sets, the Spearman between `B(R)` and the overlap with H is practically 0 in both models (0.09 and 0.005), and in TinyLlama `B(R)` goes from 0.07 to 0.85 (R5 shares 41 of 77 tensors with H and recovers only 0.070). What a set recovers does not depend on how many tensors it shares with H, which is compatible with it depending on including a few specific tensors.
- **Descriptive and later, corrected by the extension: L is not extreme.** With 3 random sets, L recovered less than all of them in both models. With 20 it stops being so: in TinyLlama it is second from the bottom (`p_L` = 0.095; only R5, with 0.070, recovers less) and in Qwen2.5, fourth (`p_L` = 0.19). The earlier observation was, in large part, an effect of having only 3 random sets. In Qwen2.5, H exceeds L by 0.3, but neither of the two separates from the random distribution.
- **Additivity:** `B(H) + B(L)` is 0.96-0.97 in TinyLlama and 1.01-1.03 in Qwen2.5, close to 1: the benefit is nearly additive per tensor.

**Extension: rank-based contrast with 20 random sets** (pre-registered in Section 3.0.2 after seeing the previous result and before seeing these data). 17 more random sets (R4 to R20) per model, evaluated only with the imatrix of repetition 1; with R1-R3, 20 in total. The 34 new evaluations used the same tensors as `sets.json` (the extended one), 8 threads, 658 and 583 chunks, and reused the "none" and "full,1" perplexities already measured (9.2084 and 8.9554; 12.4471 and 11.4282). No errors, with 17 different GGUFs per model; each perplexity took 1952-1976 s in TinyLlama and 2619-2656 s in Qwen2.5. `p_H = (1 + no. of random sets with B ≥ B(H)) / 21` and `p_L` analogous from below.

| Model | B(H) | `p_H` | B(L) | `p_L` | B(R): min / Q1 / median / Q3 / max | Spearman of B(R) with the overlap with H | Result |
|---|---|---|---|---|---|---|---|
| TinyLlama | 0.869 | 0.048 | 0.087 | 0.095 | 0.070 / 0.284 / 0.501 / 0.691 / 0.846 | 0.09 (p = 0.70) | **«H es extremo»** (H is extreme); L, second of 21 from the bottom |
| Qwen2.5 | 0.656 | 0.238 | 0.353 | 0.190 | 0.299 / 0.405 / 0.495 / 0.635 / 0.753 | 0.005 (p = 0.98) | H, fifth of 21; L, fourth of 21 from the bottom |

- **TinyLlama: «H es extremo» (H is extreme).** It exceeds the 20 random sets (the closest one is still R3, with 0.846). It agrees with the main verdict. But `p_H` = 0.048 is the minimum possible with 20 random sets, the contrasts of H and L go without adjustment for multiplicity (with Bonferroni over the two, 0.095) and it has only been measured with repetition 1. It is moderate evidence.
- **Qwen2.5:** H is in the upper quartile of the random distribution, but not at the extreme (four sets exceed it, up to 0.753). It agrees with «not distinguishable from chance».
- **Dispersion of the random sets:** much larger in TinyLlama (0.07 to 0.85) than in Qwen2.5 (0.30 to 0.75). In TinyLlama the imatrix benefit is more heterogeneous across tensors: a few seem to concentrate a good part of the effect.

**Limits.** Two models, three repetitions and a single configuration (Q3_K_M). The comparison with the maximum of 3 random sets is not a statistical test. The perplexity is on WikiText-2, the same domain as the imatrix corpus. It does not prove the mechanism: it says where the benefit is concentrated, not why.

### Ministral-3-3B: imatrix ceiling effect?

**Design** (pre-registered in `calibration_protocol_llm.md`, Section 3.0.3, before seeing these data). Ministral-3-3B at Q2_K (primary) and Q3_K_M (secondary) with the WikiText-2 imatrix of repetitions 1-3, which had never been measured because in Phase 3 there was no winning imatrix for this model. `G = (PPL_none − PPL_imatrix,k) / (PPL_none − PPL_F16)`, with `PPL_F16` = 8.8760. Criterion: «compatible with a ceiling effect» if at Q2_K `G ≥ 0.20` in the three repetitions.

**Checks.** The quantization without imatrix reproduces exactly what was recorded (Q2_K 12.9385 and Q3_K_M 9.6171, difference 0). 6 evaluations with imatrix, 8 threads and 588 chunks; 8 different GGUFs (sha256), no errors. Each perplexity took about 68 min at Q2_K and about 89 at Q3_K_M.

| Format | Without imatrix | With imatrix (rep 1 / 2 / 3) | G (rep 1 / 2 / 3) | G ≥ 0.20 in the 3? |
|---|---|---|---|---|
| **Q2_K (primary)** | 12.9385 | 11.6764 / 11.7264 / 11.6698 | 0.311 / 0.298 / 0.312 | yes: **«compatible with a ceiling effect»** |
| Q3_K_M (secondary) | 9.6171 | 9.4307 / 9.4117 / 9.4306 | 0.252 / 0.277 / 0.252 | yes (no verdict of its own) |

**Context, the same G in the three models** (mean of the 3 repetitions; Q2_K and Q3_K_M from the Phase 3 perplexity record, Q4_K_M from Phase 2):

| Model | Q4_K_M | Q3_K_M | Q2_K |
|---|---|---|---|
| TinyLlama | 0.48 | 0.36 | 0.45 |
| Qwen2.5 | 0.42 | 0.45 | 0.75 |
| Ministral-3-3B | −0.14 | 0.26 | 0.31 |

**Reading.** In Ministral, the imatrix goes from not helping at 4 bits (−14 %) to closing about 26 % of the gap at 3 bits and 31 % at 2 bits, consistently in the three repetitions. **The lack of effect at 4 bits is not a general insensitivity of Ministral to the imatrix**, and it fits a ceiling effect: at 4 bits its gap against F16 was the smallest (1.7 %). Even so, it closes less than the other two models in all formats, so the ceiling does not explain everything: Ministral responds less to the imatrix even when the gap is large.

**What it cannot say.** It is compatible with a ceiling effect, but does not prove it: it also fits with calibration mattering more at fewer bits in any model, as in ViT (Section 4.1 of `findings_vit.md`). It is a single model with three repetitions. In TinyLlama `G` does not grow monotonically as bits drop (0.48, 0.36 and 0.45), so there is no general rule about bit-width.

**Traceability (resolved).** The reference F16 perplexities were remeasured on the cluster and coincide exactly with the table «Unquantized reference (F16)» (difference 0, also in the standard error); the source is now `results/raw/llm_perplexity_f16_reference.csv`.

### Synthesis of LLM Phase 4 for RQ2

- **Activations (part A, with criterion):** the criterion on `ρ` is not met (0 of 4 types), so Ministral's lack of improvement with the imatrix is not explained by the magnitude of the activation outliers measured this way, and TinyLlama, with the lowest `ρ`, benefits from the imatrix. **The result is robust to the dimension** (with the dimension equalized, Ministral is lower in 1 of 4 types; with `p99/p50`, in 2 of 4; with the fraction of outlier channels, in 1 of 4).
- **Weights (part B, descriptive):** Ministral has weights with generally lighter tails, with no clear separation except in `ffn_up`, and with no prior prediction.
- **Per-tensor intervention (within each model):** in TinyLlama, applying the imatrix to the half of the blocks with the highest `ρ` (per type) recovers 87 % of its benefit, more than the other half, than the depth control and than 20 random sets (`p_H` = 0.048, unadjusted; moderate evidence). In Qwen2.5 it is not distinguishable from chance (H, fifth of 21). `ρ` points to where the imatrix acts in one of the two models.
- **Ceiling effect (Ministral):** at 4 bits the imatrix does not help it (−14 %), but at 3 and 2 bits it closes 26 % and 31 % of the gap in the three repetitions: compatible with a ceiling effect, although it still responds less than the other two models.
- **Scope:** 3 models, no inference between them. The per-tensor intervention was done on TinyLlama and Qwen2.5; a finer per-tensor analysis remains as future work.

## On-disk size of the artifacts (reduction versus F16)

**Source.** `results/artifact_sizes.csv` (44 rows, no warnings), generated on the cluster with `scripts/artifact_sizes.py`, guided by the measurement manifests: the 31 LLM GGUFs measured on device, the 3 base F16 and the 10 ViT deployment ONNX. It is the size of the file the device loads (`stat`, in MiB), not the memory footprint. It is joined to the results by `run_id`; the results CSVs are not modified (`model_size_mb` stays empty).

**Reduction per format** (baseline row without imatrix; in parentheses, the ratio `F16 / format` and the reduction):

| Model | F16 (MiB) | Q4_K_M | Q4_0 | Q3_K_M | Q2_K |
|---|---|---|---|---|---|
| TinyLlama | 2099.1 | 636.9 (3.30×; −69.7 %) | 607.2 (3.46×; −71.1 %) | 523.0 (4.01×; −75.1 %) | 412.1 (5.09×; −80.4 %) |
| Qwen2.5-1.5B | 2950.4 | 940.4 (3.14×; −68.1 %) | 891.6 (3.31×; −69.8 %) | 786.0 (3.75×; −73.4 %) | 645.0 (4.57×; −78.1 %) |
| Ministral-3-3B | 6548.1 | 2047.1 (3.20×; −68.7 %) | 1946.5 (3.36×; −70.3 %) | 1712.4 (3.82×; −73.8 %) | 1391.4 (4.71×; −78.8 %) |

**Observations.**
- The reduction goes from 3.1× (Q4_K_M) to 5.1× (Q2_K), and is similar across models: the largest is TinyLlama's, because the part that is not quantized (embeddings and head) weighs less. Q4_0 is somewhat smaller than Q4_K_M (3.3 to 3.5× versus 3.1 to 3.3×).
- The calibration variants of the same format have the same size: the four Q4_K_M rows of each model (corpus, calibration size, scheme) and the baseline and combined rows of Q3_K_M and Q2_K coincide. The imatrix does not change the size. The only exception is Q4_0 with WikiText-2 imatrix, which weighs 0.2-0.3 % more than without imatrix (608.6 versus 607.2 MiB in TinyLlama; 894.1 versus 891.6 in Qwen2.5; 1951.6 versus 1946.5 in Ministral).
- The F16 files take 2.05, 2.88 and 6.39 GiB, which serves to decide on which devices they fit (`technical_architecture.md`, Section 6.7).

## F16 reference on the Jetson: speedup and energy saving from quantization (RQ3a)

**Design** (`technical_architecture.md`, Section 6.7, fixed before measuring). 18 F16 rows (3 models × 6 Jetson configurations), measured with the same protocol as the 31 quantized rows (`llama-bench -p 128 -n 32 -r 30`, `--with-power --temp-cooldown`). They are compared with the baseline rows without imatrix of Q4_K_M, Q4_0 (legacy), Q3_K_M and Q2_K. `Q / F16` ratios (≥ 1 favors the quantized format): decode (`tg`), prefill (`pp`) and energy per request (`E_F16 / E_Q`); and size reduction (`size_F16 / size_Q`, from `results/artifact_sizes.csv`). Fixed reading: a ratio is only different from 1 if it falls outside [0.9, 1.1], because F16 and Q were measured in different invocations. Results in `results/investigations/llm/f16_ratios/ratios.csv` (72 ratios of each type, no missing data); `ΔPPL` against the F16 perplexities of `results/raw/llm_perplexity_f16_reference.csv`.

Each cell is **tg / pp / E**; ≈ indicates that it falls in [0.9, 1.1].

**TinyLlama** (PPL F16 = 8.5113)

| Mode | Q4_K_M | Q4_0 | Q3_K_M | Q2_K |
|---|---|---|---|---|
| Size (F16/Q) | 3.30× | 3.46× | 4.01× | 5.09× |
| ΔPPL (no imatrix) | +0.26 (+3.1 %) | +0.34 (+4.0 %) | +0.70 (+8.2 %) | +3.80 (+44.6 %) |
| CPU 15 W | 1.70 / 1.24 / 2.04 | 1.77 / 1.32 / 1.99 | 1.80 / 1.20 / 2.07 | 2.00 / ≈1.09 / 2.22 |
| CPU 30 W | 2.67 / 1.37 / 2.05 | 2.75 / 1.46 / 2.12 | 2.65 / 1.31 / 2.00 | 2.78 / 1.23 / 2.15 |
| CPU 50 W | 1.88 / 1.22 / 1.94 | 1.91 / 1.28 / 2.00 | 1.80 / 1.23 / 1.90 | 1.89 / 1.22 / 2.02 |
| GPU 15 W | 1.55 / 0.72 / 1.83 | 2.06 / 0.80 / 2.17 | 1.13 / 0.66 / 1.54 | 1.28 / 0.54 / 1.69 |
| GPU 30 W | 1.59 / 0.82 / 1.78 | 1.97 / ≈0.91 / 2.20 | 1.18 / 0.76 / 1.52 | 1.33 / 0.62 / 1.68 |
| GPU 50 W | 1.81 / ≈0.95 / 1.77 | 2.19 / ≈1.04 / 2.22 | 1.40 / 0.85 / 1.51 | 1.51 / 0.71 / 1.76 |

**Qwen2.5** (PPL F16 = 10.1692)

| Mode | Q4_K_M | Q4_0 | Q3_K_M | Q2_K |
|---|---|---|---|---|
| Size (F16/Q) | 3.14× | 3.31× | 3.75× | 4.57× |
| ΔPPL (no imatrix) | +0.62 (+6.1 %) | +1.04 (+10.2 %) | +2.28 (+22.4 %) | +15.96 (+156.9 %) |
| CPU 15 W | 2.55 / 1.25 / 2.41 | 2.63 / 1.34 / 2.32 | 2.72 / 1.22 / 2.33 | 2.98 / 1.11 / 2.52 |
| CPU 30 W | 2.58 / 1.31 / 2.02 | 2.66 / 1.38 / 2.08 | 2.59 / 1.29 / 1.96 | 2.75 / 1.18 / 2.13 |
| CPU 50 W | 2.38 / 1.18 / 1.96 | 2.40 / 1.23 / 2.04 | 2.34 / ≈1.05 / 1.95 | 2.40 / 1.15 / 2.02 |
| GPU 15 W | 2.12 / 0.81 / 2.06 | 2.81 / 0.898 / 2.67 | ≈1.05 / 0.74 / 1.56 | 1.63 / 0.62 / 1.85 |
| GPU 30 W | 2.18 / 0.78 / 1.96 | 2.81 / 0.86 / 2.47 | ≈1.09 / 0.72 / 1.40 | 1.70 / 0.60 / 1.78 |
| GPU 50 W | 2.13 / 0.85 / 1.99 | 2.67 / ≈1.00 / 2.45 | ≈1.05 / 0.77 / 1.62 | 1.63 / 0.65 / 1.94 |

**Ministral-3-3B** (PPL F16 = 8.8760)

| Mode | Q4_K_M | Q4_0 | Q3_K_M | Q2_K |
|---|---|---|---|---|
| Size (F16/Q) | 3.20× | 3.36× | 3.82× | 4.71× |
| ΔPPL (no imatrix) | +0.15 (+1.7 %) | +0.44 (+5.0 %) | +0.74 (+8.3 %) | +4.06 (+45.8 %) |
| CPU 15 W | 1.75 / 1.29 / 1.78 | 1.82 / 1.41 / 1.85 | 1.83 / 1.27 / 2.00 | 1.99 / 1.13 / 2.20 |
| CPU 30 W | 2.81 / 1.39 / 2.10 | 2.89 / 1.49 / 2.15 | 2.78 / 1.38 / 2.06 | 2.94 / 1.28 / 2.24 |
| CPU 50 W | 2.01 / 1.38 / 2.00 | 2.02 / 1.34 / 2.03 | 1.88 / 1.31 / 1.97 | 1.94 / 1.35 / 2.04 |
| GPU 15 W | 1.57 / 0.68 / 1.79 | 2.01 / 0.77 / 2.22 | 1.14 / 0.63 / 1.49 | 1.31 / 0.51 / 1.66 |
| GPU 30 W | 1.47 / 0.76 / 1.84 | 1.88 / 0.84 / 2.30 | 1.13 / 0.69 / 1.61 | 1.29 / 0.57 / 1.80 |
| GPU 50 W | 1.75 / 0.82 / 1.79 | 2.15 / ≈0.91 / 2.22 | 1.38 / 0.76 / 1.56 | 1.56 / 0.63 / 1.76 |

**Count with the fixed reading** (72 ratios of each type): energy per request, Q better in all 72; decode, Q better in 69 and 3 with no appreciable difference (Qwen Q3_K_M in the three GPU configurations), none worse; prefill on CPU, Q better in 34 and 2 with no difference; **prefill on GPU, F16 faster in 31 and 5 with no difference, none in favor of Q**. Two values graze the limit of the interval: 0.898 (Qwen Q4_0, GPU 15 W prefill) and 1.115 (Qwen Q2_K, CPU 15 W prefill).

**Reading.**
- **Quantization saves energy per request in all cases** (from 1.40× to 2.67×), on CPU and on GPU, in the three modes and the four formats. It is the LLM branch's answer to the energy part of RQ3a.
- **It speeds up decode in almost all cases** (from 1.13× to 2.98×, except Qwen Q3_K_M on GPU, with no appreciable difference), and prefill on CPU. **On GPU, prefill is faster in F16** (Q between 0.51× and 0.85× in most). The cause has not been verified; the hypothesis is the cost of dequantizing in the GPU kernels compared with the F16 matrix multiplication.
- **The energy saving and the speedup are smaller than the size reduction** (3.1× to 5.1×): quantization reduces memory considerably more than it speeds up or saves.
- **Fewer bits does not mean faster.** On GPU, Q3_K_M and Q2_K decode more slowly than Q4_K_M and Q4_0 (Qwen Q3_K_M ≈1.05× versus Q4_K_M 2.12× on GPU 15 W), and Q4_0 is the fastest format and the one that saves the most on GPU (decode from 1.88× to 2.81×; energy from 2.17× to 2.67×). Going down to Q2_K barely adds energy saving compared with Q4_K_M (TinyLlama CPU 15 W: 2.22× versus 2.04×) while its `ΔPPL` skyrockets (from +3.1 % to +44.6 % without imatrix; from +1.6 % to +24.7 % with the best calibration), so **Q4_K_M and Q4_0 dominate the trade-off between saving and quality**.
- **The energy saving depends on the shape of the request.** With 128 prompt tokens and 32 generated, decode is 81-96 % of the time of each repetition (in the compared CPU and GPU rows), and that is why quantization saves energy on GPU too, even though prefill is slower there. In prefill-dominated requests (long prompts, few generated tokens), on GPU the advantage of quantization would shrink and could reverse.

**Fixed cost of the window and its effect on the energy ratios (later analysis).** The fraction of the window's time that does not correspond to the 30 repetitions (mostly model loading) in the F16 rows is:

| | TinyLlama | Qwen2.5 | Ministral-3-3B |
|---|---|---|---|
| CPU 15 W | 7.6 % | 6.9 % | 6.5 % |
| CPU 30 W | 13.9 % | 11.8 % | 11.7 % |
| CPU 50 W | 22.5 % | 16.0 % | 19.6 % |
| GPU 15 W | 3.1 % | 2.6 % | 4.1 % |
| GPU 30 W | 4.0 % | 3.0 % | 4.7 % |
| GPU 50 W | 8.1 % | 7.0 % | 9.6 % |

In the quantized reference rows, the same fraction (with each row's window computed as `energy_raw_j / avg_power_w_raw`, which in the 18 F16 rows coincides with the recorded one with differences of ≤ 3 ms) is smaller than the F16 one in all CPU cases; on GPU there are exceptions (Qwen2.5 at 30 W: 13-16 % in Q versus 3.0 % in F16). In energy, the fixed part weighs between `k` = 0.11 and 0.39 times its time fraction on CPU and nothing distinguishable on GPU (N sweep). The ratio without the fixed energy would be `r · (1 − k·φ_F16) / (1 − k·φ_Q)`, and on CPU the measured one is inflated in the 36 cases. Maximum overestimation (`k` = 0.39), Q4_K_M / Q4_0 / Q3_K_M / Q2_K:

| Mode | TinyLlama | Qwen2.5 | Ministral-3-3B |
|---|---|---|---|
| CPU 15 W | 0.9 / 1.2 / 1.3 / 1.4 % | 0.8 / 0.8 / 0.9 / 1.0 % | 0.9 / 0.9 / 1.0 / 1.0 % |
| CPU 30 W | 0.7 / 0.6 / 1.1 / 1.6 % | 2.9 / 2.8 / 3.0 / 2.9 % | 0.2 / 0.2 / 0.2 / 1.0 % |
| CPU 50 W | 7.2 / 7.2 / 7.5 / 7.3 % | 0.2 / 0.4 / 1.2 / 1.4 % | 1.4 / 1.8 / 4.4 / 6.5 % |

**The overestimation reaches 7.5 % in TinyLlama at CPU 50 W and 4.4-6.5 % in Ministral Q3_K_M and Q2_K at CPU 50 W; in the rest of the CPU it is ≤ 3.0 %.** (It corrects an earlier bound, made with per-configuration medians, that gave ≤ 3 % outside TinyLlama.) The lowest ratio after the correction is 1.76 (TinyLlama Q3_K_M, CPU 50 W): the 36 CPU ratios remain «Q mejor» (Q better) with both extremes of `k`, so the conclusions do not change; the exact magnitudes on CPU carry that uncertainty. On GPU it is not corrected. It is a later analysis, not pre-registered, and `k` was only measured at 50 W and with Q4_K_M: applying it at 15 and 30 W, to other formats and to the F16 is an assumption. CSV columns: `r_energy_k011`, `r_energy_k039`, `energy_bias_max_pct` and `read_energy_bounds`.

**Quality loss with the best calibration.** The imatrix changes neither the format nor the size, and within a session the sibling rows vary ≤ 3.7 % in energy, so the best calibration improves accuracy at no cost in speed or energy. `ΔPPL` against the F16, without imatrix (deterministic) and with the best calibration (mean ± sd of 3 repetitions):

| Model | Format | No imatrix | Best calibration | Row of the best calibration |
|---|---|---|---|---|
| TinyLlama | Q4_K_M | +3.1 % | +1.6 % (8.6470 ± 0.0015) | Phase 2 winner (`calibration_corpus_wikitext2`) |
| | Q4_0 | +4.0 % | +3.4 % (8.7987 ± 0.0025) | `quant_scheme_legacy_corpus_wikitext2` |
| | Q3_K_M | +8.2 % | +5.3 % (8.9584 ± 0.0030) | Phase 3 combined |
| | Q2_K | +44.6 % | +24.7 % (10.6121 ± 0.0030) | Phase 3 combined |
| Qwen2.5 | Q4_K_M | +6.1 % | +3.6 % (10.5322 ± 0.0122) | Phase 2 winner |
| | Q4_0 | +10.2 % | +6.5 % (10.8325 ± 0.0113) | `quant_scheme_legacy_corpus_wikitext2` |
| | Q3_K_M | +22.4 % | +12.3 % (11.4239 ± 0.0179) | Phase 3 combined |
| | Q2_K | +156.9 % | +39.3 % (14.1671 ± 0.0114) | Phase 3 combined |
| Ministral-3-3B | Q4_K_M | +1.7 % | +1.7 % (no imatrix; there is no Phase 2 winner) | `phase1_baseline_default_q4km` |
| | Q4_0 | +5.0 % | +3.4 % (9.1754 ± 0.0046) | `quant_scheme_legacy_corpus_wikitext2` |
| | Q3_K_M | +8.3 % | +6.2 % (9.4243 ± 0.0109) | ceiling-effect test |
| | Q2_K | +45.8 % | +31.7 % (11.6909 ± 0.0309) | ceiling-effect test |

With the best calibration, the loss drops in all formats except Ministral Q4_K_M, which keeps the baseline by the selection rule (no calibration improves it; with WikiText-2 imatrix it would be +2.0 %). **Calibration roughly halves the quality loss of quantization without changing its cost**: it is the link between RQ1 and RQ3a in the LLM branch. Even calibrated, Q2_K loses between 25 % and 39 %, so Q4_K_M and Q4_0 still dominate the trade-off.

**Limits.** F16 and Q were measured in different invocations, each row in a single session, and that is why the reading uses the interval [0.9, 1.1]. The `ΔPPL` without imatrix has n = 1 (the rows are deterministic) and the calibrated one is the mean of 3 repetitions. A single platform (the Jetson); the colleague's devices will be added where memory allows. The cause of the faster prefill in F16 on GPU is a hypothesis.

## GPU versus CPU on the Jetson (RQ3a)

**Design** (reading fixed before computing). For each model, format and power mode, the same artifact on GPU and on CPU is compared: Q4_K_M, Q4_0 (legacy), Q3_K_M and Q2_K without imatrix, and F16; 5 formats × 3 models × 3 modes = 45 pairs, all with N = 30. GPU/CPU ratios (≥ 1 favors GPU): prefill `pp_GPU / pp_CPU`, decode `tg_GPU / tg_CPU` and energy per request `E_CPU / E_GPU`. A ratio is only different from 1 if it falls outside [0.9, 1.1], because each row is a single session and CPU and GPU were measured in different invocations. Results in `results/investigations/llm/gpu_vs_cpu/ratios.csv`, generated with `gpu_vs_cpu.py` (the complete per-model tables are in the CSV).

| Metric | GPU better | No difference | CPU better |
|---|---|---|---|
| Prefill | 45 | 0 | 0 |
| Decode | 37 | 5 | 3 |
| Energy per request | 26 | 12 | 7 |

**Energy ratio `E_CPU / E_GPU`** (range across the three models):

| Format | 15 W | 30 W | 50 W |
|---|---|---|---|
| Q4_K_M | 0.89-1.07 | 1.33-1.37 | 1.12-1.16 |
| Q4_0 | 1.15-1.27 | 1.63-1.64 | 1.36-1.38 |
| Q3_K_M | 0.70-0.79 | 0.99-1.20 | 0.95-0.99 |
| Q2_K | 0.77-0.80 | 1.15-1.24 | 1.07-1.10 |
| F16 | 1.04-1.06 | 1.38-1.58 | 1.15-1.25 |

Mean net power: at 15 W, 1.5-1.9 W on CPU versus 2.4-4.7 W on GPU; at 30 W, 4.8-6.5 W versus 4.1-8.2 W; at 50 W, 5.8-7.3 W versus 8.8-16.0 W.

**Reading.**
- **Prefill is always faster on GPU** (45 of 45), with an advantage that grows with power (1.3-2.9× at 15 W; 2.3-5.0× at 50 W). Decode is in 37 of 45.
- **The GPU's energy advantage depends on the mode, it is not universal.** At 30 W it wins in 14 of 15 cases; at 50 W, in 9 (the other 6, no difference); **at 15 W the CPU wins in 7 of 15** and the GPU in only 3: the GPU decodes faster, but consumes roughly twice the net power.
- **By format:** Q4_0 is the most favorable to the GPU (it wins in energy and in decode in all 9 cases; energy 1.15-1.64×); F16 wins in 6 and never loses; Q3_K_M and Q2_K are the least favorable (decode on GPU from 0.57× to 2.00×) and concentrate 6 of the 7 cases in which the CPU wins in energy. It is consistent with what was seen against F16: the low-bit K-quant kernels perform worse on GPU.
- **The 7 cases in which the CPU wins in energy are at 15 W:** Q3_K_M and Q2_K in the three models (0.70-0.80×) and Qwen2.5 Q4_K_M (0.89×). The CPU decodes faster in 3 cases, all at 30 W (Qwen2.5 Q3_K_M 0.57×, Qwen2.5 Q2_K 0.83× and TinyLlama Q3_K_M 0.88×), and in them the GPU still wins or ties in energy because of its lower power in that mode.
- **Model size gives no monotonic trend.**

**Fixed-cost bias (later analysis).** With the same bound as in the ratios against F16 (`k` from 0.11 to 0.39 on CPU and 0 on GPU), `E_CPU / E_GPU` may be overestimated by between 1.6 % and 9.6 %. No reading changes sign, and three cases go from «GPU mejor» (GPU better) to «sin diferencia» (no difference) with `k` = 0.39: Qwen2.5 Q4_K_M at 50 W (1.165 → 1.094), Qwen2.5 F16 at 50 W (1.146 → 1.075) and Ministral Q4_K_M at 50 W (1.118 → 1.046). The 7 cases in favor of the CPU are kept. On GPU, the fixed fraction reaches 13-16 % in Qwen2.5 Q4_K_M and Q4_0 at 30 W, and assuming `k` = 0 on GPU is an assumption that does not cover a possible bias in the opposite direction; `k` was only measured at 50 W and with Q4_K_M.

**Contrast with ViT (later analysis, not pre-registered).** In ViT, FP32 on GPU dominates INT8 on CPU in latency and in energy (`findings_vit.md`, Section 7.9). In LLM the opposite happens: comparing F16 on GPU with each quantized format on CPU, in the same mode and model (36 cases), the GPU always wins in prefill (1.8-4.1×), decode is split (16 in favor of the GPU, 6 with no difference and 14 in favor of the CPU) and **quantization on CPU spends less energy per request in all 36 cases** (`E_CPU,Q / E_GPU,F16` from 0.41 to 0.79). That this is due to decode being memory-bound is a hypothesis, not a verified mechanism.

**Limits.** A single session per row and a single device. The same mode name does not imply the same power on CPU and on GPU, and that is why the measured net power is given. The energy conclusions hold for requests of 128+32 tokens. The causes pointed to (the relative power of each backend per mode, the performance of low-bit kernels on GPU) are readings of what was measured, not verified mechanisms.


## Performance and energy on all devices

**Design** (same rules as «F16 reference on the Jetson», fixed before measuring the F16). `llm_` rows of `results/combined/edge_results_combined.csv` (292 rows, one measurement per row) joined by `run_id` to `results/artifact_sizes.csv`. Devices: Jetson AGX Orin (CPU and GPU × 15, 30 and 50 W), n150, rk1-02, CM4 (`kube01`) and VisionFive 2 (`visionfive_2`); all except the Jetson's GPU are CPU. On the CM4 and the VisionFive 2 only TinyLlama and Qwen2.5 were measured (F16 and the four formats without imatrix). Ratios against F16 of Q4_K_M, Q4_0, Q3_K_M and Q2_K without imatrix: `pp_Q / pp_F16`, `tg_Q / tg_F16` and `E_F16 / E_Q` (net energy per request of 128+32 tokens), so that > 1 favors the quantized format. Reading: different from 1 only outside [0.9, 1.1]. Output: `plots/out/llm_deployment_summary.csv` (292 rows: absolute values, size, window, fixed fraction, ratios and classification), generated with `python3 -m analysis.llm_deployment_summary`. The 72 Jetson ratios reproduce those of `results/investigations/llm/f16_ratios/ratios.csv` (maximum difference 5·10⁻⁵, same classifications).

**Prior checks (all passed).** `num_inferences_measured == requested_repeats` and `energy_net_j > 0` in the 292 rows; `energy_per_inference == energy_net_j / N` and `perf_per_watt == 160·N / energy_net_j` (relative tolerance 10⁻⁶); no duplicates per device, backend, mode and `run_id`; all rows with size. N = 30 in all except the CM4 and the VisionFive 2 (N = 10, except TinyLlama Q2_K with N = 30).

**Excluded row.** `llm_ministral_3_3b_reference_f16` on the n150: not valid because of disk reads due to lack of memory during the measurement. It does not enter any ratio; the 8 quantized Ministral-3-3B rows on the n150 are kept (absolute values and imatrix invariance), but have no ratio against F16.

Each cell is **pp / tg / E** against F16. Without a mark: > 1.1 (faster / less energy) or < 0.9 (slower / more energy); ≈: no appreciable difference. † = row measured with N = 30 (rest of the device, N = 10). Size `F16 / Q` in the section «On-disk size».

**TinyLlama**

| Device | Q4_K_M | Q4_0 | Q3_K_M | Q2_K |
|---|---|---|---|---|
| Jetson CPU 15 W | 1.24 / 1.70 / 2.04 | 1.32 / 1.77 / 1.99 | 1.20 / 1.80 / 2.07 | ≈1.09 / 2.00 / 2.22 |
| Jetson CPU 30 W | 1.37 / 2.67 / 2.05 | 1.46 / 2.75 / 2.12 | 1.31 / 2.65 / 2.00 | 1.23 / 2.78 / 2.15 |
| Jetson CPU 50 W | 1.22 / 1.88 / 1.94 | 1.28 / 1.91 / 2.00 | 1.23 / 1.80 / 1.90 | 1.22 / 1.89 / 2.02 |
| Jetson GPU 15 W | 0.72 / 1.55 / 1.83 | 0.80 / 2.06 / 2.17 | 0.66 / 1.13 / 1.54 | 0.54 / 1.28 / 1.69 |
| Jetson GPU 30 W | 0.82 / 1.59 / 1.78 | ≈0.91 / 1.97 / 2.20 | 0.76 / 1.18 / 1.52 | 0.62 / 1.33 / 1.68 |
| Jetson GPU 50 W | ≈0.95 / 1.81 / 1.77 | ≈1.04 / 2.19 / 2.22 | 0.85 / 1.40 / 1.51 | 0.71 / 1.51 / 1.76 |
| n150 | 2.02 / 4.19 / 2.41 | 2.39 / 4.24 / 2.72 | 1.25 / 3.57 / 1.77 | 1.33 / 4.18 / 1.97 |
| rk1-02 | 1.15 / 2.30 / 2.11 | 2.32 / 2.79 / 2.96 | 0.65 / 2.31 / 1.45 | 0.60 / 2.24 / 1.34 |
| CM4 | 1.68 / 2.87 / 1.98 | 1.66 / 2.97 / 1.98 | 1.41 / 2.73 / 1.75 | 1.46 / 3.03 / 1.96 † |
| VisionFive 2 | 2.87 / 3.23 / 2.55 | 6.10 / 6.27 / 4.84 | 2.11 / 2.44 / 1.71 | 3.15 / 3.53 / 2.64 † |

**Qwen2.5**

| Device | Q4_K_M | Q4_0 | Q3_K_M | Q2_K |
|---|---|---|---|---|
| Jetson CPU 15 W | 1.25 / 2.55 / 2.41 | 1.35 / 2.63 / 2.32 | 1.22 / 2.72 / 2.33 | 1.11 / 2.98 / 2.52 |
| Jetson CPU 30 W | 1.31 / 2.58 / 2.02 | 1.38 / 2.66 / 2.08 | 1.29 / 2.59 / 1.96 | 1.18 / 2.75 / 2.13 |
| Jetson CPU 50 W | 1.18 / 2.38 / 1.96 | 1.23 / 2.40 / 2.04 | ≈1.05 / 2.34 / 1.95 | 1.15 / 2.40 / 2.02 |
| Jetson GPU 15 W | 0.81 / 2.12 / 2.06 | 0.898 / 2.81 / 2.67 | 0.74 / ≈1.05 / 1.56 | 0.62 / 1.63 / 1.85 |
| Jetson GPU 30 W | 0.78 / 2.18 / 1.96 | 0.86 / 2.81 / 2.47 | 0.72 / ≈1.09 / 1.40 | 0.60 / 1.70 / 1.78 |
| Jetson GPU 50 W | 0.85 / 2.13 / 1.99 | ≈1.00 / 2.67 / 2.45 | 0.77 / ≈1.05 / 1.62 | 0.65 / 1.63 / 1.94 |
| n150 | 1.94 / 4.16 / 2.52 | 2.35 / 4.27 / 2.84 | 1.21 / 3.56 / 1.84 | 1.31 / 4.17 / 2.08 |
| rk1-02 | 1.19 / 2.31 / 2.13 | 2.43 / 2.85 / 2.95 | 0.65 / 2.28 / 1.45 | 0.60 / 1.54 / 1.30 |
| CM4 | 1.49 / 2.84 / 1.86 | 1.46 / 2.87 / 1.84 | 1.25 / 2.67 / 1.64 | 1.34 / 2.97 / 1.77 |
| VisionFive 2 | 3.01 / 3.36 / 2.47 | 6.36 / 5.96 / 4.89 | 2.20 / 2.60 / 1.85 | 3.36 / 3.69 / 2.78 |

**Ministral-3-3B** (n150 without ratios: F16 excluded; not measured on the CM4 or the VisionFive 2)

| Device | Q4_K_M | Q4_0 | Q3_K_M | Q2_K |
|---|---|---|---|---|
| Jetson CPU 15 W | 1.29 / 1.75 / 1.78 | 1.41 / 1.82 / 1.85 | 1.27 / 1.83 / 2.00 | 1.13 / 1.99 / 2.20 |
| Jetson CPU 30 W | 1.39 / 2.81 / 2.10 | 1.49 / 2.89 / 2.15 | 1.38 / 2.78 / 2.06 | 1.28 / 2.94 / 2.24 |
| Jetson CPU 50 W | 1.38 / 2.01 / 2.00 | 1.34 / 2.02 / 2.03 | 1.31 / 1.88 / 1.97 | 1.35 / 1.94 / 2.04 |
| Jetson GPU 15 W | 0.68 / 1.57 / 1.79 | 0.77 / 2.01 / 2.22 | 0.63 / 1.14 / 1.49 | 0.51 / 1.31 / 1.66 |
| Jetson GPU 30 W | 0.76 / 1.47 / 1.84 | 0.84 / 1.88 / 2.30 | 0.69 / 1.13 / 1.61 | 0.57 / 1.29 / 1.80 |
| Jetson GPU 50 W | 0.82 / 1.75 / 1.79 | ≈0.91 / 2.15 / 2.22 | 0.76 / 1.38 / 1.56 | 0.63 / 1.56 / 1.76 |
| rk1-02 | 1.29 / 2.42 / 2.06 | 2.52 / 3.05 / 2.92 | 0.67 / 2.51 / 1.44 | 0.60 / 2.75 / 1.37 |

**Count** (108 ratios of each type: 72 from the Jetson and 36 from the rest; in parentheses, the range):

| Group | n | Prefill: faster / ≈ / slower | Decode: faster / ≈ / slower | Energy: less / ≈ / more |
|---|---|---|---|---|
| Jetson CPU | 36 | 34 / 2 / 0 (1.05-1.49) | 36 / 0 / 0 (1.70-2.98) | 36 / 0 / 0 (1.78-2.52) |
| Jetson GPU | 36 | 0 / 5 / 31 (0.51-1.04) | 33 / 3 / 0 (1.05-2.81) | 36 / 0 / 0 (1.40-2.67) |
| n150 | 8 | 8 / 0 / 0 (1.21-2.39) | 8 / 0 / 0 (3.56-4.27) | 8 / 0 / 0 (1.77-2.84) |
| rk1-02 | 12 | 6 / 0 / 6 (0.60-2.52) | 12 / 0 / 0 (1.54-3.05) | 12 / 0 / 0 (1.30-2.96) |
| CM4 | 8 | 8 / 0 / 0 (1.25-1.68) | 8 / 0 / 0 (2.67-3.03) | 8 / 0 / 0 (1.64-1.98) |
| VisionFive 2 | 8 | 8 / 0 / 0 (2.11-6.36) | 8 / 0 / 0 (2.44-6.27) | 8 / 0 / 0 (1.71-4.89) |
| **Total** | 108 | 64 / 7 / 37 | 105 / 3 / 0 | **108 / 0 / 0** (1.30-4.89) |

**Energy ratio `E_F16 / E_Q` per format** (range across models and modes):

| Group | Q4_K_M | Q4_0 | Q3_K_M | Q2_K |
|---|---|---|---|---|
| Jetson CPU | 1.78-2.41 | 1.85-2.32 | 1.90-2.33 | 2.02-2.52 |
| Jetson GPU | 1.77-2.06 | 2.17-2.67 | 1.40-1.62 | 1.66-1.94 |
| n150 | 2.41-2.52 | 2.72-2.84 | 1.77-1.84 | 1.97-2.08 |
| rk1-02 | 2.06-2.13 | 2.92-2.96 | 1.44-1.45 | 1.30-1.37 |
| CM4 | 1.86-1.98 | 1.84-1.98 | 1.64-1.75 | 1.77-1.96 |
| VisionFive 2 | 2.47-2.55 | 4.84-4.89 | 1.71-1.85 | 2.64-2.78 |

**Imatrix invariance.** The four Q4_K_M variants with imatrix (corpus `wikitext2` and `domain_corpus`; size 50 and 1000) against Q4_K_M without imatrix, on the 8 devices or modes where they were measured (6 of the Jetson, n150 and rk1-02; 3 models × 4 variants = 12 ratios per table row, 96 in total). **All 96 ratios of each type fall within [0.9, 1.1]**, as the methodology states.

| Device | Prefill | Decode | Energy |
|---|---|---|---|
| Jetson CPU 15 W | 0.995-1.009 | 0.997-1.014 | 0.983-1.037 |
| Jetson CPU 30 W | 0.972-1.011 | 0.995-1.013 | 0.999-1.013 |
| Jetson CPU 50 W | 0.926-1.013 | 0.986-1.019 | 0.994-1.018 |
| Jetson GPU 15 W | 0.994-1.000 | 0.999-1.001 | 0.999-1.014 |
| Jetson GPU 30 W | 0.994-1.006 | 0.995-1.001 | 0.995-1.011 |
| Jetson GPU 50 W | 0.975-1.010 | 0.998-1.005 | 0.982-1.021 |
| n150 | 0.995-1.007 | 0.994-1.007 | 0.989-1.026 |
| rk1-02 | 0.996-1.005 | 0.941-1.006 | 0.971-1.010 |

The largest deviations: prefill 0.926 (Ministral, size 50, Jetson CPU 50 W), decode 0.941 (Ministral, size 1000, rk1-02) and energy 1.037 (Qwen2.5, `wikitext2`, Jetson CPU 15 W). The latter exceeds the ±3 % repeatability declared above (measured on a single combination) and is the only one of the 96 above 3 %.

**Fixed cost of the window per device and N.** Fraction of the window (`energy_raw_j / avg_power_w_raw`) not covered by the timed repetitions, `1 − N·(128/pp + 32/tg) / window`:

| Device | N | Rows | Minimum window (s) | Fixed fraction |
|---|---|---|---|---|
| Jetson CPU 15 W | 30 | 34 | 87.8 | 3.9-7.6 % |
| Jetson CPU 30 W | 30 | 34 | 33.9 | 4.5-16.2 % |
| Jetson CPU 50 W | 30 | 34 | 25.6 | 4.3-22.5 % |
| Jetson GPU 15 W | 30 | 34 | 38.0 | 1.7-4.2 % |
| Jetson GPU 30 W | 30 | 34 | 21.4 | 2.2-15.9 % |
| Jetson GPU 50 W | 30 | 34 | 9.8 | 4.8-10.6 % |
| n150 | 30 | 34 | 80.3 | 1.7-3.2 % |
| rk1-02 | 30 | 34 | 60.2 | 2.4-6.5 % |
| CM4 | 10 | 9 | 312.6 | 5.0-5.4 % |
| CM4 | 30 † | 1 | 993.3 | 1.8 % |
| VisionFive 2 | 10 | 9 | 1794.4 | 6.9-7.3 % |
| VisionFive 2 | 30 † | 1 | 9740.7 | 2.6 % |

† TinyLlama Q2_K. On the CM4 and the VisionFive 2 its fixed fraction (1.8 % and 2.6 %) is between 3.2 and 3.6 points (CM4) and between 4.3 and 4.7 points (VisionFive 2) lower than that of the other rows of the same device, consistent with a fixed part of similar duration spread over three times as many repetitions. In the ratios against F16 of that row, the F16 has N = 10. How much of that time difference passes into energy depends on `k`, which was only measured on the Jetson (0.11-0.39 on CPU); with `k` = 0.39 it would be ≤ 1.8 points, within the band, but applying that `k` to these devices is an assumption (later analysis, not pre-registered). On the n150 and the rk1-02 the fixed fraction of the F16 (1.7-2.9 % and 2.4-2.7 %) is not larger than that of the quantized rows, unlike on the Jetson's CPU.

**Readings (only what the tables show).**

*Confirmed with the fixed reading:*
- **Less energy per request in the 108 combinations** (1.30-4.89×), on the five devices, the four formats and the three models. The Jetson reading extends to all of them.
- **Faster decode in 105 of 108**; the 3 with no difference are the already known ones (Qwen2.5 Q3_K_M on the Jetson's GPU). Outside the Jetson, 36 of 36 (1.54-6.27×).
- **Prefill:** outside the Jetson it is faster in 30 of 36; the 6 in which it is slower are Q3_K_M and Q2_K on the rk1-02, in the three models (0.60-0.67×). On the n150, the CM4 and the VisionFive 2 prefill is faster in all cases. Therefore, the slower prefill in quantized is not exclusive to the Jetson's GPU.
- **Q4_0 is the format with the largest energy saving on the n150, the rk1-02 and the VisionFive 2** (2.72-4.89×), as on the Jetson's GPU; on the CM4 it ties with Q4_K_M (1.84-1.98 versus 1.86-1.98) and on the Jetson's CPU the four formats are close (1.78-2.52). The difference between Q4_0 and Q4_K_M is larger on the rk1-02 (2.92-2.96 versus 2.06-2.13) and on the VisionFive 2 (4.84-4.89 versus 2.47-2.55).
- **Q3_K_M and Q2_K save less than Q4_K_M and Q4_0 on the n150 and the rk1-02**; on the rk1-02 Q2_K is the one that saves the least (1.30-1.37×). On the VisionFive 2 Q3_K_M is the one that saves the least (1.71-1.85×) and Q2_K falls between Q4_K_M and Q4_0 (2.64-2.78×). On the Jetson's CPU Q2_K is the one that saves the most in 8 of 9 cases (in Qwen2.5 at 50 W, Q4_0).
- **The relation with the size reduction (3.1-5.1×) is not the same on all devices.** On the Jetson, the rk1-02 and the CM4, the energy saving and the decode speedup stay below it, as in the Jetson section. On the n150, the decode of Q4_K_M and Q4_0 (4.16-4.27×) exceeds it (3.14-3.46×), and on the VisionFive 2 the decode of Q4_0 (5.96-6.27×) and of Qwen2.5 Q4_K_M (3.36×), the prefill of Q4_0 (6.10-6.36×) and the energy of Q4_0 (4.84-4.89× versus 3.31-3.46×) exceed it.
- **The imatrix does not change performance or energy:** 96 of 96 ratios within the band in each metric.

*Hypotheses, not verified:*
- The advantage of Q4_0 on the rk1-02 and the VisionFive 2, and the slow prefill of Q3_K_M and Q2_K on the rk1-02, point to differences in the llama.cpp kernels by format and architecture (Q4_0-specific paths versus the K-quants). It has not been verified which kernel runs on each device.
- That the decode exceeds the size reduction on the n150 and the VisionFive 2 could indicate that the F16 is not limited only by memory bandwidth on those processors (for example, because of the absence of native F16 arithmetic). It has not been verified.

**Limits.** One measurement per row and a single session per row; F16 and Q in different invocations, hence the band. The CM4 and the VisionFive 2 use N = 10 (except TinyLlama Q2_K) and do not include Ministral-3-3B or the variants with imatrix; the n150 has no valid ratios against F16 for Ministral-3-3B. The fixed-cost bias in energy was only bounded on the Jetson; no correction is applied. The energy conclusions hold for requests of 128+32 tokens.
