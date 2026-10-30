> **Note on hypothesis numbering.** This protocol numbers the hypotheses differently from the paper; the numbering below is kept unchanged because it is the record of the study. Correspondence for the language models: protocol H1 = paper H5 and H1, protocol H2 = paper H2, protocol H3 = paper H6, protocol H4 = paper H3.

# Experimental Protocol: PTQ Calibration Ablation on LLMs (llama.cpp / GGUF)

## 1. General theoretical objective

Same causal framework as in ViT, adapted to the available tools: llama.cpp does not let one freely choose the calibration algorithm (there are no interchangeable MinMax/percentile/entropy as in Brevitas), but it does allow controlling the **calibration corpus used to generate the `imatrix`** (importance matrix), which weights which weights are most sensitive to quantization according to their contribution to the activations observed during that corpus.

Adapted research question:

> How does the choice of the imatrix calibration corpus affect the final quality of the quantization, and does that sensitivity vary with the LLM's architecture family (TinyLlama, Qwen2.5, Ministral-3-3B)?

This is intentionally more modest than the ViT ablation — it is consistent with the decision already taken to rely on the mature llama.cpp ecosystem instead of rebuilding it, and you must justify it explicitly in the Methodology (see Section 8 of the ViT protocol, "Scope with respect to LLMs").

## 1.1 Final models

| # | Model | Role in the design |
|---|---|---|
| 1 | **TinyLlama 1.1B Chat v1.0** | Classic Llama2 architecture, widely used as a reference in the small-LLM-for-edge literature |
| 2 | **Qwen2.5-1.5B-Instruct** | Family with a different tokenizer (larger vocabulary); grouped-query attention (GQA) **is not exclusive to this model** — TinyLlama also uses it, with a different group/head configuration (verify the exact values of each before citing it in the paper, do not leave it as a distinctive trait) — widely used in 2026, mature GGUF |
| 3 | **Ministral-3-3B-Instruct** | European model (Mistral AI, Ministral 3 family, a line designed specifically for edge deployment), text backbone of 3.43B parameters (`model_n_params`) — standard dense transformer, GGUF with llama.cpp support. **Instruct** variant, keeping the conversational comparability with the other two models (same reason already established: the ablation's domain corpus, Section 2, represents the use case of a conversational assistant at the edge). The published checkpoint includes a vision encoder (~0.4B parameters) in addition to the text backbone — it is explicitly discarded in the GGUF conversion (only the text backbone is quantized and evaluated, with no multimodal capability), confirmed with no loss of text tensors and no carry-over of vision weights. The checkpoint `Ministral-3-3B-Instruct-2512-**BF16**` is used explicitly, not the default repository (published in static FP8) — starting from FP8 would have broken comparability with TinyLlama/Qwen2.5, which start from full precision before the project's own quantization. |

Naming: always use **"Ministral-3-3B"** (never plain "Ministral-3B") to avoid confusion with a different 2024 Mistral model with a similar name.

The three are dense decoder transformers, within the same class of "small for edge" models (~1-3.5B), keeping direct comparability across families.

*Note: in an earlier iteration of the protocol a different European model was evaluated as the third LLM; Ministral-3-3B was prioritized instead, with the results of that evaluation kept in `results/appendix/` in case it is taken up again in the future.*

## 1.2 Model identifiers and size

| Model | Hugging Face repository | Parameters (`model_n_params`, llama.cpp) | `tie_word_embeddings` | `output.weight` in the GGUF |
|---|---|---|---|---|
| TinyLlama 1.1B Chat v1.0 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1,100,048,384 | no | yes |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | 1,543,714,304 | yes | no |
| Ministral-3-3B-Instruct | `mistralai/Ministral-3-3B-Instruct-2512-BF16` | 3,429,006,336 | yes (`text_config`) | no |

**Counting criterion (uniform across the three):** `model_n_params` is the total number of parameters stored in the GGUF, with embeddings included and without Ministral's vision encoder (only the text backbone is converted). With shared embeddings (Qwen2.5, Ministral-3-3B) the embedding matrix is stored and counted once; in TinyLlama, with unshared embeddings, the input and output ones are counted separately. That is why the Qwen2.5 figure is larger than the one usually quoted without embeddings: use this criterion in the paper's table and state it in a note.

**Bit-width nomenclature.** The llama.cpp `Qn_K` names designate per-tensor mixes of types, not a uniform bit-width. `Q3_K` is an alias of `Q3_K_M`: the string passed to `llama-quantize` is `Q3_K_M` (GGUF type `MOSTLY_Q3_K_M`) and it must be written that way in the paper. Tensor mix measured in Q3_K_M: Qwen2.5, 112 Q3_K tensors, 81 Q4_K, 3 Q5_K, 1 Q6_K and 141 F32; Ministral-3-3B, 104 Q3_K, 75 Q4_K, 3 Q5_K, 1 Q6_K and 53 F32.

## 2. Independent variables (factors)

| Factor | Proposed levels | Analogy with the ViT protocol | What it measures (see Section 6) |
|---|---|---|---|
| **imatrix calibration corpus** | (a) No imatrix (current baseline of the paper), (b) WikiText-2 (generic), (c) Dolly-15k (conversational domain corpus) | Analogous to the "calibration algorithm": here the variable is not the mathematical method but the source of information used to weight importance | Accuracy only |
| **Corpus size (no. of samples)** | 50, 200 (default), 1000 samples — non-empty lines in WikiText-2, records in Dolly-15k | Analogous to "calibration set size" in ViT — the number of real tokens/chunks per model is in Section 3.1 | Accuracy only |
| **Block/grouping scheme** | Legacy (`Q4_0`, no K) vs. K-quants (`Q4_K_M`) | Analogous to "per-tensor vs. per-channel granularity": K-quants apply per-block scaling, finer than the legacy scheme | Accuracy + performance/energy |
| **Bit-width** | `Q4_K_M` as the anchor point in Phase 1-2; `Q3_K_M`/`Q2_K` as aggressive points in Phase 3 (with and without the combined configuration) | No exhaustive sweep of the whole spectrum supported by llama.cpp (Q2-Q8) is measured — only the points relevant to contrast the ablation's hypotheses | Accuracy + performance/energy |

**Scope note:** there is no direct equivalent of "symmetry" in this context — the GGUF format handles this internally without exposing it as an adjustable parameter, so it is left out of the ablation and documented as a limitation of the tool, not as a methodological omission.

**Justification of Dolly-15k as the domain corpus (versus Alpaca or other usual alternatives):** Dolly-15k is made of human-written instructions, not generated by another LLM (unlike Alpaca, generated with GPT-3) — calibrating with synthetic text from one model to then evaluate the quantization sensitivity of another introduces an uncomfortable circular bias that is hard to defend methodologically ("is quantization sensitivity being measured, or artifacts of another LLM's generation?"). Dolly-15k avoids that ambiguity, is directly accessible without gating, and covers diverse categories (Q&A, summarization, classification, free-form generation), better representing the real conversational deployment domain of the 3 evaluated models.

## 3. Dependent variables (metrics)

Standard metrics for an edge deployment study, justified in themselves:
- Perplexity (WikiText-2 **test split**)
- Throughput (tokens per second, separated into prefill/decode — see Section 6.3 of the architecture)
- Energy — **primary metric: net joules per request** (`energy_per_inference = energy_net_j / N`, one request of 128+32 tokens); tokens per second per watt (`perf_per_watt`) as a secondary metric, marked as specific to this prefill/decode ratio (Section 6.4.1 of the architecture, and see the same criterion for ViT there)
- Model size

**New metric for the causal analysis:**
- Statistics of the weight and activation distribution per transformer layer/block, for the 3 families (TinyLlama, Qwen2.5, Ministral-3-3B), specifically including the presence of **"outlier features"** in activations — a well-documented phenomenon in the LLM quantization literature (specific channels with anomalously high magnitudes that dominate the quantization error). This gives a solid, citable theoretical anchor for Phase 4.

**Scope of Phase 4 in LLM — descriptive characterization, not causal correlation (same reason as in ViT, Section 3.0.1 of the ViT protocol):** with only 3 models (one per architectural family), there is no statistical power to correlate distribution with sensitivity, nor a way to isolate the effect of the distribution from any other architectural difference between families. Unlike ViT (where Level 1 uses dozens of layers *within* a single model to achieve a real correlation), in LLM there are not 3 models per architectural family to replicate that design — the comparison of outlier magnitude between Ministral-3-3B and the other two is presented as a **descriptive characterization** (documenting what structurally distinguishes the model that does not benefit from the imatrix), not as causal evidence. The per-tensor analysis within each model is limited to describing the metrics of Section 3.0.1; the correlation with a per-tensor sensitivity remains a future extension, and the intervention (the equivalent of ViT's Level 2) is set out separately in Section 3.0.2.

## 3.0.1 Design of Phase 4 in LLM (descriptive), fixed before seeing data

**Question.** Does Ministral-3-3B, the only model without improvement from the imatrix, have less pronounced activation outliers than TinyLlama and Qwen2.5? It is a descriptive characterization: with 3 models there is no inference across models, and the effect of the imatrix is not attributed to any metric. The starting hypothesis (Section 4, Phase 4) predicts that models with more pronounced outliers benefit more from the imatrix.

**Part A — activations, from the existing imatrix files (local).** The imatrix files (GGUF v3, despite the `.dat` extension) store, per weight tensor, the **sum** of the squared input activation per channel (`in_sum2`) and the number of accumulated rows (`counts`, one scalar per tensor: 23,552 in TinyLlama, i.e. 46 chunks × 512 tokens). The WikiText-2, 200-line ones are used, repetitions 1-3 of the 3 models (9 files). With `m_c = sqrt(in_sum2_c / counts)` per channel, per tensor: **`ρ = max_c(m_c) / (median_c(m_c) + 1e-8)`** (primary), and, as secondary ones, to control the effect of different hidden dimensions on the maximum, `p99/p50` and the fraction of channels with `m_c` > 10 times the median. Each group that shares an input (q, k and v; gate and up) is counted only once, after checking that their inputs coincide, so there are 4 types per block: qkv, attention output, gate/up and down. Before computing anything, the file format and the tensor names and dimensions are checked (done with TinyLlama, repetition 1: 154 entries = 22 blocks × 7 tensors, 2048 channels in q, k, v, output, gate and up and 5632 in down, and identical vectors in the q/k/v and gate/up groups in the 22 blocks; pending for the other 8 files) and that the llama.cpp version is `b95502ba9`. The ratios are based on per-channel RMS (not on the absolute maximum) and are not comparable with ViT's `R`.

**Part B — weights, from the F16 GGUFs (cluster).** These are the files that go into the quantization. Per weight tensor (q, k, v, attention output, gate, up and down; per block): **`ω = max_c(absmax_c) / (median_c(absmax_c) + 1e-8)`**, with `absmax_c` the absolute maximum of the column of input channel `c` (the same axis as `ρ`), and, as a secondary one, the excess kurtosis of the tensor. There is no directional prediction for the weights: the imatrix weights the error by the activation energy and does not act on the distribution of the weights, so this part is not expected to explain its effect.

**Summary, for both parts.** Per model and type: median across blocks, maximum and the block where it occurs, and values by relative-depth thirds (the models have different numbers of blocks). For `ρ`, in addition, the dispersion across the 3 repetitions. 95 % confidence intervals of each median by bootstrap over the blocks (10,000 resamples, fixed seed), which describe each model's statistic and are not a test between models. The imatrix has no entry for the output head (checked in TinyLlama), so it is only reported in part B.

**Criterion on `ρ`, fixed before seeing data.** The result is **consistent** with the imatrix not helping Ministral if its median `ρ` is lower than that of TinyLlama and that of Qwen2.5 in at least 3 of the 4 types; if not, that difference is not explained by this metric. It is also reported whether the intervals of Ministral and of the other two models overlap in each type, as a description. For `ω` there is no criterion: it is described, and an absence of differences in the weights is worded as «the weight distribution does not distinguish Ministral in these metrics», without claiming that the weights do not matter.

**Details fixed before running.**
- **imatrix used:** those of the K-quant row (`llm_<model>_phase2_calibration_corpus_wikitext2_q4km`, 200 lines, repetitions 1-3), not those of the legacy row. Each repetition uses a different sample of lines (seeds 42-44), so the dispersion across repetitions is corpus sampling variability.
- **Combination of repetitions:** the primary value of `ρ` per type (the one used by the criterion) is the median across blocks of the per-tensor mean `ρ` of the 3 repetitions; the dispersion is the range of the 3 per-repetition medians.
- **Groups with the same input** (q, k and v; gate and up): their vectors are required to coincide (relative tolerance of 1e-6); if not, the script stops.
- **Depth thirds:** block `i` of `n` goes to third `floor(3·i/n)`.
- **Bootstrap:** 2.5-97.5 percentile interval of the median, 10,000 resamples of blocks with replacement, seed 0.
- **Output head (only in part B):** `output.weight` or, if the weights are tied, `token_embd.weight`; it is reported separately and is outside the criterion.
- **Expected input dimensions:** `embedding_length` for q, k, v, gate and up; `feed_forward_length` for down; and for the attention output, `head_count × key_length` (it can differ from `embedding_length`); if the `key_length` metadata does not exist (Qwen2.5), `key_length = embedding_length / head_count`. If they do not coincide, the script stops. **Values read from the F16 GGUFs** (`artifacts/llm_base/`), with the shapes of the block 0 tensors in GGUF format `[input, output]`:

| Model | Blocks | `embedding_length` | `feed_forward_length` | `attn_output` input | Head |
|---|---|---|---|---|---|
| TinyLlama | 22 | 2048 | 5632 | 2048 (32 × 64) | `output.weight` (untied weights) |
| Qwen2.5-1.5B | 28 | 1536 | 8960 | 1536 (12 × 128, no `key_length` in the metadata) | `token_embd.weight` (tied) |
| Ministral-3-3B | 26 | 3072 | 9216 | **4096** (32 × 128) | `token_embd.weight` (tied) |

  In Ministral-3-3B the `attn_output` input (4096) differs from `embedding_length` (3072), so the check against `embedding_length` would have stopped for no reason. The block weight tensors are F16 in the three models. Outside the blocks there are only `token_embd.weight` (F16), `output_norm.weight` (F32) and, in TinyLlama, `output.weight`: Ministral's F16 GGUF contains only the text part, with no vision tensors.
- **Weight checks:** all weight matrices are F16 and contain no infinite values or NaN.

**Robustness controls (added after seeing the results of parts A and B, and before computing them).** Ministral has the largest input dimension in the four types and the maximum grows with the number of channels, which inflates its `ρ` and goes against a criterion that requires it to be lower. The controls are presented alongside the primary verdict, which does not change.
1. **Complete secondary metrics:** `p99/p50` and the fraction of channels with `m_c` > 10 times the median, in the four types and the three models, with the same shape criterion (Ministral's median lower than that of the other two in at least 3 of the 4 types), applied to each metric separately.
2. **Dimension-adjusted `ρ`:** per tensor, the channels are subsampled without replacement to a common dimension `D_ref`, the smallest of the three models in that type (1536 in qkv, attention output and gate/up; 5632 in down), 1000 times with seed 0, and the `ρ` of the subsamples is averaged. The rest of the aggregation is that of the primary result (mean of the 3 repetitions, median across blocks, bootstrap over the blocks). The same 3-of-4-types criterion is applied. **Fixed wording:** if it is met in 3 or more types, «el veredicto primario depende del tamaño de la dimensión»; if not, «el resultado negativo es robusto a la dimensión» [the two verdict strings are data values, kept in Spanish; see `docs/GLOSSARY.md`].
3. **Gaussian reference for `ω`:** for each tensor shape (rows × columns of the GGUF) `ω` is computed for 20 N(0,1) matrices (seed 0, float32) and `ω / ω_gauss` is reported next to `ω`. It is a control of the dependence on shape, without a verdict; it only captures light tails.

**Out of scope, declared.** Tests between models (n = 3) and correlation with a per-tensor sensitivity. The per-tensor imatrix intervention is set out separately (Section 3.0.2).

## 3.0.2 Per-tensor imatrix intervention (exploratory; added after seeing parts A and B and the robustness controls, and before seeing any data from this test)

**Question.** Within a model, does `ρ` identify which tensors concentrate the benefit of the imatrix? It is the equivalent of ViT's Level 2. It says where the benefit is concentrated, not why, and it is not a statistical test.

**Feasibility checked** (`future_work.md`): `llama-quantize` applies the imatrix only to the tensors indicated with `--include-weights` (repeatable, with full names and no collisions), each tensor is quantized independently, and passing all the tensors reproduces the full imatrix exactly.

**Models and configuration.** TinyLlama and Qwen2.5, which are the models where the imatrix improves the perplexity (Ministral is left out: its benefit is not defined). **Q3_K_M**, because the effect of the imatrix is larger at fewer bits and it gives more resolution. WikiText-2 imatrix, 200 lines, repetitions 1-3. **Clarification, after reading the pipeline and before running:** the Phase 3 combined row generates its own imatrix (`llm_<model>_phase3_combined_q3_k__rep{k}.dat`), a file different from those of Section 3.0.1 (`phase2_calibration_corpus_wikitext2`) although with the same corpus, size and seed. First the tensors of both sets are compared, locally and read-only. **The Phase 3 combined ones are used**, which are the ones that produced the recorded values; the sets H, L and R are computed with the `ρ` of Section 3.0.1, as fixed, **Checked locally, before running:** the imatrix files of both phases are bit-for-bit identical (`in_sum2` and `counts`) in both models and the three repetitions; only the corpus path in the metadata changes. Therefore the `ρ` rankings coincide and no additional comparison is needed.

**Tensor sets.** The unit is the groups of part A: qkv (q, k and v), attention output, gate/up (gate and up) and down, per block. They are sorted by the mean primary `ρ` of the 3 repetitions **within each type** (not globally, so as not to confound `ρ` with the tensor type); ties, by execution order. The lists are generated once, locally, and stored in a file (`sets.json`) with the full names (`blk.N.attn_q.weight`), which is the one the cluster script uses.
- **H:** the half of the blocks with the highest `ρ` in each type. **L:** the other half.
- **R1, R2 and R3:** half of the blocks at random in each type. For each seed s (1, 2 and 3), `rng = random.Random(s)` and, for the types in the order qkv, attention output, gate/up and down, the list of blocks is shuffled with `rng.shuffle` and the first n/2 are taken.
- **E (depth control; added after seeing the composition of H in `sets.json` and before seeing any perplexity data):** H is concentrated in extreme blocks (in Qwen2.5, almost all in the first and the last ones), so a benefit of H could be due to depth and not to `ρ`. E is the n/2 blocks farthest from the center `(n − 1)/2`, the same in all types (ties by lowest block index; TinyLlama, blocks 0-5 and 17-21; Qwen2.5, 0-6 and 21-27), with the same composition per type as the other sets. The overlap of H with E is reported.
- All sets have the same composition per type: 77 tensors in TinyLlama (22 blocks) and 98 in Qwen2.5 (28 blocks).
- Same ranking and same permutations in the 3 repetitions, so that only the calibration sample changes.

**Measurement.** Perplexity with `llama-perplexity` on WikiText-2 test, `n_ctx` = 512 and all chunks (658 in TinyLlama and 583 in Qwen2.5), with the same flags and binaries as the recorded values. `B(S) = (PPL_none − PPL_S) / (PPL_none − PPL_full,k)`, i.e. the fraction of the imatrix benefit that the set S recovers in repetition k. "None" is the quantization without imatrix (deterministic, the same in the 3 repetitions); "full,k" is the quantization with the whole imatrix of repetition k. **`B(S)` uses the perplexities measured in this test** (without imatrix and full,k), all on the same device as the rest of the evaluations of the test; the recorded values only serve as a control.

**Controls, before interpreting anything.**
1. The quantization without imatrix must reproduce the recorded perplexity of the Q3_K_M baseline (TinyLlama 9.2084; Qwen2.5 12.4471), and that of "full,k" the one recorded for repetition k (TinyLlama 8.9554 / 8.9613 / 8.9585; Qwen2.5 11.4282 / 11.4393 / 11.4043; from the Phase 3 perplexity record), both with a tolerance of 0.0005. If not, it stops and reports.
2. Per model, the quantization with all the imatrix tensors passed explicitly with `--include-weights` must give tensors identical to those of the full imatrix (comparison of each tensor's data, as in the feasibility test; the file's sha256 is stored as a record, but it includes metadata). If not, it stops.
3. The GGUF models are evaluated with the same binary as the recorded values (the CPU build of `b95502ba9`, without recompiling) and on the same type of node, with `llama-perplexity` on CPU.
4. **Evaluator determinism:** the perplexity of the same GGUF (the "full,1" of each model), evaluated twice on the same node, must coincide (difference ≤ 0.0005); if not, that noise is reported and compared with the differences between sets before interpreting anything.
5. **Threads:** the pipeline does not set `-t` (llama.cpp uses its default threads). In this test `-t` (and the threads of `llama-quantize`) is set to an explicit number, the same in all evaluations and configurable when launching the script (default, 8), so that performance does not depend on the threads llama.cpp detects on the node. Control 1 checks that the result does not change with respect to what was recorded.

**Criterion, fixed now.** Per model: **«`ρ` identifies the benefit»** if, in the three repetitions, `B(H)` is greater than `B(L)` and greater than the maximum of `B(R1)`, `B(R2)` and `B(R3)`. **«Reverse direction»** if the same happens with `L` instead of `H`. In any other case, **«not distinguishable from chance»**. In addition, `B(H)`, `B(L)` and `B(R1..R3)` are reported per repetition, and `B(H) + B(L)`, which would be 1 if the benefit were additive. **Secondary criterion, on depth:** «`ρ` identifies the benefit beyond depth» if, in addition to the previous criterion being met, `B(H)` is greater than `B(E)` in the three repetitions; if the previous criterion is met but not this one, the pattern is compatible with depth and is not attributed to `ρ`.

**Cost.** 22 evaluations per model (1 without imatrix, 3 full, 6 with H and L, 9 random and 3 with E), 44 in total, plus the 2 checks of control 2 (quantizations without perplexity) and those of control 4.

**Result:** TinyLlama, «`ρ` identifies the benefit» and beyond depth; Qwen2.5, «not distinguishable from chance» (`findings_llm.md`, section «Per-tensor imatrix intervention (Q3_K_M)»).

**Extension: more random sets for a rank-based contrast (added after seeing the results, and before seeing any data from the extension).** With only 3 random sets, if `ρ` were not informative, H would beat the 3 by chance with probability 1/4, and the three repetitions use the same sets. For both models, 17 random sets are added (R4 to R20, seeds 4 to 20), generated with the same procedure as R1-R3 (same types in the same order, n/2 blocks per type, the same composition as H) and evaluated only with the imatrix of repetition 1 (17 evaluations per model; `B(S)` with the "none" and "full,1" perplexities already measured). With the 20 random sets (R1-R20): `p_H = (1 + no. of random sets with B ≥ B(H)) / 21` and `p_L = (1 + no. of random sets with B ≤ B(L)) / 21`. **Fixed wording:** «H es extremo» (H is extreme) if `B(H)` exceeds the 20 random ones (`p_H = 0.048`), and «L es extremo» (L is extreme) if `B(L)` falls below the 20; if not, the rank is reported. In addition, the distribution of `B(R)` (minimum, quartiles and maximum) and the Spearman between `B(R)` and the overlap of each random set with H are reported (descriptive, to contrast the concentration of the benefit). It is an assumption-free rank contrast, with H and the random sets interchangeable if `ρ` were not informative; the two contrasts (H high and L low) are reported without adjustment. The main criterion and its result do not change. **Result of the extension:** TinyLlama, «H es extremo» (`p_H` = 0.048); Qwen2.5, H fifth of 21 (`findings_llm.md`).

**What it cannot say.** There are only two models, three repetitions and one configuration (Q3_K_M); the comparison with the maximum of 3 seeds is conservative and is not a test. It does not prove the mechanism. The perplexity is on WikiText-2, the same domain as the imatrix corpus. `ρ` is an activation metric and the weights do not enter. A result of «not distinguishable from chance» or «reverse direction» is also publishable: within a model, `ρ` would not predict where the imatrix helps.

## 3.0.3 Ministral-3-3B: imatrix ceiling effect? (added after seeing parts A and B and the intervention, and before seeing any data from this test)

**Question.** The imatrix does not improve Ministral-3-3B at Q4_K_M (it closes −14 % of the gap against F16, versus 48 % for TinyLlama and 42 % for Qwen2.5). Ministral is also the model with the smallest gap at 4 bits (1.7 %, versus 3.1 % and 6.1 %) and with the lightest-tailed weights (part B). Hypothesis: a **ceiling effect**, in which at 4 bits there is hardly any error left for the imatrix to reduce. If so, at fewer bits, with a much larger gap, the imatrix should help it. In Phase 3 there was no winning imatrix for Ministral, so it was never quantized with imatrix at Q3_K_M or at Q2_K.

**Design.** Ministral-3-3B at **Q2_K (primary)** and **Q3_K_M (secondary)**, with the WikiText-2 imatrix, 200 lines, repetitions 1-3 (those of Section 3.0.1), and perplexity with the same binary, flags and 8 threads as the intervention (Section 3.0.2). Per format and repetition: `G = (PPL_none − PPL_imatrix,k) / (PPL_none − PPL_F16)`, the fraction of the gap against F16 that the imatrix closes. `PPL_F16` = 8.8760 (recorded, deterministic).

**Controls.** The quantization without imatrix of each format, with 8 threads, must reproduce the recorded perplexity of `phase3_baseline_q2_k` and `phase3_baseline_q3_k` with a tolerance of 0.0005; if not, it stops. There are 6 evaluations with imatrix and 2 controls.

**Criterion, fixed now.** **«Compatible with a ceiling effect»** if at Q2_K the imatrix closes at least 20 % of the gap (`G ≥ 0.20`) in the three repetitions. If not, **«Ministral's insensitivity to the imatrix persists at 2 bits»**. Q3_K_M is reported as secondary, with the same threshold, without its own verdict. As context, at Q3_K_M the imatrix closes 36 % of the gap in TinyLlama and 45 % in Qwen2.5 (recorded Phase 3 values).

**Result:** compatible with a ceiling effect (`G` at Q2_K of 0.311, 0.298 and 0.312; at Q3_K_M, 0.25-0.28), although Ministral still closes less gap than the other two models (`findings_llm.md`).

**What it cannot say.** One model and three repetitions. If the imatrix helps at 2 bits, the result is compatible with a ceiling effect, but does not prove it: it also fits with calibration mattering more at fewer bits in any model, as in ViT. If it does not help, the insensitivity is a property of the model that parts A and B do not explain by the activation outliers.

## 3.1 Data separation: calibration vs. evaluation

The generic WikiText-2 imatrix corpus is drawn from the **training split** of WikiText-2 (`wikitext-2-raw-v1`, `train` partition), never from the test split used for perplexity — it is the standard convention in the llama.cpp community to prevent the imatrix from "seeing" the same data with which the quality of the quantized model is later evaluated (data leakage), which would bias the results in favor of the quantization. The domain corpus (Dolly-15k, Section 2) comes from a completely different dataset, with no overlap with WikiText-2 in any split.

**Methodological limitation to declare (comparison between corpora, not imatrix-vs-nothing):** the perplexity is evaluated on the **WikiText-2** test split — so the WikiText-2 corpus has an inherent domain advantage over Dolly-15k in that specific comparison (calibrating and evaluating in the same domain favors the matching corpus, regardless of whether it is intrinsically "better" at calibrating). This does not invalidate the main finding of H1 (having an imatrix improves over not having one, clearly confirmed) but it does limit the interpretation of "WikiText-2 beats/ties Dolly-15k" as evidence that a generic corpus calibrates better than a domain one in general — declare this limitation explicitly when writing that specific comparison.

The performance benchmarks (`llama-bench`) do not use real data either — they measure throughput with synthetic prompts of fixed length (128 input tokens, 32 generation tokens, values representative of a typical short conversational interaction in edge deployment), without WikiText-2 needing to be present on the edge devices at any moment.

**imatrix generation and perplexity evaluation (fixed parameters).**
- `llama-imatrix -m <F16> -f <corpus> -o <.dat>`, with `n_ctx = 512`, `batch = 2048` and `n_seq = 4`. Corpus: non-empty lines of `wikitext-2-raw-v1` (`train` split), sampled with seed `42 + (rep − 1)`, so that each repetition uses a different sample.
- The "corpus size" factor is expressed in **samples** (50/200/1000), not in tokens: the unit is the non-empty line in WikiText-2 and the record in Dolly-15k. Since a Dolly-15k record is longer than a WikiText-2 line, the same number of samples gives about twice as many tokens in Dolly (see below). WikiText-2 table; the number of tokens and chunks depends on the tokenizer:

| Model | 50 lines | 200 lines (default) | 1000 lines |
|---|---|---|---|
| TinyLlama | 5,852 tok → 11 chunks | 23,632 tok → 46 chunks | 115,928 tok → 226 chunks |
| Qwen2.5 | 5,184 tok → 10 chunks | 20,879 tok → 40 chunks | 103,016 tok → 201 chunks |
| Ministral-3-3B | 5,237 tok → 10 chunks | 21,117 tok → 41 chunks | 104,054 tok → 203 chunks |

(Figures from one specific run; they may vary slightly between seeds.)
- `llama-perplexity -m <gguf> -f wikitext2_test.txt`, with `n_ctx = 512` and all the chunks of the test split: 658 (TinyLlama), 583 (Qwen2.5), 588 (Ministral-3-3B).
- **Construction of the Dolly-15k corpus:** each record is formed by concatenating `instruction`, `context` (only if not empty) and `response`, separated by `\n` (`domain_corpus.py`); each record formed this way is a sampling unit, and the records are joined with `\n` in the corpus file. Including the response and the context is what makes a record longer than a WikiText-2 line.
- **Resolved — Dolly-15k at the default level (200 records), rep1:**

| Model | Tokens → chunks |
|---|---|
| TinyLlama | 44,970 → 87 |
| Qwen2.5 | 39,832 → 77 |
| Ministral-3-3B | 40,179 → 78 |

A Dolly-15k record is longer than a WikiText-2 line: at the same nominal level ("200"), Dolly contributes **about twice as many tokens** as WikiText-2 (23,632/20,879/21,117). **It confounds the corpus comparison of H1:** the result that WikiText-2 matches or beats Dolly-15k (Section 5, H1) was obtained with Dolly receiving twice the calibration data, not under equal conditions — declare this explicitly when presenting that comparison in the paper. Given that H2 (corpus size) showed little sensitivity in the three models, the effect of this asymmetry is probably small, but it has not been isolated with a control at equal tokens.

## 4. Experimental design by phases

**Phase 1 — Replication baseline**
Configuration: no imatrix, standard K-quants, **Q4_K_M** as the anchor point (intermediate bit-width, the llama.cpp community's standard choice for balancing accuracy/size). The rest of the spectrum supported by llama.cpp (Q2–Q8) is not exhaustively measured in Phase 1 — the additional aggressive points (Q3_K_M, Q2_K) are covered specifically in Phase 3, with and without the combined configuration, not as a complete sweep of the spectrum.

**Phase 2 — One-factor-at-a-time ablation**
Fixing a representative intermediate bit-width (e.g. Q4_K_M, the llama.cpp community's standard choice for its balance between accuracy and size):
- **imatrix calibration corpus** (2 new configurations): WikiText-2 and domain corpus, each with the reference size (200 samples: lines in WikiText-2, records in Dolly-15k). The baseline of this comparison is the absence of imatrix (Phase 1).
- **Corpus size** (2 new configurations): 50 and 1000 lines, both with the **corpus fixed at WikiText-2** as a secondary reference point — a specific corpus must be fixed for "size" to make sense, since the global baseline (no imatrix) has no corpus whose size can be varied. WikiText-2 + 200 lines acts as the intermediate reference point for this factor.
- **Block scheme × imatrix presence** (2 new configurations, not 3): unlike the other factors, here two variables are deliberately crossed — legacy without imatrix and legacy with imatrix (the best found above) — because the specific question (H3) is whether the imatrix compensates for the lack of granularity of the legacy scheme, something that cannot be answered by varying one factor at a time. It is the only exception to the OFAT pattern of the rest of the protocol, and it is declared so explicitly. The other two combinations needed to contrast H3 (`K-quants without imatrix` and `K-quants with imatrix`) **are not generated again**: they already exist as the Phase 1 baseline and as the `wikitext2` variant of the "corpus" factor respectively — the H3 analysis (Phase 4) reuses those existing rows by filtering on their field values, instead of duplicating the run.

Total per model: 1 baseline + 2 (corpus) + 2 (size) + 2 (scheme×imatrix) = 7 configurations, × 3 models = 21 runs.

**Phase 3 — Optimal combined configuration vs. aggressive extreme**
Apply the best imatrix combination (corpus + size) found in Phase 2 at **two** aggressive bit-widths (Q3_K_M and Q2_K), comparing each against its equivalent without imatrix. This directly answers H1: can a good calibration corpus compensate part of the accuracy loss when dropping to 2-3 bits?

With two bit-width points, H1 (which predicts a growing difference as the bit-width drops) can also be contrasted directly: a single point (Q2) does not allow distinguishing a real trend from an isolated value. Additional cost: 3 models × 2 configurations (best imatrix vs. no imatrix) × 2 aggressive bit-widths (Q3_K_M, Q2_K) = 12 runs, not a complete repetition of Phase 2. *(In practice, Ministral-3-3B does not generate a "combined" row — no factor exceeded the improvement margin in its Phase 2, see `findings_llm.md` — so its real cost is 2 rows, not 4; the other models generate the 4 complete rows.)*

**Phase 4 — Descriptive characterization** (design in Section 3.0.1)
Describe the presence and magnitude of activation outlier features and the weight distribution per model, and compare them with the results of Phases 2-3 without inference across models (n = 3). Starting hypothesis: models with more pronounced activation outliers benefit more from a well-chosen imatrix, because the imatrix weights precisely the real importance of those channels.

## 5. Specific hypotheses

- **H1:** An imatrix generated with a domain-specific corpus offers better perplexity than a generic one (WikiText-2) or than no imatrix, and the difference widens at aggressive bit-widths (Q2-Q3). **Actual status (Phase 1-3): partially confirmed.** As formulated (domain > generic), it is refuted — in the models with data, WikiText-2 (generic) ties or beats Dolly-15k (domain), see `findings_llm.md` and the domain-bias limitation declared in Section 3.1. What is clearly confirmed is the "imatrix vs. none" part, with the difference widening at aggressive bit-widths — that is the claim that must remain in the body of the paper, not the specific comparison between corpora.
- **H2:** The size of the calibration corpus shows diminishing returns, with a threshold that may vary across model families. **Actual status:** no detectable effect in Ministral-3-3B; a small monotonic trend, consistent with the direction of H2, in TinyLlama and Qwen2.5, but without exceeding the selection margin in either — it neither confirms nor refutes with the current statistical power, and no saturation threshold is identified in any model (see `findings_llm.md`).
- **H3:** The imatrix partially compensates for the lack of granularity of the legacy (non-K) scheme, reducing the perplexity gap between Q4_0 and Q4_K_M when both use the same imatrix. **Actual status: confirmed in the 3 models** (TinyLlama, Qwen2.5, Ministral-3-3B) — with the nuance that `k_quant` still beats `legacy+imatrix` in absolute terms in the 3; H3 specifically answers whether the imatrix helps legacy, not whether legacy+imatrix beats k_quant (see the interpretation note in `findings_llm.md`).
- **H4:** The three families (TinyLlama, Qwen2.5, Ministral-3-3B) show different sensitivity to the calibration corpus, correlated with the magnitude of their outlier features — the same pattern of "architecture-dependent sensitivity" already observed in the ViT branch of this study (see `findings_vit.md`). **Actual status:** the pattern (imatrix with no improvement effect) is observed in 1 of the 3 main models (Ministral-3-3B); TinyLlama and Qwen2.5 do benefit clearly. It confirms that architecture-dependent sensitivity exists, without the 3 models showing it in the same direction — with 3 models, Phase 4 can only describe what distinguishes Ministral-3-3B (Section 3.0.1); it cannot isolate the cause. **Result of part A:** the criterion on `ρ` is not met (0 of 4 types); Ministral's difference is not explained by the magnitude of the activation outliers (`findings_llm.md`, section «Phase 4 (descriptive), part A: activation outlier ratio (ρ)»). Part B (weights) is descriptive and is covered in the same section.

*Results and concrete confirmations of these hypotheses, as they are obtained: see `findings_llm.md`, not this document.*

## 6. Considerations on performance vs. accuracy (parallel to the clarification made for ViT)

- **imatrix corpus and size**: they only affect the offline imatrix generation phase and the resulting accuracy — they do not change the final format of the quantized model nor its inference speed. Expected: accuracy-only, with no performance impact.
- **Legacy vs. K-quants**: here there is a potential performance difference, since K-quants introduce additional per-block scaling metadata. *(Note: an analogous phenomenon — scaling metadata cost affecting the size/performance of RNN models in TFLite — appeared in the rejected manuscript of the original study; verify whether it is also reported in the published E-Energy '25 paper before citing it as a precedent, since the rejected manuscript is not citable as evidence.)* Throughput/energy must be measured, not assumed neutral.

## 6.1 Statistical considerations

Unlike ViT, where the calibration set is explicitly sampled at random from an image pool in each repetition, here it is advisable to fix an explicit decision: the lines that make up the imatrix corpus (WikiText-2 or domain corpus, in their different sizes) **are randomly sampled from the corresponding source in each run**, as in ViT — so the same sampling variance exists and the same minimum of **3 repetitions per configuration** applies, reporting mean ± standard deviation (see Section 6 of the ViT protocol for the detail of the criterion). If the implementation decides to fix the whole corpus without random sampling (e.g. always using the same 200 lines in the same order), this section must be explicitly updated to reflect it, because in that case the repetitions would not provide new information and it could be reduced to a single run per configuration.

## 7. Mapping to the reviewers' criticisms

| Criticism | How it is addressed |
|---|---|
| 6 / 6.5 — Old architectures, a single LLM | 3 modern LLM families (TinyLlama, Qwen2.5, Ministral-3-3B) |
| 8 — PTQ as a black box | The imatrix generation process stops being a fixed detail and becomes a variable of study |
| 8.5 / 8.6 — No calibration/hyperparameter ablation | Phase 2 (corpus, size, block scheme) — a bounded but real version, consistent with the tools of the llama.cpp ecosystem |
| 7 / 7.5 — No causal analysis of sensitivity | Phase 4 (outlier features per architectural family) |

## 8. Difference in scope with respect to the ViT protocol (to be declared explicitly in the paper)

This ablation is intentionally **lighter** than the ViT one: it does not cover interchangeable calibration algorithms nor configurable granularity/symmetry, because the llama.cpp ecosystem does not expose those degrees of freedom. It is recommended to declare this in the Methodology as a justified scope decision, not to hide it — reinforcing the argument that the study combines "fine-grained control where the tool allows it (ViT/Brevitas)" with "state-of-the-art good practices where the ecosystem is already consolidated (LLM/llama.cpp)".
