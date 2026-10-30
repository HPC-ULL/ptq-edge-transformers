"""ViT Phase 4 (RQ2): per-layer distribution analysis (Level 1) and targeted
float-layer interventions (Level 2), calibration_protocol_vit.md Section
3.0.1, with the decisions below fixed BEFORE looking at any Phase 4 data.

Lives in analysis/, not quantization/vit_brevitas/investigations/: it only
imports from quantization/ and never modifies it (analysis/ is not in
edge_bundle_code_vit.zip). Outputs: results/investigations/vit/phase4/.

PRE-REGISTERED CRITERIA (fixed 2026-10-05; two deviations from the protocol
are marked DEVIATION and are recorded by the author in the docs)

Steps, in order, each one only after the author's go-ahead:
  (0) layer inventory + type classification (inventory.py);
  (1) Level 1 (shown to the author to detect errors, not to decide whether
      Level 2 runs); (2) K=0 control; (3) targeted curve; (4) random control.
  Level 2 runs whatever the Level 1 correlation is. It stops only if the
  ranking is degenerate (fewer than 12 rankable layers, or more than half of
  them with the same R) or if the K=0 control fails.

Layers: every nn.Linear / nn.Conv2d that quantize.quantize_module_tree
substitutes.

Type classification by functional membership (fixed after the inventory,
before Level 1; inventory.EXPECTED_TYPE_COUNTS):
  Linear of the classifier heads -> other (EfficientViT-B1's head.in_conv
  Conv2d stays conv, as in the approved table); every layer of an attention module -> attn,
  including all Conv2d of EfficientViT-B1's LiteMLA (qkv, aggreg, proj);
  any other Conv2d -> conv; the remaining Linear by name: "qkv"/"attn" ->
  attn, "mlp"/"fc" -> mlp. "proj" alone does not discriminate.
  EfficientViT-B1 has no mlp layers (its FFN, local_module, is MBConv ->
  conv), so per-type comparisons across models are limited to conv and attn.
  Sensitivity analysis, EfficientViT-B1 only: the per-type breakdown again
  with the 14 LiteMLA aggreg convs as conv.

Level 1
  - Images: 512 from the calibration bank (train split), fixed seed, disjoint
    from the calibration samples of seeds 42/43/44 at sizes 32/128/512.
    DEVIATION: the protocol said the evaluation subset.
  - Outlier ratio R = max_c(absmax_c) / (median_c(absmax_c) + 1e-8), absmax_c
    over the 512 images and all tokens/positions of the layer's FP32 INPUT.
    Channel axis: last for Linear, dim=1 for Conv2d.
  - Layers with fewer than 8 input channels (patch_embed, stems) are left out
    of the ranking and the correlation, and stay quantized throughout.
    Layers with median_c(absmax_c) < 1e-6 are listed, not ranked.
    Ties: execution order.
  - Secondary, exploratory: kurtosis and range of the input activations.
  - Local error: relative L2 per image, ||y_q - y|| / ||y||, averaged over the
    same 512 images; y_q = the layer quantized alone (baseline MinMax,
    per-tensor, symmetric, INT8, rep1 scales) applied to its FP32 input.
  - Spearman per model, global and per type, always with n and rho. Primary:
    the global R vs. local error per model, Holm across the 3 models. The
    rest is descriptive. A type with fewer than 10 ranked layers is not
    correlated, only listed.

Level 2
  - Float layer: weights and input left unquantized (module not substituted),
    in a copy of the build here; quantize.py untouched. Evaluation: same QCDQ
    export, same ONNX Runtime CPU top-1/top-5 loop, same 2000 images.
  - Configurations: MobileViT-S and EfficientViT-B1 INT8 baseline; DeiT-Tiny
    INT6 baseline (DEVIATION: the protocol said INT4). Rep1 only (seed 42):
    a single calibration sample, declared as such.
  - K=0 must reproduce exactly the recorded rep1 accuracy; otherwise stop.
  - Gap = FP32 - acc(K=0, rep1). Recovery(K) = (acc_K - acc_0) / gap.
  - Targeted: top-K of the R ranking (nested prefixes), K = 1, 2, 4, 8;
    stop at the first K with recovery >= 90 %, or at K=8.
  - Random control: seeds 1, 2, 3; one permutation of the rankable layers per
    seed, K = prefix (nested); up to the same final K as the targeted curve.
  - Top-1 prediction per image saved for every evaluation.
  - "Clearly better": with n = number of K evaluated (K=0 excluded, K_final
    included), the targeted accuracy is STRICTLY greater than the maximum of
    the 3 random seeds at K_final AND at >= ceil(n/2) of the n values of K.
    Otherwise: "not distinguishable from random".
  - If at K=8 neither the targeted nor any random curve recovers 10 % of the
    gap: "no recovery with K <= 8", without concluding that the degradation
    is diffuse.
  - Decisions D1-D8 (approved before any Level 2 data): FP32 reference =
    top-1 of vit_accuracy_fp32_reference.csv (stored_q95, PyTorch; was read from
    fp32_jpeg_vs_original.json, same values); random
    permutation = rankable layers in execution order, random.Random(seed).
    shuffle; ONNX deleted after evaluation, sha256 kept; recovery on top-1;
    K=0 per-image top-1 compared with per_image_predictions, reported only;
    one K=0 failure stops everything. At most 21 evaluations per model
    (1 + 4 + 12 + 4), 63 in total.
  - Smoke first: DeiT-Tiny, K=0 and targeted K=1 only (mixed float/quantized
    export); on an export failure, stop and report the exact error before
    changing anything. The full run resumes without repeating evaluations.

Added before any Level 2 data
  - Ranking stability (stability.py, reported only): R on a second set of 512
    bank images (seed 1), disjoint from the primary set and from the
    calibration samples; per model, Spearman of the two R rankings over the
    layers ranked in both, and the overlap of the top 8. Level 2 always uses
    the primary ranking.
  - Exploratory curve: targeted by local error (descending; ties by execution
    order; same candidate layers), same K up to the K_final of the R curve,
    against the same random controls. The "clearly better" criterion and the
    main conclusions apply only to the R curve.
"""

from pathlib import Path

from quantization.vit_brevitas.paths import INVESTIGATIONS_ROOT

PHASE4_ROOT: Path = INVESTIGATIONS_ROOT / "phase4"
MODELS = ("deit_tiny", "mobilevit", "efficientvit_b1")
