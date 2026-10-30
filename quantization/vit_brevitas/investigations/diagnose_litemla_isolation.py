"""One-off diagnostic (NOT part of the pipeline, like spike_test.py): does
leaving LiteMLA (EfficientViT-B1's lightweight linear-attention block)
UNQUANTIZED, while quantizing the rest of the network exactly as the
current INT8 baseline does, rescue accuracy?

Context: diagnose_baseline_collapse.py / diagnose_layer_by_layer.py /
diagnose_manual_fakequant.py already showed (a) EfficientViT-B1's INT8
baseline (minmax, per_tensor, symmetric) collapses to near-random accuracy
already before ONNX export, (b) Brevitas's own execution matches manual
fake-quant math exactly (no execution bug), and (c) switching calibration
to MSE (which measurably rescues MobileViT) does NOTHING for
EfficientViT-B1. Reading timm's efficientvit_mit.LiteMLA source showed why
this might be qualitatively different from ordinary conv quantization
noise: its kernel_func is nn.ReLU (a hard zero threshold that quantization
noise can flip discretely) feeding an explicit division-based normalizer
(`out[..., :-1] / (out[..., -1:] + eps)`, eps=1e-5) -- a mechanism no
amount of better *scale calibration* would fix, since the sensitivity is
structural, not about scale coarseness.

This isolates that hypothesis empirically by comparing scenarios on the
SAME fixed eval-subset samples:

  1. Full float (no quantization at all) -- reference.
  2. Full INT8 (the current baseline config: minmax/per_tensor/symmetric)
     -- known near-random.
  3. Mixed: every LiteMLA instance in the model tree left entirely
     unquantized (float, using its original pretrained weights), the rest
     of the network quantized exactly as in scenario 2.
  4. Weights-only INT8, activations left float ("Dynamic Range"
     quantization, the same mode used for the CNNs in the original paper):
     every Linear/Conv2d substituted with weight_quant=<baseline
     minmax/per_tensor> and input_quant=None (no activation fake-quant at
     all). Since weight-only MinMax quantization derives its scale
     statically from the weight tensor itself (MaxStatsScaling), no
     calibration forward pass is needed for this scenario.

If scenario 3 recovers accuracy close to scenario 1 (float), LiteMLA's
ReLU+division normalizer is confirmed as the specific cause of the
collapse -- not a generic "PTQ is imprecise" story, and not something
better calibration alone can fix. If scenario 3 is still near-random, the
cause lies elsewhere in the network (the earlier per-layer error, which
was already large before LiteMLA, at the conv/MBConv stages).

Scenario 4 isolates a different axis entirely: none of scenarios 2/3 (nor
the earlier granularity/calibration-algorithm sweeps) can distinguish
whether the collapse comes from quantizing WEIGHTS, ACTIVATIONS, or their
combination, because all of them kept activations quantized throughout.
If scenario 4 (weights-only) is close to float, activation quantization is
the dominant cause. If scenario 4 is ALSO near-random, weight quantization
alone is sufficient to collapse this model -- a much stronger and more
surprising result, given weight-only INT8 is usually the gentlest PTQ mode.

Standalone: no manifest row, no results/raw/ writes, console output only.

Needs: Brevitas installed, the gated ImageNet-1K eval subset already
prepared.

Run directly:

    python quantization/vit_brevitas/diagnose_litemla_isolation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODEL_NAME = "efficientvit_b1"
CALIB_SET_SIZE = 128
REPEAT_ID = 1
N_SAMPLES = 200  # same fixed eval subset used throughout; capped for speed


def quantize_module_tree_excluding(module, weight_quant, act_quant, skip_types) -> None:
    """Same recursion as quantize.py's quantize_module_tree, except any
    subtree rooted at an instance of `skip_types` is left completely
    untouched (not recursed into at all) instead of substituted."""
    import torch.nn as nn

    from quantization.vit_brevitas.quantize import _quantize_conv2d, _quantize_linear

    for name, child in module.named_children():
        if isinstance(child, skip_types):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, _quantize_linear(child, weight_quant, act_quant))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, _quantize_conv2d(child, weight_quant, act_quant))
        else:
            quantize_module_tree_excluding(child, weight_quant, act_quant, skip_types)


def build_quantized_model_excluding_litemla(entry, repeat_id):
    """Mirrors quantize.py's build_quantized_model exactly, except LiteMLA
    instances are skipped by quantize_module_tree_excluding instead of
    substituted -- everything else (loading, input shape, calibration
    batch, calibration_mode forward pass) is identical to the real
    pipeline."""
    import torch
    from brevitas.graph.calibrate import calibration_mode
    from timm.models.efficientvit_mit import LiteMLA

    from quantization.vit_brevitas.calibration import load_calibration_batch
    from quantization.vit_brevitas.quantize import load_pretrained_model, resolve_input_shape
    from quantization.vit_brevitas.quantizer_config import resolve_act_quant, resolve_weight_quant

    model = load_pretrained_model(entry["model_name"])
    input_shape = resolve_input_shape(model)

    weight_quant = resolve_weight_quant(entry["calibration_algorithm"], entry["granularity"])
    act_quant = resolve_act_quant(entry["calibration_algorithm"], entry["symmetry"])

    calib_batch = load_calibration_batch(model, entry["calib_set_size"], repeat_id)

    quantize_module_tree_excluding(model, weight_quant, act_quant, skip_types=(LiteMLA,))
    model.eval()

    with torch.no_grad(), calibration_mode(model):
        model(calib_batch)

    return model, input_shape


def build_weights_only_quantized_model(entry):
    """Weight-only ("Dynamic Range") quantization: every Linear/Conv2d gets
    its weight fake-quantized (same baseline minmax/per_tensor weight_quant
    as the real pipeline) but input_quant=None, so activations pass through
    as plain float -- reusing quantize.py's own quantize_module_tree and
    _quantize_linear/_quantize_conv2d unchanged, just with act_quant=None.

    No calibration batch/forward pass: MinMax weight quantization
    (MaxStatsScaling) computes its scale directly and deterministically from
    the weight tensor itself, and there is no activation quantizer here to
    collect runtime stats for."""
    from quantization.vit_brevitas.quantize import (
        load_pretrained_model,
        quantize_module_tree,
        resolve_input_shape,
    )
    from quantization.vit_brevitas.quantizer_config import resolve_weight_quant

    model = load_pretrained_model(entry["model_name"])
    input_shape = resolve_input_shape(model)

    weight_quant = resolve_weight_quant(entry["calibration_algorithm"], entry["granularity"])

    quantize_module_tree(model, weight_quant, act_quant=None)
    model.eval()

    return model, input_shape


def compute_accuracy(model, preprocess, samples):
    import numpy as np
    import torch
    from PIL import Image

    top1 = top5 = 0
    with torch.no_grad():
        for sample in samples:
            image = Image.open(sample["path"]).convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
            logits = model(tensor)[0].numpy()
            top5_pred = np.argsort(logits)[-5:][::-1]
            top1 += int(top5_pred[0] == sample["label"])
            top5 += int(sample["label"] in top5_pred)
    n = len(samples)
    return (100.0 * top1 / n, 100.0 * top5 / n)


def main() -> None:
    from data_prep.imagenet import load_eval_subset
    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess
    from quantization.vit_brevitas.quantize import build_quantized_model, load_pretrained_model

    samples = load_eval_subset()[:N_SAMPLES]
    preprocess = resolve_preprocess(MODEL_NAME)
    entry = {
        "model_name": MODEL_NAME,
        "calibration_algorithm": "minmax",
        "granularity": "per_tensor",
        "symmetry": "symmetric",
        "calib_set_size": CALIB_SET_SIZE,
        "bitwidth": "int8",
    }

    print(f"Using {len(samples)} eval-subset samples, model={MODEL_NAME}\n")

    print("[1/4] Full float (no quantization)...")
    float_model = load_pretrained_model(MODEL_NAME)
    float_model.eval()
    top1_float, top5_float = compute_accuracy(float_model, preprocess, samples)

    print("[2/4] Full INT8 (minmax/per_tensor/symmetric baseline)...")
    quant_model, _ = build_quantized_model(entry, repeat_id=REPEAT_ID)
    top1_int8, top5_int8 = compute_accuracy(quant_model, preprocess, samples)

    print("[3/4] Mixed: INT8 everywhere except LiteMLA left float...")
    mixed_model, _ = build_quantized_model_excluding_litemla(entry, repeat_id=REPEAT_ID)
    top1_mixed, top5_mixed = compute_accuracy(mixed_model, preprocess, samples)

    print("[4/4] Weights-only INT8, activations float (Dynamic Range)...")
    weights_only_model, _ = build_weights_only_quantized_model(entry)
    top1_wonly, top5_wonly = compute_accuracy(weights_only_model, preprocess, samples)

    print(f"\n{'scenario':<45} {'top1':>8} {'top5':>8}")
    print(f"{'1. full float':<45} {top1_float:>7.2f}% {top5_float:>7.2f}%")
    print(f"{'2. full INT8 (current baseline)':<45} {top1_int8:>7.2f}% {top5_int8:>7.2f}%")
    print(f"{'3. INT8, LiteMLA left float':<45} {top1_mixed:>7.2f}% {top5_mixed:>7.2f}%")
    print(f"{'4. weights-only INT8, activations float':<45} {top1_wonly:>7.2f}% {top5_wonly:>7.2f}%")

    if top1_mixed > top1_int8 + 20:
        print("\n=> Leaving LiteMLA float substantially recovers accuracy: "
              "LiteMLA's ReLU+division normalizer is confirmed as the cause "
              "of the collapse, not generic quantization noise elsewhere.")
    else:
        print("\n=> Leaving LiteMLA float did NOT substantially help: the "
              "collapse is not (solely) attributable to LiteMLA -- the "
              "earlier conv/MBConv stages are already corrupting the signal "
              "before it even reaches LiteMLA.")

    if top1_wonly > top1_int8 + 20:
        print("=> Weights-only INT8 recovers most of the accuracy: activation "
              "quantization is the dominant cause of the collapse, not weight "
              "quantization.")
    else:
        print("=> Weights-only INT8 is ALSO near-random: weight quantization "
              "alone is sufficient to collapse this model -- surprising, since "
              "weight-only PTQ is usually the gentlest mode.")


if __name__ == "__main__":
    main()
