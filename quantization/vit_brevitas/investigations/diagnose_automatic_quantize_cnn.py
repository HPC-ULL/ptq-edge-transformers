"""One-off diagnostic (NOT part of the pipeline, no manifest/results
involvement, no re-launch of anything): does Brevitas's AUTOMATIC
FX-graph-based quantize() pipeline work for MobileViT / EfficientViT-B1,
and if so, does it produce better accuracy than our manual layer
substitution at the same baseline config?

Context: quantize.py uses manual nn.Linear/nn.Conv2d -> QuantLinear/
QuantConv2d substitution (return_quant_tensor=False at every boundary --
every quantized layer independently dequantizes/requantizes, no shared
scale propagation between layers) INSTEAD OF Brevitas's own automatic
quantize(), because quantize() hit 3 failures when this project first
tried it against DeiT-Tiny (see quantize.py's and spike_test.py's module
docstrings): a missing GraphModule trace, `is_causal` breaking as an FX
Proxy, and an unconditional "Residual handler could not find a solution to
align scale factors across ADDs and CATs" error. ALL THREE of those are
tied to transformer/attention residual patterns, not CNN residual
patterns -- Brevitas's automatic pipeline is designed and tested primarily
against CNN backbones (ResNet/MobileNet-style). It was never actually
retried against MobileViT/EfficientViT-B1 specifically; the decision to use
manual substitution for ALL THREE models was made once, based only on
DeiT's failures, and then applied uniformly.

Hypothesis this tests: manual substitution's conservative
return_quant_tensor=False-everywhere style is well-tolerated by DeiT
because LayerNorm renormalizes every block regardless of quantization
noise, but may be adding avoidable compounding noise for MobileViT/
EfficientViT-B1, which use BatchNorm (fixed at inference, no adaptive
reset). If Brevitas's automatic pipeline (a) avoids the transformer-specific
tracing errors for these CNN-residual architectures, and (b) produces
meaningfully better accuracy at the same nominal INT8 baseline config, that
would mean part of what's been characterized as "architecture PTQ
sensitivity" is actually attributable to our own pipeline's conservativeness
-- not an intrinsic property of the architectures.

For a fair comparison, this recomputes the MANUAL pipeline's accuracy on
the exact same N_SAMPLES eval-subset slice (rather than reusing the earlier
full-2000-image results/raw numbers, which used a different sample size).

IMPORTANT CAVEAT: Brevitas's quantize() API/defaults were not verified
against the installed version before writing this (no Brevitas available
in the authoring environment) -- if the exact call signature below is
wrong, the resulting error message will say so; report it back verbatim so
the call can be corrected. Whatever weight/activation quantizer classes
quantize() actually ends up applying are introspected and printed directly
(rather than assumed) so we know exactly what was compared.

Needs: Brevitas installed, the gated ImageNet-1K eval subset already
prepared.

Run directly:

    python quantization/vit_brevitas/diagnose_automatic_quantize_cnn.py mobilevit
    python quantization/vit_brevitas/diagnose_automatic_quantize_cnn.py efficientvit_b1
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

CALIB_SET_SIZE = 128
REPEAT_ID = 1
N_SAMPLES = 200


def fx_trace(model):
    """Try a plain FX trace first; fall back to the concrete_args used for
    DeiT's is_causal issue. Returns (traced_or_None, error_message_or_None)."""
    from brevitas.fx import symbolic_trace

    try:
        return symbolic_trace(model), None
    except Exception as e1:
        try:
            return symbolic_trace(model, concrete_args={"attn_mask": None, "is_causal": False}), None
        except Exception as e2:
            return None, (
                f"plain trace failed: {type(e1).__name__}: {e1}\n"
                f"trace with concrete_args also failed: {type(e2).__name__}: {e2}"
            )


def build_automatic_quantized_model(model_name: str):
    """Attempt Brevitas's automatic FX-graph quantize() pipeline.
    Returns (quant_model_or_None, error_message_or_None)."""
    import torch
    from brevitas.graph.calibrate import calibration_mode

    from quantization.vit_brevitas.calibration import load_calibration_batch
    from quantization.vit_brevitas.quantize import load_pretrained_model

    model = load_pretrained_model(model_name)

    print("  [1/3] FX symbolic trace...")
    traced, err = fx_trace(model)
    if traced is None:
        return None, f"[FX trace] {err}"

    print("  [2/3] brevitas.graph.quantize.quantize()...")
    try:
        from brevitas.graph.quantize import quantize
        quant_model = quantize(traced)
    except Exception as e:
        return None, f"[quantize()] {type(e).__name__}: {e}"

    # Report what quantize() actually applied, since its default
    # weight/act quantizer choices were not verified ahead of time.
    from brevitas.nn import QuantConv2d, QuantLinear
    for name, module in quant_model.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear)):
            wq = type(module.weight_quant.quant_injector).__name__ if hasattr(module, "weight_quant") else "?"
            aq = type(module.input_quant.quant_injector).__name__ if hasattr(module, "input_quant") else "?"
            print(f"  applied config (first quantized layer, {name}): weight_quant={wq}, input_quant={aq}")
            break

    print("  [3/3] calibration...")
    calib_batch = load_calibration_batch(model, CALIB_SET_SIZE, REPEAT_ID)
    quant_model.eval()
    try:
        with torch.no_grad(), calibration_mode(quant_model):
            quant_model(calib_batch)
    except Exception as e:
        return None, f"[calibration] {type(e).__name__}: {e}"

    return quant_model, None


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
    from quantization.vit_brevitas.quantize import build_quantized_model

    model_name = sys.argv[1] if len(sys.argv) > 1 else "mobilevit"
    samples = load_eval_subset()[:N_SAMPLES]
    preprocess = resolve_preprocess(model_name)

    print(f"=== {model_name}: automatic quantize() vs. manual substitution ===\n")

    print("[A] Manual substitution baseline (recomputed on this same sample slice)...")
    entry = {
        "model_name": model_name,
        "calibration_algorithm": "minmax",
        "granularity": "per_tensor",
        "symmetry": "symmetric",
        "calib_set_size": CALIB_SET_SIZE,
        "bitwidth": "int8",
    }
    manual_model, _ = build_quantized_model(entry, repeat_id=REPEAT_ID)
    top1_manual, top5_manual = compute_accuracy(manual_model, preprocess, samples)

    print("\n[B] Automatic quantize() pipeline...")
    auto_model, error = build_automatic_quantized_model(model_name)
    if auto_model is None:
        print(f"\nFAILED: {error}")
        top1_auto = top5_auto = None
    else:
        top1_auto, top5_auto = compute_accuracy(auto_model, preprocess, samples)

    print(f"\n{'pipeline':<40} {'top1':>8} {'top5':>8}")
    print(f"{'A. manual substitution (this project)':<40} {top1_manual:>7.2f}% {top5_manual:>7.2f}%")
    if top1_auto is None:
        print(f"{'B. Brevitas automatic quantize()':<40} {'FAILED':>8} {'--':>8}")
        print(f"\n=> quantize() did not run to completion for {model_name}. Error:\n{error}")
    else:
        print(f"{'B. Brevitas automatic quantize()':<40} {top1_auto:>7.2f}% {top5_auto:>7.2f}%")
        if top1_auto > top1_manual + 20:
            print("\n=> Automatic quantize() ran successfully AND substantially outperforms manual "
                  "substitution: our pipeline's conservativeness is a real, avoidable contributor "
                  "to this model's collapse -- worth redoing this model's quantization base.")
        else:
            print("\n=> Automatic quantize() ran successfully but did NOT substantially outperform "
                  "manual substitution: the collapse is not (mainly) attributable to our "
                  "substitution style either.")


if __name__ == "__main__":
    main()
