"""One-off diagnostic (NOT part of the pipeline, no manifest/results
involvement, no re-launch of anything): does independent per-layer
dequantize/requantize noise (return_quant_tensor=False on every substituted
layer, quantize.py's current, proven-working choice for all 3 models)
contribute meaningfully to MobileViT/EfficientViT-B1's collapse, compared
to sharing the quantized tensor (scale/zero_point carried along, no
dequant-requant round trip) between directly adjacent quantized layers?

This is a more direct test than chasing Brevitas's automatic quantize()
pipeline (parked: both models failed at FX tracing for unrelated,
non-quantization reasons -- MobileViT hit a genuine dynamic-control-flow
limitation of FX itself, EfficientViT-B1 hit LiteMLA's
torch.autocast(device_type=v.device.type) breaking under a traced Proxy --
and even a successful trace wouldn't guarantee clearing the same
residual-handler scale-alignment stage that blocked DeiT).

IMPORTANT STRUCTURAL CAVEAT, stated before results come in rather than
after: in these architectures, quantized Conv2d layers are overwhelmingly
separated by plain nn.BatchNorm2d/activation functions (Conv -> BN -> Act
-> Conv -> ...), which are NOT part of this project's substitution and do
not understand Brevitas's QuantTensor -- a QuantTensor hitting a plain
BatchNorm2d will, at best, silently degrade back to a bare tensor (no
different from return_quant_tensor=False), or, at worst, error outright.
The one place in these architectures with two Conv2d layers DIRECTLY
adjacent (no intervening float op) is LiteMLA's `aggreg` submodule
(nn.Sequential of two Conv2d) -- already shown (diagnose_litemla_isolation.
py, scenario 3) not to be where the collapse originates. So there is a real
possibility this change has near-zero effect simply because there is
nowhere for the "shared tensor" benefit to apply -- that would itself be a
decisive, informative negative result, not a wasted test.

Setting return_quant_tensor=True is applied uniformly (letting Brevitas's
own QuantTensor-vs-plain-op semantics determine what actually gets shared
vs. silently degrades, rather than hand-picking which specific layer pairs
are "compatible" -- there is no clean way to determine that without
runtime behavior anyway). If the forward pass errors outright, that itself
directly explains why return_quant_tensor=False was the necessary choice.

Compares accuracy against the current baseline (minmax/per_tensor/
symmetric/INT8, return_quant_tensor=False), recomputed on the SAME sample
slice for a fair comparison.

Needs: Brevitas installed, the gated ImageNet-1K eval subset already
prepared.

Run directly:

    python quantization/vit_brevitas/diagnose_return_quant_tensor.py mobilevit
    python quantization/vit_brevitas/diagnose_return_quant_tensor.py efficientvit_b1
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


def _quantize_linear_shared(child, weight_quant, act_quant):
    from brevitas.nn import QuantLinear

    quant_linear = QuantLinear(
        in_features=child.in_features,
        out_features=child.out_features,
        bias=child.bias is not None,
        weight_quant=weight_quant,
        input_quant=act_quant,
        return_quant_tensor=True,
    )
    quant_linear.weight.data = child.weight.data.clone()
    if child.bias is not None:
        quant_linear.bias.data = child.bias.data.clone()
    return quant_linear


def _quantize_conv2d_shared(child, weight_quant, act_quant):
    from brevitas.nn import QuantConv2d

    quant_conv = QuantConv2d(
        in_channels=child.in_channels,
        out_channels=child.out_channels,
        kernel_size=child.kernel_size,
        stride=child.stride,
        padding=child.padding,
        dilation=child.dilation,
        groups=child.groups,
        bias=child.bias is not None,
        weight_quant=weight_quant,
        input_quant=act_quant,
        return_quant_tensor=True,
    )
    quant_conv.weight.data = child.weight.data.clone()
    if child.bias is not None:
        quant_conv.bias.data = child.bias.data.clone()
    return quant_conv


def quantize_module_tree_shared(module, weight_quant, act_quant) -> None:
    import torch.nn as nn

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, _quantize_linear_shared(child, weight_quant, act_quant))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, _quantize_conv2d_shared(child, weight_quant, act_quant))
        else:
            quantize_module_tree_shared(child, weight_quant, act_quant)


def build_quantized_model_shared(entry, repeat_id):
    """Mirrors quantize.py's build_quantized_model exactly, except every
    substituted layer uses return_quant_tensor=True instead of False."""
    import torch
    from brevitas.graph.calibrate import calibration_mode

    from quantization.vit_brevitas.calibration import load_calibration_batch
    from quantization.vit_brevitas.quantize import load_pretrained_model, resolve_input_shape
    from quantization.vit_brevitas.quantizer_config import resolve_act_quant, resolve_weight_quant

    model = load_pretrained_model(entry["model_name"])
    input_shape = resolve_input_shape(model)

    weight_quant = resolve_weight_quant(entry["calibration_algorithm"], entry["granularity"])
    act_quant = resolve_act_quant(entry["calibration_algorithm"], entry["symmetry"])

    calib_batch = load_calibration_batch(model, entry["calib_set_size"], repeat_id)

    quantize_module_tree_shared(model, weight_quant, act_quant)
    model.eval()

    with torch.no_grad(), calibration_mode(model):
        model(calib_batch)

    return model, input_shape


def _unwrap(output):
    """QuantTensor (when return_quant_tensor=True survives all the way to
    the model's output) exposes its dequantized value via .value; a plain
    tensor doesn't have that attribute, so this is safe either way."""
    return output.value if hasattr(output, "value") else output


def compute_accuracy(model, preprocess, samples):
    import numpy as np
    import torch
    from PIL import Image

    top1 = top5 = 0
    with torch.no_grad():
        for sample in samples:
            image = Image.open(sample["path"]).convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
            output = _unwrap(model(tensor))
            logits = output[0].detach().numpy()
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
    entry = {
        "model_name": model_name,
        "calibration_algorithm": "minmax",
        "granularity": "per_tensor",
        "symmetry": "symmetric",
        "calib_set_size": CALIB_SET_SIZE,
        "bitwidth": "int8",
    }

    print(f"=== {model_name}: return_quant_tensor=False (baseline) vs. True (shared) ===\n")

    print("[A] return_quant_tensor=False (current baseline, recomputed on this sample slice)...")
    baseline_model, _ = build_quantized_model(entry, repeat_id=REPEAT_ID)
    top1_false, top5_false = compute_accuracy(baseline_model, preprocess, samples)

    print("[B] return_quant_tensor=True (shared quantized tensor where structurally possible)...")
    try:
        shared_model, _ = build_quantized_model_shared(entry, repeat_id=REPEAT_ID)
        top1_true, top5_true = compute_accuracy(shared_model, preprocess, samples)
        failed = False
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        failed = True

    print(f"\n{'scenario':<45} {'top1':>8} {'top5':>8}")
    print(f"{'A. return_quant_tensor=False (baseline)':<45} {top1_false:>7.2f}% {top5_false:>7.2f}%")
    if failed:
        print(f"{'B. return_quant_tensor=True':<45} {'FAILED':>8} {'--':>8}")
        print("\n=> return_quant_tensor=True errors outright on this architecture -- this directly "
              "explains why manual substitution had to use return_quant_tensor=False, and rules out "
              "the shared-tensor form of this hypothesis structurally, not just empirically.")
    else:
        print(f"{'B. return_quant_tensor=True':<45} {top1_true:>7.2f}% {top5_true:>7.2f}%")
        if abs(top1_true - top1_false) < 2.0:
            print("\n=> No meaningful difference: independent per-layer dequant/requant noise is NOT "
                  "a real contributor to the collapse (consistent with the structural expectation that "
                  "BatchNorm/activations between almost every quantized layer prevent the shared-tensor "
                  "benefit from applying here).")
        elif top1_true > top1_false + 2.0:
            print("\n=> return_quant_tensor=True measurably helps: independent per-layer "
                  "dequant/requant noise IS a real, avoidable contributor to the collapse.")
        else:
            print("\n=> return_quant_tensor=True measurably HURTS: unexpected, worth a closer look "
                  "at where in the graph QuantTensor propagation is behaving unexpectedly.")


if __name__ == "__main__":
    main()
