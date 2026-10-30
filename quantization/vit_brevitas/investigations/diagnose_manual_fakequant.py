"""One-off diagnostic (NOT part of the pipeline): does Brevitas's own
QuantConv2d execution match plain fake-quantization math, for the specific
mobilevit layers where diagnose_layer_by_layer.py showed the sharpest
relative-error jumps?

Ruled out so far: activation-scale clipping (diagnose_activation_scale.py --
negligible clipping at stem.conv and conv1_1x1.conv), and weight
granularity (switching minmax's weight quant from per_tensor to
per_channel changed accuracy NOT AT ALL on the real pipeline -- still
near-random). That leaves two remaining explanations for the large,
early per-layer error:

  1. It's "just" ordinary INT8 fake-quantization noise for this layer's
     specific weight/activation distributions, which then compounds across
     ~50 sequential per-tensor-quantized conv layers with nothing (no
     LayerNorm-style renormalization) resetting the drift -- a real
     architecture-sensitivity finding, not a bug.
  2. Brevitas's QuantConv2d module itself computes something different from
     straightforward dequantize(quantize(weight)) conv'd with
     dequantize(quantize(input)) using the SAME (already-calibrated)
     scale/zero_point -- an execution-level bug (e.g. in how groups, bias,
     or scale application interacts for this layer shape), not a
     calibration bug.

This builds the manual fake-quantized conv using the layer's own
already-calibrated scale/zero_point (extracted via quant_weight() and
input_quant(x), the same accessors used successfully in
diagnose_activation_scale.py) and compares its output directly against
Brevitas's own module(x) on the identical real input. If they match closely
but both differ substantially from the float conv, explanation 1 holds. If
Brevitas's own output diverges from the manual simulation (which uses
Brevitas's own calibrated scale/zero_point, so should reproduce it if
nothing else is different), explanation 2 holds.

Needs: Brevitas installed, the gated ImageNet-1K pool already prepared.

Run directly:

    python quantization/vit_brevitas/diagnose_manual_fakequant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

LAYERS_BY_MODEL = {
    "mobilevit": ["stem.conv", "stages.0.0.conv1_1x1.conv", "stages.0.0.conv3_1x1.conv"],
    "efficientvit_b1": [
        "stem.in_conv.conv",
        "stem.res0.main.point_conv.conv",
        "stages.0.blocks.0.main.point_conv.conv",
    ],
}
CALIB_SET_SIZE = 128
REPEAT_ID = 1


def main() -> None:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from data_prep.imagenet import load_eval_subset
    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess
    from quantization.vit_brevitas.quantize import build_quantized_model

    MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "mobilevit"
    LAYERS_TO_CHECK = LAYERS_BY_MODEL[MODEL_NAME]

    entry = {
        "model_name": MODEL_NAME,
        "calibration_algorithm": "minmax",
        "granularity": "per_tensor",
        "symmetry": "symmetric",
        "calib_set_size": CALIB_SET_SIZE,
        "bitwidth": "int8",
    }
    print(f"Building calibrated {MODEL_NAME}...")
    quant_model, _input_shape = build_quantized_model(entry, repeat_id=REPEAT_ID)
    quant_model.eval()

    named = dict(quant_model.named_modules())

    captured = {}

    def make_hook(name):
        def hook(_module, inputs, _output):
            captured[name] = inputs[0].detach()
        return hook

    handles = [named[name].register_forward_hook(make_hook(name)) for name in LAYERS_TO_CHECK]

    preprocess = resolve_preprocess(MODEL_NAME)
    sample = load_eval_subset()[1]
    image = Image.open(sample["path"]).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        quant_model(tensor)
    for h in handles:
        h.remove()

    for layer_name in LAYERS_TO_CHECK:
        print(f"\n=== {layer_name} ===")
        conv = named[layer_name]
        x = captured[layer_name]

        with torch.no_grad():
            # Brevitas's own, real output for this exact input.
            brevitas_out = conv(x)

            # Manual simulation: dequantize(quantize(.)) using Brevitas's OWN
            # already-calibrated weight/activation quantizers, fed through a
            # plain functional conv -- if this doesn't match conv(x), the
            # discrepancy is in Brevitas's execution, not its calibration.
            quant_weight = conv.quant_weight()
            weight_dequant = quant_weight.value if hasattr(quant_weight, "value") else quant_weight
            x_quant_tensor = conv.input_quant(x)
            x_dequant = x_quant_tensor.value if hasattr(x_quant_tensor, "value") else x_quant_tensor

            manual_out = F.conv2d(
                x_dequant, weight_dequant,
                bias=conv.bias, stride=conv.stride, padding=conv.padding,
                dilation=conv.dilation, groups=conv.groups,
            )

        diff = (brevitas_out - manual_out).float()
        rel_err = diff.norm().item() / (manual_out.float().norm().item() + 1e-12)
        print(f"brevitas_out norm={brevitas_out.float().norm().item():.4f}, "
              f"manual_fakequant_out norm={manual_out.float().norm().item():.4f}, "
              f"rel_L2_diff(brevitas_vs_manual)={rel_err:.6f}")
        if rel_err > 0.01:
            print("MISMATCH: Brevitas's own module output diverges from a manual "
                  "fake-quant simulation using its OWN calibrated scale/zero_point "
                  "-- points at an execution-level issue (groups/bias/scale "
                  "application), not calibration.")
        else:
            print("MATCH: Brevitas's module output matches manual fake-quant closely "
                  "-- the per-layer error is genuine quantization noise for this "
                  "layer's weight/activation distribution, not an execution bug.")


if __name__ == "__main__":
    main()
