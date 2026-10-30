"""One-off diagnostic (NOT part of the pipeline): is the CALIBRATED
activation quantizer's scale badly wrong for Conv2d layers specifically?

diagnose_layer_by_layer.py showed per-layer error growing/staying high from
the very first quantized conv onward in mobilevit/efficientvit_b1, and
switching weight quantization to granularity=per_channel (already generated
by --all) changed NOTHING (still near-random accuracy) -- ruling out "weight
quantization is just imprecise for depthwise convs" as the explanation,
since granularity only touches weight quant. That leaves activation
quantization/calibration (unchanged between those two rows) as the prime
suspect.

This calls the layer's *actual* calibrated input_quant module directly
(module.input_quant(x) -> QuantTensor with .scale/.zero_point/.bit_width)
on a fresh real image, for one QuantConv2d layer in mobilevit and, as a
known-good control, one QuantLinear layer in deit_tiny -- then checks what
fraction of the real captured input values would be clipped by that
calibrated scale (i.e. |x| > scale * (2**(bit_width-1) - 1)). A properly
calibrated minmax quantizer should clip ~0% of a same-distribution input;
heavy clipping means the calibrated scale is too small for what the layer
actually sees at inference.

Needs: Brevitas installed, the gated ImageNet-1K pool already prepared.

Run directly:

    python quantization/vit_brevitas/diagnose_activation_scale.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

CALIB_SET_SIZE = 128
REPEAT_ID = 1


def inspect_layer(model_name: str, layer_name: str) -> None:
    import torch
    from PIL import Image

    from data_prep.imagenet import load_eval_subset
    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess
    from quantization.vit_brevitas.quantize import build_quantized_model

    print(f"\n=== {model_name}: {layer_name} ===")

    entry = {
        "model_name": model_name,
        "calibration_algorithm": "minmax",
        "granularity": "per_tensor",
        "symmetry": "symmetric",
        "calib_set_size": CALIB_SET_SIZE,
        "bitwidth": "int8",
    }
    quant_model, _input_shape = build_quantized_model(entry, repeat_id=REPEAT_ID)
    quant_model.eval()

    target = dict(quant_model.named_modules())[layer_name]

    captured = {}

    def hook(_module, inputs, _output):
        captured["x"] = inputs[0].detach()

    handle = target.register_forward_hook(hook)

    preprocess = resolve_preprocess(model_name)
    sample = load_eval_subset()[1]  # a different image than diagnose_layer_by_layer's [0]
    image = Image.open(sample["path"]).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        quant_model(tensor)
    handle.remove()

    x = captured["x"]
    print(f"captured real input to this layer: shape={tuple(x.shape)}, "
          f"absmax={x.abs().max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")

    with torch.no_grad():
        quant_out = target.input_quant(x)

    scale = quant_out.scale
    zero_point = quant_out.zero_point
    bit_width = quant_out.bit_width
    print(f"calibrated input_quant: scale={scale.flatten()[:5].tolist()} "
          f"(numel={scale.numel()}), zero_point={zero_point.flatten()[:5].tolist()}, "
          f"bit_width={bit_width}")

    scale_scalar = scale.flatten()[0].item()
    bw_scalar = bit_width.flatten()[0].item() if hasattr(bit_width, "flatten") else float(bit_width)
    max_representable = scale_scalar * (2 ** (bw_scalar - 1) - 1)
    clipped_fraction = (x.abs() > max_representable).float().mean().item()
    print(f"max representable value at this scale: {max_representable:.4f} "
          f"(real input absmax: {x.abs().max().item():.4f})")
    print(f"fraction of real input values that would be CLIPPED: {clipped_fraction:.4%}")

    if clipped_fraction > 0.01:
        print("SUSPECT: >1% of real activations are clipped -- calibrated scale is too small "
              "for what this layer actually sees at inference.")
    else:
        print("OK: negligible clipping at this layer.")


def main() -> None:
    # Known-good control: a Linear layer in deit_tiny (accuracy is fine end-to-end).
    inspect_layer("deit_tiny", "blocks.0.attn.qkv")
    # Suspect: an early Conv2d layer in mobilevit (accuracy collapses).
    inspect_layer("mobilevit", "stem.conv")
    inspect_layer("mobilevit", "stages.0.0.conv1_1x1.conv")
    # Never actually checked before: efficientvit_b1's own early layers.
    # per_channel weight quant, MSE calibration, and excluding LiteMLA all
    # failed to rescue this model's accuracy -- activation clipping in its
    # OWN early conv stages is the remaining unverified hypothesis.
    inspect_layer("efficientvit_b1", "stem.in_conv.conv")
    inspect_layer("efficientvit_b1", "stem.res0.main.depth_conv.conv")
    inspect_layer("efficientvit_b1", "stem.res0.main.point_conv.conv")
    inspect_layer("efficientvit_b1", "stages.0.blocks.0.main.point_conv.conv")


if __name__ == "__main__":
    main()
