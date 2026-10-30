"""One-off diagnostic (NOT part of the pipeline): are EfficientViT-B1's real
activations dominated by a small number of high-magnitude OUTLIER CHANNELS,
which a per-tensor scale must stretch to cover -- without ever literally
clipping anything -- while crushing the rest of the channels into very few
effective quantization levels?

Context: diagnose_litemla_isolation.py's scenario 4 (weights-only INT8,
activations float) recovered accuracy from 0.5% to 55% top1 (float is 81%),
proving ACTIVATION quantization is the dominant cause of EfficientViT-B1's
collapse, not weights. diagnose_activation_scale.py already showed
negligible literal clipping at this model's early layers, and switching to
MSE calibration (which explicitly searches for the error-minimizing scale)
did not help either. The remaining explanation consistent with all of that:
a per-tensor scale can be "correctly" calibrated (no clipping, not even a
bad MSE-suboptimal choice) and STILL be catastrophic if a handful of
channels have much larger range than the rest -- the scale must accommodate
the outlier, so the majority of channels (the ones actually carrying
signal) get quantized into only a handful of effective levels out of 256.
This is the same activation-outlier-channel phenomenon documented for LLM
quantization (LLM.int8(), SmoothQuant), here checked for a CNN activation.

This captures the REAL float activation (no quantization at all, to see the
architecture's own natural distribution, not something already contaminated
by upstream quantization noise) feeding into one of EfficientViT-B1's early
flagged layers, computes PER-CHANNEL absmax (over batch+spatial dims), and
reports how concentrated the range is -- plus, for context, the same
measurement on a low-error EfficientViT-B1 layer and on deit_tiny's known
-good control layer.

Needs: only timm + torch (no Brevitas, no gated-data dependency beyond the
already-prepared eval subset used elsewhere).

Run directly:

    python quantization/vit_brevitas/diagnose_activation_outliers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# (model_name, layer_name, note)
LAYERS_TO_CHECK = [
    ("efficientvit_b1", "stages.0.blocks.0.main.point_conv.conv", "flagged: high per-layer error"),
    ("efficientvit_b1", "stem.res0.main.depth_conv.conv", "control: low per-layer error, same model"),
    ("deit_tiny", "blocks.0.attn.qkv", "control: known-good, different architecture"),
]


def inspect_channel_distribution(model_name: str, layer_name: str, note: str) -> None:
    import torch
    from PIL import Image

    from data_prep.imagenet import load_eval_subset
    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess
    from quantization.vit_brevitas.quantize import load_pretrained_model

    print(f"\n=== {model_name}: {layer_name} ({note}) ===")

    model = load_pretrained_model(model_name)
    model.eval()
    target = dict(model.named_modules())[layer_name]

    captured = {}

    def hook(_module, inputs, _output):
        captured["x"] = inputs[0].detach()

    handle = target.register_forward_hook(hook)

    preprocess = resolve_preprocess(model_name)
    sample = load_eval_subset()[1]
    image = Image.open(sample["path"]).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        model(tensor)
    handle.remove()

    x = captured["x"].float()
    # Channel dim is 1 for both (N,C,H,W) conv inputs and (N,*,C) would
    # differ, but every layer checked here is a Conv2d, so NCHW throughout.
    reduce_dims = tuple(d for d in range(x.dim()) if d != 1)
    channel_absmax = x.abs().amax(dim=reduce_dims)  # (C,)

    global_absmax = channel_absmax.max().item()
    median_absmax = channel_absmax.median().item()
    mean_absmax = channel_absmax.mean().item()
    ratio = global_absmax / (median_absmax + 1e-12)

    print(f"channels: {channel_absmax.numel()}")
    print(f"channel absmax: max={global_absmax:.4f}, mean={mean_absmax:.4f}, "
          f"median={median_absmax:.4f}")
    print(f"max/median channel absmax ratio: {ratio:.2f}x")

    # How many effective INT8 quantization levels (out of 256, symmetric so
    # ~254 usable) does a TYPICAL channel actually get, once the scale is
    # forced to cover the global (per-tensor) absmax?
    per_tensor_scale = global_absmax / 127.0
    typical_levels = 2 * median_absmax / per_tensor_scale
    print(f"per-tensor scale (from global absmax): {per_tensor_scale:.6f}")
    print(f"effective INT8 levels a TYPICAL (median) channel gets: "
          f"{typical_levels:.1f} / 254")

    top5_vals, top5_idx = torch.topk(channel_absmax, k=min(5, channel_absmax.numel()))
    print(f"top-5 outlier channels (idx: absmax): "
          + ", ".join(f"{i.item()}: {v.item():.3f}" for i, v in zip(top5_idx, top5_vals)))

    if ratio > 10:
        print(f"OUTLIER-DOMINATED: a handful of channels are >{ratio:.0f}x the "
              f"typical channel's range -- per-tensor scale wastes most of its "
              f"resolution on channels that carry little signal.")
    else:
        print("Reasonably balanced across channels -- outlier-channel "
              "explanation does not apply strongly here.")


def main() -> None:
    for model_name, layer_name, note in LAYERS_TO_CHECK:
        inspect_channel_distribution(model_name, layer_name, note)


if __name__ == "__main__":
    main()
