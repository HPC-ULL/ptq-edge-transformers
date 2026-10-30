"""One-off diagnostic (NOT part of the pipeline): is KLMinimizerThreshold's
num_bins default (1001) behaving badly on REAL DeiT-Tiny calibration
activations?

Motivation: calibration_algorithm=entropy collapsed real top1 accuracy from
73.20% (minmax) to 24.25%, while a synthetic-data sanity check of
KLMinimizerThreshold itself (a single injected outlier against a gaussian
bulk) gave a perfectly sensible threshold. This captures a REAL activation
tensor from an actual forward pass over the real calibration images (via a
forward hook on the first nn.Linear layer, before any quantization is
applied) and compares the resulting KL threshold across a few num_bins
values against that same real tensor -- with a calibration batch of 128
images, most of the 1001 default bins spanning [-absmax, absmax] could be
near-empty if a handful of outlier values stretch absmax far beyond where
the bulk of the distribution sits (exactly the scenario the synthetic test
reproduced), which could destabilize the histogram-merging search fewer,
wider bins wouldn't.

Nothing here writes to artifacts/ or results/ -- purely diagnostic.

Run directly:

    python quantization/vit_brevitas/diagnose_entropy_num_bins.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODEL_NAME = "deit_tiny"
CALIB_SET_SIZE = 128
NUM_BINS_TO_TRY = [1001, 255, 127, 63]


def main() -> None:
    import torch
    from brevitas.core.stats.stats_op import KLMinimizerThreshold

    from quantization.vit_brevitas.calibration import load_calibration_batch
    from quantization.vit_brevitas.quantize import load_pretrained_model
    from quantization.vit_brevitas.quantizer_config import _entropy_act_quant

    print(f"Loading {MODEL_NAME} and the calibration pool ({CALIB_SET_SIZE} images)...")
    model = load_pretrained_model(MODEL_NAME)
    calib_batch = load_calibration_batch(model, calib_set_size=CALIB_SET_SIZE, repeat_id=1)

    captured = {}

    def hook(_module, inputs, _output):
        captured["x"] = inputs[0].detach()

    target_name = None
    handle = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            target_name = name
            handle = module.register_forward_hook(hook)
            break

    if handle is None:
        print("FAILED: no nn.Linear found in the model -- nothing can be captured.")
        sys.exit(1)

    print(f"Capturing the real input activation of: {target_name}")
    with torch.no_grad():
        model(calib_batch)
    handle.remove()

    x = captured["x"]
    absmax = x.abs().max().item()
    # Also report the absmax excluding the single most extreme value, as a
    # rough proxy for "how much is one outlier stretching the range".
    flat_abs = x.abs().flatten()
    top2 = torch.topk(flat_abs, k=2).values
    absmax_excl_top1 = top2[1].item()

    print(f"\nCaptured tensor: shape={tuple(x.shape)}, {x.numel()} values")
    print(f"absmax real: {absmax:.4f}")
    print(f"absmax excluding the most extreme value: {absmax_excl_top1:.4f}")
    print(f"(if these two differ a lot, a single value is stretching the histogram range)\n")

    bit_width_impl = _entropy_act_quant().scaling_stats_impl.bit_width_impl

    print("KL threshold per num_bins (same real tensor in all cases):")
    for num_bins in NUM_BINS_TO_TRY:
        kl = KLMinimizerThreshold(signed=True, bit_width_impl=bit_width_impl, num_bins=num_bins, smoothing_eps=0.0001)
        threshold = kl(x)
        print(f"  num_bins={num_bins:>5}: threshold = {threshold.item():.4f}")


if __name__ == "__main__":
    main()
