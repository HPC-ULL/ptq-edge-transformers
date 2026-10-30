"""One-off diagnostic (NOT part of the pipeline): does KLMinimizerThreshold's
resulting threshold for deit_tiny's blocks.0.attn.qkv converge toward the
real absmax as num_bins increases (a fixable hyperparameter issue), or does
it stay degenerate (far below absmax) regardless of num_bins (evidence the
method itself doesn't suit this activation distribution, consistent with
PTQ4ViT's findings on ViT activation outliers -- a real result, not a bug)?

Follow-up to diagnose_entropy_wiring.py, which already ruled out a wiring
bug for this same tensor/layer (calibrated vs. calibration-skipped scales
differ; the real pipeline's resolved scale matches a standalone
KLMinimizerThreshold reproduction closely) and reported the current
default (num_bins=1001, smoothing_eps=0.0001) giving threshold~1.5546 /
scale~0.01378 against a real absmax of 10.44.

This captures the SAME tensor the same way (entropy-calibrated deit_tiny,
blocks.0.attn.qkv, calib_set_size=128, repeat_id=1 -- deterministic given
the same seed, so this should reproduce ~1.5546 at num_bins=1001) and
sweeps num_bins across a wider range with smoothing_eps held fixed, to see
the trend rather than a single data point.

Needs: Brevitas installed, the gated ImageNet-1K pool already prepared.

Run directly:

    python quantization/vit_brevitas/investigations/diagnose_entropy_num_bins_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODEL_NAME = "deit_tiny"
LAYER_NAME = "blocks.0.attn.qkv"
CALIB_SET_SIZE = 128
REPEAT_ID = 1
SMOOTHING_EPS = 0.0001
NUM_BINS_TO_TRY = [256, 384, 512, 750, 1001, 1500, 2048, 4096]


def main() -> None:
    import torch
    from brevitas.core.stats.stats_op import KLMinimizerThreshold

    from diagnose_entropy_wiring import build_model  # same directory

    model, calib_batch = build_model(MODEL_NAME, "entropy", REPEAT_ID)
    target = dict(model.named_modules())[LAYER_NAME]

    captured = {}

    def hook(_module, inputs, _output):
        captured["x"] = inputs[0].detach()

    handle = target.register_forward_hook(hook)
    with torch.no_grad():
        model(calib_batch)
    handle.remove()

    x = captured["x"].float()
    absmax = x.abs().max().item()
    print(f"captured tensor: shape={tuple(x.shape)}, absmax={absmax:.4f}\n")

    bit_width_impl = None
    for m in target.input_quant.modules():
        if isinstance(m, KLMinimizerThreshold):
            bit_width_impl = m.bit_width_impl
            break
    if bit_width_impl is None:
        print("Could not locate the real KLMinimizerThreshold instance to reuse its "
              "bit_width_impl -- aborting.")
        return

    print(f"{'num_bins':>10} {'threshold':>12} {'scale':>12} {'threshold/absmax':>18}")
    for num_bins in NUM_BINS_TO_TRY:
        kl = KLMinimizerThreshold(
            signed=True, bit_width_impl=bit_width_impl, num_bins=num_bins, smoothing_eps=SMOOTHING_EPS
        )
        try:
            threshold = kl(x).item()
        except Exception as e:
            print(f"{num_bins:>10}  FAILED: {type(e).__name__}: {e}")
            continue
        scale = threshold / 127.0
        ratio = threshold / absmax
        marker = "  <- current default" if num_bins == 1001 else ""
        print(f"{num_bins:>10} {threshold:>12.4f} {scale:>12.6f} {ratio:>17.2%}{marker}")

    print(f"\nreal absmax: {absmax:.4f}")
    print("If threshold/absmax climbs toward 100% as num_bins increases: hyperparameter issue, "
          "adjustable (more bins would fix it).")
    print("If threshold/absmax stays low (well under 50%) across the whole sweep: the method "
          "itself doesn't suit this activation distribution -- a real finding, not a bug.")


if __name__ == "__main__":
    main()
