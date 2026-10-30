"""One-off diagnostic (NOT part of the pipeline): follow-up to
diagnose_entropy_num_bins_sweep.py, which found that every EVEN num_bins
tested crashes with an off-by-one IndexError in Brevitas 0.13.0's
KLMinimizerThreshold.forward() (num_bins // 2 + num_bins // 2 == num_bins
exactly when num_bins is even, pushing the histogram-edge index one past
the last valid position; for odd num_bins, floor-division truncation keeps
it in bounds) -- num_bins=1001 (the library default) only "worked" because
it happens to be odd, not because it was validated as a good choice.

This sweeps several ODD num_bins values (which dodge the parity bug) on
the SAME captured tensor (blocks.0.attn.qkv, deit_tiny, entropy-calibrated,
calib_set_size=128, repeat_id=1) to see whether the severe clipping already
seen at 1001 (threshold/absmax=14.89%) is a stable property of the
algorithm for this activation distribution (consistent behavior across odd
values -> real finding, document and don't block) or itself unstable
independent of the parity bug (wildly different results across odd values
-> block with NotImplementedError for unreliability, not for the crash).

Needs: Brevitas installed, the gated ImageNet-1K pool already prepared.

Run directly:

    python quantization/vit_brevitas/investigations/diagnose_entropy_num_bins_odd_sweep.py
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
NUM_BINS_TO_TRY = [501, 1001, 2001, 3001, 5001, 8001]


def main() -> None:
    import statistics

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
        print("Could not locate the real KLMinimizerThreshold instance -- aborting.")
        return

    print(f"{'num_bins':>10} {'threshold':>12} {'scale':>12} {'threshold/absmax':>18}")
    ratios = []
    for num_bins in NUM_BINS_TO_TRY:
        assert num_bins % 2 == 1, f"{num_bins} is even -- would hit the known parity bug"
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
        ratios.append(ratio)
        marker = "  <- previously tested" if num_bins == 1001 else ""
        print(f"{num_bins:>10} {threshold:>12.4f} {scale:>12.6f} {ratio:>17.2%}{marker}")

    if len(ratios) >= 2:
        spread = max(ratios) - min(ratios)
        print(f"\nthreshold/absmax across odd num_bins: min={min(ratios):.2%}, max={max(ratios):.2%}, "
              f"spread={spread:.2%}, stdev={statistics.stdev(ratios):.2%}")
        if spread < 0.10:
            print("=> STABLE across odd num_bins: consistent severe clipping is a genuine property "
                  "of the algorithm for this activation distribution -- real finding, don't block.")
        else:
            print("=> UNSTABLE across odd num_bins: results vary substantially even avoiding the "
                  "parity bug -- the method is unreliable here independent of the crash; consider "
                  "blocking with NotImplementedError for unreliability, not for the crash.")


if __name__ == "__main__":
    main()
