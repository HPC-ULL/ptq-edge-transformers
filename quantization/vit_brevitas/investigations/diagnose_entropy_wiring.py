"""One-off diagnostic (NOT part of the pipeline): is calibration_algorithm=
entropy's severe, consistent collapse on deit_tiny (24-27% top1 in 2/3
reps, 12.85% in the third, vs. ~73% everywhere else) a real finding, or a
bug in how KLMinimizerThreshold is wired into the real pipeline?

The earlier investigation (diagnose_entropy_num_bins.py, now in this same
directory) only validated the ALGORITHM in isolation -- manually
constructing a fresh KLMinimizerThreshold and feeding it a captured real
tensor, outside of Brevitas's own calibration_mode() machinery. It never
verified that the REAL pipeline (quantizer_config.py's _entropy_act_quant,
composed via ExtendedInjector's dependency-injection framework) actually
wires calibration data INTO that instance correctly. That's the gap this
checks, directly:

  1. WIRING: (a) introspect the real, internally-instantiated
     KLMinimizerThreshold's actual num_bins/smoothing_eps/signed
     attributes -- do they match the assumed defaults (1001/0.0001), or
     did the ExtendedInjector composition silently produce something else?
     (b) decisive test: build the SAME entropy-quantized model but with
     the calibration_mode() forward pass skipped entirely (never shown any
     calibration data) and compare its resolved scale against the properly
     -calibrated one -- if identical, calibration data never reached the
     quantizer at all. (c) compare the real pipeline's resolved scale
     against a fresh, standalone KLMinimizerThreshold call on the exact
     same captured calibration-batch tensor -- if they diverge, the
     wiring/execution (not just the algorithm) is where the bug is.
  2. PARAMETERS vs. REAL RANGE: report the captured real activation's
     absmax/mean/std alongside the resulting threshold, to see whether the
     defaults produce a degenerate (near-zero or absurdly large) clipping
     range for this specific tensor.
  3. SCALE COMPARISON: minmax / percentile / entropy resolved scales, same
     layer, same calibration batch -- an order-of-magnitude difference
     confirms a calculation bug rather than "just a worse threshold
     choice".

Needs: Brevitas installed, the gated ImageNet-1K pool already prepared.

Run directly:

    python quantization/vit_brevitas/investigations/diagnose_entropy_wiring.py
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


def _capture_input(model, layer_name, tensor):
    import torch

    captured = {}

    def hook(_module, inputs, _output):
        captured["x"] = inputs[0].detach()

    target = dict(model.named_modules())[layer_name]
    handle = target.register_forward_hook(hook)
    with torch.no_grad():
        model(tensor)
    handle.remove()
    return captured["x"], target


def _resolved_scale(target, x):
    import torch

    with torch.no_grad():
        q = target.input_quant(x)
    return q.scale.flatten()[0].item(), q.zero_point.flatten()[0].item(), q.bit_width


def build_model(model_name, calibration_algorithm, repeat_id, *, skip_calibration=False):
    """Mirrors quantize.py's build_quantized_model, with an option to skip
    the calibration_mode() forward pass entirely (for the "never saw
    calibration data" control)."""
    import torch
    from brevitas.graph.calibrate import calibration_mode

    from quantization.vit_brevitas.calibration import load_calibration_batch
    from quantization.vit_brevitas.quantize import (
        load_pretrained_model,
        quantize_module_tree,
        resolve_input_shape,
    )
    from quantization.vit_brevitas.quantizer_config import resolve_act_quant, resolve_weight_quant

    model = load_pretrained_model(model_name)
    input_shape = resolve_input_shape(model)

    weight_quant = resolve_weight_quant(calibration_algorithm, "per_tensor")
    act_quant = resolve_act_quant(calibration_algorithm, "symmetric")

    calib_batch = load_calibration_batch(model, CALIB_SET_SIZE, repeat_id)

    quantize_module_tree(model, weight_quant, act_quant)
    model.eval()

    if not skip_calibration:
        with torch.no_grad(), calibration_mode(model):
            model(calib_batch)

    return model, calib_batch


def main() -> None:
    import torch
    from brevitas.core.stats.stats_op import KLMinimizerThreshold

    print(f"=== 1(a): introspect the real KLMinimizerThreshold instance's actual parameters ===")
    entropy_model, calib_batch = build_model(MODEL_NAME, "entropy", REPEAT_ID)
    target = dict(entropy_model.named_modules())[LAYER_NAME]
    found = []
    for m in target.input_quant.modules():
        if isinstance(m, KLMinimizerThreshold):
            found.append(m)
            print(f"  found KLMinimizerThreshold: num_bins={m.num_bins}, "
                  f"smoothing_eps={m.smoothing_eps}, signed={getattr(m, 'signed', '?')}")
    if not found:
        print("  NOT FOUND -- input_quant's module tree does not contain a KLMinimizerThreshold "
              "instance at all. This alone would be a direct wiring bug.")

    print(f"\n=== 1(b): calibrated vs. calibration-SKIPPED (never saw calibration data) ===")
    x, target_calibrated = _capture_input(entropy_model, LAYER_NAME, calib_batch)
    scale_calibrated, zp_calibrated, bw_calibrated = _resolved_scale(target_calibrated, x)
    print(f"  properly calibrated: scale={scale_calibrated:.8f}, zero_point={zp_calibrated}, "
          f"bit_width={bw_calibrated}")

    skipped_model, _ = build_model(MODEL_NAME, "entropy", REPEAT_ID, skip_calibration=True)
    target_skipped = dict(skipped_model.named_modules())[LAYER_NAME]
    scale_skipped, zp_skipped, bw_skipped = _resolved_scale(target_skipped, x)
    print(f"  calibration SKIPPED:  scale={scale_skipped:.8f}, zero_point={zp_skipped}, "
          f"bit_width={bw_skipped}")
    if abs(scale_calibrated - scale_skipped) < 1e-9:
        print("  MATCH: calibrated and never-calibrated scales are IDENTICAL -- calibration data "
              "is NOT reaching the quantizer. This is a direct, confirmed wiring bug.")
    else:
        print("  DIFFERENT: calibration data does appear to be reaching the quantizer (scales "
          "differ) -- wiring is not trivially broken; proceed to check 1(c)/2/3.")

    print(f"\n=== 1(c): real pipeline's resolved threshold vs. a fresh, standalone "
          f"KLMinimizerThreshold call on the SAME captured tensor ===")
    print(f"  captured real input to {LAYER_NAME}: shape={tuple(x.shape)}, "
          f"absmax={x.abs().max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
    if found:
        bit_width_impl = found[0].bit_width_impl
        manual_kl = KLMinimizerThreshold(
            signed=True, bit_width_impl=bit_width_impl, num_bins=1001, smoothing_eps=0.0001
        )
        manual_threshold = manual_kl(x.float()).item()
        manual_scale = manual_threshold / 127.0
        print(f"  standalone KLMinimizerThreshold(num_bins=1001, smoothing_eps=0.0001): "
              f"threshold={manual_threshold:.4f} -> equivalent scale={manual_scale:.8f}")
        print(f"  real pipeline's resolved scale: {scale_calibrated:.8f}")
        ratio = scale_calibrated / manual_scale if manual_scale else float("inf")
        print(f"  ratio (real / standalone): {ratio:.4f}x")
        if not (0.5 < ratio < 2.0):
            print("  MISMATCH: real pipeline's scale differs substantially from a standalone "
                  "reproduction on the identical tensor -- points at an execution/wiring "
                  "discrepancy, not just an algorithm/parameter choice.")
        else:
            print("  MATCH: real pipeline reproduces the standalone calculation closely -- "
                  "wiring and execution are faithful; any degeneracy is in the algorithm/"
                  "parameters themselves, not how they're invoked.")

    print(f"\n=== 3: scale comparison across calibration algorithms, same layer, same calib batch ===")
    for algo in ("minmax", "percentile", "entropy"):
        model, batch = build_model(MODEL_NAME, algo, REPEAT_ID)
        x_algo, target_algo = _capture_input(model, LAYER_NAME, batch)
        scale, zp, bw = _resolved_scale(target_algo, x_algo)
        print(f"  {algo:<12} scale={scale:.8f}  zero_point={zp}  bit_width={bw}")


if __name__ == "__main__":
    main()
