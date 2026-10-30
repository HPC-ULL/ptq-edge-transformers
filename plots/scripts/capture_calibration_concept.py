"""Capture, on the cluster, the input activations of the MobileViT-S layer
with the highest outlier ratio R (Phase 4, level 1: rank 1, the first of the
two depthwise 3x3 convolutions) and save what the calibration-concept figure
needs to plots/out/calibration_concept_data.npz.

Activations: FP32 timm model, CPU, hook on the INPUT of the layer, over the
128 images of the repetition-1 calibration set, loaded with the ablation's own
quantization.vit_brevitas.calibration.load_calibration_batch (seed 42, same
preprocessing). The ablation calibrates with ONE forward pass over the whole
batch, so the statistics below are over the whole tensor, as it does.

Everything exact (max |x|, the 99.9 percentile of |x|, the clipped fraction) is
computed on the FULL tensor, never on the sample. The file stores those, a
histogram that covers [-max|x|, +max|x|] with an even number of bins (so 0 is
a bin edge), and a random sample of the values kept only for checks.

Brevitas cross-check (skipped with --no-brevitas-check): builds the ablation's
minmax and percentile (99.9) models with build_quantized_model, reads the
scale of the layer's input quantizer, and compares range = scale * 127 with
the exact values. If Brevitas differs from the exact value by more than 1 %,
the Brevitas range is the one used and the log says so.

Resumable: if the output exists nothing is recomputed (move it away to redo).
Everything printed is also written to plots/out/calibration_concept_capture.log.
Run from the repo root:

    python plots/scripts/capture_calibration_concept.py --dry-run
    python plots/scripts/capture_calibration_concept.py
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

MODEL = "mobilevit"
LAYER = "stages.0.0.conv2_kxk.conv"
REPEAT_ID = 1
CALIB_SIZE = 128
CALIB_SEED = 42          # quantization.vit_brevitas.calibration.seed_for_repeat(1)
PERCENTILE_Q = 99.9      # quantizer_config._PERCENTILE_HIGH_Q (checked at run time)
N_BINS = 4000            # even: 0 is an edge; bin width = 2 * max|x| / 4000
SUBSAMPLE_N = 1_000_000
SUBSAMPLE_SEED = 0
CHUNK = 16
BREVITAS_TOL = 0.01
DEFAULT_OUT = _PROJECT_ROOT / "plots" / "out" / "calibration_concept_data.npz"
ENTRY = {"granularity": "per_tensor", "symmetry": "symmetric", "calib_set_size": CALIB_SIZE, "bitwidth": "int8"}


# --- pure numpy part (tested locally) ---------------------------------------

def summarize(x: np.ndarray, q: float = PERCENTILE_Q, n_bins: int = N_BINS,
              n_sub: int = SUBSAMPLE_N, seed: int = SUBSAMPLE_SEED) -> dict:
    """Exact statistics, histogram over [-max|x|, +max|x|] and a sample, all from x."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)
    assert n > 0 and n_bins % 2 == 0
    vmin, vmax = float(x.min()), float(x.max())
    max_abs = max(-vmin, vmax)
    a = np.abs(x)
    pos = q / 100.0 * (n - 1)                                   # numpy "linear" definition
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    k = min(max(int(math.ceil(q / 100.0 * n)), 1), n)           # nearest rank (k-th smallest, 1-based)
    part = np.partition(a, sorted({lo, hi, k - 1}))
    p_lo, p_hi = float(part[lo]), float(part[hi])
    p_linear = p_lo + (pos - lo) * (p_hi - p_lo)
    p_nearest = float(part[k - 1])
    edges = np.linspace(-max_abs, max_abs, n_bins + 1)
    counts = np.histogram(x, bins=edges)[0].astype(np.int64)
    assert int(counts.sum()) == n, "histogram does not cover every value"
    rng = np.random.default_rng(seed)
    sub = x[np.sort(rng.choice(n, size=min(n_sub, n), replace=False))]
    return dict(n_values=n, vmin=vmin, vmax=vmax, mean=float(x.mean(dtype=np.float64)),
                std=float(x.std(dtype=np.float64)), max_abs=max_abs, p_linear=p_linear,
                p_nearest=p_nearest, hist_counts=counts, hist_edges=edges, subsample=sub)


def fraction_above(x: np.ndarray, thr: float, chunk: int = 10_000_000) -> float:
    """Exact fraction of values with |x| > thr, on the full tensor."""
    x = np.asarray(x).reshape(-1)
    over = sum(int((np.abs(x[i:i + chunk]) > thr).sum()) for i in range(0, x.size, chunk))
    return over / x.size


def choose_range(exact: float, brevitas: float, tol: float = BREVITAS_TOL):
    """(range used, source). Brevitas's own value wins if it differs by more than tol."""
    if brevitas is None or not np.isfinite(brevitas):
        return exact, "exact (Brevitas check unavailable)"
    if abs(brevitas - exact) <= tol * exact:
        return exact, f"exact (matches Brevitas within {tol:.0%})"
    return float(brevitas), "Brevitas (differs from the exact value)"


# --- cluster part (torch / timm / Brevitas) ----------------------------------

def capture_activations(device: str = "cpu"):
    """Flat float32 array of the layer's input over the calibration set, and the image-list hash."""
    import torch
    import torch.nn as nn

    from data_prep.imagenet import get_calibration_batch
    from quantization.vit_brevitas.calibration import load_calibration_batch, seed_for_repeat
    from quantization.vit_brevitas.quantize import load_pretrained_model

    assert seed_for_repeat(REPEAT_ID) == CALIB_SEED
    model = load_pretrained_model(MODEL).to(device)
    modules = dict(model.named_modules())
    assert LAYER in modules, f"{LAYER} not in the model; e.g. {[n for n in modules if 'conv2_kxk' in n][:6]}"
    layer = modules[LAYER]
    assert isinstance(layer, nn.Conv2d), type(layer)
    print(f"layer {LAYER}: Conv2d {layer.in_channels}->{layer.out_channels}, kernel {layer.kernel_size}, "
          f"groups {layer.groups} (depthwise: {layer.groups == layer.in_channels})", flush=True)
    batch = load_calibration_batch(model, CALIB_SIZE, REPEAT_ID)
    names = [s["path"] for s in get_calibration_batch(repetition_idx=0, batch_size=CALIB_SIZE, seed=CALIB_SEED)]
    assert batch.shape[0] == CALIB_SIZE == len(names), (batch.shape, len(names))
    sha = hashlib.sha1("\n".join(str(n) for n in names).encode()).hexdigest()
    print(f"calibration batch: {tuple(batch.shape)}, image list sha1 {sha}", flush=True)

    got = []
    handle = layer.register_forward_hook(lambda m, inp, out: got.append(inp[0].detach().float().cpu().numpy()))
    with torch.no_grad():
        for i in range(0, CALIB_SIZE, CHUNK):
            model(batch[i:i + CHUNK].to(device))
    handle.remove()
    x = np.concatenate([g.reshape(-1) for g in got])
    print(f"captured {x.size} values ({len(got)} chunks of up to {CHUNK} images; input shape per image "
          f"{tuple(got[0].shape[1:])})", flush=True)
    return x, sha


def brevitas_input_range(algorithm: str) -> float:
    """range = scale * (2^(b-1) - 1) of the layer's input quantizer in the ablation's own model; NaN on failure."""
    try:
        from quantization.vit_brevitas.quantize import build_quantized_model

        entry = {"model_name": MODEL, "run_id": f"calibration_concept_{algorithm}",
                 "calibration_algorithm": algorithm, **ENTRY}
        qmodel, _ = build_quantized_model(entry, REPEAT_ID)
        qmodel.eval()
        q = dict(qmodel.named_modules())[LAYER].input_quant
        scale = float(q.scale().detach().reshape(-1)[0])
        bits = int(float(q.bit_width()))
        return scale * (2 ** (bits - 1) - 1)
    except Exception as exc:  # noqa: BLE001 -- a failed cross-check must not lose the capture
        print(f"WARNING: Brevitas {algorithm} scale not readable ({type(exc).__name__}: {exc})", flush=True)
        return float("nan")


def versions() -> dict:
    out = {}
    for mod in ("torch", "timm", "brevitas", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = "not available"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cpu", help="forward device (default cpu, as build and calibration)")
    ap.add_argument("--no-brevitas-check", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="check paths and the calibration image list only")
    args = ap.parse_args(argv)

    from quantization.vit_brevitas.paths import tee_output

    log = args.out.with_name("calibration_concept_capture.log")
    with tee_output(log):
        print(f"model {MODEL}, layer {LAYER}, repetition {REPEAT_ID} (seed {CALIB_SEED}), "
              f"{CALIB_SIZE} images, percentile {PERCENTILE_Q}")
        try:
            from quantization.vit_brevitas.quantizer_config import _PERCENTILE_HIGH_Q

            assert _PERCENTILE_HIGH_Q == PERCENTILE_Q, f"ablation percentile is {_PERCENTILE_HIGH_Q}"
        except ImportError as exc:
            print(f"note: quantizer_config not importable here ({exc}); percentile {PERCENTILE_Q} not cross-checked")
        if args.out.exists():
            print(f"{args.out} already exists: nothing to do (move it away to capture again)")
            return 0
        if args.dry_run:
            from data_prep.imagenet import get_calibration_batch

            names = [s["path"] for s in get_calibration_batch(repetition_idx=0, batch_size=CALIB_SIZE, seed=CALIB_SEED)]
            missing = [n for n in names if not os.path.exists(n)]
            print(f"dry run: {len(names)} images selected, {len(missing)} missing on disk; output would be {args.out}")
            return 1 if missing else 0

        x, sha = capture_activations(args.device)
        s = summarize(x)
        print(f"max|x| {s['max_abs']:.6g}; percentile {PERCENTILE_Q}: linear {s['p_linear']:.6g}, "
              f"nearest rank {s['p_nearest']:.6g}", flush=True)
        b_mm = b_p = float("nan")
        if not args.no_brevitas_check:
            b_mm, b_p = brevitas_input_range("minmax"), brevitas_input_range("percentile")
            print(f"Brevitas ranges (scale*127): minmax {b_mm:.6g} (exact {s['max_abs']:.6g}), "
                  f"percentile {b_p:.6g} (exact {s['p_linear']:.6g})", flush=True)
        r_mm, src_mm = choose_range(s["max_abs"], b_mm)
        r_p, src_p = choose_range(s["p_linear"], b_p)
        clipped = fraction_above(x, r_p)
        arrays = dict(
            layer=LAYER, model=MODEL, repeat_id=REPEAT_ID, calib_seed=CALIB_SEED, calib_set_size=CALIB_SIZE,
            percentile_q=PERCENTILE_Q, images_sha1=sha, subsample_seed=SUBSAMPLE_SEED,
            versions="; ".join(f"{k} {v}" for k, v in versions().items()),
            brevitas_range_minmax=b_mm, brevitas_range_percentile=b_p,
            range_minmax=r_mm, range_minmax_source=src_mm, range_percentile=r_p, range_percentile_source=src_p,
            step_minmax=r_mm / 127, step_percentile=r_p / 127, step_ratio=r_mm / r_p,
            clipped_fraction=clipped, clipped_fraction_linear=fraction_above(x, s["p_linear"]),
            clipped_fraction_nearest=fraction_above(x, s["p_nearest"]), **s)
        tmp = args.out.with_name(args.out.name + ".tmp")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, args.out)
        print(f"MinMax range {r_mm:.6g} ({src_mm}), step {r_mm / 127:.6g}\n"
              f"Percentile range {r_p:.6g} ({src_p}), step {r_p / 127:.6g}\n"
              f"step ratio {r_mm / r_p:.3f}; clipped by the percentile range: {clipped:.4%} of {s['n_values']} values\n"
              f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
