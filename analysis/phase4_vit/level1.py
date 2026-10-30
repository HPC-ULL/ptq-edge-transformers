"""Phase 4, step 1 (Level 1): per-layer activation outlier ratio and local
quantization error, ranking, and Spearman correlations. Pre-registered rules:
this package's __init__ docstring; nothing here may change them.

Build and calibration on CPU, exactly as the pipeline; the forward passes
(FP32 model and the quantized layers applied to its inputs) on the GPU when
there is one -- both recorded in run.log.

Per model, one FP32 pass over the 512 Level 1 images with a forward hook on
every quantized-layer position (the step-0 inventory). Each hook takes the
layer's FP32 input x and FP32 output y and
  - accumulates per-channel absmax of x (channel: last axis for Linear,
    dim 1 for Conv2d), global min/max and the moments for the kurtosis
    (central sums accumulated in float64);
  - applies the SAME layer from the baseline-quantized model (MinMax,
    per-tensor, symmetric, INT8, rep1 calibration: quantize.
    build_quantized_model, unmodified) to x, and accumulates the per-image
    relative L2 error ||y_q - y|| / ||y||.
Upstream layers are float, so this is exactly "only this layer quantized".

R = max_c(absmax_c) / (median_c(absmax_c) + 1e-8). Ranked: >= 8 input
channels and median >= 1e-6; descending R, ties by execution order.
Kurtosis (excess, pooled over every input element of the 512 images) and
range (max - min) are secondary, exploratory: recorded, not correlated.

Checks, any one failing makes the run exit non-zero (no correlations are
reported from a run that fails them):
  - the 512 images are distinct and disjoint from the calibration samples
    of seeds 42/43/44 at size 512 (which contain sizes 32 and 128);
  - the model's Linear/Conv2d set equals the inventory's, and every one is
    quantized in the built model;
  - per-image grouping, run before the pass: the first and the last image
    of a batch of 16 (Level 1 images 1 and 16), each computed alone and
    inside that batch, give the same error at EVERY layer (relative
    tolerance 1 %: batch-1 and batch-16 GPU kernels may flip a few
    quantization codes; mixing images would not stay within it). On failure
    the model's pass is skipped;
  - every layer saw 512 images, with finite errors and ||y|| > 0.
Degenerate ranking (fewer than 12 ranked, or more than half sharing the same
R to 6 significant figures) is reported; it stops Level 2, not this step.

Output: results/investigations/vit/phase4/level1/{images.csv,
layers_<model>.csv, correlations.csv, checks.csv, run.log}. Run from the
repo root:

    python -m analysis.phase4_vit.level1 --dry-run
    python -m analysis.phase4_vit.level1
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_vit import MODELS, PHASE4_ROOT  # noqa: E402
from analysis.phase4_vit.inventory import OUT_DIR as INVENTORY_DIR  # noqa: E402
from analysis.phase4_vit.inventory import TYPES  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

OUT_DIR = PHASE4_ROOT / "level1"

N_IMAGES = 512
IMAGE_SEED = 0
EXCLUDED_CALIB_SEEDS = (42, 43, 44)  # calibration.seed_for_repeat(1..3)
EXCLUDED_CALIB_SIZE = 512  # largest calib_set_size; sizes 32/128 are its prefixes
EPS = 1e-8
MIN_MEDIAN = 1e-6
MIN_RANKED = 12
MIN_TYPE_N = 10
R_SIG_FIGS = 6
HOLM_ALPHA = 0.05
GROUPING_RTOL = 1e-2  # a wrong grouping mixes images: far larger than kernel noise
CHECK_BATCH = 16
REPEAT_ID = 1
BASELINE_INT8 = {"calibration_algorithm": "minmax", "granularity": "per_tensor",
                 "symmetry": "symmetric", "calib_set_size": 128, "bitwidth": "int8"}

LAYER_COLUMNS = [
    "model", "exec_idx", "name", "kind", "layer_type", "layer_type_alt", "in_channels", "rankable_channels",
    "absmax_max", "absmax_median", "R", "ranked", "exclusion", "rank", "local_error_mean", "local_error_n",
    "kurtosis_excess", "act_min", "act_max", "act_range",
]
CORR_COLUMNS = ["model", "classification", "scope", "layer_type", "n", "rho", "p", "p_holm", "holm_reject", "status"]


# -- image selection ---------------------------------------------------------

def select_image_indices(n_pool: int, excluded: Iterable[int], n: int = N_IMAGES, seed: int = IMAGE_SEED) -> List[int]:
    """n pool indices outside `excluded`, drawn with a fixed seed, sorted."""
    candidates = sorted(set(range(n_pool)) - set(excluded))
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} candidate images, need {n}")
    return sorted(random.Random(seed).sample(candidates, n))


def level1_images() -> Tuple[List[Dict], set]:
    """(images, excluded filenames): the Level 1 images from the calibration
    bank, and the calibration samples they must avoid, taken from the
    pipeline's own sampler (data_prep.imagenet.get_calibration_batch)."""
    from data_prep.imagenet import calibration_pool_dir, get_calibration_batch, get_calibration_manifest

    rows = get_calibration_manifest()
    excluded = {s["path"].name for seed in EXCLUDED_CALIB_SEEDS
                for s in get_calibration_batch(repetition_idx=0, batch_size=EXCLUDED_CALIB_SIZE, seed=seed)}
    index_of = {r["filename"]: i for i, r in enumerate(rows)}
    picked = select_image_indices(len(rows), {index_of[f] for f in excluded})
    images_dir = calibration_pool_dir() / "images"
    images = [{"pool_idx": i, "filename": rows[i]["filename"], "path": images_dir / rows[i]["filename"]}
              for i in picked]
    return images, excluded


# -- statistics (pure python, testable without torch) ------------------------

Moments = Tuple[int, float, float, float, float]  # n, mean, M2, M3, M4 (central sums)


def merge_moments(a: Moments, b: Moments) -> Moments:
    """Combine central moment sums of two disjoint samples (Pebay 2008)."""
    na, ma, m2a, m3a, m4a = a
    nb, mb, m2b, m3b, m4b = b
    if na == 0:
        return b
    if nb == 0:
        return a
    n = na + nb
    d = mb - ma
    mean = ma + d * nb / n
    m2 = m2a + m2b + d * d * na * nb / n
    m3 = m3a + m3b + d ** 3 * na * nb * (na - nb) / n ** 2 + 3 * d * (na * m2b - nb * m2a) / n
    m4 = (m4a + m4b + d ** 4 * na * nb * (na * na - na * nb + nb * nb) / n ** 3
          + 6 * d * d * (na * na * m2b + nb * nb * m2a) / n ** 2 + 4 * d * (na * m3b - nb * m3a) / n)
    return n, mean, m2, m3, m4


def excess_kurtosis(m: Moments) -> float:
    n, _, m2, _, m4 = m
    return n * m4 / (m2 * m2) - 3.0 if m2 > 0 else float("nan")


def outlier_ratio(absmax_max: float, absmax_median: float) -> float:
    return absmax_max / (absmax_median + EPS)


def rank_layers(rows: List[Dict]) -> List[Dict]:
    """Sets ranked/exclusion/rank in place; returns the ranked rows in order
    (descending R, ties by execution order)."""
    for r in rows:
        r["rank"] = ""
        if not r["rankable_channels"]:
            r["ranked"], r["exclusion"] = False, "< 8 input channels"
        elif r["absmax_median"] < MIN_MEDIAN:
            r["ranked"], r["exclusion"] = False, f"median absmax < {MIN_MEDIAN:g}"
        else:
            r["ranked"], r["exclusion"] = True, ""
    ranked = sorted((r for r in rows if r["ranked"]), key=lambda r: (-r["R"], r["exec_idx"]))
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    return ranked


def degenerate_reasons(ranked: Sequence[Dict]) -> List[str]:
    reasons = []
    if len(ranked) < MIN_RANKED:
        reasons.append(f"{len(ranked)} ranked layers (< {MIN_RANKED})")
    if ranked:
        value, count = Counter(float(f"{r['R']:.{R_SIG_FIGS}g}") for r in ranked).most_common(1)[0]
        if count > len(ranked) / 2:
            reasons.append(f"{count} of {len(ranked)} ranked layers share R = {value:.{R_SIG_FIGS}g}")
    return reasons


def holm(pvalues: Sequence[float], alpha: float = HOLM_ALPHA) -> Tuple[List[float], List[bool]]:
    """Holm step-down: adjusted p-values (monotone, capped at 1) and rejections."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * pvalues[i]))
        adjusted[i] = running
    return adjusted, [p <= alpha for p in adjusted]


def correlations(model: str, ranked: Sequence[Dict], spearman) -> List[Dict]:
    """Global (primary) and per-type rows for one model; `spearman(x, y)`
    returns (rho, p). Per-type with the main classification, and, when the
    model has a different one, with layer_type_alt."""
    out = []
    rho, p = spearman([r["R"] for r in ranked], [r["local_error_mean"] for r in ranked])
    out.append({"model": model, "classification": "main", "scope": "global", "layer_type": "all",
                "n": len(ranked), "rho": rho, "p": p, "status": "primary"})
    columns = [("main", "layer_type")]
    if any(r["layer_type_alt"] != r["layer_type"] for r in ranked):
        columns.append(("alt", "layer_type_alt"))
    for label, column in columns:
        for t in TYPES:
            group = [r for r in ranked if r[column] == t]
            row = {"model": model, "classification": label, "scope": "type", "layer_type": t, "n": len(group),
                   "rho": "", "p": "", "status": "listed (n < 10)" if len(group) < MIN_TYPE_N else "descriptive"}
            if len(group) >= MIN_TYPE_N:
                row["rho"], row["p"] = spearman([r["R"] for r in group], [r["local_error_mean"] for r in group])
            if group:
                out.append(row)
    return out


def apply_holm(rows: List[Dict]) -> None:
    primary = [r for r in rows if r["status"] == "primary"]
    adjusted, reject = holm([r["p"] for r in primary])
    for r, a, rej in zip(primary, adjusted, reject):
        r["p_holm"], r["holm_reject"] = a, rej
    for r in rows:
        r.setdefault("p_holm", "")
        r.setdefault("holm_reject", "")


# -- the FP32 pass (torch) ---------------------------------------------------

class LayerAccumulator:
    def __init__(self, kind: str):
        self.kind = kind
        self.absmax = None
        self.min = math.inf
        self.max = -math.inf
        self.moments: Moments = (0, 0.0, 0.0, 0.0, 0.0)
        self.err_sum = 0.0
        self.err_n = 0
        self.bad = 0  # non-finite error or ||y|| == 0

    def channel_absmax(self, x):
        if self.kind == "Linear":
            return x.reshape(-1, x.shape[-1]).abs().amax(dim=0)
        return x.transpose(0, 1).reshape(x.shape[1], -1).abs().amax(dim=1)

    def update_input(self, x) -> None:
        cmax = self.channel_absmax(x).double()
        self.absmax = cmax if self.absmax is None else self.absmax.maximum(cmax)
        self.min = min(self.min, x.min().item())
        self.max = max(self.max, x.max().item())
        x64 = x.double().flatten()
        mean = x64.mean()
        d = x64 - mean
        d2 = d * d
        batch = (x64.numel(), mean.item(), d2.sum().item(), (d2 * d).sum().item(), (d2 * d2).sum().item())
        self.moments = merge_moments(self.moments, batch)

    def update_error(self, errors: List[float]) -> None:
        for e in errors:
            if math.isfinite(e):
                self.err_sum += e
                self.err_n += 1
            else:
                self.bad += 1


def per_image_errors(y_q, y, batch_size: int) -> List[float]:
    """||y_q - y|| / ||y|| per image. The batch index must be the outermost
    factor of dim 0 (true for (B, ...) and for MobileViT's unfolded
    (B * patch_area, N, C)); checked by the grouping check."""
    if y.shape[0] % batch_size:
        raise RuntimeError(f"dim 0 = {y.shape[0]} not a multiple of the batch size {batch_size}")
    diff = (y_q.float() - y.float()).reshape(batch_size, -1).norm(dim=1)
    ref = y.float().reshape(batch_size, -1).norm(dim=1)
    return (diff / ref).tolist()  # ||y|| == 0 -> inf/nan -> counted as bad


def grouping_problems(names: Sequence[str], check: Dict[str, Dict[str, Tuple[float, float]]]) -> List[str]:
    """Grouping check: per layer, the first image alone vs. first in the
    batch, and the last image alone vs. last in the batch, within
    GROUPING_RTOL. `check[mode][layer] = (error of first, error of last)`
    for mode in alone_first / alone_last / batch."""
    problems = []
    for name in names:
        try:
            pairs = [(check["alone_first"][name][0], check["batch"][name][0]),
                     (check["alone_last"][name][0], check["batch"][name][1])]
        except KeyError:
            problems.append(f"{name}: grouping check missing")
            continue
        for which, (alone, in_batch) in zip(("first", "last"), pairs):
            if not (math.isfinite(alone) and math.isfinite(in_batch)
                    and math.isclose(alone, in_batch, rel_tol=GROUPING_RTOL)):
                problems.append(f"{name}: {which} image alone {alone:.6g} vs in batch {in_batch:.6g} (grouping)")
    return problems


def run_model(model_name: str, inventory_rows: List[Dict], images: List[Dict], device: str,
              batch_size: int) -> Tuple[List[Dict], List[str]]:
    import torch
    import torch.nn as nn
    from brevitas.nn import QuantConv2d, QuantLinear
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    from quantization.vit_brevitas.quantize import build_quantized_model, load_pretrained_model

    problems: List[str] = []
    fp32 = load_pretrained_model(model_name)
    preprocess = create_transform(**resolve_data_config({}, model=fp32))
    qmodel, _ = build_quantized_model({"model_name": model_name, "run_id": f"phase4_level1_{model_name}",
                                       **BASELINE_INT8}, REPEAT_ID)  # CPU, as the pipeline
    qmodel.eval()

    by_name = {r["name"]: r for r in inventory_rows}
    fp32_layers = {n: m for n, m in fp32.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))}
    q_modules = dict(qmodel.named_modules())
    if set(fp32_layers) != set(by_name):
        problems.append(f"model layers differ from the inventory: {sorted(set(fp32_layers) ^ set(by_name))}")
        return [], problems
    not_quant = [n for n in by_name if not isinstance(q_modules.get(n), (QuantLinear, QuantConv2d))]
    if not_quant:
        problems.append(f"not quantized in the built model: {not_quant}")
        return [], problems

    print(f"   build + calibration: cpu; forward: {device}", flush=True)
    fp32.to(device)
    qmodel.to(device)
    acc = {n: LayerAccumulator(by_name[n]["kind"]) for n in by_name}
    state: Dict = {"mode": "accumulate", "bsz": batch_size, "check": {}}

    def make_hook(name):
        qmod = q_modules[name]

        def hook(_module, inputs, output):
            x = inputs[0].detach()
            errors = per_image_errors(qmod(x), output.detach(), state["bsz"])
            if state["mode"] != "accumulate":  # grouping check: record, accumulate nothing
                state["check"].setdefault(state["mode"], {})[name] = (errors[0], errors[-1])
                return
            acc[name].update_input(x)
            acc[name].update_error(errors)
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in fp32_layers.items()]

    def load(batch):
        return torch.stack([preprocess(Image.open(s["path"]).convert("RGB")) for s in batch]).to(device)

    try:
        with torch.no_grad():
            check_batch = images[:CHECK_BATCH]
            for mode, batch in (("alone_first", check_batch[:1]), ("alone_last", check_batch[-1:]),
                                ("batch", check_batch)):
                state.update(mode=mode, bsz=len(batch))
                fp32(load(batch))
            grouping = grouping_problems(list(by_name), state["check"])
            print(f"   grouping check (images 1 and {CHECK_BATCH}, every layer): "
                  f"{'OK' if not grouping else f'FAILED in {len(grouping)} layer(s)'}", flush=True)
            if grouping:
                return [], problems + grouping
            state.update(mode="accumulate")
            for start in range(0, len(images), batch_size):
                batch = images[start:start + batch_size]
                state["bsz"] = len(batch)
                fp32(load(batch))
                if start // batch_size % 8 == 0:
                    print(f"   {model_name}: {start + len(batch)}/{len(images)} images", flush=True)
    finally:
        for h in handles:
            h.remove()

    for name in by_name:
        a = acc[name]
        if a.err_n != len(images) or a.bad:
            problems.append(f"{name}: {a.err_n} finite errors of {len(images)}, {a.bad} non-finite or ||y||=0")

    rows = []
    for name, inv_row in by_name.items():
        a = acc[name]
        absmax_max = a.absmax.max().item()
        absmax_median = a.absmax.quantile(0.5).item()  # standard median (torch.median is the lower one)
        rows.append({
            "model": model_name, "exec_idx": int(inv_row["exec_idx"]), "name": name, "kind": inv_row["kind"],
            "layer_type": inv_row["layer_type"], "layer_type_alt": inv_row["layer_type_alt"],
            "in_channels": int(inv_row["in_channels"]),
            "rankable_channels": inv_row["rankable_channels"] in (True, "True"),
            "absmax_max": absmax_max, "absmax_median": absmax_median,
            "R": outlier_ratio(absmax_max, absmax_median),
            "local_error_mean": a.err_sum / a.err_n if a.err_n else float("nan"), "local_error_n": a.err_n,
            "kurtosis_excess": excess_kurtosis(a.moments), "act_min": a.min, "act_max": a.max,
            "act_range": a.max - a.min,
        })
    rows.sort(key=lambda r: r["exec_idx"])
    return rows, problems


# -- driver ------------------------------------------------------------------

def read_inventory(path: Path) -> Dict[str, List[Dict]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        out.setdefault(r["model"], []).append(r)
    return out


def _write_csv(path: Path, rows: List[Dict], columns: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _spearman(x, y):
    from scipy.stats import spearmanr

    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--inventory", type=Path, default=INVENTORY_DIR / "layers.csv")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true",
                        help="check inventory and image selection, load no model")
    args = parser.parse_args(argv)

    inventory = read_inventory(args.inventory)
    missing = [m for m in args.models if m not in inventory]
    images, excluded = level1_images()
    overlap = {s["filename"] for s in images} & excluded
    print(f"inventory: {args.inventory} ({', '.join(f'{m}: {len(inventory.get(m, []))}' for m in args.models)})")
    print(f"images: {len(images)} from the calibration bank, seed {IMAGE_SEED}; excluded calibration samples: "
          f"{len(excluded)} (seeds {EXCLUDED_CALIB_SEEDS}, size {EXCLUDED_CALIB_SIZE}); overlap: {len(overlap)}")
    problems = [f"model {m} not in the inventory" for m in missing]
    if len({s['filename'] for s in images}) != N_IMAGES or overlap:
        problems.append("image selection: not 512 distinct images disjoint from the calibration samples")
    if (args.out_dir / "correlations.csv").exists():
        problems.append(f"{args.out_dir / 'correlations.csv'} already exists: move it away or use --out-dir")
    if args.dry_run or problems:
        for p in problems:
            print(f"PROBLEM: {p}")
        return 1 if problems else 0

    device = args.device or _default_device()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(args.out_dir / "run.log"):
        print(f"forward device: {device} (build + calibration: cpu); batch size {args.batch_size}")
        _write_csv(args.out_dir / "images.csv", images, ["pool_idx", "filename"])
        corr_rows, checks = [], []
        for model in args.models:
            print(f"\n== {model}", flush=True)
            rows, model_problems = run_model(model, inventory[model], images, device, args.batch_size)
            problems += [f"{model}: {p}" for p in model_problems]
            checks.append({"model": model, "status": "OK" if not model_problems else "FAILED",
                           "problems": " | ".join(model_problems)})
            if not rows:
                continue
            ranked = rank_layers(rows)
            _write_csv(args.out_dir / f"layers_{model}.csv", rows, LAYER_COLUMNS)
            degenerate = degenerate_reasons(ranked)
            print(f"   ranked {len(ranked)} of {len(rows)}; excluded: "
                  + (", ".join(f"{r['name']} ({r['exclusion']})" for r in rows if not r["ranked"]) or "none"))
            print(f"   degenerate ranking: {'; '.join(degenerate) if degenerate else 'no'}")
            checks[-1]["degenerate"] = "; ".join(degenerate)
            if not model_problems:
                corr_rows += correlations(model, ranked, _spearman)
        _write_csv(args.out_dir / "checks.csv", checks, ["model", "status", "problems", "degenerate"])
        if problems:
            for p in problems:
                print(f"PROBLEM: {p}")
            print("checks FAILED: no correlations written")
            return 1
        apply_holm(corr_rows)
        _write_csv(args.out_dir / "correlations.csv", corr_rows, CORR_COLUMNS)
        print("\nmodel            class scope type   n     rho       p   p_holm  status")
        for r in corr_rows:
            fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)  # noqa: E731
            print(f"{r['model']:<16} {r['classification']:<5} {r['scope']:<6} {r['layer_type']:<5} {r['n']:>3} "
                  f"{fmt(r['rho']):>8} {fmt(r['p']):>8} {fmt(r['p_holm']):>8}  {r['status']}")
        print(f"\nwritten: {args.out_dir}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
