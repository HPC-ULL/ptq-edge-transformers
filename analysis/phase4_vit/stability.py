"""Phase 4, Level 1 add-on (pre-registered, reported only): stability of the
R ranking. R is recomputed on a SECOND set of 512 calibration-bank images
(seed 1), disjoint from the primary Level 1 set and from the calibration
samples of seeds 42/43/44 (size 512). Same FP32 pass, hooks, R formula and
ranking rules as level1.py; no quantized model is needed.

Per model it reports the Spearman correlation between the R values of the
two sets over the layers ranked in both (no p-value), and how many of the
top 8 coincide (set intersection, order ignored). Level 2 always uses the
primary ranking; nothing here changes it.

Output: results/investigations/vit/phase4/stability/{images.csv,
layers_<model>.csv, stability.csv, run.log}. Run from the repo root, after
level1.py:

    python -m analysis.phase4_vit.stability --dry-run
    python -m analysis.phase4_vit.stability
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_vit import MODELS, PHASE4_ROOT  # noqa: E402
from analysis.phase4_vit import level1 as l1  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

OUT_DIR = PHASE4_ROOT / "stability"
SECOND_SEED = 1
TOP = 8

LAYER_COLUMNS = ["model", "exec_idx", "name", "layer_type", "in_channels", "rankable_channels",
                 "absmax_max", "absmax_median", "R", "ranked", "exclusion", "rank"]
STABILITY_COLUMNS = ["model", "n_common", "spearman_R", "top8_overlap", "top8_primary", "top8_second"]


def second_images() -> Tuple[List[Dict], set, set]:
    """(images, excluded calibration filenames, primary filenames)."""
    from data_prep.imagenet import calibration_pool_dir, get_calibration_manifest

    primary, excluded = l1.level1_images()
    rows = get_calibration_manifest()
    index_of = {r["filename"]: i for i, r in enumerate(rows)}
    primary_names = {s["filename"] for s in primary}
    avoid = {index_of[f] for f in excluded | primary_names}
    picked = l1.select_image_indices(len(rows), avoid, seed=SECOND_SEED)
    images_dir = calibration_pool_dir() / "images"
    images = [{"pool_idx": i, "filename": rows[i]["filename"], "path": images_dir / rows[i]["filename"]}
              for i in picked]
    return images, excluded, primary_names


def compare(primary: Sequence[Dict], second: Sequence[Dict], spearman) -> Dict:
    """Ranked rows of both sets (with 'name', 'R', 'rank')."""
    p = {r["name"]: r for r in primary if r["ranked"] in (True, "True")}
    s = {r["name"]: r for r in second if r["ranked"] in (True, "True")}
    common = sorted(set(p) & set(s))
    rho, _ = spearman([float(p[n]["R"]) for n in common], [float(s[n]["R"]) for n in common])
    top_p = [r["name"] for r in sorted(p.values(), key=lambda r: int(r["rank"]))[:TOP]]
    top_s = [r["name"] for r in sorted(s.values(), key=lambda r: int(r["rank"]))[:TOP]]
    return {"n_common": len(common), "spearman_R": rho, "top8_overlap": len(set(top_p) & set(top_s)),
            "top8_primary": ";".join(top_p), "top8_second": ";".join(top_s)}


def r_pass(model_name: str, inventory_rows: List[Dict], images: List[Dict], device: str,
           batch_size: int) -> List[Dict]:
    """FP32 pass accumulating only the per-channel absmax of each layer input."""
    import torch
    import torch.nn as nn
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    from quantization.vit_brevitas.quantize import load_pretrained_model

    fp32 = load_pretrained_model(model_name)
    preprocess = create_transform(**resolve_data_config({}, model=fp32))
    by_name = {r["name"]: r for r in inventory_rows}
    layers = {n: m for n, m in fp32.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))}
    if set(layers) != set(by_name):
        raise RuntimeError(f"{model_name}: model layers differ from the inventory")
    fp32.to(device)
    acc = {n: l1.LayerAccumulator(by_name[n]["kind"]) for n in by_name}

    def make_hook(name):
        def hook(_module, inputs, _output):
            x = inputs[0].detach()
            cmax = acc[name].channel_absmax(x).double()
            acc[name].absmax = cmax if acc[name].absmax is None else acc[name].absmax.maximum(cmax)
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in layers.items()]
    try:
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = images[start:start + batch_size]
                fp32(torch.stack([preprocess(Image.open(s["path"]).convert("RGB")) for s in batch]).to(device))
    finally:
        for h in handles:
            h.remove()

    rows = []
    for name, inv in by_name.items():
        absmax_max = acc[name].absmax.max().item()
        absmax_median = acc[name].absmax.quantile(0.5).item()
        rows.append({"model": model_name, "exec_idx": int(inv["exec_idx"]), "name": name,
                     "layer_type": inv["layer_type"], "in_channels": int(inv["in_channels"]),
                     "rankable_channels": inv["rankable_channels"] in (True, "True"),
                     "absmax_max": absmax_max, "absmax_median": absmax_median,
                     "R": l1.outlier_ratio(absmax_max, absmax_median)})
    rows.sort(key=lambda r: r["exec_idx"])
    l1.rank_layers(rows)
    return rows


def _read(path: Path) -> List[Dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--inventory", type=Path, default=l1.INVENTORY_DIR / "layers.csv")
    parser.add_argument("--level1-dir", type=Path, default=l1.OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="check the image selection, load no model")
    args = parser.parse_args(argv)

    images, excluded, primary = second_images()
    names = {s["filename"] for s in images}
    print(f"second set: {len(images)} images, seed {SECOND_SEED}; overlap with calibration samples: "
          f"{len(names & excluded)}, with the primary Level 1 set: {len(names & primary)}")
    problems = []
    if len(names) != l1.N_IMAGES or names & excluded or names & primary:
        problems.append("second image set: not 512 distinct images disjoint from calibration and primary")
    missing = [m for m in args.models if not (args.level1_dir / f"layers_{m}.csv").exists()]
    problems += [f"no Level 1 ranking for {m}" for m in missing]
    if args.dry_run or problems:
        for p in problems:
            print(f"PROBLEM: {p}")
        return 1 if problems else 0

    device = args.device or l1._default_device()
    inventory = l1.read_inventory(args.inventory)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(args.out_dir / "run.log"):
        print(f"forward device: {device}; batch size {args.batch_size}")
        l1._write_csv(args.out_dir / "images.csv", images, ["pool_idx", "filename"])
        out = []
        for m in args.models:
            print(f"\n== {m}", flush=True)
            rows = r_pass(m, inventory[m], images, device, args.batch_size)
            l1._write_csv(args.out_dir / f"layers_{m}.csv", rows, LAYER_COLUMNS)
            c = {"model": m, **compare(_read(args.level1_dir / f"layers_{m}.csv"), rows, l1._spearman)}
            out.append(c)
            print(f"   layers ranked in both: {c['n_common']}; Spearman of R: {c['spearman_R']:.3f}; "
                  f"top-8 in common: {c['top8_overlap']}/8")
        l1._write_csv(args.out_dir / "stability.csv", out, STABILITY_COLUMNS)
        print(f"\nwritten: {args.out_dir}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
