"""Phase 4, Level 2 add-on: weights-or-activations attribution
(calibration_protocol_vit.md Section 3.0.1, point 8; exploratory, fixed
after the Level 2 results and before any data of this experiment).

Layers: the ones the targeted-by-R curve left in float at its K_final --
MobileViT-S K=2, EfficientViT-B1 K=8 -- read from level2/evaluations.csv and
checked against the first K of the primary Level 1 R ranking.

Two variants per model, everything else as in Level 2 (level2.build:
baseline INT8, rep1 calibration with seed 42 on CPU; export_to_onnx; ONNX
Runtime CPU on the 2000 images; top-1 per image saved):
  (a) weights_float: those layers keep their input quantizer, their weights
      are not quantized (QuantLinear/QuantConv2d built with weight_quant=None);
  (b) input_float: those layers keep their weight quantizer, their input is
      not quantized (input_quant=None).
Every other layer is quantized exactly as in Level 2. After the build it is
checked, layer by layer, which quantizers are enabled.

Recovery = (top1 - top1_K0) / (FP32 - top1_K0), K=0 and FP32 as in Level 2.
Rec_full = recovery of the targeted curve at that K (whole layer in float).
Criterion (fixed in the protocol):
  "activations" if Rec_b >= 0.75 Rec_full and Rec_a <= 0.25 Rec_full;
  "weights"     if Rec_a >= 0.75 Rec_full and Rec_b <= 0.25 Rec_full;
  otherwise "mixed or interaction".
Single calibration sample (rep1); only these two configurations.

If the QCDQ export (or anything else) fails, the exact exception is written
to run.log and the run stops: nothing is changed or retried.

Resumable: one row per (model, variant) in evaluations.csv, never recomputed.
Output: results/investigations/vit/phase4/level2/attribution/{evaluations.csv,
summary.csv, predictions/, run.log}. Run from the repo root, one model at a
time:

    python -m analysis.phase4_vit.attribution --models mobilevit --dry-run
    python -m analysis.phase4_vit.attribution --models mobilevit
    python -m analysis.phase4_vit.attribution --models efficientvit_b1
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_vit import level2 as l2  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

OUT_DIR = l2.OUT_DIR / "attribution"
TARGET_K = {"mobilevit": 2, "efficientvit_b1": 8}
VARIANTS = ("weights_float", "input_float")  # (a), (b)
HIGH, LOW = 0.75, 0.25

EVAL_COLUMNS = ["model", "bitwidth", "variant", "K", "layers", "top1", "top5", "recovery", "onnx_sha256", "seconds"]
SUMMARY_COLUMNS = ["model", "K", "fp32_top1", "k0_top1", "top1_full", "rec_full", "top1_a", "rec_a", "top1_b",
                   "rec_b", "rec_a_over_full", "rec_b_over_full", "classification"]


# -- pure ----------------------------------------------------------------------

def classify(rec_a: float, rec_b: float, rec_full: float) -> str:
    if rec_b >= HIGH * rec_full and rec_a <= LOW * rec_full:
        return "activations"
    if rec_a >= HIGH * rec_full and rec_b <= LOW * rec_full:
        return "weights"
    return "mixed or interaction"


def target_layers(model: str, level2_rows: Dict[Tuple[str, str, int], Dict], ranking: Sequence[str]) -> List[str]:
    """Layers of the targeted curve at TARGET_K, checked against the R ranking."""
    k = TARGET_K[model]
    key = (model, l2.TARGETED_R, k)
    if key not in level2_rows:
        raise KeyError(f"{key} not in level2/evaluations.csv")
    layers = level2_rows[key]["float_layers"].split(";")
    if layers != list(ranking[:k]):
        raise RuntimeError(f"{model}: Level 2 float layers {layers} != first {k} of the R ranking {list(ranking[:k])}")
    return layers


def references(model: str, level2_rows: Dict[Tuple[str, str, int], Dict], fp32: float) -> Dict[str, float]:
    k0 = level2_rows[(model, l2.K0, 0)]["top1"]
    full = level2_rows[(model, l2.TARGETED_R, TARGET_K[model])]["top1"]
    return {"fp32_top1": fp32, "k0_top1": k0, "top1_full": full, "rec_full": l2.recovery(full, k0, fp32)}


def summarize(model: str, refs: Dict[str, float], top1_a: float, top1_b: float) -> Dict:
    rec_a = l2.recovery(top1_a, refs["k0_top1"], refs["fp32_top1"])
    rec_b = l2.recovery(top1_b, refs["k0_top1"], refs["fp32_top1"])
    rf = refs["rec_full"]
    return {"model": model, "K": TARGET_K[model], **refs, "top1_a": top1_a, "rec_a": rec_a, "top1_b": top1_b,
            "rec_b": rec_b, "rec_a_over_full": rec_a / rf, "rec_b_over_full": rec_b / rf,
            "classification": classify(rec_a, rec_b, rf)}


# -- build (torch / Brevitas) ------------------------------------------------

def quantize_tree_variant(module, weight_quant, act_quant, names, variant: str, prefix: str = "") -> List[str]:
    """level2.quantize_tree_except's walk, except that the layers in `names`
    are built with weight_quant=None (weights_float) or act_quant=None
    (input_float) instead of being left as plain nn modules. Returns the
    names so built."""
    import torch.nn as nn

    from quantization.vit_brevitas.quantize import _quantize_conv2d, _quantize_linear

    if variant not in VARIANTS:
        raise ValueError(variant)
    done = []
    for name, child in module.named_children():
        full = f"{prefix}{name}"
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            wq, aq = weight_quant, act_quant
            if full in names:
                done.append(full)
                wq, aq = (None, act_quant) if variant == "weights_float" else (weight_quant, None)
            make = _quantize_linear if isinstance(child, nn.Linear) else _quantize_conv2d
            setattr(module, name, make(child, wq, aq))
        else:
            done += quantize_tree_variant(child, weight_quant, act_quant, names, variant, f"{full}.")
    return done


def quant_state(model) -> Dict[str, Tuple[bool, bool]]:
    """{layer: (weight quantizer enabled, input quantizer enabled)}."""
    from brevitas.nn import QuantConv2d, QuantLinear

    return {n: (bool(m.weight_quant.is_quant_enabled), bool(m.input_quant.is_quant_enabled))
            for n, m in model.named_modules() if isinstance(m, (QuantLinear, QuantConv2d))}


def state_problems(state: Dict[str, Tuple[bool, bool]], names: Sequence[str], variant: str, n_layers: int) -> List[str]:
    """Every layer quantized; the named ones with exactly one quantizer off."""
    expected_named = (False, True) if variant == "weights_float" else (True, False)
    problems = []
    if len(state) != n_layers:
        problems.append(f"{len(state)} Quant layers, expected {n_layers}")
    missing = [n for n in names if n not in state]
    if missing:
        problems.append(f"not built as Quant layers: {missing}")
    for n, s in state.items():
        want = expected_named if n in names else (True, True)
        if s != want:
            problems.append(f"{n}: (weight, input) quant enabled = {s}, expected {want}")
    return problems


def build_variant(entry: Dict, names: Sequence[str], variant: str):
    """level2.build with the variant tree. CPU."""
    import torch
    import torch.nn as nn
    from brevitas.graph.calibrate import calibration_mode

    from quantization.vit_brevitas.calibration import load_calibration_batch
    from quantization.vit_brevitas.quantize import load_pretrained_model, resolve_input_shape
    from quantization.vit_brevitas.quantizer_config import resolve_act_quant, resolve_weight_quant

    model = load_pretrained_model(entry["model_name"])
    input_shape = resolve_input_shape(model)
    n_layers = sum(isinstance(m, (nn.Linear, nn.Conv2d)) for m in model.modules())
    weight_quant = resolve_weight_quant(entry["calibration_algorithm"], entry["granularity"], entry["bitwidth"])
    act_quant = resolve_act_quant(entry["calibration_algorithm"], entry["symmetry"], entry["bitwidth"])
    calib_batch = load_calibration_batch(model, entry["calib_set_size"], l2.REPEAT_ID)

    done = quantize_tree_variant(model, weight_quant, act_quant, set(names), variant)
    model.eval()
    problems = ([] if sorted(done) == sorted(names) else [f"built {sorted(done)} != requested {sorted(names)}"])
    problems += state_problems(quant_state(model), names, variant, n_layers)
    if problems:
        raise RuntimeError(f"{variant}: " + " | ".join(problems))
    print(f"   {variant}: {len(names)} layer(s) with the {'weight' if variant == 'weights_float' else 'input'} "
          f"quantizer off, {n_layers - len(names)} fully quantized", flush=True)

    with torch.no_grad(), calibration_mode(model):
        model(calib_batch)
    return model, input_shape


def run_variant(entry: Dict, variant: str, names: Sequence[str], out_dir: Path) -> Dict:
    from export.vit_to_onnx.export import export_to_onnx

    model_name = entry["model_name"]
    start = time.time()
    model, input_shape = build_variant(entry, names, variant)
    onnx_file = out_dir / "onnx" / f"{model_name}__{variant}.onnx"
    export_to_onnx(model, input_shape, entry["run_id"], l2.REPEAT_ID, out_path=onnx_file)
    top1, top5, preds = l2.evaluate_onnx(model_name, onnx_file)
    sha = l2._sha256(onnx_file)
    onnx_file.unlink()
    l2._write_csv(out_dir / "predictions" / f"{model_name}__{variant}.csv", preds,
                  ["idx", "filename", "label", "pred1", "correct1"])
    return {"model": model_name, "bitwidth": entry["bitwidth"], "variant": variant, "K": len(names),
            "layers": ";".join(names), "top1": top1, "top5": top5, "onnx_sha256": sha,
            "seconds": round(time.time() - start, 1)}


# -- driver ------------------------------------------------------------------

def read_rows(path: Path) -> Dict[Tuple[str, str], Dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["top1"], r["top5"] = float(r["top1"]), float(r["top5"])
    return {(r["model"], r["variant"]): r for r in rows}


def run(models: Sequence[str], out_dir: Path, dry_run: bool, evaluate=run_variant,
        level2_dir: Path = l2.OUT_DIR, level1_dir: Path = l2.LEVEL1_DIR, fp32=l2.fp32_top1) -> int:
    level2_rows = l2.read_evaluations(level2_dir / "evaluations.csv")
    plan = {}
    for m in models:
        names = target_layers(m, level2_rows, l2.order_by_r(l2.read_level1(m, level1_dir)))
        refs = references(m, level2_rows, fp32(m))
        plan[m] = (names, refs)
        print(f"== {m}: K={TARGET_K[m]}, layers = {names}")
        print(f"   FP32 {refs['fp32_top1']}, K=0 {refs['k0_top1']}, whole layers in float {refs['top1_full']} "
              f"(Rec_full {100 * refs['rec_full']:.1f} %)", flush=True)
    if dry_run:
        return 0

    path = out_dir / "evaluations.csv"
    rows = read_rows(path)
    summary = []
    for m in models:
        names, refs = plan[m]
        entry = l2.reference_entry(m)
        print(f"\n== {m}", flush=True)
        for variant in VARIANTS:
            if (m, variant) in rows:
                if rows[(m, variant)]["layers"] != ";".join(names):
                    raise RuntimeError(f"{m} {variant}: recorded layers differ from the Level 2 layers")
                print(f"   [reuse] {variant}: top1 {rows[(m, variant)]['top1']:.2f}", flush=True)
                continue
            print(f"   [eval] {variant}", flush=True)
            row = evaluate(entry, variant, names, out_dir)
            row["recovery"] = l2.recovery(row["top1"], refs["k0_top1"], refs["fp32_top1"])
            rows[(m, variant)] = row
            l2._write_csv(path, list(rows.values()), EVAL_COLUMNS)
            print(f"   {variant}: top1 {row['top1']:.2f} top5 {row['top5']:.2f} ({row['seconds']} s)", flush=True)
        s = summarize(m, refs, rows[(m, "weights_float")]["top1"], rows[(m, "input_float")]["top1"])
        summary.append(s)
        print(f"   Rec_full {100 * s['rec_full']:.1f} %; Rec_a (weights float) {100 * s['rec_a']:.1f} % "
              f"= {s['rec_a_over_full']:.2f} Rec_full; Rec_b (input float) {100 * s['rec_b']:.1f} % "
              f"= {s['rec_b_over_full']:.2f} Rec_full -> {s['classification']}", flush=True)

    summary_path = out_dir / "summary.csv"
    previous = {}
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            previous = {r["model"]: r for r in csv.DictReader(f)}
    previous.update({s["model"]: s for s in summary})
    l2._write_csv(summary_path, list(previous.values()), SUMMARY_COLUMNS)
    print(f"\nwritten: {out_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=list(TARGET_K), choices=list(TARGET_K))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="check layers and references, build nothing")
    args = parser.parse_args(argv)
    if args.dry_run:
        return run(args.models, args.out_dir, dry_run=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(args.out_dir / "run.log"):
        print(f"attribution: models {args.models}; rep{l2.REPEAT_ID}; build + calibration + export: cpu; "
              f"evaluation: ONNX Runtime CPU", flush=True)
        try:
            return run(args.models, args.out_dir, dry_run=False)
        except Exception:
            import traceback

            traceback.print_exc()  # inside tee_output, so the exact error lands in run.log
            print("PROBLEM: stopped on the exception above -- nothing changed, nothing retried")
            return 1


if __name__ == "__main__":
    sys.exit(main())
