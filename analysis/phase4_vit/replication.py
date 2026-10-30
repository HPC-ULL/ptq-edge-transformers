"""Phase 4, Level 2 add-on: replication with other calibration samples
(calibration_protocol_vit.md Section 3.0.1, point 9; fixed after the Level 2
and attribution results, before any replication data).

Repeats 2 and 3 (calibration seeds 43 and 44), MobileViT-S with K=2 and
EfficientViT-B1 with K=8: per repeat, the K=0 control and the targeted
curve at that K, with the SAME layers as rep1 (the first K of the primary
R ranking, computed in FP32: independent of the calibration sample; read
from level2/evaluations.csv and checked against the ranking). 8 evaluations.

Same build (level2.build with repeat_id), export and evaluation as Level 2;
top-1 per image and the ONNX sha256 recorded.

Order: every K=0 first. Each must reproduce EXACTLY the recorded top-1 and
top-5 of its repeat (INT8 baseline row of vit_accuracy_offline.csv); one
mismatch stops everything before any targeted evaluation. The per-image
top-1 is also compared with per_image_predictions (reported only).

Recovery_k = (acc_k - acc_0,k) / (FP32 - acc_0,k), with each repeat's own
K=0. Ratio_k = Recovery_k / Recovery_rep1 (rep1: level2/evaluations.csv,
93.9 % and 49.6 %). Criterion: "replicates" if Ratio_k >= 0.75 in BOTH
repeats (>= 70.4 % in MobileViT-S, >= 37.2 % in EfficientViT-B1);
otherwise "does not replicate". Random control and DeiT-Tiny not replicated.

If anything fails, the exact exception is written to run.log and the run
stops; nothing is changed or retried. Resumable: one row per (model,
repeat, curve) in evaluations.csv, never recomputed.

Output: results/investigations/vit/phase4/level2/replication/
{evaluations.csv, kzero_checks.csv, summary.csv, predictions/, run.log}.
Run from the repo root:

    python -m analysis.phase4_vit.replication --dry-run
    python -m analysis.phase4_vit.replication
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_vit import attribution as at  # noqa: E402
from analysis.phase4_vit import level2 as l2  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

OUT_DIR = l2.OUT_DIR / "replication"
REPEATS = (2, 3)
TARGET_K = at.TARGET_K  # {"mobilevit": 2, "efficientvit_b1": 8}
MIN_RATIO = 0.75

EVAL_COLUMNS = ["model", "repeat", "calib_seed", "curve", "K", "float_layers", "top1", "top5", "recovery",
                "onnx_sha256", "seconds"]
CHECK_COLUMNS = ["model", "repeat", "top1", "top5", "ref_top1", "ref_top5", "per_image_top1_diff", "status"]
SUMMARY_COLUMNS = ["model", "K", "fp32_top1", "rec_rep1", "threshold", "k0_rep2", "top1_rep2", "rec_rep2",
                   "ratio_rep2", "k0_rep3", "top1_rep3", "rec_rep3", "ratio_rep3", "verdict"]


def calib_seed(repeat: int) -> int:
    from quantization.vit_brevitas.calibration import seed_for_repeat

    return seed_for_repeat(repeat)


def replication_verdict(ratios: Sequence[float]) -> str:
    return "replicates" if all(r >= MIN_RATIO for r in ratios) else "does not replicate"


# -- evaluation (torch / ONNX Runtime) ---------------------------------------

def run_evaluation(entry: Dict, repeat: int, curve: str, names: Sequence[str], out_dir: Path,
                   build: Callable = l2.build) -> Dict:
    from export.vit_to_onnx.export import export_to_onnx

    model_name = entry["model_name"]
    start = time.time()
    model, input_shape = build(entry, names, repeat_id=repeat)
    stem = f"{model_name}__rep{repeat}__{curve}__K{len(names)}"
    onnx_file = out_dir / "onnx" / f"{stem}.onnx"
    export_to_onnx(model, input_shape, entry["run_id"], repeat, out_path=onnx_file)
    top1, top5, preds = l2.evaluate_onnx(model_name, onnx_file)
    sha = l2._sha256(onnx_file)
    onnx_file.unlink()
    l2._write_csv(out_dir / "predictions" / f"{stem}.csv", preds, ["idx", "filename", "label", "pred1", "correct1"])
    return {"model": model_name, "repeat": repeat, "calib_seed": calib_seed(repeat), "curve": curve,
            "K": len(names), "float_layers": ";".join(names), "top1": top1, "top5": top5, "onnx_sha256": sha,
            "seconds": round(time.time() - start, 1)}


# -- driver ------------------------------------------------------------------

def read_rows(path: Path) -> Dict[Tuple[str, int, str], Dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["repeat"], r["K"] = int(r["repeat"]), int(r["K"])
        r["top1"], r["top5"] = float(r["top1"]), float(r["top5"])
    return {(r["model"], r["repeat"], r["curve"]): r for r in rows}


class Runner:
    def __init__(self, out_dir: Path, evaluate=run_evaluation):
        self.out_dir, self.path = out_dir, out_dir / "evaluations.csv"
        self.rows = read_rows(self.path)
        self.evaluate = evaluate

    def get(self, entry: Dict, repeat: int, curve: str, names: Sequence[str]) -> Dict:
        key = (entry["model_name"], repeat, curve)
        if key in self.rows:
            row = self.rows[key]
            if row["float_layers"] != ";".join(names):
                raise RuntimeError(f"{key}: recorded float layers differ from rep1's")
            print(f"   [reuse] rep{repeat} {curve}: top1 {row['top1']:.2f}", flush=True)
            return row
        print(f"   [eval] rep{repeat} (seed {calib_seed(repeat)}) {curve}: float = {list(names) or '-'}", flush=True)
        row = self.evaluate(entry, repeat, curve, names, self.out_dir)
        self.rows[key] = row
        l2._write_csv(self.path, list(self.rows.values()), EVAL_COLUMNS)
        print(f"   rep{repeat} {curve}: top1 {row['top1']:.2f} top5 {row['top5']:.2f} ({row['seconds']} s)", flush=True)
        return row


def run(models: Sequence[str], out_dir: Path, dry_run: bool, runner: Optional[Runner] = None,
        recorded=l2.recorded_accuracy, fp32=l2.fp32_top1, level2_dir: Path = l2.OUT_DIR,
        level1_dir: Path = l2.LEVEL1_DIR, per_image_dir: Path = l2.PER_IMAGE_DIR) -> int:
    level2_rows = l2.read_evaluations(level2_dir / "evaluations.csv")
    plan = {}
    for m in models:
        names = at.target_layers(m, level2_rows, l2.order_by_r(l2.read_level1(m, level1_dir)))
        refs = at.references(m, level2_rows, fp32(m))
        plan[m] = (l2.reference_entry(m), names, refs)
        print(f"== {m}: K={TARGET_K[m]}, layers = {names}")
        print(f"   rep1: K=0 {refs['k0_top1']}, targeted {refs['top1_full']}, recovery {100 * refs['rec_full']:.1f} %; "
              f"threshold {100 * MIN_RATIO * refs['rec_full']:.1f} %; repeats {list(REPEATS)} "
              f"(seeds {[calib_seed(k) for k in REPEATS]})", flush=True)
    if dry_run:
        return 0
    runner = runner or Runner(out_dir)

    print("\n== K=0 controls", flush=True)
    checks = []
    for m in models:
        entry = plan[m][0]
        for k in REPEATS:
            row = runner.get(entry, k, l2.K0, [])
            ref = recorded(entry["run_id"], repeat_id=k)
            pred = out_dir / "predictions" / f"{m}__rep{k}__{l2.K0}__K0.csv"
            ref_pred = per_image_dir / f"{entry['run_id']}__rep{k}.csv"
            diff = l2.per_image_differences(pred, ref_pred) if pred.exists() else None
            ok = row["top1"] == ref["top1"] and row["top5"] == ref["top5"]
            checks.append({"model": m, "repeat": k, "top1": row["top1"], "top5": row["top5"], "ref_top1": ref["top1"],
                           "ref_top5": ref["top5"], "per_image_top1_diff": "" if diff is None else diff,
                           "status": "OK" if ok else "MISMATCH"})
            print(f"   {m} rep{k}: K=0 {row['top1']} / {row['top5']} vs recorded {ref['top1']} / {ref['top5']}: "
                  f"{checks[-1]['status']}; per-image top-1 differences: {'n/a' if diff is None else diff}", flush=True)
    l2._write_csv(out_dir / "kzero_checks.csv", checks, CHECK_COLUMNS)
    if any(c["status"] != "OK" for c in checks):
        print("PROBLEM: a K=0 control does not reproduce its recorded accuracy -- stopping (nothing else evaluated)")
        return 1

    summary = []
    for m in models:
        entry, names, refs = plan[m]
        print(f"\n== {m}: targeted K={TARGET_K[m]}", flush=True)
        s = {"model": m, "K": TARGET_K[m], "fp32_top1": refs["fp32_top1"], "rec_rep1": refs["rec_full"],
             "threshold": MIN_RATIO * refs["rec_full"]}
        ratios = []
        for k in REPEATS:
            k0 = runner.rows[(m, k, l2.K0)]["top1"]
            row = runner.get(entry, k, l2.TARGETED_R, names)
            rec = l2.recovery(row["top1"], k0, refs["fp32_top1"])
            row["recovery"] = rec
            ratio = rec / refs["rec_full"]
            ratios.append(ratio)
            s.update({f"k0_rep{k}": k0, f"top1_rep{k}": row["top1"], f"rec_rep{k}": rec, f"ratio_rep{k}": ratio})
            print(f"   rep{k}: K=0 {k0}, targeted {row['top1']}, recovery {100 * rec:.1f} % "
                  f"= {ratio:.2f} x rep1 ({100 * refs['rec_full']:.1f} %)", flush=True)
        l2._write_csv(runner.path, list(runner.rows.values()), EVAL_COLUMNS)
        s["verdict"] = replication_verdict(ratios)
        summary.append(s)
        print(f"   -> {s['verdict']} (needs >= {MIN_RATIO} x rep1 in both repeats)", flush=True)
    l2._write_csv(out_dir / "summary.csv", summary, SUMMARY_COLUMNS)
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
        print(f"replication: models {args.models}; repeats {list(REPEATS)}; build + calibration + export: cpu; "
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
