"""Phase 4, Level 2 add-on: random control of the replication, EfficientViT-B1
K=8, calibration repeats 2 and 3 (seeds 43, 44). Criterion as given by the
author on 2026-10-06 (calibration_protocol_vit.md Section 3.0.1, point 10),
fixed after the replication and before any data of this control.

Float layers: the 3 random selections of rep1 (level2/evaluations.csv,
random_s1/s2/s3 at K=8), checked to equal the first 8 of level2.order_random
(rankable layers in execution order, random.Random(seed).shuffle) -- the same
8 layers per seed in every repeat. 6 evaluations (2 repeats x 3 seeds), with
the same build/export/evaluation as the replication (replication.
run_evaluation: level2.build with repeat_id, so seed 42 + repeat - 1).

Reused, never re-evaluated: the replication's K=0 (checked OK in
kzero_checks.csv) and targeted K=8 of each repeat (34.25 % and 29.15 %);
if any is missing the run stops before evaluating anything.

Recovery = (acc - acc_0,k) / (FP32 - acc_0,k), with each repeat's own K=0.
Criterion: the superiority over random holds if, in BOTH repeats, the
targeted top-1 is STRICTLY greater than the max top-1 of the 3 random seeds
of that repeat; otherwise it does not hold.

Same files as the replication (evaluations.csv, predictions/, ONNX sha256),
resumable: one row per (model, repeat, curve). Own summary and log:
random_summary.csv, random_run.log. Exact exception in the log on failure,
nothing changed or retried. Run from the repo root:

    python -m analysis.phase4_vit.replication_random --dry-run
    python -m analysis.phase4_vit.replication_random
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

from analysis.phase4_vit import level2 as l2  # noqa: E402
from analysis.phase4_vit import replication as rp  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

MODEL = "efficientvit_b1"
K = 8
REPEATS = rp.REPEATS  # (2, 3)
SEEDS = l2.RANDOM_SEEDS  # (1, 2, 3)
CURVES = l2.RANDOM_CURVES  # random_s1..3

SUMMARY_COLUMNS = ["model", "K", "repeat", "calib_seed", "k0_top1", "targeted_top1", "targeted_recovery",
                   "random_s1_top1", "random_s1_recovery", "random_s2_top1", "random_s2_recovery",
                   "random_s3_top1", "random_s3_recovery", "random_max_top1", "targeted_beats_max", "verdict"]


def random_layers(level2_rows: Dict[Tuple[str, str, int], Dict], ranked_rows: Sequence[Dict]) -> Dict[str, List[str]]:
    """{curve: 8 float layers} of rep1, checked against level2.order_random."""
    out = {}
    for seed, curve in zip(SEEDS, CURVES):
        key = (MODEL, curve, K)
        if key not in level2_rows:
            raise KeyError(f"{key} not in level2/evaluations.csv")
        layers = level2_rows[key]["float_layers"].split(";")
        expected = l2.order_random(ranked_rows, seed)[:K]
        if layers != expected or len(layers) != K:
            raise RuntimeError(f"{curve}: rep1 float layers {layers} != order_random(seed {seed})[:{K}] {expected}")
        out[curve] = layers
    return out


def replication_inputs(rows: Dict[Tuple[str, int, str], Dict], checks_path: Path,
                       targeted_layers: Sequence[str]) -> Dict[int, Dict[str, float]]:
    """{repeat: {'k0': top1, 'targeted': top1}} from the replication; every
    K=0 must be OK in kzero_checks.csv and every targeted K=8 present with the
    rep1 layers."""
    if not checks_path.exists():
        raise FileNotFoundError(f"{checks_path}: the replication's K=0 checks are missing")
    with open(checks_path, newline="") as f:
        checks = {(r["model"], int(r["repeat"])): r["status"] for r in csv.DictReader(f)}
    out = {}
    for k in REPEATS:
        if checks.get((MODEL, k)) != "OK":
            raise RuntimeError(f"{MODEL} rep{k}: K=0 check is {checks.get((MODEL, k))!r}, not OK")
        for curve in (l2.K0, l2.TARGETED_R):
            if (MODEL, k, curve) not in rows:
                raise KeyError(f"{MODEL} rep{k} {curve}: not in the replication's evaluations.csv")
        if rows[(MODEL, k, l2.TARGETED_R)]["float_layers"] != ";".join(targeted_layers):
            raise RuntimeError(f"{MODEL} rep{k}: replication targeted layers differ from rep1's")
        out[k] = {"k0": rows[(MODEL, k, l2.K0)]["top1"], "targeted": rows[(MODEL, k, l2.TARGETED_R)]["top1"]}
    return out


def repeat_summary(k: int, k0: float, targeted: float, random_top1: Dict[str, float], fp32: float) -> Dict:
    rmax = max(random_top1[c] for c in CURVES)
    s = {"model": MODEL, "K": K, "repeat": k, "calib_seed": rp.calib_seed(k), "k0_top1": k0,
         "targeted_top1": targeted, "targeted_recovery": l2.recovery(targeted, k0, fp32),
         "random_max_top1": rmax, "targeted_beats_max": targeted > rmax}
    for c in CURVES:
        s[f"{c}_top1"] = random_top1[c]
        s[f"{c}_recovery"] = l2.recovery(random_top1[c], k0, fp32)
    return s


def overall_verdict(summaries: Sequence[Dict]) -> str:
    return "superiority holds" if all(s["targeted_beats_max"] for s in summaries) else "superiority does not hold"


def run(out_dir: Path, dry_run: bool, runner: Optional[rp.Runner] = None, fp32=l2.fp32_top1,
        level2_dir: Path = l2.OUT_DIR, level1_dir: Path = l2.LEVEL1_DIR) -> int:
    level2_rows = l2.read_evaluations(level2_dir / "evaluations.csv")
    ranked = l2.read_level1(MODEL, level1_dir)
    targeted_layers = l2.order_by_r(ranked)[:K]
    layers = random_layers(level2_rows, ranked)
    runner = runner or rp.Runner(out_dir)
    inputs = replication_inputs(runner.rows, out_dir / "kzero_checks.csv", targeted_layers)
    ref = fp32(MODEL)
    print(f"== {MODEL} K={K}; FP32 {ref}; repeats {list(REPEATS)} (calibration seeds "
          f"{[rp.calib_seed(k) for k in REPEATS]}); reused from the replication: "
          + ", ".join(f"rep{k} K=0 {v['k0']} targeted {v['targeted']}" for k, v in inputs.items()))
    for c, names in layers.items():
        print(f"   {c}: {names}")
    if dry_run:
        return 0

    entry = l2.reference_entry(MODEL)
    summaries = []
    for k in REPEATS:
        print(f"\n== rep{k} (calibration seed {rp.calib_seed(k)})", flush=True)
        top1 = {}
        for c in CURVES:
            row = runner.get(entry, k, c, layers[c])
            row["recovery"] = l2.recovery(row["top1"], inputs[k]["k0"], ref)
            top1[c] = row["top1"]
        l2._write_csv(runner.path, list(runner.rows.values()), rp.EVAL_COLUMNS)
        s = repeat_summary(k, inputs[k]["k0"], inputs[k]["targeted"], top1, ref)
        summaries.append(s)
        print(f"   targeted: top1 {s['targeted_top1']} (recovery {100 * s['targeted_recovery']:.1f} %)")
        for c in CURVES:
            print(f"   {c}: top1 {s[c + '_top1']} (recovery {100 * s[c + '_recovery']:.1f} %)")
        print(f"   random max {s['random_max_top1']}; targeted strictly greater: {s['targeted_beats_max']}", flush=True)
    verdict = overall_verdict(summaries)
    for s in summaries:
        s["verdict"] = verdict
    l2._write_csv(out_dir / "random_summary.csv", summaries, SUMMARY_COLUMNS)
    print(f"\n-> {verdict} (targeted > max of the 3 random seeds in both repeats)\nwritten: {out_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=rp.OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="check layers and reused results, build nothing")
    args = parser.parse_args(argv)
    if args.dry_run:
        return run(args.out_dir, dry_run=True)
    with tee_output(args.out_dir / "random_run.log"):
        print(f"replication random control: {MODEL} K={K}, repeats {list(REPEATS)}, seeds {list(SEEDS)}; "
              f"build + calibration + export: cpu; evaluation: ONNX Runtime CPU", flush=True)
        try:
            return run(args.out_dir, dry_run=False)
        except Exception:
            import traceback

            traceback.print_exc()  # inside tee_output, so the exact error lands in the log
            print("PROBLEM: stopped on the exception above -- nothing changed, nothing retried")
            return 1


if __name__ == "__main__":
    sys.exit(main())
