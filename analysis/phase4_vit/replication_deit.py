"""Phase 4, Level 2 add-on: replication of the DeiT-Tiny INT6 negative result
with calibration repeats 2 and 3 (calibration_protocol_vit.md Section 3.0.1,
point 12; fixed before any data of this replication).

DeiT-Tiny INT6 baseline, K=8, repeats 2 and 3 (seeds 43, 44). Per repeat:
  k0            -- must reproduce EXACTLY the recorded top-1/top-5 of the INT6
                   baseline of that repeat, with 0 per-image top-1 differences
                   against per_image_predictions; otherwise stop;
  targeted_r    -- the first 8 of the primary R ranking;
  random_s1..3  -- the 3 random selections of rep1 (level2.order_random,
                   seeds 1, 2, 3, over the rankable layers);
  targeted_err  -- exploratory: the 8 ranked layers with the largest local error.
Each float set is read from level2/evaluations.csv (rep1, K=8) and checked
against its level2 order; 8 layers each. 12 evaluations; both K=0 first.

Recovery = (acc - acc_0,k) / (FP32 - acc_0,k), with each repeat's own K=0.
Criterion: the negative result REPLICATES if, in both repeats, the targeted-
by-R top-1 at K=8 is NOT strictly greater than the max of the 3 random seeds
of that repeat; if it is in either, it does not replicate. The local-error
curve is compared with the same controls and reported, outside the verdict.

Same build/export/evaluation and files as the replication (replication.
run_evaluation; evaluations.csv, predictions/, ONNX sha256), resumable; own
deit_kzero_checks.csv, deit_summary.csv and deit_run.log. Exact exception
in the log on failure, nothing changed or retried. Run from the repo root:

    python -m analysis.phase4_vit.replication_deit --dry-run
    python -m analysis.phase4_vit.replication_deit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.phase4_vit import level2 as l2  # noqa: E402
from analysis.phase4_vit import replication as rp  # noqa: E402
from quantization.vit_brevitas.paths import tee_output  # noqa: E402

MODEL = "deit_tiny"
K = 8
REPEATS = rp.REPEATS  # (2, 3)
CURVES = (l2.TARGETED_R,) + l2.RANDOM_CURVES + (l2.TARGETED_ERR,)

CHECK_COLUMNS = ["model", "repeat", "top1", "top5", "ref_top1", "ref_top5", "per_image_top1_diff", "status"]
SUMMARY_COLUMNS = (["model", "K", "repeat", "calib_seed", "k0_top1"]
                   + [f"{c}_{x}" for c in CURVES for x in ("top1", "recovery")]
                   + ["random_max_top1", "targeted_r_beats_max", "targeted_err_beats_max", "verdict"])


def expected_layers(ranked_rows: Sequence[Dict]) -> Dict[str, List[str]]:
    orders = l2.curve_orders(ranked_rows)
    return {c: orders[c][:K] for c in CURVES}


def rep1_layers(level2_rows: Dict[Tuple[str, str, int], Dict], ranked_rows: Sequence[Dict]) -> Dict[str, List[str]]:
    """{curve: 8 float layers} of rep1 (level2/evaluations.csv), checked
    against the level2 order of that curve."""
    expected = expected_layers(ranked_rows)
    out = {}
    for c in CURVES:
        key = (MODEL, c, K)
        if key not in level2_rows:
            raise KeyError(f"{key} not in level2/evaluations.csv")
        layers = level2_rows[key]["float_layers"].split(";")
        if layers != expected[c] or len(layers) != K:
            raise RuntimeError(f"{c}: rep1 float layers {layers} != level2 order {expected[c]}")
        out[c] = layers
    return out


def repeat_summary(k: int, k0: float, top1: Dict[str, float], fp32: float) -> Dict:
    rmax = max(top1[c] for c in l2.RANDOM_CURVES)
    s = {"model": MODEL, "K": K, "repeat": k, "calib_seed": rp.calib_seed(k), "k0_top1": k0,
         "random_max_top1": rmax, "targeted_r_beats_max": top1[l2.TARGETED_R] > rmax,
         "targeted_err_beats_max": top1[l2.TARGETED_ERR] > rmax}
    for c in CURVES:
        s[f"{c}_top1"], s[f"{c}_recovery"] = top1[c], l2.recovery(top1[c], k0, fp32)
    return s


def overall_verdict(summaries: Sequence[Dict]) -> str:
    if any(s["targeted_r_beats_max"] for s in summaries):
        return "negative result does not replicate"
    return "negative result replicates"


def run(out_dir: Path, dry_run: bool, runner: Optional[rp.Runner] = None, recorded=l2.recorded_accuracy,
        fp32=l2.fp32_top1, level2_dir: Path = l2.OUT_DIR, level1_dir: Path = l2.LEVEL1_DIR,
        per_image_dir: Path = l2.PER_IMAGE_DIR) -> int:
    level2_rows = l2.read_evaluations(level2_dir / "evaluations.csv")
    layers = rep1_layers(level2_rows, l2.read_level1(MODEL, level1_dir))
    entry = l2.reference_entry(MODEL)  # INT6 baseline
    ref = fp32(MODEL)
    refs = {k: recorded(entry["run_id"], repeat_id=k) for k in REPEATS}
    print(f"== {MODEL} {entry['bitwidth']} baseline, K={K}; FP32 {ref}; repeats {list(REPEATS)} (calibration seeds "
          f"{[rp.calib_seed(k) for k in REPEATS]}); recorded K=0: "
          + ", ".join(f"rep{k} {r['top1']} / {r['top5']}" for k, r in refs.items()))
    for c, names in layers.items():
        print(f"   {c}: {names}")
    if dry_run:
        return 0
    runner = runner or rp.Runner(out_dir)

    print("\n== K=0 controls", flush=True)
    checks = []
    for k in REPEATS:
        row = runner.get(entry, k, l2.K0, [])
        pred = out_dir / "predictions" / f"{MODEL}__rep{k}__{l2.K0}__K0.csv"
        ref_pred = per_image_dir / f"{entry['run_id']}__rep{k}.csv"
        diff = l2.per_image_differences(pred, ref_pred) if pred.exists() else None
        ok = row["top1"] == refs[k]["top1"] and row["top5"] == refs[k]["top5"] and diff == 0
        checks.append({"model": MODEL, "repeat": k, "top1": row["top1"], "top5": row["top5"],
                       "ref_top1": refs[k]["top1"], "ref_top5": refs[k]["top5"],
                       "per_image_top1_diff": "n/a" if diff is None else diff, "status": "OK" if ok else "MISMATCH"})
        print(f"   rep{k}: K=0 {row['top1']} / {row['top5']} vs recorded {refs[k]['top1']} / {refs[k]['top5']}; "
              f"per-image top-1 differences: {checks[-1]['per_image_top1_diff']} -> {checks[-1]['status']}", flush=True)
        l2._write_csv(out_dir / "deit_kzero_checks.csv", checks, CHECK_COLUMNS)
        if not ok:
            print(f"PROBLEM: the rep{k} K=0 does not reproduce the recorded INT6 baseline exactly -- stopping")
            return 1

    summaries = []
    for k in REPEATS:
        k0 = runner.rows[(MODEL, k, l2.K0)]["top1"]
        print(f"\n== rep{k} (calibration seed {rp.calib_seed(k)}): K=0 {k0}", flush=True)
        top1 = {}
        for c in CURVES:
            row = runner.get(entry, k, c, layers[c])
            row["recovery"] = l2.recovery(row["top1"], k0, ref)
            top1[c] = row["top1"]
        l2._write_csv(runner.path, list(runner.rows.values()), rp.EVAL_COLUMNS)
        s = repeat_summary(k, k0, top1, ref)
        summaries.append(s)
        for c in CURVES:
            role = " (exploratory)" if c == l2.TARGETED_ERR else ""
            print(f"   {c}{role}: top1 {s[c + '_top1']} (recovery {100 * s[c + '_recovery']:.1f} %)")
        print(f"   random max {s['random_max_top1']}; targeted_r strictly greater: {s['targeted_r_beats_max']}; "
              f"targeted_err strictly greater (exploratory): {s['targeted_err_beats_max']}", flush=True)
    verdict = overall_verdict(summaries)
    for s in summaries:
        s["verdict"] = verdict
    l2._write_csv(out_dir / "deit_summary.csv", summaries, SUMMARY_COLUMNS)
    print(f"\n-> {verdict} (targeted_r not > max of the 3 random seeds in both repeats)\nwritten: {out_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=rp.OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="check layers and references, build nothing")
    args = parser.parse_args(argv)
    if args.dry_run:
        return run(args.out_dir, dry_run=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(args.out_dir / "deit_run.log"):
        print(f"DeiT-Tiny replication: K={K}, repeats {list(REPEATS)}; build + calibration + export: cpu; "
              f"evaluation: ONNX Runtime CPU", flush=True)
        try:
            return run(args.out_dir, dry_run=False)
        except Exception:
            import traceback

            traceback.print_exc()  # inside tee_output, so the exact error lands in the log
            print("PROBLEM: stopped on the exception above -- nothing changed, nothing retried")
            return 1


if __name__ == "__main__":
    sys.exit(main())
