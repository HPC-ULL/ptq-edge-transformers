"""One-off reporting script (NOT part of the pipeline): summarize a
model's full Phase 1-2 ablation from results/raw/vit_accuracy_offline.csv
-- mean/stdev of top1_accuracy and top5_accuracy across the 3 repeats, for
each valid configuration (baseline + 6 other OFAT variants -- entropy is
excluded on purpose: it's blocked in quantizer_config.py, confirmed
unreliable in Brevitas 0.13.0, see investigations/diagnose_entropy_*.py),
sorted by mean top1 descending, flagging any config below a top1 floor or
with an anomalously high cross-repeat stdev relative to the others.

Generalized from the original deit_tiny-only summarize_deit_ablation.py
(now superseded/removed) to take model_name as an argument, so the same
script covers all 3 models.

Run directly, naming the model (default "deit_tiny"):

    python quantization/vit_brevitas/investigations/summarize_model_ablation.py deit_tiny
    python quantization/vit_brevitas/investigations/summarize_model_ablation.py mobilevit
    python quantization/vit_brevitas/investigations/summarize_model_ablation.py efficientvit_b1
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

TOP1_FLOOR = 20.0
STD_OUTLIER_FACTOR = 3.0  # flag a config's stdev if it's this many times the median stdev
EXCLUDED_CALIBRATION_ALGORITHMS = {"entropy"}
EXPECTED_CONFIGS = 7  # 8 OFAT configs - 1 excluded (entropy)


def main() -> None:
    from evaluation_offline.vit_accuracy.paths import VIT_ACCURACY_RESULTS_CSV

    model_name = sys.argv[1] if len(sys.argv) > 1 else "deit_tiny"

    if not VIT_ACCURACY_RESULTS_CSV.exists():
        print(f"No results file found at {VIT_ACCURACY_RESULTS_CSV}")
        return

    # values[run_id][metric_name] = list of accuracy_value across repeats
    values: dict = defaultdict(lambda: defaultdict(list))
    labels: dict = {}
    excluded_rows_seen = 0

    with open(VIT_ACCURACY_RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["model_name"] != model_name:
                continue
            if row["calibration_algorithm"] in EXCLUDED_CALIBRATION_ALGORITHMS:
                excluded_rows_seen += 1
                continue
            run_id = row["run_id"]
            values[run_id][row["accuracy_metric_name"]].append(float(row["accuracy_value"]))
            labels[run_id] = f"{row['factor_varied']}={row['factor_level']}"

    if not values:
        print(f"No rows found for model_name={model_name!r} in {VIT_ACCURACY_RESULTS_CSV}")
        return

    summary = []
    for run_id, metrics in values.items():
        top1_vals = metrics.get("top1_accuracy", [])
        top5_vals = metrics.get("top5_accuracy", [])
        summary.append({
            "run_id": run_id,
            "label": labels[run_id],
            "n_top1": len(top1_vals),
            "n_top5": len(top5_vals),
            "top1_mean": statistics.mean(top1_vals) if top1_vals else float("nan"),
            "top1_std": statistics.stdev(top1_vals) if len(top1_vals) > 1 else float("nan"),
            "top5_mean": statistics.mean(top5_vals) if top5_vals else float("nan"),
            "top5_std": statistics.stdev(top5_vals) if len(top5_vals) > 1 else float("nan"),
        })

    summary.sort(key=lambda r: r["top1_mean"], reverse=True)

    all_stds = [r["top1_std"] for r in summary if r["top1_std"] == r["top1_std"]]  # drop NaN
    all_stds += [r["top5_std"] for r in summary if r["top5_std"] == r["top5_std"]]
    median_std = statistics.median(all_stds) if all_stds else 0.0

    excl_note = f" ({excluded_rows_seen} entropy row(s) excluded)" if excluded_rows_seen else ""
    print(f"{model_name} ablation summary ({len(summary)} configurations found; "
          f"expected {EXPECTED_CONFIGS}; n reps expected 3){excl_note}\n")
    header = f"{'config':<45} {'n':>3} {'top1 mean':>10} {'top1 std':>9} {'top5 mean':>10} {'top5 std':>9}"
    print(header)
    print("-" * len(header))

    flags = []
    if len(summary) != EXPECTED_CONFIGS:
        flags.append(f"  UNEXPECTED CONFIG COUNT: {len(summary)} found, {EXPECTED_CONFIGS} expected")

    for r in summary:
        n_note = "" if r["n_top1"] == 3 and r["n_top5"] == 3 else f"  (n_top1={r['n_top1']}, n_top5={r['n_top5']})"
        print(f"{r['label']:<45} {r['n_top1']:>3} {r['top1_mean']:>9.2f}% {r['top1_std']:>8.2f}% "
              f"{r['top5_mean']:>9.2f}% {r['top5_std']:>8.2f}%{n_note}")

        if r["top1_mean"] == r["top1_mean"] and r["top1_mean"] < TOP1_FLOOR:
            flags.append(f"  BELOW {TOP1_FLOOR:.0f}% top1: {r['label']} ({r['top1_mean']:.2f}%)")
        if median_std > 0:
            if r["top1_std"] == r["top1_std"] and r["top1_std"] > STD_OUTLIER_FACTOR * median_std:
                flags.append(f"  ANOMALOUS top1 std: {r['label']} ({r['top1_std']:.2f}% vs. "
                              f"median {median_std:.2f}%)")
            if r["top5_std"] == r["top5_std"] and r["top5_std"] > STD_OUTLIER_FACTOR * median_std:
                flags.append(f"  ANOMALOUS top5 std: {r['label']} ({r['top5_std']:.2f}% vs. "
                              f"median {median_std:.2f}%)")

    print()
    if flags:
        print("Flags:")
        for flag in flags:
            print(flag)
    else:
        print("No flags: all configurations are >= %.0f%% top1, and no cross-repeat stdev is "
              "anomalously high relative to the others." % TOP1_FLOOR)


if __name__ == "__main__":
    main()
