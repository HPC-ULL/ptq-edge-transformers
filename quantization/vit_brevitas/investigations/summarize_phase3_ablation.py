"""One-off reporting script (NOT part of the pipeline): summarize ViT
Phase 3's full ablation from results/raw/vit_accuracy_offline.csv -- for
each of the 12 configurations (3 models x {baseline,combined} x
{int6,int4}), mean/stdev of top1_accuracy/top5_accuracy across the 3
repeats.

Anomaly detection is the SAME criterion already used for the Phase 1-2
summary (summarize_model_ablation.py) -- the one that originally caught the
entropy bug -- reused verbatim here, not reinvented: flag a configuration's
stdev if it exceeds STD_OUTLIER_FACTOR times the MEDIAN stdev among that
SAME MODEL's other configurations. Scoped per-model (not globally across
all 12) since "the rest of the same model's configurations" is the
explicit comparison basis, mirroring exactly how the Phase 1-2 version
computes its median within one model's own config set. Phase 3's shape
differs from Phase 1-2's (4 configs/model here vs. 7 there, no
calibration_algorithm axis to exclude), so this is a dedicated script
rather than a forced generalization of summarize_model_ablation.py.

Run directly:

    python quantization/vit_brevitas/investigations/summarize_phase3_ablation.py
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

MODELS = ["deit_tiny", "mobilevit", "efficientvit_b1"]
STD_OUTLIER_FACTOR = 3.0
EXPECTED_CONFIGS = 4  # {baseline,combined} x {int6,int4}


def _summarize_model(model_rows: list) -> list:
    values: dict = defaultdict(lambda: defaultdict(list))
    labels: dict = {}
    for row in model_rows:
        run_id = row["run_id"]
        values[run_id][row["accuracy_metric_name"]].append(float(row["accuracy_value"]))
        labels[run_id] = f"{row['factor_level']}_{row['bitwidth']}"

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
    return summary


def main() -> None:
    from evaluation_offline.vit_accuracy.paths import VIT_ACCURACY_RESULTS_CSV

    if not VIT_ACCURACY_RESULTS_CSV.exists():
        print(f"No results file found at {VIT_ACCURACY_RESULTS_CSV}")
        return

    with open(VIT_ACCURACY_RESULTS_CSV, newline="") as f:
        all_rows = [r for r in csv.DictReader(f) if "phase3" in r["run_id"]]

    if not all_rows:
        print("No Phase 3 rows found.")
        return

    any_flags = False
    for model in MODELS:
        model_rows = [r for r in all_rows if r["model_name"] == model]
        if not model_rows:
            print(f"=== {model}: no Phase 3 rows found ===\n")
            continue

        summary = _summarize_model(model_rows)

        all_stds = [r["top1_std"] for r in summary if r["top1_std"] == r["top1_std"]]
        all_stds += [r["top5_std"] for r in summary if r["top5_std"] == r["top5_std"]]
        median_std = statistics.median(all_stds) if all_stds else 0.0

        print(f"=== {model} Phase 3 ablation summary ({len(summary)} configurations found; "
              f"expected {EXPECTED_CONFIGS}; n reps expected 3) ===")
        header = f"{'config':<20} {'n':>3} {'top1 mean':>10} {'top1 std':>9} {'top5 mean':>10} {'top5 std':>9}"
        print(header)
        print("-" * len(header))

        flags = []
        if len(summary) != EXPECTED_CONFIGS:
            flags.append(f"  UNEXPECTED CONFIG COUNT: {len(summary)} found, {EXPECTED_CONFIGS} expected")

        for r in summary:
            n_note = "" if r["n_top1"] == 3 and r["n_top5"] == 3 else f"  (n_top1={r['n_top1']}, n_top5={r['n_top5']})"
            print(f"{r['label']:<20} {r['n_top1']:>3} {r['top1_mean']:>9.2f}% {r['top1_std']:>8.2f}% "
                  f"{r['top5_mean']:>9.2f}% {r['top5_std']:>8.2f}%{n_note}")

            if median_std > 0:
                if r["top1_std"] == r["top1_std"] and r["top1_std"] > STD_OUTLIER_FACTOR * median_std:
                    flags.append(f"  ANOMALOUS top1 std: {r['label']} ({r['top1_std']:.2f}% vs. "
                                  f"median {median_std:.2f}%)")
                if r["top5_std"] == r["top5_std"] and r["top5_std"] > STD_OUTLIER_FACTOR * median_std:
                    flags.append(f"  ANOMALOUS top5 std: {r['label']} ({r['top5_std']:.2f}% vs. "
                                  f"median {median_std:.2f}%)")

        if flags:
            any_flags = True
            print("Flags:")
            for flag in flags:
                print(flag)
        else:
            print("No flags: no cross-repeat stdev is anomalously high relative to this model's "
                  "other Phase 3 configurations.")
        print()

    if not any_flags:
        print("Overall: no anomalies flagged across any of the 3 models' Phase 3 configurations.")


if __name__ == "__main__":
    main()
