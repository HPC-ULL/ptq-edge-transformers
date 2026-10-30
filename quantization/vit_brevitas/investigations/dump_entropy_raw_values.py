"""One-off reporting script (NOT part of the pipeline): print the raw,
individual per-repeat accuracy_value rows (top1 and top5, not mean/std) for
calibration_algorithm=entropy, across all 3 models where the row exists in
results/raw/vit_accuracy_offline.csv -- to see whether the degradation
pattern is "one catastrophic repeat among two normal ones" or "all three
degraded similarly", and whether entropy's collapse is deit_tiny-specific
or transversal across models.

Run directly:

    python quantization/vit_brevitas/investigations/dump_entropy_raw_values.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RUN_IDS = [
    "vit_deit_tiny_phase2_calibration_algorithm_entropy_int8",
    "vit_mobilevit_phase2_calibration_algorithm_entropy_int8",
    "vit_efficientvit_b1_phase2_calibration_algorithm_entropy_int8",
]


def main() -> None:
    from evaluation_offline.vit_accuracy.paths import VIT_ACCURACY_RESULTS_CSV

    if not VIT_ACCURACY_RESULTS_CSV.exists():
        print(f"No results file found at {VIT_ACCURACY_RESULTS_CSV}")
        return

    # rows[run_id][repeat_id][metric_name] = value
    rows: dict = defaultdict(lambda: defaultdict(dict))

    with open(VIT_ACCURACY_RESULTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["run_id"] in RUN_IDS:
                rows[row["run_id"]][row["repeat_id"]][row["accuracy_metric_name"]] = row["accuracy_value"]

    for run_id in RUN_IDS:
        print(f"\n=== {run_id} ===")
        if run_id not in rows:
            print("  no rows found (not yet evaluated)")
            continue
        for repeat_id in sorted(rows[run_id], key=lambda r: int(r)):
            metrics = rows[run_id][repeat_id]
            top1 = metrics.get("top1_accuracy", "?")
            top5 = metrics.get("top5_accuracy", "?")
            print(f"  rep{repeat_id}: top1={top1}%  top5={top5}%")


if __name__ == "__main__":
    main()
