"""Writes the combined (performance + energy) row to the SAME per-device
CSV as runner_vit/results_io.py and runner_llm/results_io.py.

Unlike those two modules, this one doesn't build the row itself -- the row
is already fully built by jetson_powermeter.py (via the wrapped branch's
own results_io.build_result_row, then merged with translated energy
fields, see that module's docstring for why). append_result here just
takes the finished row dict and performs the same replace-by-result_id
write as the other two.

RESULT_COLUMNS is kept BYTE-IDENTICAL to the other two modules' copies (see
their docstrings) -- if one changes, mirror the change in all three.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

from .paths import results_csv_path

RESULT_COLUMNS = [
    "result_id", "run_id", "model_family", "model_name", "phase", "factor_varied", "factor_level",
    "calibration_algorithm", "granularity", "symmetry",
    "calibration_corpus", "quant_scheme",
    "calib_set_size", "bitwidth", "device", "repeat_id",
    "accuracy_metric_name", "accuracy_value", "latency_ms",
    "throughput", "throughput_prefill_tps", "throughput_decode_tps",
    "energy_per_inference", "perf_per_watt", "model_size_mb",
    "num_inferences_measured", "requested_repeats",
    "energy_raw_j", "energy_net_j", "avg_power_w_raw", "avg_power_w_net",
]


def append_result(device_id: str, row: Dict[str, object]) -> Path:
    """Writes `row` (already fully built, with `result_id` set) to
    results/raw/results_<device_id>.csv, REPLACING any existing row with
    the same result_id -- same semantics as the other two results_io.py
    modules."""
    path = results_csv_path(device_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = []
    if path.exists():
        with open(path, newline="") as f:
            existing_rows = list(csv.DictReader(f))

    kept_rows = [r for r in existing_rows if r["result_id"] != row["result_id"]]
    kept_rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(kept_rows)

    return path
