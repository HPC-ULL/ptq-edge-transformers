"""Append/inspect on-device LLM benchmark results -- writes to the SAME
per-device CSV as benchmarking_ondevice/runner_vit/results_io.py
(results/raw/results_<device_id>.csv, see paths.py), since a device that
benchmarks both branches gets one file, not one per branch (run_id's
"vit_"/"llm_" prefix disambiguates rows within it).

RESULT_COLUMNS is therefore kept BYTE-IDENTICAL to runner_vit's
results_io.py (same common schema, Section 4, plus the same
num_inferences_measured/requested_repeats extension) -- both modules write
rows into what is, for a shared device, literally the same file, so their
column sets and order must never drift apart. If one changes, mirror the
change in the other. This is the same kind of accepted, documented
duplication already established between evaluation_offline/vit_accuracy/
results_io.py and evaluation_offline/llm_perplexity/results_io.py (that
pair doesn't share a file, but shares the convention of independent,
schema-identical modules over a forced common import).

Unlike ViT (latency_ms/throughput), the LLM branch fills
throughput_prefill_tps/throughput_decode_tps (Section 6.3) and leaves
latency_ms/throughput blank -- the two branches' rows are distinguished by
which pair of throughput columns is populated, not by a branch column.

`repeat_id` is Optional[int], mirroring
evaluation_offline/llm_perplexity/results_io.py's exact pattern: LLM has
real deterministic configs (baseline, legacy_corpus_none, some Phase 3
combined rows) with no calibration-repeat concept at all -- forcing an int
onto them would fabricate a "repeat 1" that doesn't exist. None ->
result_id has no "__rep" suffix and the repeat_id column is blank in the
CSV, same as the offline module.

append_result REPLACES any existing row with the same result_id rather
than duplicating it -- same fix as the other results_io.py modules.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

from .paths import results_csv_path

# Keep in sync with benchmarking_ondevice/runner_vit/results_io.py and
# benchmarking_ondevice/power_logging/results_io.py -- all three write to
# the same per-device file. energy_raw_j/energy_net_j/avg_power_w_raw/
# avg_power_w_net (Section 6.4.1) populated only when power_logging/ wraps
# this runner with energy measurement.
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


def result_id(run_id: str, device_id: str, repeat_id: Optional[int]) -> str:
    # Section 4: "{run_id}__{device}__rep{repeat_id}", no "__rep" suffix at
    # all when repeat_id is None (deterministic config).
    suffix = f"__rep{repeat_id}" if repeat_id is not None else ""
    return f"{run_id}__{device_id}{suffix}"


def build_result_row(
    run_id: str, device_id: str, repeat_id: Optional[int], parsed: Dict[str, float], requested_repeats: int
) -> Dict[str, object]:
    row: Dict[str, object] = {col: "" for col in RESULT_COLUMNS}
    row.update(
        {
            "result_id": result_id(run_id, device_id, repeat_id),
            "run_id": run_id,
            "device": device_id,
            "repeat_id": repeat_id if repeat_id is not None else "",
            "throughput_prefill_tps": parsed["throughput_prefill_tps"],
            "throughput_decode_tps": parsed["throughput_decode_tps"],
            "num_inferences_measured": parsed["num_inferences_measured"],
            "requested_repeats": requested_repeats,
        }
    )
    return row


def has_result(run_id: str, device_id: str, repeat_id: Optional[int]) -> bool:
    path = results_csv_path(device_id)
    if not path.exists():
        return False
    target = result_id(run_id, device_id, repeat_id)
    with open(path, newline="") as f:
        return any(row["result_id"] == target for row in csv.DictReader(f))


def append_result(
    run_id: str, device_id: str, repeat_id: Optional[int], parsed: Dict[str, float], requested_repeats: int
) -> Path:
    """Writes one result row to results/raw/results_<device_id>.csv,
    REPLACING any existing row with the same result_id rather than
    duplicating it -- see module docstring."""
    path = results_csv_path(device_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = build_result_row(run_id, device_id, repeat_id, parsed, requested_repeats)

    existing_rows = []
    if path.exists():
        with open(path, newline="") as f:
            existing_rows = list(csv.DictReader(f))

    kept_rows = [r for r in existing_rows if r["result_id"] != new_row["result_id"]]
    kept_rows.append(new_row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(kept_rows)

    return path
