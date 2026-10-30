"""Append/inspect on-device ViT benchmark results, one CSV per device
(results/raw/results_<device_id>.csv per technical_architecture.md
Section 4's stated per-machine/per-person convention).

Same common results schema as evaluation_offline/vit_accuracy/results_io.py
and evaluation_offline/llm_perplexity/results_io.py (Section 4), so this
file can eventually be concatenated with theirs by the not-yet-built
results-combination script -- fields that don't apply to an on-device
latency/throughput row (the offline-only calibration_algorithm/
calib_set_size/accuracy_metric_name/accuracy_value fields, and the
model_family/model_name/phase/factor_varied/factor_level fields this CLI
doesn't currently resolve from the manifest -- it only ever sees a bare
run_id string + artifact path, not the manifest entry) are left blank, per
the project's established "not applicable -> blank" convention. Blank here
means "not filled by this CLI as designed", not "inapplicable to the ViT
branch" -- a later analysis step can still recover model_name/factor_varied
etc. by joining on run_id against the manifest if needed.

Two columns beyond the common schema, appended at the end rather than
inserted into the middle (keeps the shared prefix identical to the other
two results_io.py modules): `num_inferences_measured` (parsed from the
binary's own output, output_parser.py) and `requested_repeats` (the -r
value actually passed for this run) -- Section 6 requires the -r value
used per device to stay traceable, not just implied by whatever
num_inferences_measured happened to come back.

append_result REPLACES any existing row with the same result_id rather
than duplicating it -- same fix already applied to the other two
results_io.py modules after a real duplication bug there (see
evaluation_offline/vit_accuracy/results_io.py's docstring for the full
story); applied here proactively, since a rerun against the same
(run_id, device_id, repeat_id) is exactly the same kind of "recompute and
overwrite" scenario that caused it there.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

from .paths import results_csv_path

# Same common results schema as the other two results_io.py modules
# (Section 4), plus num_inferences_measured/requested_repeats (see module
# docstring) and energy_raw_j/energy_net_j/avg_power_w_raw/avg_power_w_net
# (Section 6.4.1, populated only when benchmarking_ondevice/power_logging/
# wraps this runner with energy measurement -- plain runs leave them blank)
# appended at the end. Keep in sync with
# benchmarking_ondevice/runner_llm/results_io.py and
# benchmarking_ondevice/power_logging/results_io.py -- all three write to
# the same per-device file.
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


def result_id(run_id: str, device_id: str, repeat_id: int) -> str:
    # Section 4: "{run_id}__{device}__rep{repeat_id}"
    return f"{run_id}__{device_id}__rep{repeat_id}"


def build_result_row(
    run_id: str, device_id: str, repeat_id: int, parsed: Dict[str, float], requested_repeats: int
) -> Dict[str, object]:
    row: Dict[str, object] = {col: "" for col in RESULT_COLUMNS}
    row.update(
        {
            "result_id": result_id(run_id, device_id, repeat_id),
            "run_id": run_id,
            "device": device_id,
            "repeat_id": repeat_id,
            "latency_ms": parsed["avg_latency_ms"],
            "throughput": parsed["throughput_ips"],
            "num_inferences_measured": parsed["num_inferences_measured"],
            "requested_repeats": requested_repeats,
        }
    )
    return row


def has_result(run_id: str, device_id: str, repeat_id: int) -> bool:
    path = results_csv_path(device_id)
    if not path.exists():
        return False
    target = result_id(run_id, device_id, repeat_id)
    with open(path, newline="") as f:
        return any(row["result_id"] == target for row in csv.DictReader(f))


def append_result(
    run_id: str, device_id: str, repeat_id: int, parsed: Dict[str, float], requested_repeats: int
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
