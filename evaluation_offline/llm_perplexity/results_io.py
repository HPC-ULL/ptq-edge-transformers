"""Append/inspect offline LLM perplexity results in the project's common results schema.

Columns that don't apply to an offline, device-independent accuracy row
(the ViT-only calibration_algorithm/granularity/symmetry/throughput fields,
and the on-device latency/throughput/energy/model_size fields, which this
module never measures) are left blank, per the project's established "not
applicable -> blank" convention.

`device` is the one exception: it is set to the literal string "offline",
per technical_architecture.md Section 4 ("`device`... o `offline`
para accuracy/perplexity") -- the results-combination script validates that
every row's `device` is non-empty and appears in devices.yaml (Section 4 /
10.5), so leaving it blank would fail that check, not just look incomplete.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

from .paths import LLM_PERPLEXITY_RESULTS_CSV

# Full common results schema (shared with the ViT branch), in the project's
# documented column order -- keeping every column here, not just the ones
# this module fills in, is what lets this CSV be concatenated directly with
# the ViT/on-device raw CSVs during results combination.
RESULT_COLUMNS = [
    "result_id", "run_id", "model_family", "model_name", "phase", "factor_varied", "factor_level",
    "calibration_algorithm", "granularity", "symmetry",
    "calibration_corpus", "quant_scheme",
    "calib_set_size", "bitwidth", "device", "repeat_id",
    "accuracy_metric_name", "accuracy_value", "latency_ms",
    "throughput", "throughput_prefill_tps", "throughput_decode_tps",
    "energy_per_inference", "perf_per_watt", "model_size_mb",
]


def result_id(run_id: str, repeat_id: Optional[int]) -> str:
    suffix = f"__rep{repeat_id}" if repeat_id is not None else ""
    return f"{run_id}__offline{suffix}"


def build_result_row(entry: Dict, repeat_id: Optional[int], perplexity: float) -> Dict[str, object]:
    row: Dict[str, object] = {col: "" for col in RESULT_COLUMNS}
    row.update(
        {
            "result_id": result_id(entry["run_id"], repeat_id),
            "run_id": entry["run_id"],
            "model_family": entry["model_family"],
            "model_name": entry["model_name"],
            "phase": entry["phase"],
            "factor_varied": entry["factor_varied"],
            "factor_level": entry["factor_level"],
            "calibration_corpus": entry.get("calibration_corpus", ""),
            "quant_scheme": entry.get("quant_scheme", ""),
            "calib_set_size": entry.get("calib_set_size") if entry.get("calib_set_size") is not None else "",
            "bitwidth": entry["bitwidth"],
            "device": "offline",
            "repeat_id": repeat_id if repeat_id is not None else "",
            "accuracy_metric_name": "perplexity",
            "accuracy_value": perplexity,
        }
    )
    return row


def has_result(run_id: str, repeat_id: Optional[int]) -> bool:
    if not LLM_PERPLEXITY_RESULTS_CSV.exists():
        return False
    target = result_id(run_id, repeat_id)
    with open(LLM_PERPLEXITY_RESULTS_CSV, newline="") as f:
        return any(row["result_id"] == target for row in csv.DictReader(f))


def append_result(entry: Dict, repeat_id: Optional[int], perplexity: float) -> Path:
    """Writes one result row, REPLACING any existing row with the same
    result_id rather than duplicating it -- mirrors the identical fix in
    evaluation_offline/vit_accuracy/results_io.py, applied here too since
    the same unconditional-append bug exists in this file (confirmed by
    inspection, not yet actually triggered on this branch, but the same
    latent risk the moment --force is used to re-evaluate an already
    -recorded result_id). See that module's append_result for the full
    story of how this was caught.

    Rewrites the whole file rather than a true in-place update (CSVs don't
    support that) -- an O(n) cost per write, fine at this project's scale."""
    LLM_PERPLEXITY_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_row = build_result_row(entry, repeat_id, perplexity)

    existing_rows = []
    if LLM_PERPLEXITY_RESULTS_CSV.exists():
        with open(LLM_PERPLEXITY_RESULTS_CSV, newline="") as f:
            existing_rows = list(csv.DictReader(f))

    kept_rows = [r for r in existing_rows if r["result_id"] != new_row["result_id"]]
    kept_rows.append(new_row)

    with open(LLM_PERPLEXITY_RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(kept_rows)

    return LLM_PERPLEXITY_RESULTS_CSV
