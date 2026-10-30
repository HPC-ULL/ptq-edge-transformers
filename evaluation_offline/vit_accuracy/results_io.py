"""Append/inspect offline ViT accuracy results in the project's common results schema.

Two rows per (run_id, repeat_id) -- one for top1_accuracy, one for
top5_accuracy -- since the common schema has a single accuracy_metric_name/
accuracy_value pair per row (mirrors how a single LLM perplexity row works,
just doubled since ViT reports two metrics instead of one).

Columns that don't apply to an offline, device-independent accuracy row
(the LLM-only calibration_corpus/quant_scheme fields, and the on-device
latency/throughput/energy/model_size fields, which this module never
measures) are left blank. `device` is set to the literal string "offline"
per technical_architecture.md Section 4 -- see
evaluation_offline/llm_perplexity/results_io.py for the same convention.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

from .paths import VIT_ACCURACY_RESULTS_CSV

# Full common results schema (shared with the LLM branch), in the project's
# documented column order.
RESULT_COLUMNS = [
    "result_id", "run_id", "model_family", "model_name", "phase", "factor_varied", "factor_level",
    "calibration_algorithm", "granularity", "symmetry",
    "calibration_corpus", "quant_scheme",
    "calib_set_size", "bitwidth", "device", "repeat_id",
    "accuracy_metric_name", "accuracy_value", "latency_ms",
    "throughput", "throughput_prefill_tps", "throughput_decode_tps",
    "energy_per_inference", "perf_per_watt", "model_size_mb",
]


def result_id(run_id: str, repeat_id: int, metric_name: str) -> str:
    return f"{run_id}__offline__rep{repeat_id}__{metric_name}"


def build_result_row(entry: Dict, repeat_id: int, metric_name: str, value: float) -> Dict[str, object]:
    row: Dict[str, object] = {col: "" for col in RESULT_COLUMNS}
    row.update(
        {
            "result_id": result_id(entry["run_id"], repeat_id, metric_name),
            "run_id": entry["run_id"],
            "model_family": entry["model_family"],
            "model_name": entry["model_name"],
            "phase": entry["phase"],
            "factor_varied": entry["factor_varied"],
            "factor_level": entry["factor_level"],
            "calibration_algorithm": entry.get("calibration_algorithm", ""),
            "granularity": entry.get("granularity", ""),
            "symmetry": entry.get("symmetry", ""),
            "calib_set_size": entry.get("calib_set_size", ""),
            "bitwidth": entry["bitwidth"],
            "device": "offline",
            "repeat_id": repeat_id,
            "accuracy_metric_name": metric_name,
            "accuracy_value": value,
        }
    )
    return row


def has_result(run_id: str, repeat_id: int, metric_name: str) -> bool:
    if not VIT_ACCURACY_RESULTS_CSV.exists():
        return False
    target = result_id(run_id, repeat_id, metric_name)
    with open(VIT_ACCURACY_RESULTS_CSV, newline="") as f:
        return any(row["result_id"] == target for row in csv.DictReader(f))


def append_result(entry: Dict, repeat_id: int, metric_name: str, value: float) -> Path:
    """Writes one result row, REPLACING any existing row with the same
    result_id rather than duplicating it -- confirmed as a real, previously
    -latent bug: evaluate_accuracy(force=True) recomputes and calls this
    to overwrite a stale value, but a plain unconditional append left BOTH
    the old and new rows in the CSV under the identical result_id, silently
    corrupting any aggregate statistic (mean/stdev) computed over all
    matching rows. Surfaced when Phase 3's INT6/INT4 fix was re-evaluated
    with --force without first removing the stale pre-fix rows -- see
    quantization/vit_brevitas/investigations/
    remove_phase3_duplicate_results.py for the one-off cleanup that caused.

    Rewrites the whole file rather than a true in-place update (CSVs don't
    support that) -- an O(n) cost per write, fine at this project's scale
    (low hundreds of rows), traded deliberately for correctness."""
    VIT_ACCURACY_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_row = build_result_row(entry, repeat_id, metric_name, value)

    existing_rows = []
    if VIT_ACCURACY_RESULTS_CSV.exists():
        with open(VIT_ACCURACY_RESULTS_CSV, newline="") as f:
            existing_rows = list(csv.DictReader(f))

    kept_rows = [r for r in existing_rows if r["result_id"] != new_row["result_id"]]
    kept_rows.append(new_row)

    with open(VIT_ACCURACY_RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(kept_rows)

    return VIT_ACCURACY_RESULTS_CSV
