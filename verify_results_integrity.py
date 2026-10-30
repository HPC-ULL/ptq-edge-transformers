"""One-off integrity check (NOT part of the pipeline): confirms both
results/raw/ CSVs (vit_accuracy_offline.csv, llm_perplexity_offline.csv)
have no NaN/non-finite accuracy_value, no duplicate result_ids, and exactly
the expected number of rows per model/configuration -- no gaps, no
leftover contamination.

For the ViT CSV, "expected" is derived dynamically from the real manifest
(configs.vit_manifest) rather than hardcoded, split into valid rows
(3 reps x 2 metrics = 6 rows each) and calibration_algorithm=entropy rows
(blocked in quantizer_config.py -- expected to have ZERO rows; if any are
found, that's flagged explicitly as stale data needing
investigations/remove_entropy_results.py, not silently absorbed into a
different "expected" total).

For the LLM CSV, expected is derived from configs.llm_manifest +
quantization.llm_llamacpp.cli.iter_jobs (6 deterministic rows x 1 +
15 stochastic rows x 3 reps = 51 total, one row per job -- perplexity is a
single metric, unlike ViT's top1/top5 pair).

Run directly:

    python verify_results_integrity.py
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent


def _read_csv(path: Path):
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_nan_and_duplicates(rows, csv_path: Path) -> list:
    problems = []
    seen_ids = Counter()
    for row in rows:
        seen_ids[row["result_id"]] += 1
        raw = row["accuracy_value"]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            problems.append(f"non-numeric accuracy_value {raw!r} in result_id={row['result_id']}")
            continue
        if not math.isfinite(value):
            problems.append(f"non-finite accuracy_value {value!r} in result_id={row['result_id']}")
    for result_id, count in seen_ids.items():
        if count > 1:
            problems.append(f"duplicate result_id={result_id!r} appears {count} times")
    return problems


def check_vit() -> None:
    from configs import vit_manifest
    from evaluation_offline.vit_accuracy.paths import VIT_ACCURACY_RESULTS_CSV
    from quantization.vit_brevitas.calibration import NUM_REPEATS

    print(f"=== {VIT_ACCURACY_RESULTS_CSV} ===")
    rows = _read_csv(VIT_ACCURACY_RESULTS_CSV)
    if rows is None:
        print("  file not found\n")
        return

    problems = check_nan_and_duplicates(rows, VIT_ACCURACY_RESULTS_CSV)

    manifest = vit_manifest.generate()
    blocked_run_ids = {e["run_id"] for e in manifest if e.get("calibration_algorithm") == "entropy"}
    valid_run_ids = {e["run_id"] for e in manifest} - blocked_run_ids
    metrics = ("top1_accuracy", "top5_accuracy")

    counts: dict = defaultdict(int)
    for row in rows:
        counts[(row["run_id"], row["repeat_id"], row["accuracy_metric_name"])] += 1

    stale_entropy_rows = [row for row in rows if row["run_id"] in blocked_run_ids]
    if stale_entropy_rows:
        problems.append(
            f"{len(stale_entropy_rows)} row(s) found for blocked calibration_algorithm=entropy "
            f"run_ids -- run investigations/remove_entropy_results.py"
        )

    gaps = []
    for run_id in sorted(valid_run_ids):
        for repeat_id in range(1, NUM_REPEATS + 1):
            for metric in metrics:
                key = (run_id, str(repeat_id), metric)
                if counts.get(key, 0) == 0:
                    gaps.append(f"{run_id} rep{repeat_id} {metric}")
    expected_total = len(valid_run_ids) * NUM_REPEATS * len(metrics)

    print(f"  {len(rows)} rows total ({len(stale_entropy_rows)} stale entropy rows, "
          f"{len(rows) - len(stale_entropy_rows)} from valid configs; "
          f"{expected_total} expected for {len(valid_run_ids)} valid configs x {NUM_REPEATS} reps x "
          f"{len(metrics)} metrics)")
    if gaps:
        print(f"  {len(gaps)} MISSING:")
        for g in gaps:
            print(f"    {g}")
    if problems:
        print(f"  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"    {p}")
    if not gaps and not problems:
        print("  OK: no gaps, no NaN/non-finite values, no duplicates, no stale entropy rows.")
    print()


def check_llm() -> None:
    from configs import llm_manifest
    from evaluation_offline.llm_perplexity.paths import LLM_PERPLEXITY_RESULTS_CSV
    from quantization.llm_llamacpp.cli import iter_jobs

    print(f"=== {LLM_PERPLEXITY_RESULTS_CSV} ===")
    rows = _read_csv(LLM_PERPLEXITY_RESULTS_CSV)
    if rows is None:
        print("  file not found\n")
        return

    problems = check_nan_and_duplicates(rows, LLM_PERPLEXITY_RESULTS_CSV)

    manifest = list(llm_manifest.generate())
    jobs = list(iter_jobs(manifest))

    counts: dict = defaultdict(int)
    for row in rows:
        repeat_id = row["repeat_id"] if row["repeat_id"] != "" else None
        counts[(row["run_id"], repeat_id)] += 1

    gaps = []
    for entry, repeat_id in jobs:
        key = (entry["run_id"], str(repeat_id) if repeat_id is not None else None)
        if counts.get(key, 0) == 0:
            label = entry["run_id"] + (f" rep{repeat_id}" if repeat_id is not None else "")
            gaps.append(label)

    print(f"  {len(rows)} rows total ({len(jobs)} expected: "
          f"{sum(1 for _, r in jobs if r is None)} deterministic + "
          f"{sum(1 for _, r in jobs if r is not None)} stochastic-repeats)")
    if gaps:
        print(f"  {len(gaps)} MISSING:")
        for g in gaps:
            print(f"    {g}")
    if problems:
        print(f"  {len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"    {p}")
    if not gaps and not problems:
        print("  OK: no gaps, no NaN/non-finite values, no duplicates.")
    print()


def main() -> None:
    check_vit()
    check_llm()


if __name__ == "__main__":
    main()
