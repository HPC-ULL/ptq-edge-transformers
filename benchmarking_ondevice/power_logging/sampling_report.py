"""Builds the power-sampling characterization files from what is on disk:

    python -m benchmarking_ondevice.power_logging.sampling_report

Reads, for every device that has results/raw/results_<device_id>.csv:
  - its energy-measured rows (those with energy_per_inference), and
  - its instrumented windows, results/characterization/power_sampling_<device_id>.csv
    (written by jetson_powermeter.measure_and_record on every measurement).

Writes, under results/characterization/ (regenerated every run):
  - power_sampling_summary.csv        one row per device (= power mode +
        backend): effective Hz, min/median samples per window, longest gap,
        and how many rows fall in each class. THE file for the
        Methodology sentence.
  - power_sampling_classification.csv one row per energy-measured results
        row and its status, so re-measurements can be targeted.

Statuses (criterion pre-registered in sampling_characterization.py):
  MEASURED_OK / MEASURED_BELOW   real instrumentation, >= / < the threshold
  ESTIMATED_OK                   no instrumentation, estimate clears the
                                 threshold by the wide margin -> accepted
  REMEASURE                      no instrumentation and the estimate does
                                 NOT clear the wide margin -> re-measure it
  NO_EFFECTIVE_HZ                nothing instrumented on this device yet, so
                                 there is no measured rate to estimate with
                                 (the configured 30 Hz is never assumed)
  NO_ESTIMATE                    the row stores no usable timing
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import paths
from . import sampling_characterization as sc

CLASSIFICATION_COLUMNS = [
    "device_id", "backend", "power_mode_w", "result_id", "run_id", "source", "n_samples", "window_s", "status",
]
SUMMARY_COLUMNS = [
    "device_id", "backend", "power_mode_w",
    "n_energy_rows", "n_measured", "n_estimated",
    "n_windows", "effective_hz_median", "effective_hz_min",
    "n_samples_min", "n_samples_median", "dt_max_s_max", "n_windows_below_threshold",
    "est_n_samples_min",
    "n_estimated_ok", "n_remeasure", "n_no_effective_hz", "n_no_estimate",
    "min_samples_threshold", "wide_margin_factor",
]


def discover_device_ids() -> List[str]:
    if not paths.RESULTS_RAW_DIR.exists():
        return []
    return sorted(p.name[len("results_"):-len(".csv")] for p in paths.RESULTS_RAW_DIR.glob("results_*.csv"))


def _energy_rows(device_id: str) -> List[Dict[str, str]]:
    path = paths.results_csv_path(device_id)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("energy_per_inference") or "").strip() != ""]


def build_report(device_ids: Sequence[str]):
    classification: List[Dict[str, object]] = []
    summary: List[Dict[str, object]] = []
    skipped: List[str] = []

    for device_id in device_ids:
        rows = _energy_rows(device_id)
        if not rows:
            continue
        try:
            entry = sc.resolve_device_entry(device_id)
        except ValueError:
            skipped.append(device_id)
            continue
        backend = entry.get("backend", "")
        power_mode_w = "" if entry.get("power_mode_w") is None else entry["power_mode_w"]

        records = sc.read_sampling_records(device_id)
        by_id = {r["result_id"]: r for r in records}
        hz = sc.device_effective_hz(records)

        statuses = []
        for row in rows:
            verdict = sc.classify_row(row, by_id.get(row["result_id"]), hz)
            statuses.append(verdict)
            classification.append({
                "device_id": device_id, "backend": backend, "power_mode_w": power_mode_w,
                "result_id": row["result_id"], "run_id": row["run_id"], **verdict,
            })

        counts = Counter(v["status"] for v in statuses)
        estimated = [v["n_samples"] for v in statuses if v["source"] == "estimated" and v["n_samples"] is not None]
        instrumented_ids = {row["result_id"] for row in rows} & set(by_id)
        group = sc.summarize_group([by_id[i] for i in sorted(instrumented_ids)])
        summary.append({
            "device_id": device_id, "backend": backend, "power_mode_w": power_mode_w,
            "n_energy_rows": len(rows),
            "n_measured": counts[sc.MEASURED_OK] + counts[sc.MEASURED_BELOW],
            "n_estimated": len(rows) - counts[sc.MEASURED_OK] - counts[sc.MEASURED_BELOW],
            **group,
            "est_n_samples_min": min(estimated) if estimated else "",
            "n_estimated_ok": counts[sc.ESTIMATED_OK], "n_remeasure": counts[sc.REMEASURE],
            "n_no_effective_hz": counts[sc.NO_EFFECTIVE_HZ], "n_no_estimate": counts[sc.NO_ESTIMATE],
            "min_samples_threshold": sc.MIN_SAMPLES_PER_WINDOW, "wide_margin_factor": sc.WIDE_MARGIN_FACTOR,
        })
    return classification, summary, skipped


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value, spec: str = "g") -> str:
    return "-" if value == "" or value is None else format(value, spec)


def format_table(summary: Sequence[Dict[str, object]]) -> str:
    header = (f"{'device_id':<18}{'backend':<8}{'W':>4}{'rows':>6}{'meas':>6}{'est':>5}"
              f"{'Hz_med':>8}{'Hz_min':>8}{'n_min':>7}{'n_med':>7}{'gap_max_ms':>11}"
              f"{'<thr':>6}{'REMEAS':>7}{'noHz':>6}")
    lines = [header, "-" * len(header)]
    for s in summary:
        gap_ms = "" if s["dt_max_s_max"] == "" else s["dt_max_s_max"] * 1000.0
        lines.append(
            f"{s['device_id']:<18}{s['backend']:<8}{_fmt(s['power_mode_w']):>4}{s['n_energy_rows']:>6}"
            f"{s['n_measured']:>6}{s['n_estimated']:>5}{_fmt(s['effective_hz_median'], '.1f'):>8}"
            f"{_fmt(s['effective_hz_min'], '.1f'):>8}{_fmt(s['n_samples_min'], '.0f'):>7}"
            f"{_fmt(s['n_samples_median'], '.0f'):>7}{_fmt(gap_ms, '.1f'):>11}"
            f"{s['n_windows_below_threshold']:>6}{s['n_remeasure']:>7}{s['n_no_effective_hz']:>6}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    classification, summary, skipped = build_report(discover_device_ids())

    _write_csv(paths.sampling_classification_csv_path(), CLASSIFICATION_COLUMNS, classification)
    _write_csv(paths.sampling_summary_csv_path(), SUMMARY_COLUMNS, summary)

    print(f"Pre-registered criterion: >= {sc.MIN_SAMPLES_PER_WINDOW} samples per window; "
          f"an estimate is only accepted with >= {sc.WIDE_MARGIN_FACTOR * sc.MIN_SAMPLES_PER_WINDOW} "
          f"({sc.WIDE_MARGIN_FACTOR}x the threshold).\n")
    print(format_table(summary) if summary else "(no rows with measured energy)")
    for device_id in skipped:
        print(f"[warning] {device_id}: has results but is not in configs/devices.yaml -- skipped")

    todo = [c for c in classification if c["status"] in (sc.MEASURED_BELOW, sc.REMEASURE)]
    if todo:
        print(f"\n{len(todo)} row(s) to remeasure with real instrumentation (MEASURED_BELOW / REMEASURE):")
        for c in todo:
            print(f"  - {c['device_id']}: {c['result_id']}  [{c['status']}, n={_fmt(c['n_samples'], '.0f')}]")
    print(f"\nWritten: {paths.sampling_summary_csv_path()}\n         {paths.sampling_classification_csv_path()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
