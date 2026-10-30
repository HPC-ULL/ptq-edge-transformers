"""One-off offline fix: recompute perf_per_watt of every LLM row of the
per-device results CSVs with the corrected formula
(benchmarking_ondevice/power_logging/perf_per_watt.py):

    perf_per_watt = (PROMPT_TOKENS + GEN_TOKENS) * N / energy_net_j

with each row's own N (num_inferences_measured). Rows written before the
fix hold (n_prompt + n_gen) / energy_net_j, i.e. the correct value divided
by N, so every changed row should report factor == N.

- Only LLM rows (run_id "llm_...") with an energy measurement are
  recomputed. ViT rows and every other column are written back exactly
  as read (same strings, same column order).
- Before rewriting a CSV, a copy is saved as <name>.csv.bak_<timestamp> in
  results/backups/ (the sibling of the results/raw/ folder the CSV is in --
  raw/ holds only data); a CSV that is not in a results/raw/ folder gets its
  backup next to it. Files with nothing to change are not rewritten (and get
  no backup).
- Idempotent: the value is derived from energy_net_j and N, never from the
  stored perf_per_watt, so running it again changes nothing.

    python scripts/recompute_llm_perf_per_watt.py                  # results/raw/results_*.csv
    python scripts/recompute_llm_perf_per_watt.py path/a.csv b.csv # specific files (e.g. a colleague's)
    python scripts/recompute_llm_perf_per_watt.py --dry-run        # report only, write nothing
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarking_ondevice.power_logging.paths import RESULTS_RAW_DIR  # noqa: E402
from benchmarking_ondevice.power_logging.perf_per_watt import is_llm_row, llm_perf_per_watt_from_row  # noqa: E402

REL_TOL = 1e-9


def _float_or_none(value: str) -> Optional[float]:
    value = (value or "").strip()
    return float(value) if value else None


def recompute_rows(rows: List[Dict[str, str]]) -> Dict:
    """Rewrites perf_per_watt in place on the LLM rows. Returns a report."""
    report = {"llm_rows": 0, "vit_or_other_rows": 0, "no_energy": 0, "unchanged": 0, "changed": 0,
              "factors": Counter(), "factor_not_n": []}
    for row in rows:
        if not is_llm_row(row):
            report["vit_or_other_rows"] += 1
            continue
        report["llm_rows"] += 1
        new = llm_perf_per_watt_from_row(row)
        if new is None:
            report["no_energy"] += 1
            continue
        old = _float_or_none(row.get("perf_per_watt", ""))
        if old is not None and abs(new - old) <= REL_TOL * abs(new):
            report["unchanged"] += 1
            continue
        report["changed"] += 1
        n = int(float(row["num_inferences_measured"]))
        if old:
            factor = new / old
            report["factors"][round(factor, 6)] += 1
            if abs(factor - n) > 1e-6 * n:
                report["factor_not_n"].append((row["result_id"], n, factor))
        else:
            report["factors"]["(previously empty)"] += 1
        row["perf_per_watt"] = repr(new)
    return report


def backup_dir_for(path: Path) -> Path:
    """results/backups/ for a CSV in a results/raw/ folder, else the CSV's own folder."""
    folder = path.resolve().parent
    if folder.name == "raw" and folder.parent.name == "results":
        return folder.parent / "backups"
    return path.parent


def process_file(path: Path, *, dry_run: bool, stamp: str) -> Dict:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    report = recompute_rows(rows)
    report["backup"] = None
    if report["changed"] and not dry_run:
        backup = backup_dir_for(path) / f"{path.name}.bak_{stamp}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        report["backup"] = backup
    return report


def format_report(path: Path, report: Dict, dry_run: bool) -> str:
    lines = [f"== {path}",
             f"   LLM rows: {report['llm_rows']} (no energy: {report['no_energy']}, already correct: "
             f"{report['unchanged']}, {'would change' if dry_run else 'changed'}: {report['changed']}); "
             f"ViT/other rows untouched: {report['vit_or_other_rows']}"]
    if report["factors"]:
        lines.append("   new/old factor -> number of rows: "
                     + ", ".join(f"{k}: {v}" for k, v in sorted(report["factors"].items(), key=str)))
    for result_id, n, factor in report["factor_not_n"]:
        lines.append(f"   WARNING: {result_id}: factor {factor:.6g} != N={n} -- review this row")
    if report["backup"]:
        lines.append(f"   backup: {report['backup']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csvs", nargs="*", type=Path,
                        help="results CSVs to fix (default: results/raw/results_*.csv)")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args(argv)

    paths = args.csvs or sorted(RESULTS_RAW_DIR.glob("results_*.csv"))
    if not paths:
        print(f"No CSV to process (searched in {RESULTS_RAW_DIR}/results_*.csv)")
        return 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    any_warning = False
    for path in paths:
        report = process_file(path, dry_run=args.dry_run, stamp=stamp)
        any_warning |= bool(report["factor_not_n"])
        print(format_report(path, report, args.dry_run))
    return 2 if any_warning else 0


if __name__ == "__main__":
    sys.exit(main())
