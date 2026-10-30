"""Confirms that a results CSV only GAINED rows with respect to its backup --
for run_batch invocations that add rows to an official CSV (e.g. the F16
reference rows, technical_architecture.md Section 6.7).

append_result (and append_sampling_record) rewrite the whole file, keeping
the existing rows in order and appending the new one, so after an
invocation that only added rows the backup's rows must be the current
file's first rows, unchanged. Checked:

  - same header;
  - the current file starts with exactly the backup's rows, field by field
    and in the same order (none changed, removed or reordered);
  - every row after them has a result_id the backup did not have;
  - with --manifest: every new row's run_id is in that manifest (manifest
    rows with no new row are listed, not an error: a row may have failed).

Also prints the sha256 of the backup and of the current file's first
len(backup) bytes (byte-identical prefix: informative; the verdict is the
row comparison). Standard library only, runs on the edge device. Exit 0 if
only new rows appeared, 1 otherwise. Usage, from the repo root:

    python3 scripts/check_csv_additions.py <backup.csv> <current.csv> [--manifest run_manifest_llm_f16.csv]

Works for results/raw/results_<device>.csv and
results/characterization/power_sampling_<device>.csv alike (both keyed by
result_id).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

KEY = "result_id"


def _read(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _manifest_run_ids(path: Path) -> List[str]:
    with open(path, newline="") as f:
        return [r["run_id"] for r in csv.DictReader(f)]


def compare(backup: Path, current: Path, manifest: Optional[Path] = None) -> Tuple[bool, List[str]]:
    """(ok, report lines)."""
    lines: List[str] = []
    ok = True
    b_header, b_rows = _read(backup)
    c_header, c_rows = _read(current)

    if b_header != c_header:
        return False, [f"[ERROR] different header:\n  backup : {b_header}\n  current: {c_header}"]

    prefix = c_rows[:len(b_rows)]
    changed = [i for i, (b, c) in enumerate(zip(b_rows, prefix)) if b != c]
    if len(prefix) < len(b_rows):
        ok = False
        lines.append(f"[ERROR] the current file has {len(c_rows)} rows, fewer than the backup ({len(b_rows)})")
    for i in changed:
        ok = False
        diff = [k for k in b_header if b_rows[i].get(k) != prefix[i].get(k)]
        lines.append(f"[ERROR] row {i + 1} ({b_rows[i].get(KEY)}) changed in: {diff}")

    old_ids = {r.get(KEY) for r in b_rows}
    new_rows = c_rows[len(b_rows):]
    for r in new_rows:
        if r.get(KEY) in old_ids:
            ok = False
            lines.append(f"[ERROR] appended row with a result_id already in the backup: {r.get(KEY)}")

    if manifest is not None:
        expected = _manifest_run_ids(manifest)
        for r in new_rows:
            if r.get("run_id") not in expected:
                ok = False
                lines.append(f"[ERROR] new row outside the manifest: {r.get(KEY)}")
        got = {r.get("run_id") for r in new_rows}
        for run_id in expected:
            if run_id not in got:
                lines.append(f"[warning] manifest row without a new row: {run_id}")

    b_bytes, c_bytes = backup.read_bytes(), current.read_bytes()
    sha_b = hashlib.sha256(b_bytes).hexdigest()
    sha_p = hashlib.sha256(c_bytes[:len(b_bytes)]).hexdigest()
    lines.append(f"sha256 backup                    : {sha_b}")
    lines.append(f"sha256 current prefix (bytes)    : {sha_p}  ({'identical' if sha_b == sha_p else 'DIFFERENT'})")
    lines.append(f"rows: backup {len(b_rows)}, current {len(c_rows)}, new {len(new_rows)}")
    lines += [f"  + {r.get(KEY)}" for r in new_rows]
    lines.append("OK: only new rows have appeared" if ok else "FAILED: the CSV did not only gain rows")
    return ok, lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("backup", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    ok, lines = compare(args.backup, args.current, args.manifest)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
