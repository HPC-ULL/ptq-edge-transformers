"""One-off cleanup (NOT part of the pipeline): remove duplicate rows from
results/raw/vit_accuracy_offline.csv, caused by a real bug in
evaluation_offline/vit_accuracy/results_io.py's append_result() (now fixed
-- it always appended, never replaced an existing row with the same
result_id). Triggered here by re-evaluating Phase 3 with --force right
after the INT6/INT4 quantizer fix, without first removing the stale
pre-fix rows (unlike the entropy and symmetry=asymmetric invalidations
earlier, which did clean up first) -- confirmed via a direct result_id
count: 144 Phase 3 rows for 72 distinct result_ids, each exactly 2x.

General de-duplication (not scoped to just Phase 3 run_ids), since the
underlying bug could in principle have duplicated any row: keeps the LAST
occurrence of each result_id (the most recently appended -- for the actual
Phase 3 case, this is confirmed to be the post-fix, correct value, since
the stale pre-fix rows were written first and the corrected re-evaluation
second) and drops any earlier ones.

Run directly (prints what it removed):

    python quantization/vit_brevitas/investigations/remove_phase3_duplicate_results.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    from evaluation_offline.vit_accuracy.paths import VIT_ACCURACY_RESULTS_CSV

    if not VIT_ACCURACY_RESULTS_CSV.exists():
        print(f"No results CSV found at {VIT_ACCURACY_RESULTS_CSV}")
        return

    with open(VIT_ACCURACY_RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    counts_before = Counter(r["result_id"] for r in rows)
    duplicated_ids = {rid for rid, c in counts_before.items() if c > 1}

    # Keep the LAST occurrence of each result_id, in original row order.
    last_index = {}
    for i, r in enumerate(rows):
        last_index[r["result_id"]] = i
    kept_rows = [r for i, r in enumerate(rows) if last_index[r["result_id"]] == i]

    removed = len(rows) - len(kept_rows)

    with open(VIT_ACCURACY_RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"{VIT_ACCURACY_RESULTS_CSV}: {len(rows)} rows -> {len(kept_rows)} rows "
          f"({removed} duplicate(s) removed across {len(duplicated_ids)} result_id(s)).")
    for rid in sorted(duplicated_ids)[:20]:
        print(f"  deduplicated: {rid} (kept the last of {counts_before[rid]} occurrences)")
    if len(duplicated_ids) > 20:
        print(f"  ... and {len(duplicated_ids) - 20} more")


if __name__ == "__main__":
    main()
