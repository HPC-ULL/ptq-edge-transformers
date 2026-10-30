"""One-off cleanup (NOT part of the pipeline): remove the stale
symmetry=asymmetric rows from results/raw/vit_accuracy_offline.csv and
their .onnx artifacts, now that quantizer_config.py's "minmax"+"asymmetric"
mapping has been corrected (ShiftedUint8ActPerTensorFloat turned out to be
percentile-interval-based internally, mirroring the earlier symmetric
minmax discovery -- see quantizer_config.py's module docstring). These are
real, recorded results computed under the OLD (secretly percentile-based)
mapping; leaving them in place would make Phase 3 selection (or any other
analysis) silently use results from a baseline that was never actually
minmax.

Run directly (prints what it removed):

    python quantization/vit_brevitas/investigations/remove_symmetry_asymmetric_results.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SYMMETRY_ASYMMETRIC_RUN_IDS = [
    "vit_deit_tiny_phase2_symmetry_asymmetric_int8",
    "vit_mobilevit_phase2_symmetry_asymmetric_int8",
    "vit_efficientvit_b1_phase2_symmetry_asymmetric_int8",
]


def clean_csv() -> None:
    from evaluation_offline.vit_accuracy.paths import VIT_ACCURACY_RESULTS_CSV

    if not VIT_ACCURACY_RESULTS_CSV.exists():
        print(f"No results CSV found at {VIT_ACCURACY_RESULTS_CSV}")
        return

    with open(VIT_ACCURACY_RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept = [r for r in rows if r["run_id"] not in SYMMETRY_ASYMMETRIC_RUN_IDS]
    removed = len(rows) - len(kept)

    with open(VIT_ACCURACY_RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"{VIT_ACCURACY_RESULTS_CSV}: removed {removed} row(s), {len(kept)} remaining.")


def clean_artifacts() -> None:
    from quantization.vit_brevitas.paths import VIT_ONNX_DIR

    removed = 0
    for run_id in SYMMETRY_ASYMMETRIC_RUN_IDS:
        for repeat_id in (1, 2, 3):
            path = VIT_ONNX_DIR / f"{run_id}__rep{repeat_id}.onnx"
            if path.exists():
                path.unlink()
                removed += 1
                print(f"  removed {path}")
    print(f"{VIT_ONNX_DIR}: removed {removed} .onnx artifact(s).")


def main() -> None:
    clean_csv()
    clean_artifacts()


if __name__ == "__main__":
    main()
