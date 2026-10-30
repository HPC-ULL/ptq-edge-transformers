"""Shared paths for power_logging/. Same results/raw/results_<device_id>.csv
target as benchmarking_ondevice/runner_vit/paths.py and
benchmarking_ondevice/runner_llm/paths.py (computed independently here
rather than importing from either branch, so this module stays symmetric
between the two -- see results_io.py's docstring for why RESULT_COLUMNS is
likewise duplicated rather than imported from one branch specifically).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_RAW_DIR = RESULTS_ROOT / "raw"

# Power-sampling characterization (sampling_characterization.py): a separate
# tree from results/raw/ on purpose -- these files describe the MEASUREMENT
# INSTRUMENT (effective sampling rate, samples per window), not benchmark
# results, and must never be concatenated with the common results schema.
CHARACTERIZATION_DIR = RESULTS_ROOT / "characterization"


def results_csv_path(device_id: str) -> Path:
    return RESULTS_RAW_DIR / f"results_{device_id}.csv"


def sampling_csv_path(device_id: str) -> Path:
    """One record per instrumented measurement window, per device."""
    return CHARACTERIZATION_DIR / f"power_sampling_{device_id}.csv"


def sampling_hw_json_path(device_id: str) -> Path:
    return CHARACTERIZATION_DIR / f"power_sampling_hw_{device_id}.json"


def sampling_summary_csv_path() -> Path:
    return CHARACTERIZATION_DIR / "power_sampling_summary.csv"


def sampling_classification_csv_path() -> Path:
    return CHARACTERIZATION_DIR / "power_sampling_classification.csv"
