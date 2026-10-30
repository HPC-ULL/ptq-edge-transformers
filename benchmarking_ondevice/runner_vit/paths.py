"""Shared paths for the ViT on-device runner.

onnxruntime_perf_test is located via the ONNXRUNTIME_DIR environment
variable, pointing at an ONNX Runtime checkout compiled as described in the README
(binary at $ONNXRUNTIME_DIR/build/Release/onnxruntime_perf_test -- confirmed
this is the real layout produced by build.sh when --build_dir is passed
explicitly, NOT the $ONNXRUNTIME_DIR/build/<Platform>/Release/ layout
build.py uses with its own default build_dir). If unset, the bare command
name is used and resolved against PATH -- same convention as
quantization/llm_llamacpp/paths.py's LLAMA_CPP_DIR, for a device where
onnxruntime_perf_test is already installed system-wide.

configs/devices.yaml is imported from configs/ directly (CONFIGS_DIR),
rather than duplicating its path computation here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from configs.common import CONFIGS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICES_YAML_PATH = CONFIGS_DIR / "devices.yaml"

RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_RAW_DIR = RESULTS_ROOT / "raw"


def _onnxruntime_dir() -> Optional[Path]:
    raw = os.environ.get("ONNXRUNTIME_DIR")
    return Path(raw).expanduser().resolve() if raw else None


def onnxruntime_perf_test_bin() -> str:
    d = _onnxruntime_dir()
    return str(d / "build" / "Release" / "onnxruntime_perf_test") if d else "onnxruntime_perf_test"


def results_csv_path(device_id: str) -> Path:
    """Per-device raw results file -- technical_architecture.md
    Section 4's stated convention ("each person/machine writes to its own
    file: results/raw/results_<device_or_person>.csv"), so two devices
    benchmarking concurrently never contend for the same file."""
    return RESULTS_RAW_DIR / f"results_{device_id}.csv"
