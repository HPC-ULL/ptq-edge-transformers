"""Shared paths for the LLM on-device runner.

llama-bench is located via quantization/llm_llamacpp/paths.py's existing
LLAMA_CPP_DIR resolution (llama_bench_bin()) -- reused rather than
duplicated, same as evaluation_offline/llm_perplexity/paths.py already
does for llama_perplexity_bin().

configs/devices.yaml is imported from configs/ directly (CONFIGS_DIR), same
as benchmarking_ondevice/runner_vit/paths.py.

results_csv_path() targets the SAME results/raw/results_<device_id>.csv
file runner_vit writes to -- a device that benchmarks both branches (e.g.
the Jetson, per technical_architecture.md Section 6.2) gets one file
per device, not one per branch (run_id's "vit_"/"llm_" prefix already
disambiguates rows). See results_io.py for why RESULT_COLUMNS is therefore
kept byte-identical to runner_vit's, not independently redefined.
"""

from __future__ import annotations

from pathlib import Path

from configs.common import CONFIGS_DIR
from quantization.llm_llamacpp.paths import PROJECT_ROOT, llama_bench_bin

DEVICES_YAML_PATH = CONFIGS_DIR / "devices.yaml"

RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_RAW_DIR = RESULTS_ROOT / "raw"

__all__ = [
    "DEVICES_YAML_PATH", "PROJECT_ROOT", "RESULTS_ROOT", "RESULTS_RAW_DIR", "llama_bench_bin", "results_csv_path",
]


def results_csv_path(device_id: str) -> Path:
    return RESULTS_RAW_DIR / f"results_{device_id}.csv"
