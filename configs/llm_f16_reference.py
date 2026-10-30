"""The base F16 reference rows of the LLM branch (technical_architecture.md
Section 6.7): one row per model, whose artifact is the base F16 GGUF
(artifacts/llm_base/<model>_f16.gguf) -- no quantization step, no imatrix.

Kept OUT of llm_manifest.generate() and quant_type._LLAMA_QUANT_TYPES on
purpose: nothing here is quantized, so the quantization CLI and
verify_edge_bundle must never see these rows. Measured on device like any
other LLM row (run_batch, same llama-bench flags and N); deterministic, so
repeat_id is blank and the result_id is llm_<model>_reference_f16__<device>.

The descriptive fields mirror the ViT deployment FP32 rows
(configs/vit_deployment_manifest.py): factor_varied "reference", factor_level
and bitwidth "f16". The on-device results rows leave them blank, like every
LLM row; the analysis takes model and format from results/artifact_sizes.csv
by run_id.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .llm_manifest import FAMILY, MODELS

PHASE = "reference"
BITWIDTH = "f16"
FORMAT = "F16"


def run_id(model: str) -> str:
    return f"{FAMILY}_{model}_{PHASE}_{BITWIDTH}"


def artifact_path(model: str):
    """The base F16 GGUF, from the pipeline's own base_model_path (absolute)."""
    from quantization.llm_llamacpp.base_model import base_model_path

    return base_model_path(model)


def generate() -> List[Dict[str, Any]]:
    return [
        {"run_id": run_id(model), "model_family": FAMILY, "model_name": model, "phase": PHASE,
         "factor_varied": "reference", "factor_level": BITWIDTH, "calibration_corpus": "none",
         "calib_set_size": None, "quant_scheme": "", "bitwidth": BITWIDTH}
        for model in MODELS
    ]
