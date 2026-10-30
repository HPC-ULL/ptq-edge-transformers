"""Shared paths and environment configuration for the llama.cpp LLM toolchain.

Artifacts (base F16 models, imatrix files, quantized GGUFs) live under the
repo's `artifacts/` directory per the project's stated architecture -- they
are shared, filename-versioned outputs, unlike the raw datasets in
data_prep/ (which deliberately live outside the repo under DATASETS_ROOT).

llama.cpp itself (the `convert_hf_to_gguf.py` script and the built
`llama-quantize` / `llama-imatrix` / `llama-perplexity` / `llama-bench`
binaries) is located via the LLAMA_CPP_DIR environment variable, pointing
at a llama.cpp checkout with a `build/bin/` directory. If unset, bare
command names are used and resolved against PATH -- convenient for
--dry-run and for machines where llama.cpp is already installed
system-wide.

evaluation_offline/llm_perplexity/ and benchmarking_ondevice/runner_llm/
both import this module's binary/artifact path helpers rather than
duplicating the LLAMA_CPP_DIR resolution logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"

LLM_BASE_DIR = ARTIFACTS_ROOT / "llm_base"
LLM_IMATRIX_DIR = ARTIFACTS_ROOT / "llm_imatrix"
LLM_QUANTIZED_DIR = ARTIFACTS_ROOT / "llm"

# Hugging Face repo IDs for the 3 LLM-branch models (calibration_protocol_llm.md
# Section 1.1).
MODEL_HF_REPOS = {
    "tinyllama_1_1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen2_5_1_5b": "Qwen/Qwen2.5-1.5B-Instruct",
    # The -BF16 repo, NOT the default mistralai/Ministral-3-3B-Instruct-2512:
    # that one is FP8 (static), which would start this model from a
    # quantized checkpoint, unlike the other two (full-precision bfloat16).
    "ministral_3_3b": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
}

# Files to skip when downloading a model's HF snapshot. The Ministral BF16
# repo ships the same weights twice: HF-format shards (model-*.safetensors,
# what convert_hf_to_gguf.py reads) and a Mistral-native consolidated.safetensors
# (7.7 GB, ignored by the converter unless --mistral-format).
MODEL_HF_EXCLUDE_PATTERNS = {
    "ministral_3_3b": ["consolidated.safetensors"],
}


def _llama_cpp_dir() -> Optional[Path]:
    raw = os.environ.get("LLAMA_CPP_DIR")
    return Path(raw).expanduser().resolve() if raw else None


def convert_script_path() -> str:
    d = _llama_cpp_dir()
    return str(d / "convert_hf_to_gguf.py") if d else "convert_hf_to_gguf.py"


def llama_quantize_bin() -> str:
    d = _llama_cpp_dir()
    return str(d / "build" / "bin" / "llama-quantize") if d else "llama-quantize"


def llama_imatrix_bin() -> str:
    d = _llama_cpp_dir()
    return str(d / "build" / "bin" / "llama-imatrix") if d else "llama-imatrix"


def llama_perplexity_bin() -> str:
    d = _llama_cpp_dir()
    return str(d / "build" / "bin" / "llama-perplexity") if d else "llama-perplexity"


def llama_bench_bin() -> str:
    d = _llama_cpp_dir()
    return str(d / "build" / "bin" / "llama-bench") if d else "llama-bench"


def hf_exclude_patterns(model_name: str) -> list:
    return list(MODEL_HF_EXCLUDE_PATTERNS.get(model_name, []))


def hf_repo_id(model_name: str) -> str:
    try:
        return MODEL_HF_REPOS[model_name]
    except KeyError:
        raise ValueError(
            f"No Hugging Face repo mapped for model_name={model_name!r}. Known models: {sorted(MODEL_HF_REPOS)}."
        ) from None
