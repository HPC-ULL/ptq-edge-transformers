"""llama.cpp/GGUF quantization for the LLM branch.

Pipeline per manifest row: base_model (HF weights -> GGUF F16, once per
model) -> imatrix_gen (only if calibration_corpus != "none") -> quantize
(resolves the real llama-quantize type via quant_type, never the raw
`bitwidth` field). See cli.py / `python -m quantization.llm_llamacpp` for
the orchestrated entry point.
"""

from .quant_type import resolve_llama_quant_type

__all__ = ["resolve_llama_quant_type"]
