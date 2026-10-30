"""Brevitas PTQ quantization + ONNX export orchestration for the ViT branch.

See cli.py / `python -m quantization.vit_brevitas` for the entry point,
quantize.py for the manual-substitution quantization pipeline (see its
docstring for why it's manual rather than brevitas.graph.quantize.quantize()),
and quantizer_config.py for which (calibration_algorithm, granularity,
symmetry) combinations are actually confirmed to work.
"""

from .quantize import build_quantized_model

__all__ = ["build_quantized_model"]
