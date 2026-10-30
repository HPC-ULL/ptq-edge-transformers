"""Export an in-memory Brevitas-quantized model to ONNX.

QCDQ style (standard ONNX QuantizeLinear/DequantizeLinear ops, runnable by
stock ONNX Runtime -- matching what onnxruntime_perf_test needs later for
on-device benchmarking), not QONNX: that is only revisited separately if/
when Phase 3 needs custom handling for INT6/INT4 (Phase 1-2 fixes INT8).

opset_version and keep_initializers_as_inputs are pinned explicitly rather
than left at Brevitas's defaults, for two reasons confirmed during spike
testing: reproducibility (an unstated default could silently change between
Brevitas versions), and keep_initializers_as_inputs=False specifically
avoids ONNX Runtime warning that weights/biases appear as both graph inputs
and initializers, which blocks constant folding.

Artifact: artifacts/vit/{run_id}__rep{repeat_id}.onnx -- every ViT-branch
manifest row is stochastic (calibration sampling is always random), so
there is no deterministic/unsuffixed exception like the LLM branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from quantization.vit_brevitas.paths import VIT_ONNX_DIR

ONNX_OPSET = 13


def onnx_path(run_id: str, repeat_id: int) -> Path:
    return VIT_ONNX_DIR / f"{run_id}__rep{repeat_id}.onnx"


def export_to_onnx(
    quant_model, input_shape: Tuple[int, int, int, int], run_id: str, repeat_id: int, out_path: Optional[Path] = None,
    single_gelu: bool = False,
) -> Path:
    """out_path defaults to the pipeline artifact path; a one-off check can
    pass its own so it never writes under artifacts/.

    single_gelu (default False: the ablation's export, untouched) rewrites each
    GELU of the exported graph as ONE com.microsoft::Gelu node -- see
    single_gelu.py for why. Only the deployment set of models with an exact
    GELU (DeiT-Tiny) asks for it; it raises for a model without one."""
    import torch
    from brevitas.export import export_onnx_qcdq

    out_path = Path(out_path) if out_path is not None else onnx_path(run_id, repeat_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*input_shape)
    export_onnx_qcdq(
        quant_model,
        args=dummy_input,
        export_path=str(out_path),
        opset_version=ONNX_OPSET,
        keep_initializers_as_inputs=False,
    )
    if single_gelu:
        from .single_gelu import fuse_gelu_in_place

        print(f"[export] {run_id}: {fuse_gelu_in_place(out_path)} GELU chain(s) written as one Gelu node each")
    return out_path
