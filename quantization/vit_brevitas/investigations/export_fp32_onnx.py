"""One-off probe (NOT part of the pipeline, like the other investigations/
scripts): export the UNQUANTIZED float model to ONNX, so its
onnxruntime_perf_test latency can be compared against the QCDQ INT8
artifacts (see scripts/probe_int8_kernels.py).

"Same export process, minus the Brevitas step": same model loader
(quantization.vit_brevitas.quantize.load_pretrained_model -- timm
pretrained, eval mode, fused SDPA attention disabled), same input shape
(resolve_input_shape), same pinned opset and keep_initializers_as_inputs=False
as export/vit_to_onnx/export.py -- only torch.onnx.export replaces
brevitas.export.export_onnx_qcdq, because there is no quantized model to
hand it. Brevitas itself is not needed to run this.

Writes NOTHING under artifacts/ (that tree feeds verify_edge_bundle.py and
the run manifests); the default output is a scratch directory in the
current working directory.

After exporting it checks, when the packages are available, that the graph
really is float (no Quantize/DequantizeLinear nodes) and that ONNX Runtime
reproduces PyTorch's output on the same random input.

Needs: torch, timm (pretrained weights, downloaded on first use);
optionally onnx and onnxruntime for the two checks above.

Run directly:

    python quantization/vit_brevitas/investigations/export_fp32_onnx.py --model deit_tiny
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantization.vit_brevitas.paths import FUSION_PROBES, investigation_dir  # noqa: E402

MODELS = ["deit_tiny", "mobilevit", "efficientvit_b1"]
QDQ_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"}


def export_fp32(model_name: str, out_path: Path) -> Path:
    import torch

    from export.vit_to_onnx.export import ONNX_OPSET
    from quantization.vit_brevitas.quantize import load_pretrained_model, resolve_input_shape

    model = load_pretrained_model(model_name)
    input_shape = resolve_input_shape(model)
    dummy_input = torch.randn(*input_shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        reference = model(dummy_input).numpy()
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            opset_version=ONNX_OPSET,
            keep_initializers_as_inputs=False,
        )

    print(f"Exported {model_name} FP32 -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"opset {ONNX_OPSET}, input {tuple(input_shape)})")
    _check_graph_is_float(out_path)
    _check_against_torch(out_path, dummy_input.numpy(), reference)
    return out_path


def _check_graph_is_float(path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("[skip] onnx not installed: cannot list the graph's op types")
        return
    ops = Counter(node.op_type for node in onnx.load(str(path)).graph.node)
    qdq = {op: n for op, n in ops.items() if op in QDQ_OPS}
    print(f"Graph op types: {dict(ops.most_common(8))}{' ...' if len(ops) > 8 else ''}")
    if qdq:
        raise SystemExit(f"UNEXPECTED: quantize/dequantize nodes in what should be a float graph: {qdq}")
    print("OK: no Quantize/DequantizeLinear nodes -- a plain float graph.")


def _check_against_torch(path: Path, input_array, reference) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[skip] onnxruntime/numpy not installed: cannot compare against PyTorch")
        return
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = session.run(None, {session.get_inputs()[0].name: input_array.astype(np.float32)})[0]
    max_diff = float(np.abs(out - reference).max())
    print(f"max |ONNX Runtime - PyTorch| on a random input: {max_diff:.2e}")
    if max_diff > 1e-3:
        print("WARNING: larger than 1e-3 -- check the export before trusting any comparison built on it.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="deit_tiny", choices=MODELS)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output .onnx (default: results/investigations/vit/fusion_probes/models/vit_<model>_fp32.onnx)")
    args = parser.parse_args()
    export_fp32(args.model, args.out or investigation_dir(FUSION_PROBES, "models") / f"vit_{args.model}_fp32.onnx")


if __name__ == "__main__":
    main()
