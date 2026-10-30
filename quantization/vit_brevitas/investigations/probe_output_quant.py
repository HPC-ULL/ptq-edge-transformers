"""One-off probe (NOT part of the pipeline): does adding OUTPUT quantization to
the Brevitas Conv/Linear layers let ONNX Runtime fuse them into integer
kernels?

Context (probe_qdq_fusion.py / scripts/probe_int8_kernels.py on the cluster): the
pipeline's Brevitas export then quantized only the INPUT of each Conv/Linear
(quantize.py has since adopted output_quant as a result of this probe), and
ORT runs it as explicit Q/DQ + float compute (0 % integer kernels), while
ORT's own quantizer -- with a Q after each op output -- gets 100 % integer
kernels on the same model. Dropping that output Q from ORT's own graph
reproduced the Brevitas behaviour (the 48 Linear layers stay float Gemm);
removing Brevitas's weight Clips changed nothing. So the missing output Q is
the suspect; this builds the same Brevitas model WITH it.

Model: EXACTLY the pipeline's deit_tiny INT8 baseline (minmax / per_tensor /
symmetric, 128 calibration images of repeat 1, same loader, same quantizer
classes, same calibration pass), except each QuantLinear/QuantConv2d also
gets output_quant = the same activation quantizer. One open point it also
tests: with output_quant the Q sits AFTER the bias add (MatMul -> Add -> Q),
not right after the MatMul, so the fusion may additionally need an integer
bias -- --bias-quant adds Brevitas's Int32Bias (scale = input x weight
scale, the usual accumulator bias).

Writes only under results/investigations/vit/fusion_probes/models/ (never artifacts/). Prints the
probe_qdq_fusion structure report of the export. Optionally (--eval-samples
N) compares top-1 on the first N eval-subset images against the pipeline's
own baseline artifact -- a quick relative check that output quantization is
not destroying accuracy, NOT a result: the first N images are not
class-stratified (2 per class in class order), so only the difference
between the two models on the same images means anything.

The pipeline itself is untouched; the two substitution helpers below mirror
quantization/vit_brevitas/quantize.py's _quantize_linear/_quantize_conv2d
plus the two extra keyword arguments.

Needs (cluster): torch, timm, brevitas, onnx, onnxruntime, the calibration
pool, and for --eval-samples the eval subset.

    python quantization/vit_brevitas/investigations/probe_output_quant.py
    python quantization/vit_brevitas/investigations/probe_output_quant.py --bias-quant
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantization.vit_brevitas.paths import FUSION_PROBES, investigation_dir  # noqa: E402

MODEL_NAME = "deit_tiny"
BASELINE = {"calibration_algorithm": "minmax", "granularity": "per_tensor", "symmetry": "symmetric", "bitwidth": "int8"}
CALIB_SET_SIZE = 128
REPEAT_ID = 1
OUT_DIR = investigation_dir(FUSION_PROBES, "models")
PIPELINE_BASELINE_ARTIFACT = Path("artifacts/vit/vit_deit_tiny_phase1_baseline_default_int8__rep1.onnx")


def _quant_linear(child, weight_quant, act_quant, bias_quant):
    from brevitas.nn import QuantLinear

    layer = QuantLinear(
        in_features=child.in_features, out_features=child.out_features, bias=child.bias is not None,
        weight_quant=weight_quant, input_quant=act_quant, output_quant=act_quant, bias_quant=bias_quant,
        return_quant_tensor=False,
    )
    layer.weight.data = child.weight.data.clone()
    if child.bias is not None:
        layer.bias.data = child.bias.data.clone()
    return layer


def _quant_conv2d(child, weight_quant, act_quant, bias_quant):
    from brevitas.nn import QuantConv2d

    layer = QuantConv2d(
        in_channels=child.in_channels, out_channels=child.out_channels, kernel_size=child.kernel_size,
        stride=child.stride, padding=child.padding, dilation=child.dilation, groups=child.groups,
        bias=child.bias is not None, weight_quant=weight_quant, input_quant=act_quant, output_quant=act_quant,
        bias_quant=bias_quant, return_quant_tensor=False,
    )
    layer.weight.data = child.weight.data.clone()
    if child.bias is not None:
        layer.bias.data = child.bias.data.clone()
    return layer


def _substitute(module, weight_quant, act_quant, bias_quant) -> None:
    import torch.nn as nn

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, _quant_linear(child, weight_quant, act_quant, bias_quant))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, _quant_conv2d(child, weight_quant, act_quant, bias_quant))
        else:
            _substitute(child, weight_quant, act_quant, bias_quant)


def build_and_export(out: Path, bias_quant_on: bool) -> Path:
    import torch
    from brevitas.export import export_onnx_qcdq
    from brevitas.graph.calibrate import calibration_mode

    from export.vit_to_onnx.export import ONNX_OPSET
    from quantization.vit_brevitas.calibration import load_calibration_batch
    from quantization.vit_brevitas.quantize import load_pretrained_model, resolve_input_shape
    from quantization.vit_brevitas.quantizer_config import resolve_act_quant, resolve_weight_quant

    model = load_pretrained_model(MODEL_NAME)
    input_shape = resolve_input_shape(model)
    weight_quant = resolve_weight_quant(BASELINE["calibration_algorithm"], BASELINE["granularity"], BASELINE["bitwidth"])
    act_quant = resolve_act_quant(BASELINE["calibration_algorithm"], BASELINE["symmetry"], BASELINE["bitwidth"])
    bias_quant = None
    if bias_quant_on:
        from brevitas.quant.scaled_int import Int32Bias

        bias_quant = Int32Bias

    calib_batch = load_calibration_batch(model, CALIB_SET_SIZE, REPEAT_ID)  # before substitution, like the pipeline
    _substitute(model, weight_quant, act_quant, bias_quant)
    model.eval()
    with torch.no_grad(), calibration_mode(model):
        model(calib_batch)

    out.parent.mkdir(parents=True, exist_ok=True)
    export_onnx_qcdq(model, args=torch.randn(*input_shape), export_path=str(out),
                     opset_version=ONNX_OPSET, keep_initializers_as_inputs=False)
    print(f"Exported {out}  (baseline recipe + output_quant{' + Int32Bias' if bias_quant_on else ''})")
    return out


def top1_on_first(paths, n: int):
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    from data_prep.imagenet import load_eval_subset
    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess

    preprocess = resolve_preprocess(MODEL_NAME)
    samples = load_eval_subset()[:n]
    inputs = [(preprocess(Image.open(s["path"]).convert("RGB")).unsqueeze(0).numpy().astype(np.float32), s["label"])
              for s in samples]
    results = {}
    for label, path in paths:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        hits = sum(int(np.argmax(session.run(None, {name: x})[0][0]) == y) for x, y in inputs)
        results[label] = 100.0 * hits / len(inputs)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bias-quant", action="store_true", help="also quantize the bias (Brevitas Int32Bias)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--eval-samples", type=int, default=0,
                        help="quick top-1 on the first N eval images vs the pipeline baseline artifact (0 = skip)")
    args = parser.parse_args(argv)

    suffix = "outq_biasq" if args.bias_quant else "outq"
    out = args.out or OUT_DIR / f"vit_deit_tiny_int8_brevitas_{suffix}.onnx"
    build_and_export(out, args.bias_quant)

    import onnx

    from quantization.vit_brevitas.investigations.probe_qdq_fusion import format_report, inspect_graph

    print(format_report(out.name, inspect_graph(onnx.load(str(out)))))

    if args.eval_samples > 0:
        paths = [("pipeline_baseline", PIPELINE_BASELINE_ARTIFACT), (suffix, out)]
        acc = top1_on_first([(k, p) for k, p in paths if p.exists()], args.eval_samples)
        print(f"\nQuick top-1 on the first {args.eval_samples} eval images (relative check only, not stratified):")
        for label, value in acc.items():
            print(f"  {label:<18} {value:6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
