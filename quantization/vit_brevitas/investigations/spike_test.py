"""Technical spike: is <model> -> Brevitas PTQ -> ONNX -> ONNX Runtime viable,
for each of the 3 ViT-branch candidate models?

NOT part of the final pipeline (that lives in the rest of
quantization/vit_brevitas/, built only once this is confirmed to work for
all 3 models). No idempotency, no unit tests -- for each model, four steps
run once, answering one question end to end:

  1. Load the pretrained model (timm).
  2. PTQ-quantize it with Brevitas at the Phase-1 baseline config (MinMax,
     per-tensor, symmetric, INT8) via manual nn.Linear/nn.Conv2d ->
     QuantLinear/QuantConv2d substitution (see MANUAL SUBSTITUTION below),
     calibrated on 32 images from the calibration pool already prepared by
     data_prep/.
  3. Export the quantized model to ONNX (QCDQ style, opset 13, initializers
     not duplicated as graph inputs).
  4. Load that .onnx in ONNX Runtime and run one inference, checking the
     output is (1, 1000) logits with no error.

One model's failure does not stop the others from being tried -- the whole
point of this run is to see all 3 results at once, not just the first
failure. A summary table prints at the end.

MANUAL SUBSTITUTION, not brevitas.graph.quantize.quantize(): the automatic,
FX-graph-based quantize() helper hit three separate, ViT-specific failures
in a row on DeiT-Tiny (missing GraphModule trace; is_causal traced as an FX
Proxy, breaking timm's internal "if is_causal:"; and finally an
unconditional "Residual handler could not find a solution to align scale
factors across ADDs and CATs", with no parameter exposed to disable it --
confirmed by reading brevitas/graph/quantize.py directly). Manual
substitution needs no FX tracing at all: it just walks the module tree and
swaps nn.Linear/nn.Conv2d for brevitas.nn.QuantLinear/QuantConv2d in place,
copying the pretrained weights over. Everything else (LayerNorm, GELU,
softmax, the residual adds themselves) stays plain float between quantized
layers -- exactly the QCDQ dequantize/float-op/requantize pattern most real
INT8 transformer deployments already use, so this is not a lesser
approximation of the automatic path: the actual per-layer weight/activation
quantization policy (Int8WeightPerTensorFloat / Int8ActPerTensorFloat) is
identical either way, only the handling of the residual adds differs.

STALE relative to the real pipeline: this spike still uses Brevitas's own
Int8ActPerTensorFloat directly for input_quant below. quantizer_config.py's
"minmax" baseline was later changed to Int8ActPerTensorFloatMinMax, once
Int8ActPerTensorFloat turned out to be percentile-based internally
(high_percentile_q=99.999), not literal minmax -- see quantizer_config.py's
module docstring and investigations/diagnose_manual_fakequant.py for the
full story. This spike was never updated to match, since it only needed to
prove ONNX export viability, not reproduce the exact baseline config -- do
not treat its numbers as representative of the current baseline.
Using the same manual mechanism for all 3 models (rather than automatic for
whichever one happens to trace cleanly) also matters for the ablation's
internal validity: any precision/performance difference measured between
architectures should reflect real architectural sensitivity, not which
quantization code path each one happened to go through.

Run directly:

    python quantization/vit_brevitas/spike_test.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

# When run as "python quantization/vit_brevitas/spike_test.py"
# (direct path, not -m), Python only adds the script's own directory to
# sys.path, not the project root -- without this, "from data_prep..."
# fails with ModuleNotFoundError even if the project is well structured.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Manifest token (configs/vit_manifest.py) -> timm model name.
# calibration_protocol_vit.md lists "MobileViT S/XS" and "EfficientViT-B1/
# EfficientFormer" as alternatives; here the variant named by the
# manifest is tested (mobilevit_s, efficientvit_b1).
MODELS_TO_TEST = {
    "deit_tiny": "deit_tiny_patch16_224",
    "mobilevit": "mobilevit_s",
    "efficientvit_b1": "efficientvit_b1",
}

NUM_CALIBRATION_IMAGES = 32
EXPECTED_OUTPUT_SHAPE = (1, 1000)


class SpikeStepFailure(Exception):
    """Aborts only the spike of the current model, not the whole script."""

    def __init__(self, step: str, exc: BaseException):
        self.step = step
        self.exc = exc
        super().__init__(f"{step}: {type(exc).__name__}: {exc}")


def fail(step: str, exc: BaseException) -> None:
    print(f"\n[FAILED] step: {step}")
    print(f"        {type(exc).__name__}: {exc}")
    print()
    traceback.print_exc()
    raise SpikeStepFailure(step, exc) from exc


def resolve_input_shape(model):
    """(1, C, H, W) according to the preprocessing config that timm resolves for
    this specific model -- not all use 224x224 (e.g. some MobileViT variants
    expect 256x256), so it cannot be hardcoded."""
    from timm.data import resolve_data_config

    config = resolve_data_config({}, model=model)
    c, h, w = config["input_size"]
    return (1, c, h, w)


def load_calibration_batch(model, n: int):
    """n preprocessed tensors from the ImageNet calibration pool already
    prepared by data_prep/, with the exact preprocessing this timm model
    expects (resize/crop/normalize)."""
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    from data_prep.imagenet import get_calibration_batch

    config = resolve_data_config({}, model=model)
    preprocess = create_transform(**config)

    samples = get_calibration_batch(repetition_idx=0, batch_size=n)
    tensors = []
    for sample in samples:
        image = Image.open(sample["path"]).convert("RGB")
        tensors.append(preprocess(image))
    return tensors


def run_spike_for_model(display_name: str, timm_model_name: str) -> None:
    print(f"\n{'=' * 70}\n{display_name} ({timm_model_name})\n{'=' * 70}")

    # ------------------------------------------------------------------
    # 1. Load the pretrained model
    # ------------------------------------------------------------------
    step = f"[{display_name}] 1/4 model loading (timm)"
    print(f"{step}: loading {timm_model_name}...")
    try:
        import timm
        import torch
        from timm.layers import set_fused_attn

        # Forces the "classic" attention implementation (matmul+softmax+matmul
        # as separate ops) instead of F.scaled_dot_product_attention -- it only
        # affects models that really have SDPA-based attention, and is a
        # harmless no-op for the rest.
        set_fused_attn(False)

        model = timm.create_model(timm_model_name, pretrained=True)
        model.eval()
        input_shape = resolve_input_shape(model)
    except Exception as e:
        fail(step, e)
    print(f"{step}: OK (input_shape={input_shape})")

    # ------------------------------------------------------------------
    # 2. Quantize with Brevitas (manual substitution) + calibrate
    # ------------------------------------------------------------------
    step = f"[{display_name}] 2/4 quantization + calibration (Brevitas)"
    print(f"{step}: quantizing and calibrating with {NUM_CALIBRATION_IMAGES} images...")
    try:
        from brevitas.graph.calibrate import calibration_mode
        from brevitas.nn import QuantConv2d, QuantLinear
        from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

        def quantize_linear(child: torch.nn.Linear) -> QuantLinear:
            quant_linear = QuantLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                return_quant_tensor=False,
            )
            quant_linear.weight.data = child.weight.data.clone()
            if child.bias is not None:
                quant_linear.bias.data = child.bias.data.clone()
            return quant_linear

        def quantize_conv2d(child: torch.nn.Conv2d) -> QuantConv2d:
            quant_conv = QuantConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                weight_quant=Int8WeightPerTensorFloat,
                input_quant=Int8ActPerTensorFloat,
                return_quant_tensor=False,
            )
            quant_conv.weight.data = child.weight.data.clone()
            if child.bias is not None:
                quant_conv.bias.data = child.bias.data.clone()
            return quant_conv

        def quantize_module_tree(module: torch.nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(module, name, quantize_linear(child))
                elif isinstance(child, torch.nn.Conv2d):
                    setattr(module, name, quantize_conv2d(child))
                else:
                    quantize_module_tree(child)

        calib_tensors = load_calibration_batch(model, NUM_CALIBRATION_IMAGES)
        calib_batch = torch.stack(calib_tensors)

        quantize_module_tree(model)
        quant_model = model
        quant_model.eval()

        with torch.no_grad(), calibration_mode(quant_model):
            quant_model(calib_batch)
    except Exception as e:
        fail(step, e)
    print(f"{step}: OK")

    # ------------------------------------------------------------------
    # 3. Export to ONNX (QCDQ: standard ONNX ops, no custom ops)
    # ------------------------------------------------------------------
    step = f"[{display_name}] 3/4 export to ONNX"
    onnx_dir = Path(tempfile.mkdtemp(prefix=f"vit_brevitas_spike_{display_name}_"))
    onnx_path = onnx_dir / f"{timm_model_name}_int8.onnx"
    print(f"{step}: exporting to {onnx_path}...")
    try:
        from brevitas.export import export_onnx_qcdq

        dummy_input = torch.randn(*input_shape)
        export_onnx_qcdq(
            quant_model,
            args=dummy_input,
            export_path=str(onnx_path),
            opset_version=13,
            keep_initializers_as_inputs=False,
        )
    except Exception as e:
        fail(step, e)
    print(f"{step}: OK")

    # ------------------------------------------------------------------
    # 4. Load in ONNX Runtime and run a test inference
    # ------------------------------------------------------------------
    step = f"[{display_name}] 4/4 loading in ONNX Runtime + inference"
    print(f"{step}: loading {onnx_path.name} and running a test inference...")
    try:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        dummy_np_input = np.random.randn(*input_shape).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_np_input})
        logits = outputs[0]

        if logits.shape != EXPECTED_OUTPUT_SHAPE:
            raise AssertionError(f"unexpected output shape: {logits.shape}, expected {EXPECTED_OUTPUT_SHAPE}")
    except Exception as e:
        fail(step, e)
    print(f"{step}: OK -- logits with shape {logits.shape}")

    print(f"\n[{display_name}] SPIKE PASSED. Test ONNX artifact (disposable): {onnx_path}")


def main() -> int:
    results = {}
    for display_name, timm_model_name in MODELS_TO_TEST.items():
        try:
            run_spike_for_model(display_name, timm_model_name)
            results[display_name] = "OK"
        except SpikeStepFailure as e:
            results[display_name] = f"FAILED at {e.step}: {type(e.exc).__name__}: {e.exc}"

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for display_name, result in results.items():
        print(f"  {display_name}: {result}")

    return 0 if all(r == "OK" for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
