"""One-off diagnostic (NOT part of the pipeline): WHICH layer is where
mobilevit/efficientvit_b1's quantized output first diverges from the float
model's?

diagnose_baseline_collapse.py already pinned the bug to Stage B (quantized,
in-memory, calibrated -- BEFORE ONNX export/onnxruntime are even involved):
float is correct (~80% top1) but the quantized model is near-random,
already before export, for mobilevit/efficientvit_b1 but not deit_tiny.
Since deit_tiny's only Conv2d (a single, ungrouped patch-embed conv) is
quantized fine, while mobilevit/efficientvit_b1 (many Conv2d, mostly
grouped/depthwise, throughout the network) collapse, the next question is
whether ONE specific early layer is already badly wrong (a calibration/
substitution bug at that layer) or whether small errors compound gradually
across many layers (an inherent per-tensor-PTQ sensitivity, not a bug).

This hooks every quantized Conv2d/Linear's output (return_quant_tensor=False
means these are already plain, comparable float tensors) and the
correspondingly-named float module's output, runs both models on the SAME
real calibration-quality image, and prints relative L2 error per layer in
execution order -- so the exact point of divergence (a specific layer name,
or a smooth gradual buildup) is visible directly instead of guessed at.

Needs: Brevitas installed, the gated ImageNet-1K pool already prepared.

Run directly, optionally naming the model (default "mobilevit"):

    python quantization/vit_brevitas/diagnose_layer_by_layer.py mobilevit
    python quantization/vit_brevitas/diagnose_layer_by_layer.py efficientvit_b1
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

CALIB_SET_SIZE = 128
REPEAT_ID = 1


def main() -> None:
    MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "mobilevit"
    import torch
    import torch.nn as nn
    from PIL import Image

    from data_prep.imagenet import load_eval_subset
    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess
    from quantization.vit_brevitas.quantize import build_quantized_model, load_pretrained_model

    print(f"Loading float {MODEL_NAME} and quantized+calibrated {MODEL_NAME}...")
    float_model = load_pretrained_model(MODEL_NAME)

    entry = {
        "model_name": MODEL_NAME,
        "calibration_algorithm": "minmax",
        "granularity": "per_tensor",
        "symmetry": "symmetric",
        "calib_set_size": CALIB_SET_SIZE,
        "bitwidth": "int8",
    }
    quant_model, _input_shape = build_quantized_model(entry, repeat_id=REPEAT_ID)
    quant_model.eval()

    preprocess = resolve_preprocess(MODEL_NAME)
    sample = load_eval_subset()[0]
    image = Image.open(sample["path"]).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    float_outputs = {}
    quant_outputs = {}

    def make_hook(store, name):
        def hook(_module, _inputs, output):
            store[name] = output.detach()
        return hook

    handles = []
    for name, module in float_model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_hook(make_hook(float_outputs, name)))
    for name, module in quant_model.named_modules():
        # QuantConv2d/QuantLinear -- match by name against the float model's
        # nn.Conv2d/nn.Linear at the same position in the tree.
        if name in {n for n, m in float_model.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))}:
            handles.append(module.register_forward_hook(make_hook(quant_outputs, name)))

    print("Running float forward pass...")
    with torch.no_grad():
        float_logits = float_model(tensor)[0]
    print("Running quantized forward pass...")
    with torch.no_grad():
        quant_logits = quant_model(tensor)[0]

    for h in handles:
        h.remove()

    print(f"\n{'layer':<50} {'float_norm':>12} {'quant_norm':>12} {'rel_L2_err':>12}")
    first_bad = None
    for name in float_outputs:
        if name not in quant_outputs:
            print(f"{name:<50} MISSING FROM QUANTIZED MODEL (not substituted?)")
            continue
        f = float_outputs[name].float()
        q = quant_outputs[name].float()
        if f.shape != q.shape:
            print(f"{name:<50} SHAPE MISMATCH float={tuple(f.shape)} quant={tuple(q.shape)}")
            continue
        f_norm = f.norm().item()
        q_norm = q.norm().item()
        rel_err = (q - f).norm().item() / (f_norm + 1e-12)
        flag = ""
        if rel_err > 0.5 and first_bad is None:
            first_bad = name
            flag = "  <-- first large divergence"
        print(f"{name:<50} {f_norm:>12.4f} {q_norm:>12.4f} {rel_err:>12.4f}{flag}")

    print(f"\nfinal logits rel L2 error: "
          f"{(quant_logits.float() - float_logits.float()).norm().item() / float_logits.float().norm().item():.4f}")
    if first_bad:
        print(f"First layer with relative L2 error > 0.5: {first_bad}")
    else:
        print("No single layer exceeded 0.5 relative error -- divergence is gradual/compounding, not a single bad layer.")


if __name__ == "__main__":
    main()
