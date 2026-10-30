"""One-off diagnostic (NOT part of the pipeline, like the other investigations/
scripts): FP32 (unquantized) top-1/top-5 accuracy for the 3 ViT-branch
models, on the SAME fixed evaluation subset as the rest of the ablation
(calibration_protocol_vit.md Section 3.1 -- ImageNet-1K validation split,
never train; data_prep.imagenet.load_eval_subset(), full 2000 images,
class-stratified, never truncated).

Written to correct a specific mistake: diagnose_litemla_isolation.py's
scenario 1 ("full float") reported 81.00%/93.00% for efficientvit_b1, but
that script caps the eval subset to its first 200 samples ("capped for
speed") -- 10% of the real 2000-sample subset, and NOT class-stratified
over that truncated slice (the manifest is built 2 images/class in class
order, so 200 samples covers only ~100 of the 1000 classes). That number
was for a faster isolation experiment, not the model's real accuracy on
the protocol's actual eval subset -- it must NOT be reused as-is for
deit_tiny/mobilevit either, since no equivalent run exists for them at all.

Reuses the exact same code paths the real (quantized) pipeline uses for
comparability: evaluation_offline.vit_accuracy.accuracy.resolve_preprocess
for preprocessing and data_prep.imagenet.load_eval_subset() for the
samples -- the only difference from evaluate_accuracy() is that the model
here is the float pretrained model itself (quantization.vit_brevitas.
quantize.load_pretrained_model), never quantized, run directly in PyTorch
instead of via an exported ONNX artifact.

Standalone: no manifest row, no results/raw/ writes (that CSV is for
quantized artifacts only) -- console output only, exactly like the other
diagnose_*.py scripts here. Redirect to a file yourself if you want a
durable copy (e.g. `> results/appendix/vit_float_baseline.txt`).

Needs: timm (pretrained weights, downloaded on first use), the ImageNet-1K
eval subset already prepared (data_prep.imagenet.prepare_eval_subset()).

Run directly:

    python quantization/vit_brevitas/investigations/float_baseline_accuracy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODELS = ["deit_tiny", "mobilevit", "efficientvit_b1"]


def compute_float_accuracy(model_name: str, preprocess, samples):
    import numpy as np
    import torch
    from PIL import Image

    from quantization.vit_brevitas.quantize import load_pretrained_model

    model = load_pretrained_model(model_name)
    model.eval()

    top1 = top5 = 0
    with torch.no_grad():
        for sample in samples:
            image = Image.open(sample["path"]).convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
            logits = model(tensor)[0].numpy()
            top5_pred = np.argsort(logits)[-5:][::-1]
            top1 += int(top5_pred[0] == sample["label"])
            top5 += int(sample["label"] in top5_pred)

    n = len(samples)
    return (100.0 * top1 / n, 100.0 * top5 / n)


def main() -> None:
    from data_prep.imagenet import load_eval_subset
    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess

    samples = load_eval_subset()  # full fixed subset, never truncated
    print(f"Eval subset: {len(samples)} samples (calibration_protocol_vit.md Section 3.1)\n")

    print(f"{'model':<18} {'top1':>8} {'top5':>8}")
    for model_name in MODELS:
        preprocess = resolve_preprocess(model_name)
        top1, top5 = compute_float_accuracy(model_name, preprocess, samples)
        print(f"{model_name:<18} {top1:>7.2f}% {top5:>7.2f}%")


if __name__ == "__main__":
    main()
