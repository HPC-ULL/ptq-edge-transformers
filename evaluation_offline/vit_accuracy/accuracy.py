"""Top-1/top-5 accuracy evaluation for a quantized ViT-branch ONNX artifact.

Locates the artifact via export/vit_to_onnx's naming convention
({run_id}__rep{repeat_id}.onnx -- every ViT-branch row is stochastic, no
deterministic exception like the LLM branch), runs it over the *fixed*
ImageNet-1K validation subset already prepared by data_prep/ (never the
calibration pool -- no data leakage), and validates both computed values
(0 <= value <= 100, finite) before they are ever written to results -- a
silently-broken evaluation must fail loudly, not land a nonsense number in
the results table.

Idempotent per (run_id, repeat_id, metric): skipped if both top1_accuracy
and top5_accuracy are already recorded, unless force=True.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

from export.vit_to_onnx import onnx_path
from quantization.vit_brevitas.paths import timm_model_name

ComputeFn = Callable[[Dict, int], Tuple[float, float]]


def resolve_preprocess(model_name: str):
    """The same preprocessing timm resolves for this model at quantization
    time -- reloaded here architecture-only (pretrained=False, no weight
    download) purely for its config metadata; the actual weights already
    live in the exported ONNX artifact."""
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    arch = timm.create_model(timm_model_name(model_name), pretrained=False)
    config = resolve_data_config({}, model=arch)
    return create_transform(**config)


def compute_topk_accuracy(entry: Dict, repeat_id: int) -> Tuple[float, float]:
    """Run the exported ONNX artifact over the fixed ImageNet-1K eval
    subset; return (top1_accuracy, top5_accuracy) as percentages."""
    return compute_topk_accuracy_for_file(entry["model_name"], onnx_path(entry["run_id"], repeat_id))


def compute_topk_accuracy_for_file(model_name: str, model_path) -> Tuple[float, float]:
    """Same evaluation as compute_topk_accuracy, for any .onnx file --
    lets a one-off check (e.g. investigations/gate_output_quant.py) score a
    model that is not a pipeline artifact with exactly the pipeline's code."""
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    from data_prep.imagenet import load_eval_subset

    preprocess = resolve_preprocess(model_name)
    samples = load_eval_subset()

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    top1_correct = 0
    top5_correct = 0
    for sample in samples:
        image = Image.open(sample["path"]).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).numpy().astype(np.float32)
        logits = session.run(None, {input_name: tensor})[0][0]
        top5_pred = np.argsort(logits)[-5:][::-1]
        if top5_pred[0] == sample["label"]:
            top1_correct += 1
        if sample["label"] in top5_pred:
            top5_correct += 1

    n = len(samples)
    return (100.0 * top1_correct / n, 100.0 * top5_correct / n)


def validate_accuracy(value: float, metric_name: str) -> float:
    assert 0.0 <= value <= 100.0 and math.isfinite(value), (
        f"Invalid {metric_name} value: {value!r} -- refusing to write it to results."
    )
    return value


def evaluate_accuracy(
    entry: Dict,
    repeat_id: int,
    *,
    dry_run: bool = False,
    force: bool = False,
    compute_fn: ComputeFn = compute_topk_accuracy,
) -> Optional[Tuple[float, float]]:
    from . import results_io

    run_id = entry["run_id"]
    label = f"{run_id} rep{repeat_id}"

    has_top1 = results_io.has_result(run_id, repeat_id, "top1_accuracy")
    has_top5 = results_io.has_result(run_id, repeat_id, "top5_accuracy")
    if has_top1 and has_top5 and not force:
        print(f"[skip] {label}: accuracy already recorded")
        return None

    if dry_run:
        print(f"[dry-run] {label}: accuracy: evaluate {onnx_path(run_id, repeat_id)} over ImageNet-1K eval subset")
        return None

    top1, top5 = compute_fn(entry, repeat_id)
    validate_accuracy(top1, "top1_accuracy")
    validate_accuracy(top5, "top5_accuracy")

    # Only (re)write a metric that isn't already recorded, unless --force --
    # both are always computed together above, but writing is per-metric so
    # a prior partial write (e.g. a crash between the two appends) can't
    # turn into a duplicate row on a later rerun.
    if force or not has_top1:
        results_io.append_result(entry, repeat_id, "top1_accuracy", top1)
    if force or not has_top5:
        results_io.append_result(entry, repeat_id, "top5_accuracy", top5)
    print(f"{label}: top1={top1:.2f}% top5={top5:.2f}%")
    return (top1, top5)
