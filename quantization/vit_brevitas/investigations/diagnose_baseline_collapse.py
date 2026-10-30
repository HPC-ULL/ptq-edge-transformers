"""One-off diagnostic (NOT part of the pipeline): WHERE in the pipeline does
mobilevit/efficientvit_b1's near-random baseline accuracy get introduced?

quantization.vit_brevitas --all completed for all 72 jobs without a single
crash, yet mobilevit/efficientvit_b1 give near-chance-level top1/top5 even
at calibration_algorithm=minmax (the least aggressive setting) while
deit_tiny is consistently correct across every tested configuration. A
silent, non-crashing corruption that is architecture-specific rules out
"quantization is just imprecise for these architectures" as the sole
explanation (chance-level accuracy is far more severe than typical
per-tensor-PTQ degradation on depthwise-conv-heavy nets) and points at a
correctness bug somewhere in this project's own code.

This bisects the pipeline into 3 stages for one model at a time, run over a
SMALL slice of the real eval subset (fast) so the collapse point is
identified directly instead of guessed at from source reading:

  Stage A -- FLOAT, no Brevitas at all: load_pretrained_model() + accuracy.py's
             own resolve_preprocess(), run directly in PyTorch.
             (this is what evaluation_offline/vit_accuracy/accuracy.py's
             resolve_preprocess() will preprocess with at eval time)
  Stage B -- QUANTIZED, in-memory, NOT exported to ONNX: build_quantized_model()
             (the real pipeline through calibration), run directly in
             PyTorch (same resolve_preprocess() as stage A, bypassing
             export/onnxruntime entirely).
  Stage C -- QUANTIZED + ONNX-exported (the real, already-produced artifact):
             loaded via onnxruntime, exactly as accuracy.py's
             compute_topk_accuracy() does.

If A is already near-random: the bug is in preprocessing/data/labels, not
in Brevitas or export at all -- rerun with model_name="deit_tiny" as a
control to confirm deit_tiny's stage A is NOT near-random on the same code
path.
If A is correct but B collapses: the bug is in quantize.py's
quantize_module_tree()/calibration (weight/activation substitution or
calibration itself).
If A and B are correct but C collapses: the bug is in export.py's ONNX
export (export_onnx_qcdq) or in onnxruntime inference / accuracy.py's own
preprocessing (resolve_preprocess uses a pretrained=False model instance,
unlike stage A/B, which both go through quantize.py's pretrained=True
instance -- see diagnose_model_config.py, which checks whether these two
configs actually differ, without needing Brevitas or gated data at all).

Needs: Brevitas installed, and the gated ImageNet-1K eval subset already
prepared (data_prep.imagenet.prepare_eval_subset()) -- run on the cluster,
not locally. Also needs quantization.vit_brevitas --all to have already
produced the artifact for the (model, minmax baseline) row being tested.

Nothing here writes to artifacts/ or results/ -- purely diagnostic.

Run directly (edit MODEL_NAME / N_SAMPLES below as needed):

    python quantization/vit_brevitas/diagnose_baseline_collapse.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# One row per model: the phase1 baseline (minmax, per_tensor, symmetric)
# run_id, whose artifact --all already generated. Edit if your baseline
# run_ids differ.
MODEL_RUN_IDS = {
    "deit_tiny": "vit_deit_tiny_phase1_baseline_default_int8",
    "mobilevit": "vit_mobilevit_phase1_baseline_default_int8",
    "efficientvit_b1": "vit_efficientvit_b1_phase1_baseline_default_int8",
}
N_SAMPLES = 100  # small slice of the 2000-image eval subset, for speed
REPEAT_ID = 1


def _topk_from_logits(logits, label: "object", k5: bool = True):
    import numpy as np

    top5 = np.argsort(logits)[-5:][::-1]
    top1_hit = int(top5[0] == label)
    top5_hit = int(label in top5)
    return top1_hit, top5_hit


def _load_samples(n_samples: int):
    from data_prep.imagenet import load_eval_subset

    samples = load_eval_subset()
    return samples[:n_samples]


def stage_a_float(model_name: str, samples) -> tuple:
    """FLOAT baseline, using accuracy.py's own resolve_preprocess (the exact
    function that will preprocess images at real evaluation time)."""
    import numpy as np
    import torch
    from PIL import Image

    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess
    from quantization.vit_brevitas.quantize import load_pretrained_model

    model = load_pretrained_model(model_name)
    preprocess = resolve_preprocess(model_name)

    top1 = top5 = 0
    with torch.no_grad():
        for sample in samples:
            image = Image.open(sample["path"]).convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
            logits = model(tensor)[0].numpy()
            h1, h5 = _topk_from_logits(logits, sample["label"])
            top1 += h1
            top5 += h5
    n = len(samples)
    return (100.0 * top1 / n, 100.0 * top5 / n)


def stage_b_quantized_in_memory(model_name: str, samples) -> tuple:
    """Quantized + calibrated, but evaluated directly in PyTorch -- never
    exported to ONNX. Uses accuracy.py's resolve_preprocess (NOT
    quantize.py's own resolve_input_shape/load_calibration_batch
    preprocessing), matching what evaluation actually does, to isolate
    export/onnxruntime as a variable."""
    import numpy as np
    import torch
    from PIL import Image

    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess
    from quantization.vit_brevitas.quantize import build_quantized_model

    entry = {
        "model_name": model_name,
        "calibration_algorithm": "minmax",
        "granularity": "per_tensor",
        "symmetry": "symmetric",
        "calib_set_size": 128,
        "bitwidth": "int8",
    }
    quant_model, _input_shape = build_quantized_model(entry, repeat_id=REPEAT_ID)
    preprocess = resolve_preprocess(model_name)

    top1 = top5 = 0
    with torch.no_grad():
        for sample in samples:
            image = Image.open(sample["path"]).convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
            logits = quant_model(tensor)[0].numpy()
            h1, h5 = _topk_from_logits(logits, sample["label"])
            top1 += h1
            top5 += h5
    n = len(samples)
    return (100.0 * top1 / n, 100.0 * top5 / n)


def stage_c_onnx(model_name: str, run_id: str, samples) -> tuple:
    """The real, already-exported artifact via onnxruntime -- identical
    code path to accuracy.py's compute_topk_accuracy(), just capped to
    N_SAMPLES for speed."""
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess
    from export.vit_to_onnx import onnx_path

    preprocess = resolve_preprocess(model_name)
    path = onnx_path(run_id, REPEAT_ID)
    if not path.exists():
        print(f"  SKIP stage C: {path} does not exist")
        return None

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    top1 = top5 = 0
    for sample in samples:
        image = Image.open(sample["path"]).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).numpy().astype("float32")
        logits = session.run(None, {input_name: tensor})[0][0]
        h1, h5 = _topk_from_logits(logits, sample["label"])
        top1 += h1
        top5 += h5
    n = len(samples)
    return (100.0 * top1 / n, 100.0 * top5 / n)


def main() -> None:
    samples = _load_samples(N_SAMPLES)
    print(f"Using {len(samples)} eval-subset samples per stage.\n")

    for model_name, run_id in MODEL_RUN_IDS.items():
        print(f"=== {model_name} ({run_id}) ===")

        a = stage_a_float(model_name, samples)
        print(f"  Stage A (float, no quantization):        top1={a[0]:.2f}% top5={a[1]:.2f}%")

        b = stage_b_quantized_in_memory(model_name, samples)
        print(f"  Stage B (quantized, in-memory, no ONNX):  top1={b[0]:.2f}% top5={b[1]:.2f}%")

        c = stage_c_onnx(model_name, run_id, samples)
        if c is not None:
            print(f"  Stage C (quantized + ONNX export, real):  top1={c[0]:.2f}% top5={c[1]:.2f}%")
        print()


if __name__ == "__main__":
    main()
