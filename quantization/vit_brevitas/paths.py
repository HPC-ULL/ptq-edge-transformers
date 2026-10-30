"""Shared paths and model-name mapping for the Brevitas ViT toolchain.

Unlike the LLM/llama.cpp branch, there is no separate "convert to a shared
base artifact" step here: timm downloads and caches pretrained weights
itself (its own cache dir, not something this project manages under
artifacts/), and Brevitas quantization + calibration + ONNX export all
happen in one in-process pipeline per (run_id, repeat_id) job -- see
quantize.py and export/vit_to_onnx/export.py.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
VIT_ONNX_DIR = ARTIFACTS_ROOT / "vit"

# Where the ONE-OFF investigations write (gates, fusion probes, per-image
# predictions...): one folder per kind of experiment, one subfolder per run,
# each with its own run.log -- never loose files in the repo root, and never
# under artifacts/ or results/raw/ (those feed the study).
#
#   results/investigations/vit/
#     output_quant_gate/<tag>/      investigations/gate_output_quant.py (ablation rows)
#     deployment_gate/<tag>/        investigations/gate_output_quant.py --deployment
#     fusion_probes/                scripts/probe_int8_kernels.py JSONs (one per run)
#     fusion_probes/models/         the .onnx files those probes take (FP32, native, gate copies)
#     fp32_jpeg_vs_original/        investigations/fp32_jpeg_vs_original.py
#     per_image_predictions/        investigations/per_image_predictions.py
INVESTIGATIONS_ROOT = PROJECT_ROOT / "results" / "investigations" / "vit"
OUTPUT_QUANT_GATE = "output_quant_gate"
DEPLOYMENT_GATE = "deployment_gate"
FUSION_PROBES = "fusion_probes"
FP32_JPEG = "fp32_jpeg_vs_original"
PER_IMAGE = "per_image_predictions"


def investigation_dir(group: str, tag: Optional[str] = None) -> Path:
    """results/investigations/vit/<group>[/<tag>] (not created here)."""
    return INVESTIGATIONS_ROOT / group / tag if tag else INVESTIGATIONS_ROOT / group


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


@contextmanager
def tee_output(log_path: Path) -> Iterator[Path]:
    """Copies everything printed (stdout AND stderr, so warnings and tracebacks
    too) into `log_path`, next to the run's results, while still printing it.
    The log is created before the first line, so a crashed run leaves it too."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out, err = sys.stdout, sys.stderr
    with open(log_path, "a", buffering=1) as log:
        sys.stdout, sys.stderr = _Tee(out, log), _Tee(err, log)
        try:
            yield log_path
        finally:
            sys.stdout, sys.stderr = out, err


# manifest model_name token (configs/vit_manifest.py) -> timm model name.
# calibration_protocol_vit.md lists "MobileViT S/XS" and "EfficientViT-B1/
# EfficientFormer" as alternatives; these are the exact three variants
# confirmed viable end to end via quantization/vit_brevitas/spike_test.py.
MODEL_TIMM_NAMES: Dict[str, str] = {
    "deit_tiny": "deit_tiny_patch16_224",
    "mobilevit": "mobilevit_s",
    "efficientvit_b1": "efficientvit_b1",
}


def timm_model_name(model_name: str) -> str:
    try:
        return MODEL_TIMM_NAMES[model_name]
    except KeyError:
        raise ValueError(
            f"No timm model mapped for model_name={model_name!r}. Known models: {sorted(MODEL_TIMM_NAMES)}."
        ) from None
