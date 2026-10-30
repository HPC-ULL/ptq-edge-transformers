"""The ViT deployment set (findings_vit.md 7.6): 8 artifacts, INT8 with
output_quant, measured on the edge devices for REAL integer-kernel speed-up
-- a separate study from the calibration ablation (configs/vit_manifest.py
and vit_phase3_manifest.py, input-only quantization), never mixed with it:
its own run_ids ("vit_<model>_deploy_..."), its own artifacts directory.

Per model (DeiT-Tiny, MobileViT-S; EfficientViT-B1 is excluded, 7.6): the
FP32 reference, the CORE (the configuration deployed) and two variants, each
the core with ONE factor changed -- critique 9: does that factor's cost
depend on the hardware?

  DeiT-Tiny    (mse):        core per_tensor/symmetric;
                             variants per_channel/symmetric, per_tensor/asymmetric
  MobileViT-S  (percentile): core per_channel/asymmetric;
                             variants per_tensor/asymmetric, per_channel/symmetric

MobileViT-S's core is per_channel + asymmetric, decided from the deployment
gate (output_quant, 2000 images, n=3, pre-registered margin std_a + std_b):
per_channel beat the first core per_tensor/symmetric (58.35 +/- 4.01 vs
47.48 +/- 2.09 % top-1), and asymmetric on top of it beat per_channel
(63.33 +/- 0.60 vs 58.35 +/- 4.01). Its two variants are both already
measured configurations.

run_ids name the CONFIGURATION, not the role: vit_<model>_deploy_<granularity>
_<symmetry>_int8 (and vit_<model>_deploy_fp32), so a run_id always means one
configuration whatever role it plays; the role is in factor_varied /
factor_level ("core", or the factor changed and its level).
DeiT-Tiny's quantized rows also carry single_gelu: their export writes each GELU as
one node (see SINGLE_GELU_MODELS); MobileViT-S's rows, the FP32 rows and the two
fallback references do not.

Every quantized row folds BatchNorm into the convolutions and keeps the
folded BN as an identity (bn_fold.py; the removal variant is NOT used, see
7.4.1). DeiT-Tiny has nothing to fold, so for it fold_bn is a no-op.

Plus 2 FALLBACK REFERENCES, packaged next to the deployment set but not part
of it (fallback_references()): the ablation's own INT8 rep1 artifacts of each
model's chosen calibration -- deit_tiny mse and mobilevit percentile, the
Phase 2 rows -- input-only (no output_quant), so ONNX Runtime runs them as
explicit Q/DQ + float compute, slower than FP32. Measured on each device as
the "no integer kernels" contrast to the deployment set. They are used
exactly as the ablation generated them -- never rebuilt, never moved out of
artifacts/vit/ -- and keep their ablation run_id and labels (phase2 /
calibration_algorithm / mse|percentile), which is how results tell them apart
from the deployment rows (run_id "..._deploy_...", phase "deployment").

ViT rows are always stochastic in this project (calibration sampling), so
the quantized rows are delivered as their rep1 artifact, like the ablation.
The fp32 rows are deterministic, but the ViT runner and run_batch require an
integer repeat_id, so they carry repeat_id 1: their one and only export.
"""

from __future__ import annotations

from typing import Any, Dict, List

FAMILY = "vit"
PHASE = "deployment"
MODELS = ["deit_tiny", "mobilevit"]
# Models whose quantized rows are exported with the GELU as ONE node (export/vit_to_onnx/single_gelu.py):
# DeiT-Tiny's MLP uses the exact erf GELU, which ORT cannot fuse around a quantized activation.
# MobileViT-S has no GELU (SiLU), so it is not touched.
SINGLE_GELU_MODELS = {"deit_tiny"}
CORE_CALIBRATION = {"deit_tiny": "mse", "mobilevit": "percentile"}
CALIB_SET_SIZE = 128
QUANTIZED_BITWIDTH = "int8"

# model -> (granularity, symmetry) of its core; each variant flips one of them
CORE_CONFIG = {"deit_tiny": ("per_tensor", "symmetric"), "mobilevit": ("per_channel", "asymmetric")}
_OTHER = {"per_tensor": "per_channel", "per_channel": "per_tensor",
          "symmetric": "asymmetric", "asymmetric": "symmetric"}


def run_id(model: str, granularity: str = "", symmetry: str = "") -> str:
    """vit_<model>_deploy_fp32 without a configuration, else
    vit_<model>_deploy_<granularity>_<symmetry>_int8."""
    suffix = f"{granularity}_{symmetry}_{QUANTIZED_BITWIDTH}" if granularity else "fp32"
    return f"{FAMILY}_{model}_deploy_{suffix}"


def _base(model: str) -> Dict[str, Any]:
    return {"model_family": FAMILY, "model_name": model, "phase": PHASE}


def _fp32(model: str) -> Dict[str, Any]:
    return {**_base(model), "run_id": run_id(model), "factor_varied": "reference", "factor_level": "fp32",
            "quantized": False, "calibration_algorithm": "", "granularity": "", "symmetry": "",
            "calib_set_size": "", "bitwidth": "fp32", "output_quant": False, "fold_bn": False,
            "single_gelu": False}


def _quantized(model: str, granularity: str, symmetry: str, factor_varied: str, factor_level: str) -> Dict[str, Any]:
    return {**_base(model), "run_id": run_id(model, granularity, symmetry), "factor_varied": factor_varied,
            "factor_level": factor_level, "quantized": True, "calibration_algorithm": CORE_CALIBRATION[model],
            "granularity": granularity, "symmetry": symmetry, "calib_set_size": CALIB_SET_SIZE,
            "bitwidth": QUANTIZED_BITWIDTH, "output_quant": True, "fold_bn": True,
            "single_gelu": model in SINGLE_GELU_MODELS}


def model_rows(model: str) -> List[Dict[str, Any]]:
    """fp32, core, granularity variant, symmetry variant -- in that order."""
    g, sym = CORE_CONFIG[model]
    return [
        _fp32(model),
        _quantized(model, g, sym, "core", "core"),
        _quantized(model, _OTHER[g], sym, "granularity", _OTHER[g]),
        _quantized(model, g, _OTHER[sym], "symmetry", _OTHER[sym]),
    ]


def generate() -> List[Dict[str, Any]]:
    """The 8 deployment rows."""
    return [row for model in MODELS for row in model_rows(model)]


def fallback_references() -> List[Dict[str, Any]]:
    """The 2 ablation rows (unchanged, straight from configs/vit_manifest.py)
    packaged as fallback references -- see the module docstring."""
    from . import vit_manifest

    by_id = {e["run_id"]: e for e in vit_manifest.generate()}
    rows = []
    for model in MODELS:
        entry = dict(by_id[f"{FAMILY}_{model}_phase2_calibration_algorithm_{CORE_CALIBRATION[model]}_{QUANTIZED_BITWIDTH}"])
        entry.update(quantized=True, output_quant=False, fold_bn=False, single_gelu=False, fallback_reference=True)
        rows.append(entry)
    return rows


def package_rows() -> List[Dict[str, Any]]:
    """Everything run_manifest_code_vit.csv lists: 8 deployment rows + 2
    fallback references."""
    return generate() + fallback_references()


def quantized_entries() -> List[Dict[str, Any]]:
    return [e for e in generate() if e["quantized"]]
