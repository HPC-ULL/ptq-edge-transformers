"""One-off diagnostic (NOT part of the pipeline): does timm resolve a
DIFFERENT preprocessing config for pretrained=True vs pretrained=False, for
any of the 3 models -- and does the manual nn.Conv2d/nn.Linear substitution
in quantize.py's quantize_module_tree() actually cover every
weight-bearing layer in each architecture?

Motivation: mobilevit_s and efficientvit_b1 give near-random top1/top5
accuracy even at calibration_algorithm=minmax baseline, while deit_tiny is
consistently correct. The quantized pipeline never crashes for these two
models -- it just produces garbage -- which points at either (a) a
preprocessing mismatch between quantization time (quantize.py's
resolve_input_shape/load_calibration_batch, both built from a
pretrained=True model instance) and evaluation time
(evaluation_offline/vit_accuracy/accuracy.py's resolve_preprocess, built
from a pretrained=False *architecture-only* instance -- same model name,
different `pretrained` flag), or (b) some weight-bearing layer type in
these architectures that quantize_module_tree's nn.Conv2d/nn.Linear
isinstance checks silently skip (leaving it float, which wouldn't crash or
explain near-random accuracy by itself, but is worth ruling out).

No Brevitas, no gated ImageNet download needed -- just timm's own model
construction and config resolution. Safe/fast to run anywhere timm is
installed (including a local, non-cluster environment).

Run directly:

    python quantization/vit_brevitas/diagnose_model_config.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantization.vit_brevitas.paths import MODEL_TIMM_NAMES  # noqa: E402


def _cfg_summary(config: dict) -> dict:
    keys = ["input_size", "mean", "std", "crop_pct", "crop_mode", "interpolation"]
    return {k: config.get(k) for k in keys}


def check_preprocess_config(model_name: str, timm_name: str) -> None:
    import timm
    from timm.data import resolve_data_config

    print(f"\n=== {model_name} ({timm_name}) ===")

    model_pretrained = timm.create_model(timm_name, pretrained=True)
    model_arch_only = timm.create_model(timm_name, pretrained=False)

    print(f"num_classes: pretrained=True -> {model_pretrained.num_classes}, "
          f"pretrained=False -> {model_arch_only.num_classes}")

    cfg_pretrained = _cfg_summary(resolve_data_config({}, model=model_pretrained))
    cfg_arch_only = _cfg_summary(resolve_data_config({}, model=model_arch_only))

    print(f"resolve_data_config (pretrained=True):  {cfg_pretrained}")
    print(f"resolve_data_config (pretrained=False): {cfg_arch_only}")

    if cfg_pretrained != cfg_arch_only:
        print("MISMATCH: quantize.py (pretrained=True) and accuracy.py's "
              "resolve_preprocess (pretrained=False) would preprocess images "
              "DIFFERENTLY for this model.")
        for key in cfg_pretrained:
            if cfg_pretrained[key] != cfg_arch_only[key]:
                print(f"  -> {key}: pretrained=True={cfg_pretrained[key]!r} "
                      f"vs pretrained=False={cfg_arch_only[key]!r}")
    else:
        print("OK: identical config regardless of pretrained flag.")

    _check_layer_coverage(model_pretrained, model_name)


def _check_layer_coverage(model, model_name: str) -> None:
    import torch.nn as nn

    uncovered = []
    for name, module in model.named_modules():
        has_params = any(True for _ in module.parameters(recurse=False))
        if not has_params:
            continue
        is_leaf_with_own_weight = any(
            p.dim() >= 2 for p in module.parameters(recurse=False)
        )
        if not is_leaf_with_own_weight:
            continue
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        # Something with its own >=2D weight parameter that ISN'T a plain
        # nn.Linear/nn.Conv2d -- quantize_module_tree's isinstance checks
        # would recurse into it (since it's not Linear/Conv2d) and, unless
        # ITS children are themselves Linear/Conv2d, leave it entirely
        # unquantized.
        uncovered.append((name, type(module).__name__))

    print(f"weight-bearing modules NOT nn.Linear/nn.Conv2d (would stay float "
          f"unless their own children are Linear/Conv2d): {len(uncovered)}")
    for name, cls in uncovered[:20]:
        print(f"  {name}: {cls}")
    if len(uncovered) > 20:
        print(f"  ... and {len(uncovered) - 20} more")


def main() -> None:
    for model_name, timm_name in MODEL_TIMM_NAMES.items():
        check_preprocess_config(model_name, timm_name)


if __name__ == "__main__":
    main()
