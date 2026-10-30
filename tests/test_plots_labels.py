"""plots/paper_labels.py: fixed paper wording and canonical orders."""

from __future__ import annotations

from plots import paper_labels as L


def test_model_labels_and_orders():
    assert [L.MODELS[m] for m in L.VIT_MODEL_ORDER] == ["DeiT-Tiny", "MobileViT-S", "EfficientViT-B1"]
    assert [L.MODELS[m] for m in L.LLM_MODEL_ORDER] == ["TinyLlama-1.1B", "Qwen2.5-1.5B", "Ministral-3-3B"]


def test_bitwidth_and_format_labels():
    assert L.BITWIDTHS_VIT == {"fp32": "FP32", "int8": "INT8", "int6": "INT6", "int4": "INT4"}
    assert L.FORMATS_LLM["Q3_K"] == "Q3_K_M" and L.FORMATS_LLM["legacy"] == "Q4_0"
    assert L.FORMATS_LLM["q4km"] == "Q4_K_M"


def test_calibration_labels_and_palette():
    assert L.CALIBRATION["mse"] == "MSE" and L.CALIBRATION["per_channel"] == "per-channel"
    assert {k: L.COLORS[k] for k in ("baseline", "selected", "fp32")} == {
        "baseline": "#E69F00", "selected": "#0072B2", "fp32": "#000000"}
    assert L.SERIES_LABELS["baseline"] == "Baseline (MinMax)"
