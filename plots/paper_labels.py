"""CSV value -> paper label, canonical orders and the fixed palette.

Single source of labels for every paper figure: no figure script spells a
label out by hand. Underscores are kept literally (Q4_K_M); never render them
through mathtext.
"""

from __future__ import annotations

VIT_MODEL_ORDER = ("deit_tiny", "mobilevit", "efficientvit_b1")
LLM_MODEL_ORDER = ("tinyllama_1_1b", "qwen2_5_1_5b", "ministral_3_3b")

MODELS = {
    "deit_tiny": "DeiT-Tiny",
    "mobilevit": "MobileViT-S",
    "mobilevit_s": "MobileViT-S",  # spelling used by the deployment summary CSV
    "efficientvit_b1": "EfficientViT-B1",
    "tinyllama_1_1b": "TinyLlama-1.1B",
    "qwen2_5_1_5b": "Qwen2.5-1.5B",
    "ministral_3_3b": "Ministral-3-3B",
}

BITWIDTHS_VIT = {"fp32": "FP32", "int8": "INT8", "int6": "INT6", "int4": "INT4"}

FORMATS_LLM = {
    "f16": "F16",
    "q4km": "Q4_K_M",
    "Q3_K": "Q3_K_M",
    "Q2_K": "Q2_K",
    "legacy": "Q4_0",
}

CALIBRATION = {
    "minmax": "MinMax",
    "percentile": "Percentile",
    "mse": "MSE",
    "per_tensor": "per-tensor",
    "per_channel": "per-channel",
    "symmetric": "symmetric",
    "asymmetric": "asymmetric",
}

SERIES_LABELS = {
    "baseline": f"Baseline ({CALIBRATION['minmax']})",
    "selected": "Selected calibration",
    "best_single": "Best single factor (INT8)",
    "fp32": BITWIDTHS_VIT["fp32"],
}
AXIS_LABELS = {
    "bitwidth": "Bit-width",
    "top1": "Top-1 accuracy (%)",
    "k_float": "Layers kept in floating point (K)",
    "recovered_gap": "Recovered accuracy gap (%)",
}

# ViT CPU deployment (device ids as in results/raw/results_<device>.csv).
# ARM CPUs together, RISC-V last.
DEPLOY_DEVICE_ORDER = ("jetson_cpu_15w", "jetson_cpu_30w", "jetson_cpu_50w", "n150", "rk1-02", "kube01",
                       "visionfive_2")
DEPLOY_MODEL_ORDER = ("deit_tiny", "mobilevit_s")
DEVICES = {
    "jetson_cpu_15w": "Jetson 15 W",
    "jetson_cpu_30w": "Jetson 30 W",
    "jetson_cpu_50w": "Jetson 50 W",
    "n150": "N150",
    "rk1-02": "RK1",
    "kube01": "CM4",
    "visionfive_2": "VisionFive 2",
}
# Two-line variants for the x tick labels (the Jetson ones do not fit in one line).
DEVICE_TICK_LABELS = {**DEVICES, "jetson_cpu_15w": "Jetson\n15 W", "jetson_cpu_30w": "Jetson\n30 W",
                      "jetson_cpu_50w": "Jetson\n50 W"}
DEPLOY_LABELS = {
    "core": "INT8, integer kernels",
    "fallback": "INT8, floating-point fallback",
    "band": "\u00b115% band",
    "speedup": "Speedup (FP32 / INT8)",
    "energy_ratio": "Energy ratio (FP32 / INT8)",
}

# LLM deployment heatmap. Keys are the `config` values of llm_deployment_summary.csv;
# the imatrix and phase3_combined_* configs are deliberately not listed.
LLM_DEPLOY_FORMATS = {"Q4_K_M": "Q4_K_M", "Q4_0": "Q4_0", "Q3_K_M": "Q3_K_M", "Q2_K": "Q2_K"}
LLM_DEPLOY_DEVICES = {
    "jetson_cpu_15w": "Jetson CPU 15 W",
    "jetson_cpu_30w": "Jetson CPU 30 W",
    "jetson_cpu_50w": "Jetson CPU 50 W",
    "jetson_gpu_15w": "Jetson GPU 15 W",
    "jetson_gpu_30w": "Jetson GPU 30 W",
    "jetson_gpu_50w": "Jetson GPU 50 W",
    "n150": "N150",
    "rk1-02": "RK1",
    "kube01": "CM4",
    "visionfive_2": "VisionFive 2",
}
LLM_DEPLOY_LABELS = {
    "energy": "Energy per request (F16 / quantized)",
    "decode": "Decode speedup (quantized / F16)",
    "colorbar": "Ratio (> 1 favors quantized)",
    "neutral": "Within 0.9 to 1.1",
    "missing": "Not measured or excluded",
}

# Calibration concept figure (Introduction).
CONCEPT_LABELS = {"xlabel": "Activation value", "ylabel": "Count"}

# Phase 4 (float-layer recovery) series; keys are figure-internal, not CSV values.
RECOVERY_LABELS = {
    "targeted": "Targeted (highest R)",
    "random": "Random",
    "targeted_rep": "Targeted, reps. 2-3",
    "random_rep": "Random, reps. 2-3",
    "stop": "90% stopping criterion",
}

# Okabe-Ito, fixed for all paper figures.
COLORS = {"baseline": "#E69F00", "selected": "#0072B2", "fp32": "#000000",
          "random": "#999999", "random_light": "#B8B8B8", "stop": "#7F7F7F", "cell_missing": "#D0D0D0", "hist": "#595959", "cell_neutral_hatch": "#8C8C8C",
          "core": "#0072B2", "fallback": "#D55E00", "band": "#E6E6E6"}
