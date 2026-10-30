"""Shared paths for offline ViT top-1/top-5 accuracy evaluation."""

from __future__ import annotations

from quantization.vit_brevitas.paths import PROJECT_ROOT

RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_RAW_DIR = RESULTS_ROOT / "raw"

# Accuracy is computed once, offline, independent of any on-device
# benchmarking (see technical_architecture.md) -- its own raw CSV,
# joined with per-device rows by run_id later during results combination,
# mirroring evaluation_offline/llm_perplexity/paths.py.
VIT_ACCURACY_RESULTS_CSV = RESULTS_RAW_DIR / "vit_accuracy_offline.csv"
