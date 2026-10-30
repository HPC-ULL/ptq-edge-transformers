"""Shared paths for offline LLM perplexity evaluation.

Reuses the LLAMA_CPP_DIR-based binary resolution and the artifacts/ layout
already established by quantization/llm_llamacpp/ (same llama.cpp checkout,
same shared artifacts tree) rather than duplicating that logic.
"""

from __future__ import annotations

from quantization.llm_llamacpp.paths import ARTIFACTS_ROOT, PROJECT_ROOT

RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_RAW_DIR = RESULTS_ROOT / "raw"

# Perplexity is computed once, offline, independent of any on-device
# benchmarking (see technical_architecture.md) -- it gets its own raw
# CSV here, to be joined with per-device rows by run_id later during
# results combination, rather than living under a per-device raw file.
LLM_PERPLEXITY_RESULTS_CSV = RESULTS_RAW_DIR / "llm_perplexity_offline.csv"

# Full WikiText-2 *test* split, materialized once as plain text for
# `llama-perplexity -f` (shared across every run_id/repeat_id -- the
# evaluation corpus never changes per row, unlike the imatrix corpus).
LLM_EVAL_DIR = ARTIFACTS_ROOT / "llm_eval"
WIKITEXT2_TEST_TEXT_PATH = LLM_EVAL_DIR / "wikitext2_test.txt"
