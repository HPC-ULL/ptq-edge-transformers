"""LLM on-device benchmarking: invokes llama.cpp's llama-bench against an
already-quantized .gguf artifact, parses its JSON output, and appends the
result to results/raw/results_<device_id>.csv. See cli.py /
`python -m benchmarking_ondevice.runner_llm` for the entry point, and
results_io.py for the per-device CSV schema.

Mirrors benchmarking_ondevice/runner_vit/'s structure and conventions
throughout -- see that package's module docstrings for the reasoning
behind patterns reused here without re-explaining them (devices.yaml
resolution never hardcoded, fail-loud on missing config, --self-test's
narrow documented scope, replace-by-result_id CSV writes, etc.).
"""
