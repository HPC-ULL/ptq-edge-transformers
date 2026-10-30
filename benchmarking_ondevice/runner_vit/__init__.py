"""ViT on-device benchmarking: invokes ONNX Runtime's onnxruntime_perf_test
against an already-quantized .onnx artifact, parses its output, and appends
the result to results/raw/results_<device_id>.csv. See cli.py /
`python -m benchmarking_ondevice.runner_vit` for the entry point, and
results_io.py for the per-device CSV schema.
"""
