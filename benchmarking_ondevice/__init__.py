"""On-device benchmarking (latency/throughput) for the ViT and LLM branches.

Deliberately separate from quantization/ and evaluation_offline/ (Section 1
of technical_architecture.md): those run once, offline, on the
workstation/cluster; this runs once per (artifact, device) on each edge
device in the open catalog (configs/devices.yaml).
"""
