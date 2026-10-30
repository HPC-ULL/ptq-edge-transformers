"""Resolves a devices.yaml device_id into the llama-bench flags it drives --
-t (threads, from `cpu_cores`) and -ngl (from `backend`: 0 for cpu, or the
device's own `n_gpu_layers` field for cuda). Neither is ever hardcoded or
left at llama-bench's own default (`-t` defaults to 40, `-ngl` defaults to
-1 = "use GPU if available") -- mirrors benchmarking_ondevice/runner_vit/
devices.py's -x/-e resolution, same fail-loud-don't-guess rationale.

`n_gpu_layers` is a new devices.yaml field, not part of
technical_architecture.md Section 10.5's originally documented
schema -- only meaningful for backend: cuda entries (none exist yet in
configs/devices.yaml). Add it alongside `backend: cuda` when the first GPU
device is catalogued; a cpu-backend entry never needs it (resolved to 0
unconditionally, see resolve_n_gpu_layers).
"""

from __future__ import annotations

from typing import Any, Dict, List

import yaml

from .paths import DEVICES_YAML_PATH


def load_devices() -> List[Dict[str, Any]]:
    if not DEVICES_YAML_PATH.exists():
        raise FileNotFoundError(
            f"No device catalog found at {DEVICES_YAML_PATH} -- see technical_architecture.md Section 10.5."
        )
    with open(DEVICES_YAML_PATH) as f:
        devices = yaml.safe_load(f) or []
    return devices


def resolve_device(device_id: str) -> Dict[str, Any]:
    devices = load_devices()
    for device in devices:
        if device.get("device_id") == device_id:
            return device
    known = sorted(d.get("device_id", "?") for d in devices)
    raise ValueError(f"Unknown device_id={device_id!r} in {DEVICES_YAML_PATH}. Known device_ids: {known}.")


def resolve_threads(device: Dict[str, Any]) -> int:
    cores = device.get("cpu_cores")
    if cores is None:
        raise ValueError(
            f"device_id={device.get('device_id')!r}: cpu_cores is not set in {DEVICES_YAML_PATH} -- "
            "refusing to guess a value for -t (must be the device's real physical core count, "
            "confirmed via `nproc`/`lscpu`, never an assumed default). Fill it in before running "
            "real benchmarks against this device."
        )
    return int(cores)


def resolve_n_gpu_layers(device: Dict[str, Any]) -> int:
    backend = device.get("backend")
    if backend == "cpu":
        return 0
    if backend == "cuda":
        n_gpu_layers = device.get("n_gpu_layers")
        if n_gpu_layers is None:
            raise ValueError(
                f"device_id={device.get('device_id')!r}: backend=cuda but n_gpu_layers is not set in "
                f"{DEVICES_YAML_PATH} -- refusing to fall back to llama-bench's own default (-ngl -1, "
                "\"use GPU if available\"), which is exactly the kind of ambiguous auto-select this "
                "project avoids (technical_architecture.md Section 6). Fill it in explicitly."
            )
        return int(n_gpu_layers)
    raise ValueError(
        f"device_id={device.get('device_id')!r}: no -ngl resolution defined for backend={backend!r}. "
        "Known backends: ['cpu', 'cuda']."
    )
