"""Resolves a devices.yaml device_id into the onnxruntime_perf_test flags
it drives -- -e (execution provider, from `backend`) and -x (from
`cpu_cores`). Neither is ever hardcoded in runner.py: both always come from
here, so adding/correcting a device only ever means editing
configs/devices.yaml, never this codebase.

Every resolution step fails loud rather than substituting a default:
technical_architecture.md Section 6 is explicit that -x must be an
unambiguous physical-core count (never hyperthreads, never the "0" auto
-select value) -- a device_id with cpu_cores left unset (e.g.
cluster_dev_test right now, pending `nproc`/`lscpu` confirmation on that
node) must be a loud error here, not a silent fallback to some arbitrary
number.
"""

from __future__ import annotations

from typing import Any, Dict, List

import yaml

from .paths import DEVICES_YAML_PATH

# backend (devices.yaml) -> onnxruntime_perf_test's -e execution provider name.
_BACKEND_TO_EP = {
    "cpu": "cpu",
    "cuda": "cuda",
}


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


def resolve_execution_provider(device: Dict[str, Any]) -> str:
    backend = device.get("backend")
    try:
        return _BACKEND_TO_EP[backend]
    except KeyError:
        raise ValueError(
            f"device_id={device.get('device_id')!r}: no onnxruntime_perf_test execution provider mapped for "
            f"backend={backend!r}. Known backends: {sorted(_BACKEND_TO_EP)}."
        ) from None


def resolve_cpu_cores(device: Dict[str, Any]) -> int:
    cores = device.get("cpu_cores")
    if cores is None:
        raise ValueError(
            f"device_id={device.get('device_id')!r}: cpu_cores is not set in {DEVICES_YAML_PATH} -- "
            "refusing to guess a value for -x (must be the device's real physical core count, "
            "confirmed via `nproc`/`lscpu`, never an assumed default). Fill it in before running "
            "real benchmarks against this device."
        )
    return int(cores)
