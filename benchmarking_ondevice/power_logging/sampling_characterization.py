"""Characterization of the power-sampling window: how many samples does each
energy measurement actually contain, and at what effective rate.

Why this exists: an energy figure integrates INA3221 readings taken every
~33 ms (configured 30 Hz) over the WHOLE wrapped benchmark invocation --
model load included, not just the timed inferences. A reviewer can fairly
ask whether fast inferences leave too few samples for that to mean
anything. This module records the answer per measurement, in a file
SEPARATE from results/raw/ (option C: the common results schema is shared
with the other branch and the colleague's meter, and is deliberately not
touched), and classifies already-recorded rows that predate the
instrumentation.

PRE-REGISTERED CRITERION -- fixed on 2026-09-24, BEFORE any instrumented
measurement existed, and not to be adjusted after seeing data (a test pins
both constants, so changing them is a deliberate, visible act):

  MIN_SAMPLES_PER_WINDOW = 30   a window is "sufficient" with >= 30 samples
      (~1 s at the configured 30 Hz). 30 is the conventional minimum for
      trusting a mean: the standard error of the mean power is sigma/sqrt(N),
      i.e. ~0.18 sigma at N=30 versus ~0.32 sigma at N=10.
  WIDE_MARGIN_FACTOR = 3        a RETROACTIVE ESTIMATE is only accepted when it
      clears the threshold by a wide margin, i.e. >= 3 * 30 = 90 samples
      (~3 s). Anything below is re-measured with real instrumentation
      instead of trusting arithmetic.

Retroactive estimates are deliberately conservative: the window is
estimated from the benchmark's own timed portion only (n x latency for ViT;
n x (prompt/prefill_tps + gen/decode_tps) for LLM), which EXCLUDES process
start-up and model loading that the real window also contains -- so the
estimate can only under-state the true sample count.
"""

from __future__ import annotations

import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .paths import sampling_csv_path

MIN_SAMPLES_PER_WINDOW = 30
WIDE_MARGIN_FACTOR = 3

SAMPLING_COLUMNS = [
    "result_id", "run_id", "device_id", "repeat_id", "backend", "power_mode_w",
    "configured_hz", "n_samples", "window_s", "elapsed_s", "effective_hz", "dt_max_s",
    "min_samples_threshold", "meets_threshold", "measured_at",
]

MEASURED_OK = "MEASURED_OK"
MEASURED_BELOW = "MEASURED_BELOW"
ESTIMATED_OK = "ESTIMATED_OK"
REMEASURE = "REMEASURE"
NO_EFFECTIVE_HZ = "NO_EFFECTIVE_HZ"
NO_ESTIMATE = "NO_ESTIMATE"


def resolve_device_entry(device_id: str) -> Dict:
    """The device_id's entry in configs/devices.yaml (backend, power_mode_w,
    ...). Raises if it is not catalogued -- same rule as the rest of the
    on-device tooling."""
    import yaml

    from configs.common import CONFIGS_DIR

    devices_yaml = CONFIGS_DIR / "devices.yaml"
    with open(devices_yaml) as f:
        devices = yaml.safe_load(f) or []
    for device in devices:
        if device.get("device_id") == device_id:
            return device
    known = sorted(d.get("device_id", "?") for d in devices)
    raise ValueError(f"Unknown device_id={device_id!r} in {devices_yaml}. Known device_ids: {known}.")


def effective_hz(n_samples: float, window_s: float) -> Optional[float]:
    if not window_s or window_s <= 0:
        return None
    return n_samples / window_s


def meets_threshold(n_samples: float) -> bool:
    return n_samples >= MIN_SAMPLES_PER_WINDOW


def build_sampling_record(
    row: Dict, device_id: str, device_entry: Dict, sampling: Dict, elapsed_s: float,
    *, measured_at: Optional[str] = None,
) -> Dict[str, object]:
    """One characterization record for one measured window. `row` is the
    finished results row (its result_id is the join key back to it);
    `sampling` is orin_power.estimate_fn_power's own "sampling" dict."""
    n_samples = sampling["n_samples"]
    return {
        "result_id": row["result_id"],
        "run_id": row["run_id"],
        "device_id": device_id,
        "repeat_id": row.get("repeat_id", ""),
        "backend": device_entry.get("backend", ""),
        "power_mode_w": "" if device_entry.get("power_mode_w") is None else device_entry["power_mode_w"],
        "configured_hz": sampling["configured_hz"],
        "n_samples": n_samples,
        "window_s": sampling["window_s"],
        "elapsed_s": elapsed_s,
        "effective_hz": effective_hz(n_samples, sampling["window_s"]) or "",
        "dt_max_s": sampling["dt_max_s"],
        "min_samples_threshold": MIN_SAMPLES_PER_WINDOW,
        "meets_threshold": meets_threshold(n_samples),
        "measured_at": measured_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def read_sampling_records(device_id: str) -> List[Dict[str, str]]:
    path = sampling_csv_path(device_id)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def append_sampling_record(device_id: str, record: Dict[str, object]) -> Path:
    """Writes `record` to results/characterization/power_sampling_<device_id>.csv,
    REPLACING any existing record with the same result_id -- same semantics
    as the results CSVs, so a forced re-measurement of a row replaces its
    record instead of leaving a stale one next to it."""
    path = sampling_csv_path(device_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    kept = [r for r in read_sampling_records(device_id) if r["result_id"] != record["result_id"]]
    kept.append(record)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLING_COLUMNS)
        writer.writeheader()
        writer.writerows(kept)
    return path


def _num(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def estimate_window_s(row: Dict) -> Optional[float]:
    """Lower-bound duration of the window of an ALREADY-RECORDED results row,
    from what the row itself stores. None if the row has no usable timing."""
    n = _num(row.get("num_inferences_measured"))
    if not n:
        return None

    latency_ms = _num(row.get("latency_ms"))
    if latency_ms:  # ViT: n timed inferences of average latency
        return n * latency_ms / 1000.0

    prefill_tps = _num(row.get("throughput_prefill_tps"))
    decode_tps = _num(row.get("throughput_decode_tps"))
    if prefill_tps and decode_tps:  # LLM: n repeats of the prefill test + the decode test
        from benchmarking_ondevice.runner_llm.runner import GEN_TOKENS, PROMPT_TOKENS

        return n * (PROMPT_TOKENS / prefill_tps + GEN_TOKENS / decode_tps)
    return None


def classify_estimate(estimated_samples: float) -> str:
    """ESTIMATED_OK only with a wide margin over the threshold; otherwise the
    row must be re-measured with real instrumentation."""
    return ESTIMATED_OK if estimated_samples >= WIDE_MARGIN_FACTOR * MIN_SAMPLES_PER_WINDOW else REMEASURE


def classify_row(row: Dict, record: Optional[Dict], device_effective_hz: Optional[float]) -> Dict[str, object]:
    """Status of one results row: measured if it has an instrumented record,
    otherwise estimated from the device's own measured effective rate (never
    from the configured 30 Hz, which is only an upper bound)."""
    if record is not None:
        n = float(record["n_samples"])
        return {
            "source": "measured", "n_samples": n, "window_s": _num(record["window_s"]),
            "status": MEASURED_OK if meets_threshold(n) else MEASURED_BELOW,
        }

    window_s = estimate_window_s(row)
    if window_s is None:
        return {"source": "estimated", "n_samples": None, "window_s": None, "status": NO_ESTIMATE}
    if device_effective_hz is None:
        return {"source": "estimated", "n_samples": None, "window_s": window_s, "status": NO_EFFECTIVE_HZ}
    n_est = window_s * device_effective_hz
    return {"source": "estimated", "n_samples": n_est, "window_s": window_s, "status": classify_estimate(n_est)}


def device_effective_hz(records: Iterable[Dict[str, str]]) -> Optional[float]:
    values = [v for v in (_num(r.get("effective_hz")) for r in records) if v]
    return statistics.median(values) if values else None


def summarize_group(records: List[Dict[str, str]]) -> Dict[str, object]:
    """Statistics of the INSTRUMENTED windows of one (device, backend, power
    mode) group -- the numbers for the Methodology sentence."""
    hz = [v for v in (_num(r.get("effective_hz")) for r in records) if v]
    samples = [float(r["n_samples"]) for r in records]
    gaps = [v for v in (_num(r.get("dt_max_s")) for r in records) if v is not None]
    return {
        "n_windows": len(records),
        "effective_hz_median": statistics.median(hz) if hz else "",
        "effective_hz_min": min(hz) if hz else "",
        "n_samples_min": min(samples) if samples else "",
        "n_samples_median": statistics.median(samples) if samples else "",
        "dt_max_s_max": max(gaps) if gaps else "",
        "n_windows_below_threshold": sum(1 for n in samples if not meets_threshold(n)),
    }
