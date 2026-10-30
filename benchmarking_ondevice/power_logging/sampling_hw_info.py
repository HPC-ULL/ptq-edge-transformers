"""Read-only look at the sensor side of power sampling, plus a short probe:

    python -m benchmarking_ondevice.power_logging.sampling_hw_info --device-id jetson_gpu_30w

Records, into results/characterization/power_sampling_hw_<device_id>.json
(and prints them):

  - which INA3221 chips the kernel driver bound, which hwmon directory
    orin_power actually reads (it takes os.listdir(...)[0], mirrored here),
    every attribute that directory exposes, and -- as raw strings, nothing
    assumed -- update_interval, samples, and each channel's label / enable /
    shunt resistor. The driver's data refresh is update_interval = C x S x
    (BC + SC) (enabled channels x averaged samples x bus+shunt conversion
    times); if it is slower than our polling period, consecutive reads
    return the same value and sampling faster adds no information.
  - which channels PowerMeter reads (CHANNEL_INDEXES) versus what the chip
    has, and PowerMeter's default sampling rate.
  - a PROBE: PowerMeter measuring time.sleep(N) on an idle system, i.e. the
    loop's own overhead with no benchmark load -> effective Hz and longest
    gap. (Under benchmark load the real windows are what count; those are
    recorded per measurement by jetson_powermeter.)

Nothing here writes to results/raw/ or changes any configuration.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from . import orin_power, paths
from . import sampling_characterization as sc

INA3221_DRIVERS_DIR = "/sys/bus/i2c/drivers/ina3221"
CHIP_CHANNELS = ("1", "2", "3")


def _read(path: Path) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def collect_hw_info(hwmon_base: str = orin_power.INA3221_HWMON_BASE, drivers_dir: str = INA3221_DRIVERS_DIR) -> Dict:
    info: Dict = {
        "ina3221_devices": None, "hwmon_base": hwmon_base, "hwmon_entries": None, "hwmon_dir_used": None,
        "attributes_present": None, "update_interval_ms": None, "samples": None, "chip_channels": {},
        "channels_read_by_powermeter": list(orin_power.CHANNEL_INDEXES),
    }

    drivers = Path(drivers_dir)
    if drivers.is_dir():
        info["ina3221_devices"] = sorted(p.name for p in drivers.iterdir() if re.fullmatch(r"\d+-00[0-9a-fA-F]{2}", p.name))

    base = Path(hwmon_base)
    if not base.is_dir():
        return info
    entries = os.listdir(base)
    info["hwmon_entries"] = sorted(entries)
    if not entries:
        return info

    hwmon_dir = base / entries[0]  # same pick as orin_power.PowerMeter._channel_scan
    info["hwmon_dir_used"] = str(hwmon_dir)
    info["attributes_present"] = sorted(os.listdir(hwmon_dir))

    raw_interval = _read(hwmon_dir / "update_interval")
    info["update_interval_ms"] = int(raw_interval) if raw_interval is not None and raw_interval.isdigit() else raw_interval
    info["samples"] = _read(hwmon_dir / "samples")
    for i in CHIP_CHANNELS:
        info["chip_channels"][i] = {
            "label": _read(hwmon_dir / f"in{i}_label"),
            "enable": _read(hwmon_dir / f"in{i}_enable"),
            "shunt_resistor": _read(hwmon_dir / f"shunt{i}_resistor"),
        }
    return info


def run_probe(seconds: float) -> Dict:
    """PowerMeter with its production defaults (30 Hz), measuring a sleep."""
    meter = orin_power.PowerMeter(idle_load_duration=2.0, idle_load_samples=2)
    sampling = meter.estimate_fn_power(time.sleep, seconds)["sampling"]
    return {
        "workload": f"time.sleep({seconds:g})  (idle system: loop overhead only, no benchmark load)",
        "configured_hz": sampling["configured_hz"], "n_samples": sampling["n_samples"],
        "window_s": sampling["window_s"], "dt_max_s": sampling["dt_max_s"],
        "effective_hz": sc.effective_hz(sampling["n_samples"], sampling["window_s"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only INA3221 / power-sampling info for one device.")
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--probe-seconds", type=float, default=10.0, help="0 disables the probe.")
    args = parser.parse_args(argv)

    info = {
        "device_id": args.device_id,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kernel": platform.release(),
        "power_mode": orin_power.get_power_mode(),
        "powermeter_default_sampling_hz": inspect.signature(orin_power.PowerMeter.__init__).parameters["sampling_rate"].default,
        "sensor": collect_hw_info(),
    }
    if args.probe_seconds > 0:
        info["probe"] = run_probe(args.probe_seconds)

    out = paths.sampling_hw_json_path(args.device_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
