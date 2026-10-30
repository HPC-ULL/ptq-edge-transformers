from __future__ import annotations

from time import sleep
import threading
import time
import os
import logging
import subprocess

# Indexes of the INA3221 channels we are interested in.
# On the Jetson AGX Orin:
#   Channel 1 -> VDD_CPU_CV  (CPU + CV cores)
#   Channel 2 -> VDD_SOC     (GPU / SOC)
CHANNEL_INDEXES = ["1", "2"]

# Base sysfs directory of the INA3221 hwmon driver (the per-chip hwmonN
# subdirectory is discovered under it, see PowerMeter._channel_scan).
INA3221_HWMON_BASE = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/"


class Channel:
    def __init__(self, label, volt_path, curr_path):
        """
        Represents a single INA3221 rail channel.

        :param label:     Human-readable name read from the sysfs label file.
        :param volt_path: sysfs path to the instantaneous voltage (mV).
        :param curr_path: sysfs path to the instantaneous current (mA).
        """
        self.label = label
        self.volt_path = volt_path
        self.curr_path = curr_path
        self.idle_load = 0.0  # mW - estimated during __init__


class PowerMeter:
    def __init__(
        self,
        idle_load_duration: float = 10.0,
        idle_load_samples: int = 10,
        sampling_rate: float = 30.0,
    ):
        """
        Initialises the PowerMeter: discovers INA3221 channels and estimates
        the idle (background) power draw so it can be subtracted later.

        :param idle_load_duration: Seconds spent estimating idle power.
        :param idle_load_samples:  Number of samples taken during that window.
        :param sampling_rate:      Measurement frequency in Hz during a run.
        """
        self.channels: list[Channel] = []
        self._channel_scan(INA3221_HWMON_BASE)
        logging.info("[POWER METER] Found %d power channel(s).", len(self.channels))

        self.total_idle_load = 0.0  # mW
        self._estimate_standby_load(idle_load_duration, idle_load_samples)
        self.sampling_rate = sampling_rate  # Hz, as CONFIGURED (not the effective rate)
        self.sampling_interval = 1.0 / sampling_rate  # seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _channel_scan(self, base_path: str) -> None:
        """Discover hwmon sub-directory and register the configured channels."""
        entries = os.listdir(base_path)
        if not entries:
            raise FileNotFoundError(f"No hwmon entry found under {base_path}")
        hwmon_index = entries[0]
        channel_path = os.path.join(base_path, hwmon_index)

        for i in CHANNEL_INDEXES:
            label_file = f"in{i}_label"
            label_full_path = os.path.join(channel_path, label_file)
            with open(label_full_path) as f:
                channel_label = f.readline().strip()
            logging.info("[POWER METER] Registered channel: %s", channel_label)

            self.channels.append(
                Channel(
                    label=channel_label,
                    volt_path=os.path.join(channel_path, f"in{i}_input"),
                    curr_path=os.path.join(channel_path, f"curr{i}_input"),
                )
            )

    def _get_channel_power(self, channel: Channel) -> float:
        """
        Read voltage (mV) and current (mA) from sysfs and return power in mW.
        Returns 0.0 and logs a warning if either file is unreadable.
        """
        try:
            with open(channel.volt_path, "r") as f:
                voltage = float(f.read().strip())  # mV
            with open(channel.curr_path, "r") as f:
                current = float(f.read().strip())  # mA
        except OSError as exc:
            logging.warning("[POWER METER] Could not read channel %s: %s", channel.label, exc)
            return 0.0

        return (voltage * current) / 1000.0  # mW

    def _estimate_standby_load(self, duration: float, samples: int) -> None:
        """
        Measure idle power over *duration* seconds using *samples* readings,
        then store the per-channel and total averages.

        All channels are sampled together within each tick (same pattern as
        the load-measurement loop in estimate_fn_power) -- NOT one channel's
        full window followed by the next channel's. Sampling them serially
        would give each channel's idle baseline a different, non-overlapping
        time window (e.g. channel 1 over the first half of *duration*,
        channel 2 over the second half), which is a different methodology
        than the load phase (where every tick reads all channels at the same
        instant) and would silently skew the idle baseline if background
        power drifts at all during the estimation window.
        """
        logging.info(
            "[POWER METER] Estimating idle power for %.1f s "
            "(%d samples). Please keep the system idle.",
            duration,
            samples,
        )
        self.total_idle_load = 0.0
        interval = duration / samples

        readings = {ch.label: [] for ch in self.channels}
        for _ in range(samples):
            for channel in self.channels:
                readings[channel.label].append(self._get_channel_power(channel))
            sleep(interval)

        for channel in self.channels:
            channel_readings = readings[channel.label]
            channel.idle_load = sum(channel_readings) / len(channel_readings)
            self.total_idle_load += channel.idle_load
            logging.info(
                "[POWER METER] Idle  %-20s  %7.2f mW",
                channel.label,
                channel.idle_load,
            )

        logging.info("[POWER METER] Total idle power: %.2f mW", self.total_idle_load)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_fn_power(self, fn, *args):
        """
        Run *fn(*args)* in a background thread while sampling power at the
        configured rate.  Returns a results dictionary when *fn* completes.

        Energy values
        -------------
        ``energy_total_j``  - raw energy integrated from all samples (Joules).
        ``energy_net_j``    - energy above the idle baseline (Joules, >= 0).
                             Negative instantaneous deltas are clamped to zero
                             so they never produce a negative net energy figure.

        :param fn:   Callable to measure.
        :param args: Positional arguments forwarded to *fn*.
        :return:     Dict with keys ``elapsed``, one sub-dict per channel label,
                    ``total``, ``sampling`` and ``output``.

        ``sampling`` describes the measurement window itself, purely
        observational (it never feeds the energy integration):
        ``n_samples`` (ticks actually taken), ``window_s`` (start until the
        LAST sample, i.e. exactly the span the integrated dt values add up
        to), ``dt_max_s`` (longest gap between two consecutive samples,
        the first one measured from the start) and ``configured_hz`` (the
        rate that was requested; the effective rate is n_samples / window_s
        and is always a bit lower, since each tick is the sleep plus the
        sysfs reads plus scheduling jitter).
        """
        # Accumulators - keyed by channel label, values in mW*s = mJ
        energy_total = {ch.label: 0.0 for ch in self.channels}
        energy_net = {ch.label: 0.0 for ch in self.channels}

        output_box = {"value": None, "exception": None}

        def _worker():
            try:
                output_box["value"] = fn(*args)
            except Exception as exc:
                output_box["exception"] = exc

        th = threading.Thread(target=_worker, daemon=True)

        logging.info("[POWER METER] Starting power measurement.")
        start_time = time.perf_counter()
        last_sample_time = start_time
        n_samples = 0
        dt_max = 0.0
        th.start()

        try:
            while th.is_alive():
                sleep(self.sampling_interval)

                now = time.perf_counter()
                # Actual elapsed time since the previous sample - this is the
                # correct integration window regardless of OS scheduling jitter.
                dt = now - last_sample_time
                last_sample_time = now
                n_samples += 1
                dt_max = max(dt_max, dt)

                for ch in self.channels:
                    pwr = self._get_channel_power(ch)  # mW
                    energy_total[ch.label] += pwr * dt  # mJ
                    # Clamp negative deltas to zero: a transient dip below idle
                    # does not represent "negative consumption" - it is noise.
                    net_pwr = max(0.0, pwr - ch.idle_load)
                    energy_net[ch.label] += net_pwr * dt  # mJ

        except Exception as exc:
            logging.error("[POWER METER] Measurement loop error: %s", exc)
            th.join()
            raise
        finally:
            th.join()

        elapsed = time.perf_counter() - start_time

        # ---- Build results dict ------------------------------------------
        results = {"elapsed": elapsed}

        for ch in self.channels:
            results[ch.label] = {
                "energy_total_j": energy_total[ch.label] / 1000.0,
                "energy_net_j": energy_net[ch.label] / 1000.0,
            }

        results["total"] = {
            "energy_total_j": sum(energy_total.values()) / 1000.0,
            "energy_net_j": sum(energy_net.values()) / 1000.0,
        }

        results["sampling"] = {
            "n_samples": n_samples,
            "window_s": last_sample_time - start_time,
            "dt_max_s": dt_max,
            "configured_hz": self.sampling_rate,
        }

        results["output"] = output_box["value"]

        if output_box["exception"] is not None:
            raise output_box["exception"]

        self._log_results(results)
        return results

    def _log_results(self, results: dict) -> None:
        """Pretty-print a results dict produced by estimate_fn_power."""
        logging.info("[POWER METER] -- Results -----------------------------------------------")
        logging.info("[POWER METER]   Elapsed time : %.3f s", results["elapsed"])
        sampling = results["sampling"]
        logging.info(
            "[POWER METER]   Samples      : %d in %.3f s (configured %.1f Hz, longest gap %.1f ms)",
            sampling["n_samples"], sampling["window_s"], sampling["configured_hz"], sampling["dt_max_s"] * 1000.0,
        )
        for ch in self.channels:
            r = results[ch.label]
            logging.info(
                "[POWER METER]   %-20s  total=%7.4f J   net=%7.4f J",
                ch.label,
                r["energy_total_j"],
                r["energy_net_j"],
            )
        t = results["total"]
        logging.info(
            "[POWER METER]   %-20s  total=%7.4f J   net=%7.4f J",
            "TOTAL",
            t["energy_total_j"],
            t["energy_net_j"],
        )
        logging.info("[POWER METER] ---------------------------------------------------------------")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def get_power_mode() -> str | None:
    """
    Query nvpmodel for the active NV Power Mode name.
    Returns the mode string (e.g. '15W', 'MAXN') or None on failure.
    """
    try:
        output = subprocess.check_output(
            ["nvpmodel", "-q"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except FileNotFoundError:
        # nvpmodel does not exist outside the Jetson — expected behavior on HPC.
        return None
    except subprocess.CalledProcessError as exc:
        logging.warning("[POWER METER] nvpmodel query failed: %s", exc)
        return None
    except OSError as exc:
        # nvpmodel exists but cannot be run (PermissionError, exec format
        # error, ...): "cannot answer", same outcome as the cases above.
        # jetson_powermeter._verify_power_mode documents exactly this
        # (nvpmodel missing / permission error / unparseable -> None, then
        # warn and continue) -- an uncaught exception here would instead
        # abort the run from the very check meant to protect it.
        logging.warning("[POWER METER] nvpmodel could not be executed: %s", exc)
        return None

    for line in output.splitlines():
        if "NV Power Mode:" in line:
            # Format: "NV Power Mode: MODE_15W"
            _, _, mode_full = line.partition(":")
            parts = mode_full.strip().split("_", 1)
            return parts[-1] if parts else mode_full.strip()

    return None


def get_cpu_temperature(zone: int = 0) -> float:
    """
    Read CPU temperature from the thermal sysfs interface.

    :param zone: Thermal zone index (default 0).
    :return:     Temperature in degrees C.
    """
    path = f"/sys/class/thermal/thermal_zone{zone}/temp"
    with open(path, "r") as f:
        return int(f.read().strip()) / 1000.0
