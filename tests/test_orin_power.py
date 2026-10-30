"""Tests for orin_power.py's idle-power estimation.

Regression coverage for a real bug: _estimate_standby_load used to sample
one channel's full window, then the next channel's full window, serially
-- giving each channel's idle baseline a different, non-overlapping time
window (unlike the load-measurement loop in estimate_fn_power, which reads
all channels together every tick) and doubling the actual wall-clock
duration of the idle phase beyond what idle_load_duration documents.

Tests construct a PowerMeter without running __init__ (which needs real
INA3221 sysfs paths) -- channels and total_idle_load are set up by hand,
then _estimate_standby_load is called directly.
"""

from __future__ import annotations

import time

import pytest

from benchmarking_ondevice.power_logging import orin_power


def _bare_meter(labels=("CH1", "CH2")) -> orin_power.PowerMeter:
    """A PowerMeter with channels set up but __init__ never run (avoids
    touching real sysfs paths)."""
    meter = orin_power.PowerMeter.__new__(orin_power.PowerMeter)
    meter.channels = [orin_power.Channel(label=label, volt_path="", curr_path="") for label in labels]
    meter.total_idle_load = 0.0
    return meter


def test_idle_phase_interleaves_channel_reads_per_tick(monkeypatch):
    """Both channels must be read within the same tick, in alternation
    (CH1, CH2, CH1, CH2, ...) -- not one channel's full block of samples
    followed by the other's (CH1, CH1, ..., CH2, CH2, ...)."""
    meter = _bare_meter()
    call_order = []

    def fake_get_channel_power(self, channel):
        call_order.append(channel.label)
        return 100.0

    monkeypatch.setattr(orin_power.PowerMeter, "_get_channel_power", fake_get_channel_power)
    monkeypatch.setattr(orin_power, "sleep", lambda s: None)

    meter._estimate_standby_load(duration=2.0, samples=4)

    assert call_order == ["CH1", "CH2"] * 4


def test_idle_phase_sleeps_once_per_sample_not_per_channel(monkeypatch):
    """The old, buggy structure slept once per (channel, sample) pair --
    samples * len(channels) sleeps -- doubling the real duration for 2
    channels. Fixed: exactly `samples` sleeps total, each of `duration /
    samples`, regardless of channel count."""
    meter = _bare_meter()
    sleep_calls = []

    monkeypatch.setattr(orin_power.PowerMeter, "_get_channel_power", lambda self, ch: 100.0)
    monkeypatch.setattr(orin_power, "sleep", lambda s: sleep_calls.append(s))

    meter._estimate_standby_load(duration=10.0, samples=10)

    assert len(sleep_calls) == 10  # not 20 (samples * 2 channels)
    assert sleep_calls == pytest.approx([1.0] * 10)


def test_idle_phase_real_duration_matches_documented_duration_not_double(monkeypatch):
    """End-to-end timing check with real (tiny) sleeps -- confirms the fix
    holds regardless of internal call-counting details. Uses a small
    duration to keep the test fast; asserts elapsed time is close to
    `duration`, and clearly not ~2x it (the old bug's signature for 2
    channels)."""
    meter = _bare_meter()
    monkeypatch.setattr(orin_power.PowerMeter, "_get_channel_power", lambda self, ch: 100.0)
    # orin_power.sleep is real time.sleep here -- deliberately not mocked,
    # this test measures actual wall-clock duration.

    duration = 0.2
    start = time.perf_counter()
    meter._estimate_standby_load(duration=duration, samples=4)
    elapsed = time.perf_counter() - start

    assert elapsed < duration * 1.5, f"elapsed={elapsed:.3f}s -- expected close to duration={duration}s, not ~2x it"


def test_get_instant_power_removed():
    """Dead code cleanup: _get_instant_power was never called anywhere and
    has been removed."""
    assert not hasattr(orin_power.PowerMeter, "_get_instant_power")


# --------------------------------------------------------------------------
# estimate_fn_power's sampling-window counters (sampling_characterization.py
# builds on these). Purely observational: they must never change the energy.
# --------------------------------------------------------------------------

def _meter_with_rate(rate=100.0) -> orin_power.PowerMeter:
    meter = _bare_meter()
    meter.sampling_rate = rate
    meter.sampling_interval = 1.0 / rate
    for channel in meter.channels:
        channel.idle_load = 0.0
    return meter


def test_estimate_fn_power_counts_every_tick_and_reports_the_window(monkeypatch):
    import threading

    meter = _meter_with_rate(100.0)
    monkeypatch.setattr(orin_power.PowerMeter, "_get_channel_power", lambda self, ch: 100.0)

    release = threading.Event()
    ticks = []

    def fake_sleep(seconds):
        ticks.append(seconds)
        if len(ticks) == 7:
            release.set()
            time.sleep(0.05)  # let the worker thread exit, so the loop ends right after this tick

    monkeypatch.setattr(orin_power, "sleep", fake_sleep)

    results = meter.estimate_fn_power(release.wait)
    sampling = results["sampling"]

    assert sampling["n_samples"] == 7
    assert sampling["configured_hz"] == 100.0
    assert sampling["window_s"] > 0
    assert sampling["dt_max_s"] > 0
    assert sampling["dt_max_s"] <= sampling["window_s"]
    assert results["elapsed"] >= sampling["window_s"]  # elapsed also covers the tail after the last sample


def test_sampling_counters_do_not_change_the_energy_integration(monkeypatch):
    """Two channels at a constant 100 mW: total energy must be exactly
    200 mW x window_s -- the window the integrated dt values add up to."""
    import threading

    meter = _meter_with_rate(100.0)
    monkeypatch.setattr(orin_power.PowerMeter, "_get_channel_power", lambda self, ch: 100.0)

    release = threading.Event()
    ticks = []

    def fake_sleep(seconds):
        ticks.append(seconds)
        if len(ticks) == 5:
            release.set()
            time.sleep(0.05)

    monkeypatch.setattr(orin_power, "sleep", fake_sleep)

    results = meter.estimate_fn_power(release.wait)

    assert results["total"]["energy_total_j"] == pytest.approx(0.2 * results["sampling"]["window_s"])
    assert results["total"]["energy_net_j"] == pytest.approx(0.2 * results["sampling"]["window_s"])  # idle_load = 0


# --------------------------------------------------------------------------
# get_power_mode(): every way nvpmodel can fail must mean "cannot answer"
# (None), never an exception -- jetson_powermeter._verify_power_mode is the
# check that stops a measurement from being filed under the wrong power
# mode, and it documents exactly these outcomes.
# --------------------------------------------------------------------------

def _nvpmodel_raises(monkeypatch, exc):
    def fake(*args, **kwargs):
        raise exc

    monkeypatch.setattr(orin_power.subprocess, "check_output", fake)


def test_get_power_mode_returns_none_on_permission_error_and_warns(monkeypatch, caplog):
    _nvpmodel_raises(monkeypatch, PermissionError(13, "Permission denied", "nvpmodel"))

    with caplog.at_level("WARNING"):
        assert orin_power.get_power_mode() is None

    assert "could not be executed" in caplog.text  # unexpected on a Jetson, so it is reported


def test_get_power_mode_returns_none_on_other_exec_failures(monkeypatch):
    _nvpmodel_raises(monkeypatch, OSError(8, "Exec format error", "nvpmodel"))

    assert orin_power.get_power_mode() is None


def test_get_power_mode_missing_nvpmodel_stays_silent(monkeypatch, caplog):
    """Expected outside a Jetson (HPC): None, and NOT a warning."""
    _nvpmodel_raises(monkeypatch, FileNotFoundError(2, "No such file", "nvpmodel"))

    with caplog.at_level("WARNING"):
        assert orin_power.get_power_mode() is None

    assert caplog.text == ""


def test_get_power_mode_returns_none_when_nvpmodel_fails(monkeypatch, caplog):
    import subprocess

    _nvpmodel_raises(monkeypatch, subprocess.CalledProcessError(1, ["nvpmodel", "-q"]))

    with caplog.at_level("WARNING"):
        assert orin_power.get_power_mode() is None

    assert "nvpmodel query failed" in caplog.text


@pytest.mark.parametrize(
    "output, expected",
    [
        ("NV Power ModeName\nNV Power Mode: MODE_15W\n2\n", "15W"),
        ("NV Power Mode: MAXN\n0\n", "MAXN"),
        ("something unexpected\n", None),
    ],
)
def test_get_power_mode_parses_the_active_mode(monkeypatch, output, expected):
    monkeypatch.setattr(orin_power.subprocess, "check_output", lambda *a, **k: output)

    assert orin_power.get_power_mode() == expected
