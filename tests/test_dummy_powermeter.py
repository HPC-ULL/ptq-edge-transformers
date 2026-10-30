"""Tests for benchmarking_ondevice/power_logging/dummy_powermeter.py.

This module is a documented template, not a real implementation -- every
test here confirms it stays that way: constructing it or calling its
functions raises NotImplementedError immediately, never touching real
hardware, sleeping, or writing anything, and never silently succeeding
with made-up values.
"""

from __future__ import annotations

import pytest

from benchmarking_ondevice.power_logging import dummy_powermeter


def test_constructing_dummy_power_meter_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        dummy_powermeter.DummyPowerMeter()


def test_estimate_fn_power_raises_not_implemented_without_calling_fn():
    """Confirms the dummy's estimate_fn_power doesn't even try to call fn()
    -- fn raising if invoked would prove that."""
    def fn_that_must_not_be_called():
        raise AssertionError("fn() must never be called by the dummy interface")

    # __new__ bypasses __init__ (which itself raises) so we can reach
    # estimate_fn_power directly and confirm IT ALSO raises on its own.
    meter = dummy_powermeter.DummyPowerMeter.__new__(dummy_powermeter.DummyPowerMeter)
    with pytest.raises(NotImplementedError):
        meter.estimate_fn_power(fn_that_must_not_be_called)


def test_get_power_meter_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        dummy_powermeter.get_power_meter()


def test_measure_and_record_raises_not_implemented_without_running_anything():
    """Confirms none of run_fn_bound/build_result_row are ever called --
    only the interface's shape (the call succeeding in binding arguments)
    is exercised, never real behavior."""
    def run_fn_bound_that_must_not_be_called():
        raise AssertionError("run_fn_bound must never be called by the dummy interface")

    def build_result_row_that_must_not_be_called(*args, **kwargs):
        raise AssertionError("build_result_row must never be called by the dummy interface")

    with pytest.raises(NotImplementedError):
        dummy_powermeter.measure_and_record(
            run_fn_bound_that_must_not_be_called,
            build_result_row_that_must_not_be_called,
            "run_x", "dev1", 1, 5,
            perf_per_watt_fn=lambda row, output, energy: None,
        )
