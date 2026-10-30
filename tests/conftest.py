"""Shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_power_characterization_dir(tmp_path_factory, monkeypatch):
    """jetson_powermeter.measure_and_record writes a sampling-characterization
    record next to every results row. Without this, any test that reaches it
    (tests/test_power_logging.py, tests/test_run_batch.py, ...) would write
    into the repo's real results/characterization/."""
    from benchmarking_ondevice.power_logging import paths as power_paths

    monkeypatch.setattr(power_paths, "CHARACTERIZATION_DIR", tmp_path_factory.mktemp("characterization"))
