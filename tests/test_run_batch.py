"""Tests for scripts/run_batch.py.

Isolates results paths to tmp_path for both branches (same pattern as
tests/test_runner_vit.py / tests/test_runner_llm.py) and monkeypatches
each branch's run_benchmark to a Mock so no real subprocess/binary is ever
touched. has_result's real logic is exercised (not mocked) against a
minimal stub CSV, so the resumability check is verified for real, not
assumed.
"""

from __future__ import annotations

import csv
import sys
import time
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_batch as run_batch
from benchmarking_ondevice.power_logging import jetson_powermeter
from benchmarking_ondevice.power_logging import results_io as power_results_io
from benchmarking_ondevice.runner_llm import results_io as llm_results_io
from benchmarking_ondevice.runner_llm import runner as llm_runner_module
from benchmarking_ondevice.runner_vit import results_io as vit_results_io
from benchmarking_ondevice.runner_vit import runner as vit_runner_module


def _write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "repeat_id", "artifact_path"])
        for row in rows:
            writer.writerow(row)


def _write_stub_results_csv(path, result_ids):
    """A minimal results CSV -- has_result only ever reads the result_id
    column, so this is enough to exercise its real logic without needing
    the full RESULT_COLUMNS schema."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["result_id"])
        for rid in result_ids:
            writer.writerow([rid])


def _write_partial_results_csv(path, rows):
    """Like _write_stub_results_csv, but for rows that need an explicit
    (possibly blank) energy_per_inference column -- mimics a real results
    CSV without needing the full RESULT_COLUMNS schema, since
    _resumability_status's --with-power completeness check only ever reads
    result_id/energy_per_inference."""
    fieldnames = sorted({k for row in rows for k in row} | {"result_id", "energy_per_inference"})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _isolate_vit_results(tmp_path, monkeypatch):
    monkeypatch.setattr(vit_results_io, "results_csv_path", lambda device_id: tmp_path / f"results_{device_id}.csv")


def _isolate_llm_results(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_results_io, "results_csv_path", lambda device_id: tmp_path / f"results_{device_id}.csv")


def _isolate_power_results(tmp_path, monkeypatch):
    """--with-power writes via power_logging/results_io.py (a separate
    module from vit_results_io/llm_results_io, though it targets the same
    physical file) -- both must point at the same tmp_path location, or
    has_result's pre-check and jetson_powermeter's actual write would
    disagree on where the file lives."""
    monkeypatch.setattr(power_results_io, "results_csv_path", lambda device_id: tmp_path / f"results_{device_id}.csv")


class _FakeMeter:
    """Stand-in for orin_power.PowerMeter -- no real sensor, no real idle
    wait. Records every construction call so the cache test can confirm
    exactly one PowerMeter is built across an entire batch."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def estimate_fn_power(self, fn):
        output = fn()
        return {
            "elapsed": 2.0, "total": {"energy_total_j": 10.0, "energy_net_j": 4.0},
            "sampling": {"n_samples": 60, "window_s": 2.0, "dt_max_s": 0.04, "configured_hz": 30.0},
            "output": output,
        }


@pytest.fixture(autouse=True)
def _no_real_sleep_by_default(monkeypatch):
    """The default cooldown strategy calls time.sleep(--cooldown-seconds,
    default 30s) after every real run -- without this, every test in this
    file that calls main() in real (non-dry-run) mode would actually wait
    that long per row (confirmed for real: the file took 150s before this
    fixture existed, exactly 30s x 4 real single-row runs + 60s for the one
    two-row test). Tests that specifically exercise sleep/cooldown timing
    re-patch time.sleep themselves inside their own body, which takes
    precedence over this default."""
    monkeypatch.setattr(run_batch.time, "sleep", lambda seconds: None)


@pytest.fixture(autouse=True)
def _no_min_uptime_for_the_fake_devices(monkeypatch):
    """main() reads the device's min_uptime_s from configs/devices.yaml;
    the device_ids these tests use ("dev1", ...) are not in it. No wait by
    default -- tests of the wait itself re-patch this."""
    monkeypatch.setattr(run_batch, "resolve_min_uptime_s", lambda device_id: 0.0)


# --------------------------------------------------------------------------
# read_manifest: blank repeat_id handling
# --------------------------------------------------------------------------

def test_read_manifest_parses_int_repeat_id(tmp_path):
    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "2", "artifacts/vit/run_a__rep2.onnx"]])

    rows = list(run_batch.read_manifest(manifest, "vit"))
    assert rows == [("run_a", 2, "artifacts/vit/run_a__rep2.onnx")]


def test_read_manifest_llm_blank_repeat_id_becomes_none(tmp_path):
    manifest = tmp_path / "run_manifest_llm_x.csv"
    _write_manifest(manifest, [["baseline_run", "", "artifacts/llm/baseline_run.gguf"]])

    rows = list(run_batch.read_manifest(manifest, "llm"))
    assert rows == [("baseline_run", None, "artifacts/llm/baseline_run.gguf")]


def test_read_manifest_vit_blank_repeat_id_raises():
    """ViT is always stochastic -- a blank repeat_id in a vit manifest is a
    real data error (wrong branch, or a corrupted manifest), not a valid
    deterministic case."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "run_manifest_code_vit.csv"
        _write_manifest(manifest, [["run_a", "", "artifacts/vit/run_a.onnx"]])
        with pytest.raises(ValueError, match="always stochastic"):
            list(run_batch.read_manifest(manifest, "vit"))


# --------------------------------------------------------------------------
# main(): resumability, --dry-run, branch dispatch
# --------------------------------------------------------------------------

def test_skips_rows_already_recorded_and_runs_the_rest(tmp_path, monkeypatch):
    _isolate_vit_results(tmp_path, monkeypatch)
    fake_run_benchmark = Mock(return_value=None)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", fake_run_benchmark)

    results_csv = tmp_path / "results_dev1.csv"
    _write_stub_results_csv(results_csv, ["run_a__dev1__rep1"])  # already done

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(
        manifest,
        [
            ["run_a", "1", "artifacts/vit/run_a__rep1.onnx"],  # already recorded -> skip
            ["run_b", "1", "artifacts/vit/run_b__rep1.onnx"],  # not recorded -> run
        ],
    )

    rc = run_batch.main(["--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1"])

    assert rc == 0
    fake_run_benchmark.assert_called_once()
    called_run_id = fake_run_benchmark.call_args[0][0]
    assert called_run_id == "run_b"


def test_skip_notice_is_printed_not_silent(tmp_path, monkeypatch, capsys):
    _isolate_vit_results(tmp_path, monkeypatch)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock())

    results_csv = tmp_path / "results_dev1.csv"
    _write_stub_results_csv(results_csv, ["run_a__dev1__rep1"])

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    run_batch.main(["--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1"])

    out = capsys.readouterr().out
    assert "[skip]" in out
    assert "run_a" in out
    assert "already in" in out


def test_dry_run_never_invokes_subprocess_but_still_checks_skips(tmp_path, monkeypatch, capsys):
    _isolate_vit_results(tmp_path, monkeypatch)
    fake_run_benchmark = Mock(return_value=None)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", fake_run_benchmark)

    results_csv = tmp_path / "results_dev1.csv"
    _write_stub_results_csv(results_csv, ["run_a__dev1__rep1"])

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(
        manifest,
        [
            ["run_a", "1", "artifacts/vit/run_a__rep1.onnx"],
            ["run_b", "1", "artifacts/vit/run_b__rep1.onnx"],
        ],
    )

    run_batch.main(["--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1", "--dry-run"])

    fake_run_benchmark.assert_called_once()
    _, kwargs = fake_run_benchmark.call_args
    assert kwargs["dry_run"] is True
    out = capsys.readouterr().out
    assert "[skip]" in out  # run_a still reported as skipped even in dry-run


def test_llm_branch_passes_none_repeat_id_through_for_deterministic_row(tmp_path, monkeypatch):
    _isolate_llm_results(tmp_path, monkeypatch)
    fake_run_benchmark = Mock(return_value=None)
    monkeypatch.setattr(llm_runner_module, "run_benchmark", fake_run_benchmark)

    manifest = tmp_path / "run_manifest_llm_x.csv"
    _write_manifest(manifest, [["baseline_run", "", "artifacts/llm/baseline_run.gguf"]])

    run_batch.main(["--branch", "llm", "--manifest", str(manifest), "--device-id", "dev1"])

    fake_run_benchmark.assert_called_once()
    args = fake_run_benchmark.call_args[0]
    assert args[0] == "baseline_run"
    assert args[3] is None  # repeat_id positional arg


def test_branch_dispatch_only_calls_the_selected_branchs_run_benchmark(tmp_path, monkeypatch):
    _isolate_vit_results(tmp_path, monkeypatch)
    _isolate_llm_results(tmp_path, monkeypatch)
    fake_vit_run_benchmark = Mock(return_value=None)
    fake_llm_run_benchmark = Mock(return_value=None)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", fake_vit_run_benchmark)
    monkeypatch.setattr(llm_runner_module, "run_benchmark", fake_llm_run_benchmark)

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    run_batch.main(["--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1"])

    fake_vit_run_benchmark.assert_called_once()
    fake_llm_run_benchmark.assert_not_called()


def test_summary_counts_total_skipped_and_run(tmp_path, monkeypatch, capsys):
    _isolate_vit_results(tmp_path, monkeypatch)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=None))

    results_csv = tmp_path / "results_dev1.csv"
    _write_stub_results_csv(results_csv, ["run_a__dev1__rep1"])

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(
        manifest,
        [
            ["run_a", "1", "artifacts/vit/run_a__rep1.onnx"],
            ["run_b", "1", "artifacts/vit/run_b__rep1.onnx"],
            ["run_c", "1", "artifacts/vit/run_c__rep1.onnx"],
        ],
    )

    run_batch.main(["--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1"])

    out = capsys.readouterr().out
    assert "3 manifest row(s)" in out
    assert "1 already existed" in out
    assert "2 run" in out


def test_missing_manifest_file_errors_clearly(tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_batch.main(
            ["--branch", "vit", "--manifest", str(tmp_path / "does_not_exist.csv"), "--device-id", "dev1"]
        )


# --------------------------------------------------------------------------
# Cooldown strategies: pluggable, one interchangeable per run
# --------------------------------------------------------------------------

def test_fixed_cooldown_sleeps_the_configured_duration(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(run_batch.time, "sleep", lambda s: sleep_calls.append(s))

    cooldown = run_batch.make_fixed_cooldown(12.5)
    cooldown()

    assert sleep_calls == [12.5]


def test_fixed_cooldown_skips_sleep_when_zero(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(run_batch.time, "sleep", lambda s: sleep_calls.append(s))

    run_batch.make_fixed_cooldown(0)()

    assert sleep_calls == []


def _fake_temp_module(name: str, get_cpu_temperature_fn):
    module = types.ModuleType(name)
    module.get_cpu_temperature = get_cpu_temperature_fn
    return module


def test_temperature_cooldown_returns_once_simulated_temperature_drops_below_threshold():
    readings = iter([80.0, 70.0, 40.0])
    fake_module = _fake_temp_module("tests._fake_temp_impl_readings", lambda: next(readings))

    cooldown = run_batch.make_temperature_cooldown(
        threshold_c=55.0,
        max_wait_seconds=5.0,
        temp_impl_module=fake_module,
        poll_interval_s=0.01,  # real, tiny sleep -- keeps the test fast
    )

    cooldown()  # must return once it reads 40.0 (< 55.0), not hang or raise StopIteration


def test_temperature_cooldown_respects_max_wait_when_temperature_never_drops(capsys):
    fake_module = _fake_temp_module("tests._fake_temp_impl_never_drops", lambda: 99.0)

    cooldown = run_batch.make_temperature_cooldown(
        threshold_c=55.0,
        max_wait_seconds=0.05,
        temp_impl_module=fake_module,
        poll_interval_s=0.01,
    )

    start = time.monotonic()
    cooldown()
    elapsed = time.monotonic() - start

    # Bounded by the safety cap, not hanging indefinitely -- generous
    # tolerance since this is real wall-clock timing in a test.
    assert elapsed < 1.0
    assert "safety limit" in capsys.readouterr().out


def test_resolve_temp_impl_module_raises_if_orin_power_not_available(monkeypatch):
    # The standard trick to simulate an unimportable module: sys.modules[name]
    # = None makes any subsequent `import name` (or `from name import x`)
    # raise ImportError immediately.
    monkeypatch.setitem(sys.modules, "benchmarking_ondevice.power_logging.orin_power", None)

    with pytest.raises(RuntimeError, match="could not be imported"):
        run_batch.resolve_temp_impl_module("jetson")


def test_main_fails_fast_on_temp_cooldown_without_orin_power_before_any_row_runs(tmp_path, monkeypatch):
    _isolate_vit_results(tmp_path, monkeypatch)
    fake_run_benchmark = Mock(side_effect=AssertionError("run_benchmark must never be called"))
    monkeypatch.setattr(vit_runner_module, "run_benchmark", fake_run_benchmark)
    monkeypatch.setitem(sys.modules, "benchmarking_ondevice.power_logging.orin_power", None)

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    with pytest.raises(RuntimeError, match="could not be imported"):
        run_batch.main(
            ["--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1", "--temp-cooldown"]
        )

    fake_run_benchmark.assert_not_called()


def test_jetson_cooldown_flag_is_deprecated_but_still_fails_fast_without_orin_power(tmp_path, monkeypatch, capsys):
    """--jetson-cooldown (deprecated alias) must still behave exactly like
    --temp-cooldown (--temp-cooldown-impl already defaults to jetson) --
    same fail-fast check, plus a printed deprecation notice."""
    _isolate_vit_results(tmp_path, monkeypatch)
    fake_run_benchmark = Mock(side_effect=AssertionError("run_benchmark must never be called"))
    monkeypatch.setattr(vit_runner_module, "run_benchmark", fake_run_benchmark)
    monkeypatch.setitem(sys.modules, "benchmarking_ondevice.power_logging.orin_power", None)

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    with pytest.raises(RuntimeError, match="could not be imported"):
        run_batch.main(
            ["--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1", "--jetson-cooldown"]
        )

    fake_run_benchmark.assert_not_called()
    assert "deprecation" in capsys.readouterr().out.lower()


def test_temp_cooldown_impl_jetson_explicit_matches_jetson_cooldown_flag_behavior(tmp_path, monkeypatch):
    """Regression: --temp-cooldown --temp-cooldown-impl jetson must behave
    like the pre-existing --jetson-cooldown flag -- both resolve to the
    same orin_power-backed strategy and complete the batch the same way."""
    _isolate_vit_results(tmp_path, monkeypatch)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=None))
    # Already below any --temp-threshold-c used here -> cooldown returns on
    # its first poll, no real waiting in this test.
    monkeypatch.setattr(jetson_powermeter.orin_power, "get_cpu_temperature", lambda: 30.0)

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    rc_new = run_batch.main(
        [
            "--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1",
            "--temp-cooldown", "--temp-cooldown-impl", "jetson", "--temp-threshold-c", "55",
        ]
    )

    manifest2 = tmp_path / "run_manifest_code_vit_2.csv"
    _write_manifest(manifest2, [["run_b", "1", "artifacts/vit/run_b__rep1.onnx"]])

    rc_old = run_batch.main(
        [
            "--branch", "vit", "--manifest", str(manifest2), "--device-id", "dev1",
            "--jetson-cooldown", "--temp-threshold-c", "55",
        ]
    )

    assert rc_new == 0 == rc_old


def test_temp_cooldown_impl_dummy_module_is_invoked_without_touching_orin_power(tmp_path, monkeypatch):
    """A fake temperature reader registered under a different name
    (simulating a different board's own sensor, following the
    get_cpu_temperature() -> float contract) is invoked correctly through
    the same --temp-cooldown-impl mechanism, with the REAL
    orin_power.get_cpu_temperature never touched anywhere in this run."""
    _isolate_vit_results(tmp_path, monkeypatch)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=None))

    calls = []
    fake_module = _fake_temp_module("tests._fake_temp_impl_dummy_board", lambda: (calls.append(1), 30.0)[1])
    monkeypatch.setitem(sys.modules, "tests._fake_temp_impl_dummy_board", fake_module)

    def _must_not_be_called():
        raise AssertionError("orin_power.get_cpu_temperature must never be called with a different --temp-cooldown-impl")

    monkeypatch.setattr(jetson_powermeter.orin_power, "get_cpu_temperature", _must_not_be_called)

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    rc = run_batch.main(
        [
            "--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1",
            "--temp-cooldown", "--temp-cooldown-impl", "tests._fake_temp_impl_dummy_board",
        ]
    )

    assert rc == 0
    assert calls == [1]  # the fake implementation was actually invoked


def test_main_calls_fixed_cooldown_between_real_runs(tmp_path, monkeypatch):
    _isolate_vit_results(tmp_path, monkeypatch)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=None))
    sleep_calls = []
    monkeypatch.setattr(run_batch.time, "sleep", lambda s: sleep_calls.append(s))

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(
        manifest,
        [
            ["run_a", "1", "artifacts/vit/run_a__rep1.onnx"],
            ["run_b", "1", "artifacts/vit/run_b__rep1.onnx"],
        ],
    )

    run_batch.main(
        ["--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1", "--cooldown-seconds", "7"]
    )

    assert sleep_calls == [7.0, 7.0]  # once after each of the 2 real runs


def test_main_never_cools_down_during_dry_run(tmp_path, monkeypatch):
    _isolate_vit_results(tmp_path, monkeypatch)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=None))
    sleep_calls = []
    monkeypatch.setattr(run_batch.time, "sleep", lambda s: sleep_calls.append(s))

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    run_batch.main(
        ["--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1", "--dry-run"]
    )

    assert sleep_calls == []


def test_main_never_cools_down_after_a_skipped_row(tmp_path, monkeypatch):
    _isolate_vit_results(tmp_path, monkeypatch)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=None))
    sleep_calls = []
    monkeypatch.setattr(run_batch.time, "sleep", lambda s: sleep_calls.append(s))

    results_csv = tmp_path / "results_dev1.csv"
    _write_stub_results_csv(results_csv, ["run_a__dev1__rep1"])

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])  # already recorded -> skip

    run_batch.main(["--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1"])

    assert sleep_calls == []


# --------------------------------------------------------------------------
# --with-power: substitutes jetson_powermeter.measure_and_record for the
# plain run_benchmark(), reusing power_logging/cli.py's own branch
# resolution rather than reimplementing it.
# --------------------------------------------------------------------------

_VIT_POWER_OUTPUT = {"avg_latency_ms": 28.07, "throughput_ips": 35.6, "num_inferences_measured": 5}


def test_with_power_writes_combined_energy_row(tmp_path, monkeypatch):
    _isolate_vit_results(tmp_path, monkeypatch)
    _isolate_power_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter, "_power_meter", None)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=dict(_VIT_POWER_OUTPUT)))

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    run_batch.main(
        ["--branch", "vit", "--manifest", str(manifest), "--device-id", "cluster_dev_test", "--with-power"]
    )

    out_path = tmp_path / "results_cluster_dev_test.csv"
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["result_id"] == "run_a__cluster_dev_test__rep1"
    assert row["latency_ms"] == "28.07"  # from build_result_row, reused not reimplemented
    assert row["energy_net_j"] == "4.0"  # from jetson_powermeter's real translation
    assert row["energy_per_inference"] != ""


def test_with_power_reuses_the_same_power_meter_across_batch_rows(tmp_path, monkeypatch):
    """The concrete confirmation the driver's in-process design implies:
    get_power_meter()'s module-level cache is shared across every row of
    the SAME batch, not reset per row -- verified here, not assumed. A
    subprocess-per-row design would defeat this (a fresh interpreter would
    re-measure idle on every single row)."""
    _isolate_vit_results(tmp_path, monkeypatch)
    _isolate_power_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter, "_power_meter", None)

    construct_calls = []

    class TrackedMeter(_FakeMeter):
        def __init__(self, **kwargs):
            construct_calls.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", TrackedMeter)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=dict(_VIT_POWER_OUTPUT)))

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(
        manifest,
        [
            ["run_a", "1", "artifacts/vit/run_a__rep1.onnx"],
            ["run_b", "1", "artifacts/vit/run_b__rep1.onnx"],
        ],
    )

    run_batch.main(
        ["--branch", "vit", "--manifest", str(manifest), "--device-id", "cluster_dev_test", "--with-power"]
    )

    assert len(construct_calls) == 1  # ONE PowerMeter, reused for the 2nd row


def test_with_power_has_no_effect_during_dry_run(tmp_path, monkeypatch):
    _isolate_vit_results(tmp_path, monkeypatch)
    _isolate_power_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter, "_power_meter", None)

    construct_calls = []

    class TrackedMeter(_FakeMeter):
        def __init__(self, **kwargs):
            construct_calls.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", TrackedMeter)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=None))

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    run_batch.main(
        [
            "--branch", "vit", "--manifest", str(manifest), "--device-id", "cluster_dev_test",
            "--with-power", "--dry-run",
        ]
    )

    assert construct_calls == []  # no PowerMeter ever constructed for a dry-run
    assert not (tmp_path / "results_cluster_dev_test.csv").exists()


def test_with_power_and_jetson_cooldown_together(tmp_path, monkeypatch):
    """Not mutually exclusive -- both active at once is the expected real
    usage on the Jetson (energy measurement AND temperature-gated cooldown
    between runs)."""
    _isolate_vit_results(tmp_path, monkeypatch)
    _isolate_power_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter, "_power_meter", None)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=dict(_VIT_POWER_OUTPUT)))
    # Already below any --temp-threshold-c used here -> cooldown returns on
    # its first poll, no real waiting in this test.
    monkeypatch.setattr(jetson_powermeter.orin_power, "get_cpu_temperature", lambda: 30.0)

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(
        manifest,
        [
            ["run_a", "1", "artifacts/vit/run_a__rep1.onnx"],
            ["run_b", "1", "artifacts/vit/run_b__rep1.onnx"],
        ],
    )

    run_batch.main(
        [
            "--branch", "vit", "--manifest", str(manifest), "--device-id", "cluster_dev_test",
            "--with-power", "--jetson-cooldown", "--temp-threshold-c", "55",
        ]
    )

    out_path = tmp_path / "results_cluster_dev_test.csv"
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert {r["result_id"] for r in rows} == {
        "run_a__cluster_dev_test__rep1",
        "run_b__cluster_dev_test__rep1",
    }


# --------------------------------------------------------------------------
# --with-power resumability: "done" requires energy data, not just a
# matching result_id -- a row recorded without it gets re-run to complete
# it, never skipped forever.
# --------------------------------------------------------------------------

def test_with_power_reruns_row_recorded_without_energy_data(tmp_path, monkeypatch):
    _isolate_vit_results(tmp_path, monkeypatch)
    _isolate_power_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter, "_power_meter", None)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)
    fake_run_benchmark = Mock(return_value=dict(_VIT_POWER_OUTPUT))
    monkeypatch.setattr(vit_runner_module, "run_benchmark", fake_run_benchmark)

    results_csv = tmp_path / "results_cluster_dev_test.csv"
    _write_partial_results_csv(
        results_csv,
        [{"result_id": "run_a__cluster_dev_test__rep1", "latency_ms": "28.07", "energy_per_inference": ""}],
    )

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    run_batch.main(
        ["--branch", "vit", "--manifest", str(manifest), "--device-id", "cluster_dev_test", "--with-power"]
    )

    fake_run_benchmark.assert_called_once()  # NOT skipped
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["energy_per_inference"] != ""  # now completed


def test_with_power_skips_row_already_complete_with_energy_data(tmp_path, monkeypatch):
    _isolate_vit_results(tmp_path, monkeypatch)
    _isolate_power_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter, "_power_meter", None)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)
    fake_run_benchmark = Mock(return_value=dict(_VIT_POWER_OUTPUT))
    monkeypatch.setattr(vit_runner_module, "run_benchmark", fake_run_benchmark)

    results_csv = tmp_path / "results_cluster_dev_test.csv"
    _write_partial_results_csv(
        results_csv,
        [{"result_id": "run_a__cluster_dev_test__rep1", "latency_ms": "28.07", "energy_per_inference": "0.8"}],
    )

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    run_batch.main(
        ["--branch", "vit", "--manifest", str(manifest), "--device-id", "cluster_dev_test", "--with-power"]
    )

    fake_run_benchmark.assert_not_called()  # skipped, already complete
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["energy_per_inference"] == "0.8"  # untouched


# --------------------------------------------------------------------------
# --power-impl: generalizes --with-power beyond jetson_powermeter
# --------------------------------------------------------------------------

def test_resolve_power_impl_module_registry_lookup_and_raw_module_path():
    via_registry = run_batch.resolve_power_impl_module("jetson")
    via_raw_path = run_batch.resolve_power_impl_module("benchmarking_ondevice.power_logging.jetson_powermeter")

    assert via_registry.__name__ == "benchmarking_ondevice.power_logging.jetson_powermeter"
    assert via_raw_path is via_registry  # same module object either way


def test_resolve_power_impl_module_raises_if_no_measure_and_record(monkeypatch):
    bad_module = types.ModuleType("tests._bad_power_impl")  # deliberately no measure_and_record
    monkeypatch.setitem(sys.modules, "tests._bad_power_impl", bad_module)

    with pytest.raises(RuntimeError, match="measure_and_record"):
        run_batch.resolve_power_impl_module("tests._bad_power_impl")


def test_power_impl_jetson_explicit_matches_default_with_power_behavior(tmp_path, monkeypatch):
    """Regression: --power-impl jetson (explicit) must behave exactly like
    the existing --with-power default (implicit jetson)."""
    _isolate_vit_results(tmp_path, monkeypatch)
    _isolate_power_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter, "_power_meter", None)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=dict(_VIT_POWER_OUTPUT)))

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    run_batch.main(
        [
            "--branch", "vit", "--manifest", str(manifest), "--device-id", "cluster_dev_test",
            "--with-power", "--power-impl", "jetson",
        ]
    )

    out_path = tmp_path / "results_cluster_dev_test.csv"
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["result_id"] == "run_a__cluster_dev_test__rep1"
    assert row["energy_net_j"] == "4.0"
    assert row["energy_per_inference"] != ""


def test_power_impl_dummy_module_is_invoked_without_touching_jetson_powermeter(tmp_path, monkeypatch):
    """A fake implementation registered under a different name (simulating
    a colleague's own accelpowercape_powermeter.py, following
    dummy_powermeter.py's contract) is invoked correctly through the same
    --power-impl mechanism, with jetson_powermeter never touched anywhere
    in this run."""
    _isolate_vit_results(tmp_path, monkeypatch)
    _isolate_power_results(tmp_path, monkeypatch)

    calls = []

    def fake_measure_and_record(
        run_fn_bound, build_result_row, run_id, device_id, repeat_id, requested_repeats,
        *, perf_per_watt_fn, power_meter_kwargs=None,
    ):
        calls.append(run_id)
        output = run_fn_bound()
        row = build_result_row(run_id, device_id, repeat_id, output, requested_repeats)
        row["energy_per_inference"] = 0.5  # pretend a real measurement happened
        return power_results_io.append_result(device_id, row)

    fake_module = types.ModuleType("tests._fake_power_impl")
    fake_module.measure_and_record = fake_measure_and_record
    monkeypatch.setitem(sys.modules, "tests._fake_power_impl", fake_module)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("jetson_powermeter.get_power_meter must never be called with a different --power-impl")

    monkeypatch.setattr(jetson_powermeter, "get_power_meter", _must_not_be_called)
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock(return_value=dict(_VIT_POWER_OUTPUT)))

    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "artifacts/vit/run_a__rep1.onnx"]])

    run_batch.main(
        [
            "--branch", "vit", "--manifest", str(manifest), "--device-id", "cluster_dev_test",
            "--with-power", "--power-impl", "tests._fake_power_impl",
        ]
    )

    assert calls == ["run_a"]  # the fake implementation was actually invoked
    out_path = tmp_path / "results_cluster_dev_test.csv"
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["energy_per_inference"] == "0.5"


# --------------------------------------------------------------------------
# min_uptime_s: per device, from devices.yaml, absent = 0
# --------------------------------------------------------------------------

def _devices(monkeypatch, entries):
    from benchmarking_ondevice.runner_vit import devices as vit_devices

    monkeypatch.setattr(vit_devices, "load_devices", lambda: entries)


@pytest.mark.parametrize("entry, expected", [
    ({"device_id": "jetson_cpu_50w", "min_uptime_s": 600}, 600.0),
    ({"device_id": "jetson_cpu_50w", "min_uptime_s": 0}, 0.0),
    ({"device_id": "jetson_cpu_50w"}, 0.0),  # absent -> no wait
])
def test_min_uptime_comes_from_the_device_entry(monkeypatch, entry, expected):
    monkeypatch.undo()
    _devices(monkeypatch, [entry])
    assert run_batch.resolve_min_uptime_s("jetson_cpu_50w") == expected


@pytest.mark.parametrize("bad", [-1, "600", True])
def test_a_malformed_min_uptime_raises_instead_of_disabling_the_wait(monkeypatch, bad):
    monkeypatch.undo()
    _devices(monkeypatch, [{"device_id": "d", "min_uptime_s": bad}])
    with pytest.raises(ValueError, match="min_uptime_s"):
        run_batch.resolve_min_uptime_s("d")


def test_there_is_no_command_line_override():
    with pytest.raises(SystemExit):
        run_batch.build_parser().parse_args(["--branch", "vit", "--manifest", "m", "--device-id", "d",
                                             "--min-uptime-seconds", "0"])


def test_main_waits_for_the_devices_min_uptime_before_the_first_real_row(monkeypatch, tmp_path):
    _isolate_vit_results(tmp_path, monkeypatch)
    manifest = tmp_path / "run_manifest_code_vit.csv"
    _write_manifest(manifest, [["run_a", "1", "a.onnx"], ["run_b", "1", "b.onnx"]])
    monkeypatch.setattr(vit_runner_module, "run_benchmark", Mock())
    monkeypatch.setattr(run_batch, "resolve_min_uptime_s", lambda device_id: 600.0)
    waits = []
    monkeypatch.setattr(run_batch, "wait_for_min_uptime", lambda s, dry_run=False: waits.append(s))

    run_batch.main(["--branch", "vit", "--manifest", str(manifest), "--device-id", "dev1"])

    assert waits == [600.0]
