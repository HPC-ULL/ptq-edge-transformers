"""Tests for benchmarking_ondevice/power_logging/.

The real orin_power.PowerMeter (needs real INA3221 sysfs paths) is always
replaced with a fake that records construction calls and returns a
power_results dict shaped exactly like the real estimate_fn_power's output
(elapsed/total/output keys) -- see conftest-less _FakeMeter below. Results
paths are always isolated to tmp_path, same pattern as
tests/test_runner_vit.py and tests/test_runner_llm.py.
"""

from __future__ import annotations

import csv
from unittest.mock import Mock

import pytest

from benchmarking_ondevice.power_logging import cli as power_cli
from benchmarking_ondevice.power_logging import jetson_powermeter
from benchmarking_ondevice.power_logging import results_io as power_results_io
from benchmarking_ondevice.runner_llm import results_io as llm_results_io
from benchmarking_ondevice.runner_vit import results_io as vit_results_io


def _isolate_results(tmp_path, monkeypatch):
    monkeypatch.setattr(
        power_results_io, "results_csv_path", lambda device_id: tmp_path / f"results_{device_id}.csv"
    )


def _fake_power_results(output, energy_total_j=10.0, energy_net_j=4.0, elapsed=2.0):
    return {
        "elapsed": elapsed,
        "total": {"energy_total_j": energy_total_j, "energy_net_j": energy_net_j},
        "sampling": {"n_samples": 60, "window_s": 2.0, "dt_max_s": 0.04, "configured_hz": 30.0},
        "output": output,
    }


class _FakeMeter:
    """Stand-in for orin_power.PowerMeter: no real sensor, no real idle
    wait, returns a fixed energy reading and simply calls through to
    whatever fn() estimate_fn_power was given (mirroring the real one's
    'output' contract)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def estimate_fn_power(self, fn):
        return _fake_power_results(fn())


@pytest.fixture(autouse=True)
def _reset_power_meter_singleton(monkeypatch):
    # jetson_powermeter's module-level cache persists across tests
    # otherwise -- every test starts from an unconstructed state.
    monkeypatch.setattr(jetson_powermeter, "_power_meter", None)


_VIT_OUTPUT = {"avg_latency_ms": 28.07, "throughput_ips": 35.6, "num_inferences_measured": 5}
_LLM_OUTPUT = {
    "throughput_prefill_tps": 43.57, "throughput_decode_tps": 7.89, "num_inferences_measured": 3,
    "n_prompt": 128, "n_gen": 32,
}


# --------------------------------------------------------------------------
# PowerMeter construction: once, reused
# --------------------------------------------------------------------------

def test_power_meter_constructed_once_and_reused_across_consecutive_calls(monkeypatch, tmp_path):
    _isolate_results(tmp_path, monkeypatch)
    construct_calls = []

    class TrackedMeter(_FakeMeter):
        def __init__(self, **kwargs):
            construct_calls.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", TrackedMeter)

    jetson_powermeter.measure_and_record(
        lambda: dict(_VIT_OUTPUT), vit_results_io.build_result_row, "run_a", "cluster_dev_test", 1, 5,
        perf_per_watt_fn=power_cli._vit_perf_per_watt,
    )
    jetson_powermeter.measure_and_record(
        lambda: dict(_VIT_OUTPUT), vit_results_io.build_result_row, "run_b", "cluster_dev_test", 2, 5,
        perf_per_watt_fn=power_cli._vit_perf_per_watt,
    )

    assert len(construct_calls) == 1


def test_get_power_meter_returns_the_same_instance(monkeypatch):
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)

    first = jetson_powermeter.get_power_meter()
    second = jetson_powermeter.get_power_meter()

    assert first is second


# --------------------------------------------------------------------------
# Combined row: performance + energy in one row
# --------------------------------------------------------------------------

def test_combined_row_merges_vit_performance_and_energy_fields(monkeypatch, tmp_path):
    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)

    result = jetson_powermeter.measure_and_record(
        lambda: dict(_VIT_OUTPUT), vit_results_io.build_result_row, "vit_run", "cluster_dev_test", 3, 5,
        perf_per_watt_fn=power_cli._vit_perf_per_watt,
    )
    row = result["row"]

    assert row["result_id"] == "vit_run__cluster_dev_test__rep3"
    assert row["latency_ms"] == 28.07
    assert row["throughput"] == 35.6
    assert row["energy_raw_j"] == 10.0
    assert row["energy_net_j"] == 4.0
    assert row["avg_power_w_raw"] == pytest.approx(10.0 / 2.0)
    assert row["avg_power_w_net"] == pytest.approx(4.0 / 2.0)
    # LLM-only fields stay blank on a ViT row
    assert row["throughput_prefill_tps"] == ""


def test_combined_row_merges_llm_performance_and_energy_fields(monkeypatch, tmp_path):
    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)

    result = jetson_powermeter.measure_and_record(
        lambda: dict(_LLM_OUTPUT), llm_results_io.build_result_row, "llm_run", "cluster_dev_test", 2, 3,
        perf_per_watt_fn=power_cli._llm_perf_per_watt,
    )
    row = result["row"]

    assert row["throughput_prefill_tps"] == 43.57
    assert row["throughput_decode_tps"] == 7.89
    assert row["energy_raw_j"] == 10.0
    assert row["energy_net_j"] == 4.0
    # ViT-only fields stay blank on an LLM row
    assert row["latency_ms"] == ""


def test_row_is_actually_written_to_the_shared_results_csv(monkeypatch, tmp_path):
    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)

    jetson_powermeter.measure_and_record(
        lambda: dict(_VIT_OUTPUT), vit_results_io.build_result_row, "vit_run", "cluster_dev_test", 3, 5,
        perf_per_watt_fn=power_cli._vit_perf_per_watt,
    )

    out_path = tmp_path / "results_cluster_dev_test.csv"
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["result_id"] == "vit_run__cluster_dev_test__rep3"


# --------------------------------------------------------------------------
# energy_per_inference: from the REAL parsed output, not hardcoded
# --------------------------------------------------------------------------

def test_energy_per_inference_uses_real_num_inferences_measured_not_requested_repeats(monkeypatch, tmp_path):
    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)

    # num_inferences_measured (7) deliberately differs from requested_repeats
    # (1000) -- energy_per_inference must divide by the former.
    output = {"avg_latency_ms": 1.0, "throughput_ips": 1.0, "num_inferences_measured": 7}

    result = jetson_powermeter.measure_and_record(
        lambda: dict(output), vit_results_io.build_result_row, "run_x", "cluster_dev_test", 1, 1000,
        perf_per_watt_fn=power_cli._vit_perf_per_watt,
    )

    assert result["row"]["energy_per_inference"] == pytest.approx(4.0 / 7)


# --------------------------------------------------------------------------
# perf_per_watt: branch-specific FORMULA, computed from real energy/tokens
# --------------------------------------------------------------------------

def test_perf_per_watt_vit_is_inferences_over_net_energy(monkeypatch, tmp_path):
    """N / energy_net_j -- NOT throughput / avg_power_w_net, whose throughput
    excludes the start-up/load/warm-up the power window includes."""
    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)

    result = jetson_powermeter.measure_and_record(
        lambda: dict(_VIT_OUTPUT), vit_results_io.build_result_row, "vit_run", "cluster_dev_test", 1, 5,
        perf_per_watt_fn=power_cli._vit_perf_per_watt,
    )

    # N = 5 inferences, energy_net_j = 4.0 J -> 1.25 inferences/J = 1 / energy_per_inference
    row = result["row"]
    assert row["perf_per_watt"] == pytest.approx(5 / 4.0)
    assert row["perf_per_watt"] == pytest.approx(1 / row["energy_per_inference"])
    assert row["perf_per_watt"] != pytest.approx(35.6 / 2.0)  # the old throughput / avg_power_w_net


def test_perf_per_watt_llm_uses_total_tokens_over_energy_per_inference(monkeypatch, tmp_path):
    """The formula: (n_prompt + n_gen) / energy_per_inference -- NOT any
    combination of throughput_prefill_tps/throughput_decode_tps, which are
    measured over sub-phase-only windows shorter than the whole-run energy
    figure they'd otherwise be divided by."""
    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)

    result = jetson_powermeter.measure_and_record(
        lambda: dict(_LLM_OUTPUT), llm_results_io.build_result_row, "llm_run", "cluster_dev_test", 1, 3,
        perf_per_watt_fn=power_cli._llm_perf_per_watt,
    )

    # n_prompt=128, n_gen=32, N=3, energy_net_j=4.0 -> energy_per_inference = 4/3,
    # perf_per_watt = 160 / (4/3) = 120.0
    expected = (_LLM_OUTPUT["n_prompt"] + _LLM_OUTPUT["n_gen"]) / (4.0 / _LLM_OUTPUT["num_inferences_measured"])
    assert result["row"]["perf_per_watt"] == pytest.approx(expected) == pytest.approx(120.0)

    # Explicitly confirm it is NOT any throughput-based combination --
    # neither decode-only, prefill-only, nor their sum/average over
    # avg_power_w_net.
    decode_over_power = _LLM_OUTPUT["throughput_decode_tps"] / 2.0  # avg_power_w_net = 4.0/2.0
    prefill_over_power = _LLM_OUTPUT["throughput_prefill_tps"] / 2.0
    combined_over_power = (_LLM_OUTPUT["throughput_decode_tps"] + _LLM_OUTPUT["throughput_prefill_tps"]) / 2.0
    assert result["row"]["perf_per_watt"] != pytest.approx(decode_over_power)
    assert result["row"]["perf_per_watt"] != pytest.approx(prefill_over_power)
    assert result["row"]["perf_per_watt"] != pytest.approx(combined_over_power)


def test_perf_per_watt_llm_ignores_throughput_fields_entirely(monkeypatch, tmp_path):
    """Even if the throughput fields were wildly different, perf_per_watt
    must not move -- confirms _llm_perf_per_watt never reads them at all,
    only the token counts, energy_fields['energy_net_j'] and N."""
    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)

    weird_output = dict(_LLM_OUTPUT, throughput_prefill_tps=999999.0, throughput_decode_tps=0.0001)

    result = jetson_powermeter.measure_and_record(
        lambda: dict(weird_output), llm_results_io.build_result_row, "llm_run", "cluster_dev_test", 1, 3,
        perf_per_watt_fn=power_cli._llm_perf_per_watt,
    )

    expected = (_LLM_OUTPUT["n_prompt"] + _LLM_OUTPUT["n_gen"]) * _LLM_OUTPUT["num_inferences_measured"] / 4.0
    assert result["row"]["perf_per_watt"] == pytest.approx(expected)


@pytest.mark.parametrize("n", [1, 3, 5, 10])
def test_perf_per_watt_llm_is_tokens_per_repetition_over_energy_per_inference(monkeypatch, tmp_path, n):
    """energy_net_j covers all N llama-bench repetitions: perf_per_watt must be
    (128 + 32) / (energy_net_j / N) = 160 * N / energy_net_j -- NOT
    160 / energy_net_j, which was tokens/J divided by N (the bug fixed here)."""
    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)
    output = dict(_LLM_OUTPUT, num_inferences_measured=n)

    result = jetson_powermeter.measure_and_record(
        lambda: dict(output), llm_results_io.build_result_row, "llm_run", "cluster_dev_test", 1, n,
        perf_per_watt_fn=power_cli._llm_perf_per_watt,
    )
    row = result["row"]

    assert row["energy_per_inference"] == pytest.approx(4.0 / n)
    assert row["perf_per_watt"] == pytest.approx(160 * n / 4.0)
    assert row["perf_per_watt"] == pytest.approx(160 / row["energy_per_inference"])


def test_llm_tokens_per_repetition_come_from_the_runner_constants():
    from benchmarking_ondevice.power_logging.perf_per_watt import TOKENS_PER_REPETITION
    from benchmarking_ondevice.runner_llm.runner import GEN_TOKENS, PROMPT_TOKENS

    assert (PROMPT_TOKENS, GEN_TOKENS) == (128, 32)
    assert TOKENS_PER_REPETITION == 160


def test_perf_per_watt_llm_refuses_a_run_with_other_token_counts():
    with pytest.raises(ValueError, match="-p 128 -n 32"):
        power_cli._llm_perf_per_watt({}, dict(_LLM_OUTPUT, n_prompt=512), {"energy_net_j": 4.0})


def test_perf_per_watt_vit_blank_when_net_energy_is_zero(monkeypatch, tmp_path):
    """energy_net_j == 0 (net energy rounded to zero, a real if unusual
    outcome for a very fast/light benchmark) must not raise a
    ZeroDivisionError -- perf_per_watt stays unset."""
    _isolate_results(tmp_path, monkeypatch)

    class ZeroNetMeter(_FakeMeter):
        def estimate_fn_power(self, fn):
            return _fake_power_results(fn(), energy_total_j=10.0, energy_net_j=0.0, elapsed=2.0)

    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", ZeroNetMeter)

    result = jetson_powermeter.measure_and_record(
        lambda: dict(_VIT_OUTPUT), vit_results_io.build_result_row, "vit_run", "cluster_dev_test", 1, 5,
        perf_per_watt_fn=power_cli._vit_perf_per_watt,
    )

    assert result["row"]["perf_per_watt"] == ""


def test_perf_per_watt_llm_blank_when_energy_net_j_is_zero(monkeypatch, tmp_path):
    """LLM's formula divides by energy_net_j directly (not avg_power_w_net)
    -- confirm the zero-denominator guard checks the right quantity."""
    _isolate_results(tmp_path, monkeypatch)

    class ZeroNetMeter(_FakeMeter):
        def estimate_fn_power(self, fn):
            return _fake_power_results(fn(), energy_total_j=10.0, energy_net_j=0.0, elapsed=2.0)

    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", ZeroNetMeter)

    result = jetson_powermeter.measure_and_record(
        lambda: dict(_LLM_OUTPUT), llm_results_io.build_result_row, "llm_run", "cluster_dev_test", 1, 3,
        perf_per_watt_fn=power_cli._llm_perf_per_watt,
    )

    assert result["row"]["perf_per_watt"] == ""


# --------------------------------------------------------------------------
# Power-mode pre-flight check (_verify_power_mode) -- self-contained in
# jetson_powermeter.py: it resolves the expected power_mode_w from the
# device_id itself (via _resolve_power_mode_w -> configs/devices.yaml).
# These tests monkeypatch _resolve_power_mode_w to force a declared mode;
# the rest of the suite uses cluster_dev_test, whose real devices.yaml
# entry has power_mode_w: null, so the check is a silent no-op there.
# --------------------------------------------------------------------------

class _RecordingMeter(_FakeMeter):
    """Records whether estimate_fn_power was actually reached."""

    measured = False

    def estimate_fn_power(self, fn):
        type(self).measured = True
        return super().estimate_fn_power(fn)


def test_resolve_power_mode_w_reads_devices_yaml_and_fails_loud_on_unknown():
    # cluster_dev_test is real, with power_mode_w: null
    assert jetson_powermeter._resolve_power_mode_w("cluster_dev_test") is None
    with pytest.raises(ValueError, match="Unknown device_id"):
        jetson_powermeter._resolve_power_mode_w("no_such_device")


def test_power_mode_query_failure_warns_and_continues(monkeypatch, tmp_path, capsys):
    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)
    monkeypatch.setattr(jetson_powermeter, "_resolve_power_mode_w", lambda device_id: 50)
    monkeypatch.setattr(jetson_powermeter.orin_power, "get_power_mode", lambda: None)

    result = jetson_powermeter.measure_and_record(
        lambda: dict(_VIT_OUTPUT), vit_results_io.build_result_row, "vit_run", "cluster_dev_test", 1, 5,
        perf_per_watt_fn=power_cli._vit_perf_per_watt,
    )

    assert "could not verify the active power mode" in capsys.readouterr().out
    assert result["row"]["result_id"] == "vit_run__cluster_dev_test__rep1"  # measurement still ran


def test_power_mode_match_is_silent_and_continues(monkeypatch, tmp_path, capsys):
    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)
    monkeypatch.setattr(jetson_powermeter, "_resolve_power_mode_w", lambda device_id: 50)
    monkeypatch.setattr(jetson_powermeter.orin_power, "get_power_mode", lambda: "50W")

    result = jetson_powermeter.measure_and_record(
        lambda: dict(_VIT_OUTPUT), vit_results_io.build_result_row, "vit_run", "cluster_dev_test", 1, 5,
        perf_per_watt_fn=power_cli._vit_perf_per_watt,
    )

    assert "power mode" not in capsys.readouterr().out  # no message at all
    assert result["row"]["result_id"] == "vit_run__cluster_dev_test__rep1"


@pytest.mark.parametrize("actual_mode", ["15W", "MAXN"])
def test_power_mode_mismatch_raises_and_never_measures(monkeypatch, tmp_path, actual_mode):
    _isolate_results(tmp_path, monkeypatch)
    _RecordingMeter.measured = False
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _RecordingMeter)
    monkeypatch.setattr(jetson_powermeter, "_resolve_power_mode_w", lambda device_id: 50)
    monkeypatch.setattr(jetson_powermeter.orin_power, "get_power_mode", lambda: actual_mode)

    def run_fn_must_not_run():
        raise AssertionError("run_fn_bound must never be called on a power-mode mismatch")

    with pytest.raises(jetson_powermeter.PowerModeMismatchError):
        jetson_powermeter.measure_and_record(
            run_fn_must_not_run, vit_results_io.build_result_row, "vit_run", "cluster_dev_test", 1, 5,
            perf_per_watt_fn=power_cli._vit_perf_per_watt,
        )

    assert _RecordingMeter.measured is False  # estimate_fn_power never reached
    assert not (tmp_path / "results_cluster_dev_test.csv").exists()  # nothing written


def test_power_mode_check_skipped_entirely_when_no_expected_mode(monkeypatch, tmp_path):
    """Device declares no power_mode_w -> get_power_mode() must not even be
    queried."""
    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)
    monkeypatch.setattr(jetson_powermeter, "_resolve_power_mode_w", lambda device_id: None)
    queried = []
    monkeypatch.setattr(jetson_powermeter.orin_power, "get_power_mode", lambda: queried.append(True))

    jetson_powermeter.measure_and_record(
        lambda: dict(_VIT_OUTPUT), vit_results_io.build_result_row, "vit_run", "cluster_dev_test", 1, 5,
        perf_per_watt_fn=power_cli._vit_perf_per_watt,
    )

    assert queried == []


# --------------------------------------------------------------------------
# CLI: write_result=False propagation, --dry-run bypass
# --------------------------------------------------------------------------

def test_cli_real_run_passes_write_result_false_to_underlying_runner(monkeypatch, tmp_path):
    import benchmarking_ondevice.runner_vit.runner as vit_runner_module

    _isolate_results(tmp_path, monkeypatch)
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", _FakeMeter)
    fake_run_benchmark = Mock(return_value=dict(_VIT_OUTPUT))
    monkeypatch.setattr(vit_runner_module, "run_benchmark", fake_run_benchmark)

    # cluster_dev_test is a real configs/devices.yaml entry with
    # power_mode_w: null -> _verify_power_mode is skipped, no nvpmodel query.
    power_cli.main(
        ["--branch", "vit", "--run-id", "run_x", "--artifact", "model.onnx",
         "--device-id", "cluster_dev_test", "--repeat-id", "1"]
    )

    fake_run_benchmark.assert_called_once()
    _, kwargs = fake_run_benchmark.call_args
    assert kwargs["write_result"] is False


def test_dry_run_never_constructs_power_meter_or_writes_results(monkeypatch, tmp_path):
    import benchmarking_ondevice.runner_vit.runner as vit_runner_module

    _isolate_results(tmp_path, monkeypatch)
    construct_calls = []

    class TrackedMeter(_FakeMeter):
        def __init__(self, **kwargs):
            construct_calls.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", TrackedMeter)
    fake_run_benchmark = Mock(return_value=None)  # the real dry-run path returns None
    monkeypatch.setattr(vit_runner_module, "run_benchmark", fake_run_benchmark)

    power_cli.main(
        ["--branch", "vit", "--run-id", "run_x", "--artifact", "model.onnx",
         "--device-id", "cluster_dev_test", "--repeat-id", "1", "--dry-run"]
    )

    assert construct_calls == []
    fake_run_benchmark.assert_called_once()
    _, kwargs = fake_run_benchmark.call_args
    assert kwargs["dry_run"] is True
    assert not (tmp_path / "results_cluster_dev_test.csv").exists()


def test_verify_power_mode_warns_and_continues_when_nvpmodel_is_not_executable(monkeypatch, capsys):
    """End to end through the REAL get_power_mode: a declared mode + an
    nvpmodel that raises PermissionError must take the documented
    'cannot verify -> warn and continue' path, not crash the run."""
    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied", "nvpmodel")

    monkeypatch.setattr(jetson_powermeter, "_resolve_power_mode_w", lambda device_id: 50)
    monkeypatch.setattr(jetson_powermeter.orin_power.subprocess, "check_output", denied)

    jetson_powermeter._verify_power_mode("jetson_gpu_50w")  # must not raise

    assert "could not verify the active power mode" in capsys.readouterr().out


@pytest.mark.parametrize("n, energy", [(1, 0.5), (1000, 50.0), (7, 3.0)])
def test_vit_perf_per_watt_formula(n, energy):
    from benchmarking_ondevice.power_logging.perf_per_watt import vit_perf_per_watt

    assert vit_perf_per_watt(energy, n) == pytest.approx(n / energy)
    assert vit_perf_per_watt(0.0, n) is None and vit_perf_per_watt(energy, 0) is None


def test_vit_perf_per_watt_ignores_throughput_and_power(monkeypatch):
    out = power_cli._vit_perf_per_watt({"throughput": 999.0}, {"num_inferences_measured": 10},
                                       {"energy_net_j": 4.0, "avg_power_w_net": 0.001})
    assert out == pytest.approx(10 / 4.0)
