"""Tests for the power-sampling characterization (sampling_characterization,
sampling_report, sampling_hw_info) and its hook in jetson_powermeter.

The results/characterization/ directory is isolated for every test by
tests/conftest.py.
"""

from __future__ import annotations

import csv
import json

import pytest

from benchmarking_ondevice.power_logging import cli as power_cli
from benchmarking_ondevice.power_logging import jetson_powermeter
from benchmarking_ondevice.power_logging import paths as power_paths
from benchmarking_ondevice.power_logging import results_io as power_results_io
from benchmarking_ondevice.power_logging import sampling_characterization as sc
from benchmarking_ondevice.power_logging import sampling_hw_info, sampling_report
from benchmarking_ondevice.runner_vit import results_io as vit_results_io


def _stub_devices(monkeypatch):
    def entry(device_id):
        return {
            "device_id": device_id,
            "backend": "cuda" if "gpu" in device_id else "cpu",
            "power_mode_w": 30,
        }

    monkeypatch.setattr(sc, "resolve_device_entry", entry)


# --------------------------------------------------------------------------
# The pre-registered criterion
# --------------------------------------------------------------------------

def test_criterion_is_pre_registered():
    """Fixed on 2026-09-24, before any instrumented data existed. Changing
    these numbers after seeing results defeats the point -- if this test
    has to be edited, that edit should be a deliberate, visible decision."""
    assert sc.MIN_SAMPLES_PER_WINDOW == 30
    assert sc.WIDE_MARGIN_FACTOR == 3


def test_threshold_boundary_is_inclusive():
    assert sc.meets_threshold(29) is False
    assert sc.meets_threshold(30) is True


def test_an_estimate_is_only_accepted_with_the_wide_margin():
    assert sc.classify_estimate(89.9) == sc.REMEASURE
    assert sc.classify_estimate(90) == sc.ESTIMATED_OK  # 3 x 30


def test_effective_hz():
    assert sc.effective_hz(60, 2.0) == pytest.approx(30.0)
    assert sc.effective_hz(0, 0.0) is None
    assert sc.effective_hz(5, 0) is None


# --------------------------------------------------------------------------
# Per-window record
# --------------------------------------------------------------------------

_SAMPLING = {"n_samples": 45, "window_s": 1.6, "dt_max_s": 0.05, "configured_hz": 30.0}
_ROW = {"result_id": "run_a__dev__rep1", "run_id": "run_a", "repeat_id": 1}


def test_build_sampling_record_fields():
    record = sc.build_sampling_record(
        _ROW, "dev", {"backend": "cuda", "power_mode_w": 30}, _SAMPLING, 2.5, measured_at="2026-01-01T00:00:00+00:00"
    )

    assert record["result_id"] == "run_a__dev__rep1"  # join key back to results/raw
    assert record["backend"] == "cuda" and record["power_mode_w"] == 30
    assert record["n_samples"] == 45
    assert record["effective_hz"] == pytest.approx(45 / 1.6)
    assert record["elapsed_s"] == 2.5
    assert record["min_samples_threshold"] == 30
    assert record["meets_threshold"] is True
    assert set(record) == set(sc.SAMPLING_COLUMNS)


def test_build_sampling_record_below_threshold_and_no_power_mode():
    record = sc.build_sampling_record(
        _ROW, "dev", {"backend": "cpu", "power_mode_w": None}, {**_SAMPLING, "n_samples": 12}, 1.0
    )

    assert record["meets_threshold"] is False
    assert record["power_mode_w"] == ""


def test_append_sampling_record_replaces_the_same_result_id_and_keeps_others():
    first = sc.build_sampling_record(_ROW, "dev", {"backend": "cpu"}, _SAMPLING, 2.0)
    other = sc.build_sampling_record(
        {**_ROW, "result_id": "run_b__dev__rep1", "run_id": "run_b"}, "dev", {"backend": "cpu"}, _SAMPLING, 2.0
    )
    redone = sc.build_sampling_record(_ROW, "dev", {"backend": "cpu"}, {**_SAMPLING, "n_samples": 99}, 2.0)

    sc.append_sampling_record("dev", first)
    sc.append_sampling_record("dev", other)
    sc.append_sampling_record("dev", redone)  # forced re-measurement of run_a

    records = sc.read_sampling_records("dev")
    assert sorted(r["result_id"] for r in records) == ["run_a__dev__rep1", "run_b__dev__rep1"]
    assert next(r for r in records if r["run_id"] == "run_a")["n_samples"] == "99"


# --------------------------------------------------------------------------
# Retroactive estimate (lower bound: excludes process start-up / model load)
# --------------------------------------------------------------------------

def test_estimate_window_vit_is_repeats_times_latency():
    assert sc.estimate_window_s({"num_inferences_measured": "1000", "latency_ms": "2.5"}) == pytest.approx(2.5)


def test_estimate_window_llm_is_repeats_times_prefill_plus_decode_time():
    # 128 prompt tokens at 64 tok/s = 2 s; 32 generated at 8 tok/s = 4 s; 5 repeats
    row = {"num_inferences_measured": "5", "throughput_prefill_tps": "64", "throughput_decode_tps": "8"}
    assert sc.estimate_window_s(row) == pytest.approx(30.0)


def test_estimate_window_is_none_without_usable_timing():
    assert sc.estimate_window_s({"num_inferences_measured": "5"}) is None
    assert sc.estimate_window_s({"latency_ms": "2.5"}) is None
    assert sc.estimate_window_s({"num_inferences_measured": "", "latency_ms": "2.5"}) is None


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

_VIT_ROW = {"num_inferences_measured": "1000", "latency_ms": "5"}  # 5 s window


def test_a_measured_record_always_wins_over_any_estimate():
    record = {"n_samples": "12", "window_s": "0.4"}

    verdict = sc.classify_row(_VIT_ROW, record, device_effective_hz=25.0)

    assert verdict["source"] == "measured" and verdict["status"] == sc.MEASURED_BELOW


def test_measured_ok_at_the_threshold():
    assert sc.classify_row(_VIT_ROW, {"n_samples": "30", "window_s": "1.0"}, None)["status"] == sc.MEASURED_OK


def test_without_a_measured_rate_nothing_is_estimated_with_the_configured_30hz():
    verdict = sc.classify_row(_VIT_ROW, None, device_effective_hz=None)

    assert verdict["status"] == sc.NO_EFFECTIVE_HZ
    assert verdict["n_samples"] is None


def test_estimated_ok_only_with_wide_margin_otherwise_remeasure():
    assert sc.classify_row(_VIT_ROW, None, 25.0)["status"] == sc.ESTIMATED_OK  # 5 s x 25 Hz = 125 >= 90
    short = {"num_inferences_measured": "1000", "latency_ms": "2.5"}  # 2.5 s x 25 Hz = 62.5: above 30, below 90
    verdict = sc.classify_row(short, None, 25.0)
    assert verdict["status"] == sc.REMEASURE
    assert verdict["n_samples"] == pytest.approx(62.5)


def test_row_without_timing_is_no_estimate():
    assert sc.classify_row({"num_inferences_measured": "5"}, None, 25.0)["status"] == sc.NO_ESTIMATE


def test_summarize_group_statistics():
    records = [
        {"n_samples": "100", "effective_hz": "25", "dt_max_s": "0.05"},
        {"n_samples": "50", "effective_hz": "28", "dt_max_s": "0.08"},
        {"n_samples": "20", "effective_hz": "30", "dt_max_s": "0.06"},
    ]

    group = sc.summarize_group(records)

    assert group["n_windows"] == 3
    assert group["effective_hz_median"] == 28 and group["effective_hz_min"] == 25
    assert group["n_samples_min"] == 20 and group["n_samples_median"] == 50
    assert group["dt_max_s_max"] == pytest.approx(0.08)
    assert group["n_windows_below_threshold"] == 1  # only the 20-sample window


# --------------------------------------------------------------------------
# The report (reads results/raw + results/characterization, writes both files)
# --------------------------------------------------------------------------

def _write_results(tmp_path, device_id, rows):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    columns = ["result_id", "run_id", "energy_per_inference", "num_inferences_measured", "latency_ms",
               "throughput_prefill_tps", "throughput_decode_tps"]
    with open(raw / f"results_{device_id}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _vit(run_id, device_id, latency_ms, energy="0.5"):
    return {"result_id": f"{run_id}__{device_id}__rep1", "run_id": run_id, "energy_per_inference": energy,
            "num_inferences_measured": "1000", "latency_ms": latency_ms}


def _instrument(device_id, run_id, n_samples, window_s):
    record = sc.build_sampling_record(
        {"result_id": f"{run_id}__{device_id}__rep1", "run_id": run_id, "repeat_id": 1},
        device_id, {"backend": "cuda", "power_mode_w": 30},
        {"n_samples": n_samples, "window_s": window_s, "dt_max_s": 0.05, "configured_hz": 30.0}, window_s,
    )
    sc.append_sampling_record(device_id, record)


def test_report_classifies_measured_estimated_and_remeasure_rows(monkeypatch, tmp_path):
    _stub_devices(monkeypatch)
    monkeypatch.setattr(power_paths, "RESULTS_RAW_DIR", tmp_path / "raw")
    device = "jetson_gpu_30w"
    _write_results(tmp_path, device, [
        _vit("measured_ok", device, "5"),
        _vit("measured_low", device, "1"),
        _vit("old_long", device, "5"),    # no record: 5 s x 25 Hz = 125 -> accepted estimate
        _vit("old_short", device, "2.5"),  # no record: 62.5 -> too close to the threshold -> re-measure
        {"result_id": "no_energy__x__rep1", "run_id": "no_energy", "energy_per_inference": ""},  # not power-measured
    ])
    _instrument(device, "measured_ok", n_samples=125, window_s=5.0)   # 25 Hz
    _instrument(device, "measured_low", n_samples=20, window_s=0.8)   # 25 Hz

    classification, summary, skipped = sampling_report.build_report([device])

    status = {c["run_id"]: c["status"] for c in classification}
    assert status == {
        "measured_ok": sc.MEASURED_OK, "measured_low": sc.MEASURED_BELOW,
        "old_long": sc.ESTIMATED_OK, "old_short": sc.REMEASURE,
    }
    assert skipped == []
    (row,) = summary
    assert row["backend"] == "cuda" and row["power_mode_w"] == 30
    assert row["n_energy_rows"] == 4 and row["n_measured"] == 2 and row["n_estimated"] == 2
    assert row["effective_hz_median"] == pytest.approx(25.0)
    assert row["n_samples_min"] == 20 and row["n_windows_below_threshold"] == 1
    assert row["est_n_samples_min"] == pytest.approx(62.5)
    assert row["n_estimated_ok"] == 1 and row["n_remeasure"] == 1
    assert row["min_samples_threshold"] == 30 and row["wide_margin_factor"] == 3


def test_report_main_writes_both_files_and_lists_rows_to_remeasure(monkeypatch, tmp_path, capsys):
    _stub_devices(monkeypatch)
    monkeypatch.setattr(power_paths, "RESULTS_RAW_DIR", tmp_path / "raw")
    device = "jetson_gpu_15w"
    _write_results(tmp_path, device, [_vit("a", device, "5"), _vit("b", device, "2.5")])
    _instrument(device, "a", n_samples=125, window_s=5.0)

    assert sampling_report.main([]) == 0

    with open(power_paths.sampling_summary_csv_path(), newline="") as f:
        summary = list(csv.DictReader(f))
    with open(power_paths.sampling_classification_csv_path(), newline="") as f:
        classification = list(csv.DictReader(f))
    assert list(summary[0]) == sampling_report.SUMMARY_COLUMNS
    assert [c["status"] for c in classification] == [sc.MEASURED_OK, sc.REMEASURE]
    out = capsys.readouterr().out
    assert "1 row(s) to remeasure" in out and "b__jetson_gpu_15w__rep1" in out


def test_report_without_any_instrumented_window_never_falls_back_to_the_configured_rate(monkeypatch, tmp_path):
    _stub_devices(monkeypatch)
    monkeypatch.setattr(power_paths, "RESULTS_RAW_DIR", tmp_path / "raw")
    _write_results(tmp_path, "jetson_cpu_15w", [_vit("a", "jetson_cpu_15w", "5")])

    classification, summary, _ = sampling_report.build_report(["jetson_cpu_15w"])

    assert classification[0]["status"] == sc.NO_EFFECTIVE_HZ
    assert summary[0]["n_no_effective_hz"] == 1 and summary[0]["effective_hz_median"] == ""


def test_report_skips_a_device_missing_from_devices_yaml(monkeypatch, tmp_path):
    def unknown(device_id):
        raise ValueError("Unknown device_id")

    monkeypatch.setattr(sc, "resolve_device_entry", unknown)
    monkeypatch.setattr(power_paths, "RESULTS_RAW_DIR", tmp_path / "raw")
    _write_results(tmp_path, "ghost", [_vit("a", "ghost", "5")])

    classification, summary, skipped = sampling_report.build_report(["ghost"])

    assert classification == [] and summary == [] and skipped == ["ghost"]


# --------------------------------------------------------------------------
# Hardware info (fake sysfs tree)
# --------------------------------------------------------------------------

def _fake_sysfs(tmp_path, *, with_samples=True):
    drivers = tmp_path / "drivers"
    (drivers / "1-0040").mkdir(parents=True)
    (drivers / "1-0041").mkdir()
    (drivers / "module").mkdir()  # not a bound chip: must be ignored
    hwmon = tmp_path / "hwmon" / "hwmon3"
    hwmon.mkdir(parents=True)
    (hwmon / "update_interval").write_text("35\n")
    if with_samples:
        (hwmon / "samples").write_text("1\n")
    for i, label in (("1", "VDD_GPU_SOC"), ("2", "VDD_CPU_CV"), ("3", "VIN_SYS_5V0")):
        (hwmon / f"in{i}_label").write_text(label + "\n")
        (hwmon / f"in{i}_enable").write_text("1\n")
    (hwmon / "shunt1_resistor").write_text("5000\n")
    return tmp_path / "hwmon", drivers


def test_collect_hw_info_reads_raw_values(tmp_path):
    hwmon_base, drivers = _fake_sysfs(tmp_path)

    info = sampling_hw_info.collect_hw_info(str(hwmon_base), str(drivers))

    assert info["ina3221_devices"] == ["1-0040", "1-0041"]
    assert info["hwmon_entries"] == ["hwmon3"] and info["hwmon_dir_used"].endswith("hwmon3")
    assert info["update_interval_ms"] == 35 and info["samples"] == "1"
    assert info["chip_channels"]["1"] == {"label": "VDD_GPU_SOC", "enable": "1", "shunt_resistor": "5000"}
    assert info["chip_channels"]["3"]["label"] == "VIN_SYS_5V0"
    assert info["chip_channels"]["2"]["shunt_resistor"] is None  # absent file -> None, never guessed
    assert info["channels_read_by_powermeter"] == ["1", "2"]  # the chip has 3, PowerMeter reads 2
    assert "update_interval" in info["attributes_present"]


def test_collect_hw_info_records_missing_attributes_as_none(tmp_path):
    hwmon_base, drivers = _fake_sysfs(tmp_path, with_samples=False)

    assert sampling_hw_info.collect_hw_info(str(hwmon_base), str(drivers))["samples"] is None


def test_collect_hw_info_without_sysfs_does_not_crash(tmp_path):
    info = sampling_hw_info.collect_hw_info(str(tmp_path / "nope"), str(tmp_path / "nope2"))

    assert info["hwmon_dir_used"] is None and info["update_interval_ms"] is None and info["ina3221_devices"] is None


def test_hw_info_main_writes_the_json_and_reports_the_default_rate(monkeypatch):
    monkeypatch.setattr(sampling_hw_info, "collect_hw_info", lambda: {"update_interval_ms": 35})
    monkeypatch.setattr(sampling_hw_info.orin_power, "get_power_mode", lambda: "30W")

    assert sampling_hw_info.main(["--device-id", "dev_x", "--probe-seconds", "0"]) == 0

    data = json.loads(power_paths.sampling_hw_json_path("dev_x").read_text())
    assert data["device_id"] == "dev_x" and data["power_mode"] == "30W"
    assert data["powermeter_default_sampling_hz"] == 30.0
    assert data["sensor"] == {"update_interval_ms": 35}
    assert "probe" not in data


def test_hw_info_survives_an_unexecutable_nvpmodel(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied", "nvpmodel")

    monkeypatch.setattr(sampling_hw_info, "collect_hw_info", lambda: {})
    monkeypatch.setattr(sampling_hw_info.orin_power.subprocess, "check_output", denied)  # the REAL get_power_mode

    assert sampling_hw_info.main(["--device-id", "dev_y", "--probe-seconds", "0"]) == 0

    assert json.loads(power_paths.sampling_hw_json_path("dev_y").read_text())["power_mode"] is None


def test_hw_info_probe_reports_the_effective_rate(monkeypatch):
    class ProbeMeter:
        def __init__(self, **kwargs):
            pass

        def estimate_fn_power(self, fn, *args):
            return {"sampling": {"n_samples": 270, "window_s": 10.0, "dt_max_s": 0.06, "configured_hz": 30.0}}

    monkeypatch.setattr(sampling_hw_info.orin_power, "PowerMeter", ProbeMeter)

    probe = sampling_hw_info.run_probe(10.0)

    assert probe["effective_hz"] == pytest.approx(27.0)
    assert probe["configured_hz"] == 30.0 and probe["n_samples"] == 270


# --------------------------------------------------------------------------
# Hook in jetson_powermeter.measure_and_record
# --------------------------------------------------------------------------

class _MeterWithSampling:
    n_samples = 60

    def __init__(self, **kwargs):
        pass

    def estimate_fn_power(self, fn):
        return {
            "elapsed": 2.0, "total": {"energy_total_j": 10.0, "energy_net_j": 4.0},
            "sampling": {"n_samples": self.n_samples, "window_s": 2.0, "dt_max_s": 0.04, "configured_hz": 30.0},
            "output": fn(),
        }


_VIT_OUTPUT = {"avg_latency_ms": 28.07, "throughput_ips": 35.6, "num_inferences_measured": 5}


def _measure(monkeypatch, tmp_path, run_id="vit_run", repeat_id=1, meter=_MeterWithSampling):
    _stub_devices(monkeypatch)
    monkeypatch.setattr(jetson_powermeter, "_power_meter", None)
    monkeypatch.setattr(jetson_powermeter, "_verify_power_mode", lambda device_id: None)
    monkeypatch.setattr(power_results_io, "results_csv_path", lambda d: tmp_path / f"results_{d}.csv")
    monkeypatch.setattr(jetson_powermeter.orin_power, "PowerMeter", meter)
    return jetson_powermeter.measure_and_record(
        lambda: dict(_VIT_OUTPUT), vit_results_io.build_result_row, run_id, "jetson_gpu_30w", repeat_id, 5,
        perf_per_watt_fn=power_cli._vit_perf_per_watt,
    )


def test_measure_and_record_writes_a_sampling_record_joined_to_the_results_row(monkeypatch, tmp_path, capsys):
    result = _measure(monkeypatch, tmp_path)

    records = sc.read_sampling_records("jetson_gpu_30w")
    assert len(records) == 1
    assert records[0]["result_id"] == result["row"]["result_id"] == "vit_run__jetson_gpu_30w__rep1"
    assert records[0]["n_samples"] == "60" and records[0]["meets_threshold"] == "True"
    assert records[0]["backend"] == "cuda" and records[0]["power_mode_w"] == "30"
    assert result["sampling_path"] == power_paths.sampling_csv_path("jetson_gpu_30w")
    out = capsys.readouterr().out
    assert "60 samples" in out and "WARNING" not in out


def test_measure_and_record_flags_a_window_below_the_threshold(monkeypatch, tmp_path, capsys):
    class ShortWindowMeter(_MeterWithSampling):
        n_samples = 12

    _measure(monkeypatch, tmp_path, meter=ShortWindowMeter)

    assert sc.read_sampling_records("jetson_gpu_30w")[0]["meets_threshold"] == "False"
    assert "WARNING" in capsys.readouterr().out


def test_a_forced_remeasurement_replaces_its_record_instead_of_duplicating_it(monkeypatch, tmp_path):
    _measure(monkeypatch, tmp_path)
    _measure(monkeypatch, tmp_path)  # same run_id/repeat_id, as run_batch --force does

    assert len(sc.read_sampling_records("jetson_gpu_30w")) == 1


def test_the_results_row_itself_is_unchanged_by_the_instrumentation(monkeypatch, tmp_path):
    row = _measure(monkeypatch, tmp_path)["row"]

    assert set(row) == set(power_results_io.RESULT_COLUMNS)  # no new columns in the shared schema
    assert row["energy_net_j"] == 4.0
