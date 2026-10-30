"""scripts/thermal_series.py: the pre-registered criterion, tegrastats parsing,
the cold wait, and the order of the whole session with a fake run_batch."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

import scripts.dispersion_rows as dr
import scripts.thermal_series as ts
from benchmarking_ondevice.power_logging.results_io import RESULT_COLUMNS

DEVICE = "jetson_cpu_50w_serie"


def test_preregistered_constants_are_pinned():
    assert (ts.RISE_MIN, ts.FLAT_RANGE, ts.TEMP_DELTA_C, ts.FREQ_DELTA, ts.CONTROL_TOL) == (0.03, 0.02, 2.0, -0.02, 0.02)
    assert (ts.COLD_TOLERANCE_C, ts.COLD_WAIT_MAX_S, ts.SERIES_N) == (1.0, 3600, 5)
    assert (ts.BATCH_FASTER_MIN, ts.BATCH_SAME_BAND) == (0.03, 0.02)
    assert ts.BASE_FLAGS == ["--with-power", "--temp-cooldown", "--force", "--temp-log"]


FLAT = [80.0, 80.5, 79.8, 80.2, 80.4]
RISING = [80.0, 81.0, 82.0, 83.0, 84.0]       # +5 %
FP32_FLAT = [75.0, 75.2, 74.9, 75.1, 75.0]


@pytest.mark.parametrize("int8, fp32, kw, expected", [
    (FLAT, FP32_FLAT, dict(temp_delta=5.0, freq_delta=-0.1, control=80.0, control_valid=True), "NO REPRODUCIBLE"),
    ([80.0, 81.0, 80.5, 81.8, 81.9], FP32_FLAT, dict(temp_delta=0, freq_delta=0, control=80.0, control_valid=True), "NO CONCLUYENTE"),  # +2.4 %
    ([84.0, 83.0, 82.0, 81.0, 80.0], FP32_FLAT, dict(temp_delta=0, freq_delta=0, control=80.0, control_valid=True), "NO CONCLUYENTE"),  # fall
    (RISING, [75.0, 76.0, 77.0, 78.0, 79.0], dict(temp_delta=3.0, freq_delta=0, control=80.0, control_valid=True), "NO CONCLUYENTE"),  # FP32 rose as much
    (RISING, FP32_FLAT, dict(temp_delta=1.0, freq_delta=-0.01, control=80.0, control_valid=True), "TERMICA REFUTADA"),
    (RISING, FP32_FLAT, dict(temp_delta=1.0, freq_delta=-0.01, control=84.0, control_valid=True), "TERMICA REFUTADA"),  # no control needed
    (RISING, FP32_FLAT, dict(temp_delta=2.0, freq_delta=0, control=80.5, control_valid=True), "TERMICA ACEPTADA"),     # +2.0 C counts
    (RISING, FP32_FLAT, dict(temp_delta=0.0, freq_delta=-0.02, control=80.5, control_valid=True), "TERMICA ACEPTADA"),  # -2 % freq counts
    (RISING, FP32_FLAT, dict(temp_delta=3.0, freq_delta=0, control=82.0, control_valid=True), "DERIVA PERSISTENTE"),    # control +2.5 %
    (RISING, FP32_FLAT, dict(temp_delta=3.0, freq_delta=0, control=80.0, control_valid=False), "NO CONCLUYENTE"),
    (RISING, FP32_FLAT, dict(temp_delta=None, freq_delta=None, control=80.0, control_valid=True), "NO CONCLUYENTE"),
])
def test_the_five_categories(int8, fp32, kw, expected):
    category, _ = ts.classify_series(int8, fp32, **kw)
    assert category.startswith(expected)


def test_control_must_come_back_strictly_below_two_percent():
    just_below, _ = ts.classify_series(RISING, FP32_FLAT, temp_delta=3, freq_delta=0, control=80.0 * 1.0199, control_valid=True)
    at_limit, _ = ts.classify_series(RISING, FP32_FLAT, temp_delta=3, freq_delta=0, control=80.0 * 1.02 + 1e-9, control_valid=True)
    assert just_below == "TERMICA ACEPTADA" and at_limit.startswith("DERIVA PERSISTENTE")


TEGRA = ("10-02-2026 11:07:18 RAM 2345/30697MB (lfb 4x4MB) SWAP 0/15348MB CPU [12%@1497,5%@1497,off,100%@2188] "
         "EMC_FREQ 0%@2133 cpu@43.5C soc2@41.4C tj@43.5C VDD_GPU_SOC 1000mW/1000mW")


def test_parse_tegrastats_and_window_means(tmp_path):
    log = tmp_path / "t.log"
    lines = [TEGRA.replace("11:07:18", f"11:07:{s:02d}").replace("cpu@43.5C", f"cpu@{40 + s}.0C") for s in (18, 19, 20)]
    log.write_text("\n".join(["garbage line"] + lines) + "\n")

    samples = ts.parse_tegrastats(log, utc_offset_s=3600)  # local = UTC+1 -> 10:07:xx UTC

    assert len(samples) == 3 and samples[0][0] == datetime(2026, 10, 2, 10, 7, 18, tzinfo=timezone.utc)
    assert samples[0][2] == [1497, 1497, 2188]            # the offline core is skipped
    start = datetime(2026, 10, 2, 10, 7, 19, tzinfo=timezone.utc)
    m = ts.window_means(samples, start, start + timedelta(seconds=1))
    assert m["samples"] == 2 and m["temp_c"] == pytest.approx(59.5) and m["freq_mhz"] == pytest.approx((1497 * 2 + 2188) / 3)


def test_row_window_is_measured_at_minus_elapsed():
    start, end = ts.row_window({"measured_at": "2026-10-02T11:31:07+00:00", "elapsed_s": "79.0"})
    assert end == datetime(2026, 10, 2, 11, 31, 7, tzinfo=timezone.utc) and (end - start).total_seconds() == 79.0


def test_wait_cold_reaches_the_target_or_gives_up():
    temps = iter([50.0, 49.0, 46.0, 44.9])
    now = [0.0]

    def sleep(s):
        now[0] += s

    ok = ts.wait_cold(lambda: next(temps), 45.0, sleep=sleep, now=lambda: now[0], log=lambda *_: None)
    assert ok["reached"] is True and ok["t_start"] == 50.0 and ok["t_end"] == 44.9

    now[0] = 0.0
    never = ts.wait_cold(lambda: 60.0, 45.0, max_s=30, poll_s=5, sleep=sleep, now=lambda: now[0], log=lambda *_: None)
    assert never["reached"] is False and never["waited_s"] >= 30


def _fake_env(tmp_path):
    results, sampling = tmp_path / "results.csv", tmp_path / "sampling.csv"
    calls = []

    def runner(name, manifest, flags, log_path):
        calls.append((name, manifest.name, tuple(flags)))
        rows, samps = [], []
        for i, rid in enumerate(dr.read_csv(manifest)[1]):
            run_id = rid["run_id"]
            lat = 75.0 if run_id == ts.FP32_ID else 80.0 + (len(calls) if run_id == ts.INT8_ID else 0)
            r = {c: "" for c in RESULT_COLUMNS}
            r.update(result_id=f"{run_id}__{DEVICE}__rep1", run_id=run_id, device=DEVICE, repeat_id=1, latency_ms=lat,
                     avg_power_w_net=2.7, avg_power_w_raw=4.7, energy_net_j=90, num_inferences_measured=1000)
            rows.append(r)
            samps.append({"result_id": r["result_id"], "run_id": run_id, "elapsed_s": 80, "measured_at": "2026-10-02T11:00:00+00:00"})
        dr.write_csv(results, RESULT_COLUMNS, rows)
        dr.write_csv(sampling, ["result_id", "run_id", "elapsed_s", "measured_at"], samps)
        return 0

    return results, sampling, runner, calls


def _manifest(path, run_ids):
    dr.write_csv(path, ["run_id", "repeat_id", "artifact_path"], [{"run_id": r, "repeat_id": 1, "artifact_path": "x"} for r in run_ids])
    return path


def test_session_runs_batch10_then_cold_series_then_cold_control_and_saves_every_invocation(tmp_path):
    results, sampling, runner, calls = _fake_env(tmp_path)
    ten = _manifest(tmp_path / "m10.csv", ts.EXPECTED_BATCH10_ORDER)
    two = _manifest(tmp_path / "m2.csv", [ts.FP32_ID, ts.INT8_ID])
    order = []
    out = tmp_path / "d"

    def fake_wait(get_temp, target):
        order.append("wait")
        return {"t_start": 50.0, "t_end": target, "waited_s": 10.0, "reached": True}

    wrapped = lambda name, m, f, log: order.append(name) or runner(name, m, f, log)  # noqa: E731
    ts.run_session(out_dir=out, device_id=DEVICE, manifest10=ten, manifest2=two, extra_flags=ts.BASE_FLAGS, runner=wrapped,
                   get_temp=lambda: 41.0, results_csv=results, sampling_csv=sampling, wait=fake_wait)

    assert order == ["batch10", "wait", "series1", "series2", "series3", "series4", "series5", "wait", "control"]
    assert all(c[2] == tuple(ts.BASE_FLAGS) for c in calls)
    assert json.loads((out / "session.json").read_text())["t0_c"] == 41.0
    assert len(dr.read_csv(out / "batch10.csv")[1]) == 10 and len(dr.read_csv(out / "control.csv")[1]) == 2
    assert float(dr.read_csv(out / "series1.csv")[1][0]["idle_implicit_w"]) == pytest.approx(2.0)
    assert [w["target_c"] for w in dr.read_csv(out / "waits.csv")[1]] == ["42.0", "42.0"]  # T0 + 1 C
    assert [i["name"] for i in dr.read_csv(out / "invocations.csv")[1]][0] == "batch10"


def test_a_failed_invocation_stops_the_session(tmp_path):
    results, sampling, _, _ = _fake_env(tmp_path)
    two = _manifest(tmp_path / "m2.csv", [ts.FP32_ID, ts.INT8_ID])
    ten = _manifest(tmp_path / "m10.csv", ts.EXPECTED_BATCH10_ORDER)
    with pytest.raises(SystemExit, match="exited with 3"):
        ts.run_session(out_dir=tmp_path / "d", device_id=DEVICE, manifest10=ten, manifest2=two, extra_flags=[],
                       runner=lambda *a: 3, get_temp=lambda: 40.0, results_csv=results, sampling_csv=sampling)


def test_the_fp32_rising_as_much_gets_the_general_effect_label():
    category, _ = ts.classify_series(RISING, [75.0, 76.0, 77.0, 78.0, 79.5], temp_delta=3, freq_delta=0, control=80.0,
                                     control_valid=True)
    assert category == "NO CONCLUYENTE (efecto general: el FP32 tambien sube)"


@pytest.mark.parametrize("control, valid, expected", [(80.0, True, "no"), (81.7, True, "si"), (80.0, False, "control no valido")])
def test_persistent_field_is_given_whatever_the_category(control, valid, expected):
    for int8, fp32 in ((FLAT, FP32_FLAT), (RISING, FP32_FLAT)):  # a "not reproducible" and a "thermal refuted/..." series
        _, info = ts.classify_series(int8, fp32, temp_delta=0, freq_delta=0, control=control, control_valid=valid)
        assert info["persistent"] == expected


@pytest.mark.parametrize("batch, series1, expected", [
    (77.0, 80.0, "COMPOSICION DEL LOTE IDENTIFICADA"),   # 3.75 % faster
    (77.6, 80.0, "COMPOSICION DEL LOTE IDENTIFICADA"),   # exactly 3 % faster
    (78.0, 80.0, "NO CONCLUYENTE"),                      # 2.5 % faster: between the bands
    (78.4, 80.0, "COMPOSICION DEL LOTE DESCARTADA"),     # 2 % faster: inside +-2 %
    (81.6, 80.0, "COMPOSICION DEL LOTE DESCARTADA"),     # 2 % slower
    (82.0, 80.0, "NO CONCLUYENTE"),                      # 2.5 % slower
])
def test_batch10_criterion(batch, series1, expected):
    verdict, faster = ts.batch_verdict(batch, series1)
    assert verdict == expected and faster == pytest.approx((series1 - batch) / series1)


def test_manifests_must_keep_the_original_batch_order(tmp_path):
    good10 = _manifest(tmp_path / "m10.csv", ts.EXPECTED_BATCH10_ORDER)
    good2 = _manifest(tmp_path / "m2.csv", [ts.FP32_ID, ts.INT8_ID])
    ts.check_manifests(good10, good2)  # no error

    swapped = _manifest(tmp_path / "bad10.csv", ts.EXPECTED_BATCH10_ORDER[1:] + ts.EXPECTED_BATCH10_ORDER[:1])
    with pytest.raises(SystemExit, match="original batch"):
        ts.check_manifests(swapped, good2)
    with pytest.raises(SystemExit, match="exactly"):
        ts.check_manifests(good10, _manifest(tmp_path / "bad2.csv", [ts.INT8_ID, ts.FP32_ID]))


def test_the_expected_order_is_the_current_manifests_order():
    from configs import vit_deployment_manifest as dm

    assert ts.EXPECTED_BATCH10_ORDER == [r["run_id"] for r in dm.package_rows()]


def test_telemetry_alive_requires_the_log_to_keep_growing(tmp_path):
    log = tmp_path / "t.log"
    log.write_text("10-02-2026 13:24:44 x\n")
    size = log.stat().st_size

    with pytest.raises(SystemExit, match="not writing"):
        ts.telemetry_alive(log, size, timeout_s=0)           # never grew
    log.write_text(log.read_text() + "10-02-2026 13:24:45 x\n")
    assert ts.telemetry_alive(log, size, timeout_s=0) > size  # grew
    with pytest.raises(SystemExit, match="not writing"):
        ts.telemetry_alive(tmp_path / "missing.log", 0, timeout_s=0)


def test_telemetry_alive_waits_for_the_next_line_instead_of_failing_on_two_close_checks(tmp_path):
    """tegrastats writes one line a second: a check made right after another may see the same size."""
    log = tmp_path / "t.log"
    log.write_text("x\n")
    size = log.stat().st_size
    clock = [0.0]

    def sleep(s):
        clock[0] += s
        if clock[0] >= 1.0:  # the next line arrives after a second
            log.write_text(log.read_text() + "y\n")

    assert ts.telemetry_alive(log, size, timeout_s=5.0, poll_s=0.5, sleep=sleep, now=lambda: clock[0]) > size
    assert clock[0] == 1.0  # it did not wait the whole timeout


def test_the_session_stops_before_an_invocation_if_tegrastats_died(tmp_path):
    results, sampling, runner, calls = _fake_env(tmp_path)
    ten = _manifest(tmp_path / "m10.csv", ts.EXPECTED_BATCH10_ORDER)
    two = _manifest(tmp_path / "m2.csv", [ts.FP32_ID, ts.INT8_ID])
    alive = iter([True, True, False])

    def check():
        if not next(alive):
            raise SystemExit("tegrastats is not writing")

    with pytest.raises(SystemExit, match="not writing"):
        ts.run_session(out_dir=tmp_path / "d", device_id=DEVICE, manifest10=ten, manifest2=two, extra_flags=[],
                       runner=runner, get_temp=lambda: 40.0, results_csv=results, sampling_csv=sampling,
                       wait=lambda g, t: {"t_start": 40.0, "t_end": 40.0, "waited_s": 0.0, "reached": True},
                       check_telemetry=check)

    assert [c[0] for c in calls] == ["batch10", "series1"]  # the third check failed: series2 never ran


CPU_ENTRY = {"device_id": "jetson_cpu_50w", "display_name": "x", "architecture": "ARM", "backend": "cpu",
             "power_mode_w": 50, "ram_gb": 29, "cpu_cores": 12, "power_measurement": "jetson_ina3221", "min_uptime_s": 600,
             "added_by": "n"}


def test_temp_entry_must_be_a_cpu_copy_of_the_reference():
    ts.check_temp_entry({**CPU_ENTRY, "device_id": "jetson_cpu_50w_serie", "display_name": "TEMP"}, CPU_ENTRY)  # ok

    gpu = {**CPU_ENTRY, "device_id": "jetson_cpu_50w_serie", "backend": "cuda", "n_gpu_layers": 99}
    with pytest.raises(SystemExit, match="CPU only"):
        ts.check_temp_entry(gpu, CPU_ENTRY)
    for change in ({"power_mode_w": 30}, {"cpu_cores": 8}, {"min_uptime_s": 0}, {"n_gpu_layers": 99}):
        with pytest.raises(SystemExit, match="differs from jetson_cpu_50w"):
            ts.check_temp_entry({**CPU_ENTRY, "device_id": "x", **change}, CPU_ENTRY)


def test_report_warns_loudly_when_the_session_was_not_on_the_cpu(tmp_path):
    results, sampling, runner, _ = _fake_env(tmp_path)
    ten = _manifest(tmp_path / "m10.csv", ts.EXPECTED_BATCH10_ORDER)
    two = _manifest(tmp_path / "m2.csv", [ts.FP32_ID, ts.INT8_ID])
    out = tmp_path / "d"
    ts.run_session(out_dir=out, device_id=DEVICE, manifest10=ten, manifest2=two, extra_flags=[], runner=runner,
                   get_temp=lambda: 40.0, results_csv=results, sampling_csv=sampling,
                   wait=lambda g, t: {"t_start": 40.0, "t_end": 40.0, "waited_s": 0.0, "reached": True})
    # tegrastats log + a "cuda" backend in the sampling records
    (out / "tegrastats.log").write_text("")
    for name in ["batch10", "control"] + [f"series{k}" for k in range(1, 6)]:
        rows = dr.read_csv(out / f"{name}_sampling.csv")[1]
        dr.write_csv(out / f"{name}_sampling.csv", ["invocation", "result_id", "run_id", "elapsed_s", "measured_at", "backend"],
                     [{**r, "backend": "cuda"} for r in rows])

    text = ts.build_report(out)

    assert "NOT A CPU SESSION" in text and "backend of the measured rows: cuda" in text
