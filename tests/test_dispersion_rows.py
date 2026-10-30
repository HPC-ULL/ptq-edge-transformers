"""scripts/dispersion_rows.py on tmp CSVs shaped like results_<device>.csv and
power_sampling_<device>.csv."""

from __future__ import annotations

import csv

import pytest

import scripts.dispersion_rows as dr
from benchmarking_ondevice.power_logging.results_io import RESULT_COLUMNS

DEVICE = "jetson_cpu_50w"
ROWS = ["vit_deit_tiny_deploy_fp32", "vit_deit_tiny_deploy_per_tensor_symmetric_int8"]
S_FIELDS = ["result_id", "run_id", "n_samples"]


def _row(run_id, latency, net, raw, energy, n=1000):
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(result_id=f"{run_id}__{DEVICE}__rep1", run_id=run_id, device=DEVICE, repeat_id=1,
               latency_ms=latency, avg_power_w_net=net, avg_power_w_raw=raw, energy_net_j=energy,
               num_inferences_measured=n, requested_repeats=n)
    return row


def _write(path, fields, rows):
    dr.write_csv(path, fields, rows)


def _setup(tmp_path, values):
    """values: {run_id: (latency, net, raw, energy)} -> results + sampling CSVs, plus an unrelated LLM row."""
    results = tmp_path / "results.csv"
    sampling = tmp_path / "sampling.csv"
    rows = [_row(r, *v) for r, v in values.items()]
    rows.append({**_row("llm_x", 1, 1, 1, 1, 30), "result_id": "llm_x__d"})
    _write(results, RESULT_COLUMNS, rows)
    _write(sampling, S_FIELDS, [{"result_id": f"{r}__{DEVICE}__rep1", "run_id": r, "n_samples": 100} for r in values])
    return results, sampling


def _run_all(tmp_path, per_rep, out):
    """snapshot rep1..3 given {rep: {run_id: (lat, net, raw, energy)}}; returns the live CSV paths (left at rep 3)."""
    for rep, values in per_rep.items():
        results, sampling = _setup(tmp_path, values)
        for r in values:  # a distinct sampling record per repetition
            dr.replace_row(sampling, {"result_id": f"{r}__{DEVICE}__rep1", "run_id": r, "n_samples": 100 + rep})
        dr.snapshot(DEVICE, list(values), rep, out, results, sampling)
    return results, sampling


def test_snapshot_saves_full_rows_idle_and_sampling_and_never_overwrites(tmp_path):
    results, sampling = _setup(tmp_path, {ROWS[0]: (38.0, 1.5, 3.5, 60.0)})

    paths = dr.snapshot(DEVICE, [ROWS[0]], 1, tmp_path / "d", results, sampling)

    rows = dr.read_csv(paths[0])[1]
    assert rows[0]["latency_ms"] == "38.0" and rows[0]["rep"] == "1" and float(rows[0]["idle_implicit_w"]) == pytest.approx(2.0)
    assert set(RESULT_COLUMNS) <= set(rows[0])  # the FULL row
    assert dr.read_csv(paths[1])[1][0]["n_samples"] == "100"
    with pytest.raises(SystemExit, match="never overwritten"):
        dr.snapshot(DEVICE, [ROWS[0]], 1, tmp_path / "d", results, sampling)


def test_snapshot_refuses_a_missing_row_or_one_without_energy(tmp_path):
    results, sampling = _setup(tmp_path, {ROWS[0]: (38.0, 1.5, 3.5, 60.0)})
    with pytest.raises(SystemExit, match="no row"):
        dr.snapshot(DEVICE, ["vit_nope"], 1, tmp_path / "d", results, sampling)
    rows = dr.read_csv(results)[1]
    rows[0]["energy_net_j"] = ""
    _write(results, RESULT_COLUMNS, rows)
    with pytest.raises(SystemExit, match="no energy"):
        dr.snapshot(DEVICE, [ROWS[0]], 2, tmp_path / "d", results, sampling)


def _three(lat=(38.0, 40.0, 39.0), net=(1.0, 1.2, 1.1), raw=(3.0, 3.1, 3.2), energy=(50.0, 60.0, 55.0)):
    return {rep: {ROWS[0]: (lat[rep - 1], net[rep - 1], raw[rep - 1], energy[rep - 1])} for rep in (1, 2, 3)}


def test_finalize_keeps_the_median_latency_row_whole_and_leaves_other_rows_untouched(tmp_path):
    out = tmp_path / "d"
    results, sampling = _run_all(tmp_path, _three(), out)  # live CSV now holds rep 3 (latency 39.0)
    llm_before = [r for r in dr.read_csv(results)[1] if r["run_id"] == "llm_x"]

    dr.finalize(DEVICE, [ROWS[0]], out, results, sampling)

    kept = next(r for r in dr.read_csv(results)[1] if r["run_id"] == ROWS[0])
    assert (kept["latency_ms"], kept["avg_power_w_net"], kept["avg_power_w_raw"], kept["energy_net_j"]) == \
        ("39.0", "1.1", "3.2", "55.0")  # all four from rep3: the median latency, no field from another run
    assert [r for r in dr.read_csv(results)[1] if r["run_id"] == "llm_x"] == llm_before
    assert next(r for r in dr.read_csv(sampling)[1])["n_samples"] == "103"  # rep3's sampling record
    assert (out / "summary.csv").exists() and (out / "selection.csv").exists()


def test_summary_medians_and_derived_metrics_are_the_transform_of_the_median(tmp_path):
    out = tmp_path / "d"
    results, sampling = _run_all(tmp_path, _three(), out)

    summary, _ = dr.finalize(DEVICE, [ROWS[0]], out, results, sampling, apply=False)

    s = {x["metric"]: x for x in summary}
    assert s["latency_ms"]["median"] == 39.0 and s["avg_power_w_net"]["median"] == 1.1
    assert s["energy_net_j"]["median"] == 55.0 and s["energy_net_j"]["derived"] is False
    # ViT formula N / energy_net_j (not the LLM one), derived, and equal to the transform of the median
    assert s["perf_per_watt"]["derived"] is True and s["perf_per_watt"]["median"] == pytest.approx(1000 / 55.0)
    assert s["energy_per_inference"]["median"] == pytest.approx(55.0 / 1000)
    assert s["idle_implicit_w"]["median"] == pytest.approx(2.0)
    assert s["latency_ms"]["cv_pct"] == pytest.approx(100 * 1.0 / 39.0)


def test_selection_flags_when_the_kept_run_is_not_the_median_power_run(tmp_path):
    out = tmp_path / "d"
    # latency median = rep3, but net power median = rep1
    results, sampling = _run_all(tmp_path, _three(net=(1.1, 2.5, 1.0)), out)

    _, selection = dr.finalize(DEVICE, [ROWS[0]], out, results, sampling, apply=False)

    assert selection[0]["chosen_rep"] == 3 and selection[0]["median_power_rep"] == 1
    assert selection[0]["chosen_is_median_power"] is False


def test_report_prints_idle_variation_between_reps_and_marks_flagged_rows(tmp_path, capsys):
    out = tmp_path / "d"
    results, sampling = _run_all(tmp_path, _three(net=(1.1, 2.5, 1.0)), out)

    dr.finalize(DEVICE, [ROWS[0]], out, results, sampling, apply=False)

    text = capsys.readouterr().out
    assert "FLAGGED" in text and "implicit idle W" in text and "within one repetition" in text
    assert "38.000 / 40.000 / 39.000" in text


def test_finalize_needs_three_repetitions_with_the_same_n(tmp_path):
    out = tmp_path / "d"
    results, sampling = _setup(tmp_path, {ROWS[0]: (38.0, 1.5, 3.5, 60.0)})
    dr.snapshot(DEVICE, [ROWS[0]], 1, out, results, sampling)
    with pytest.raises(SystemExit, match="rep2.csv is missing"):
        dr.finalize(DEVICE, [ROWS[0]], out, results, sampling)


def test_the_net_power_threshold_is_the_preregistered_midpoint():
    assert dr.NET_POWER_THRESHOLD_W == 1.27  # fixed before the re-measurement; changing it is a visible act


def _decide_setup(tmp_path, new_net):
    backup = tmp_path / "backup"
    old = {"vit_a": (40.0, 1.58, 3.58, 63.0), "vit_s1": (40.0, 1.046, 3.04, 42.0), "vit_s2": (40.0, 1.136, 3.13, 45.0)}
    _setup(tmp_path, old)
    backup.mkdir()
    for name in ("results.csv", "sampling.csv"):
        (backup / name).write_text((tmp_path / name).read_text())
    results, sampling = _setup(tmp_path, {**old, "vit_a": (41.0, new_net, new_net + 2.0, 50.0)})
    return backup, results, sampling


@pytest.mark.parametrize("new_net, verdict, kept_net", [(1.05, "one_off", "1.05"), (1.27, "one_off", "1.27"),
                                                       (1.58, "reproducible", "1.58"), (1.28, "reproducible", "1.28")])
def test_decide_keeps_the_new_row_only_at_or_below_the_threshold(tmp_path, capsys, new_net, verdict, kept_net):
    backup, results, sampling = _decide_setup(tmp_path, new_net)

    assert dr.decide(DEVICE, "vit_a", ["vit_s1", "vit_s2"], backup, tmp_path / "d", results, sampling) == verdict

    row = next(r for r in dr.read_csv(results)[1] if r["run_id"] == "vit_a")
    # one-off: the new row stays; reproducible: the OLD row (net 1.58) is restored
    assert row["avg_power_w_net"] == (kept_net if verdict == "one_off" else "1.58")
    saved = dr.read_csv(tmp_path / "d" / "vit_a_remeasure.csv")[1][0]
    assert saved["avg_power_w_net"] == str(new_net)  # the new run is kept for the investigation either way
    out = capsys.readouterr().out
    assert "context vit_s1" in out and ("nothing is replaced" in out) == (verdict == "reproducible")


def test_decide_dry_run_changes_nothing(tmp_path):
    backup, results, sampling = _decide_setup(tmp_path, 1.58)
    before = results.read_text()

    dr.decide(DEVICE, "vit_a", [], backup, tmp_path / "d", results, sampling, apply=False)

    assert results.read_text() == before


# --------------------------------------------------------------------------
# classify: speed-up rule and sibling rule
# --------------------------------------------------------------------------

def _summary_rows(device, run_id, lat, power):
    """summary.csv rows (latency and net power) for 3 runs given as lists."""
    out = []
    for metric, v in (("latency_ms", lat), ("avg_power_w_net", power)):
        out.append({"device_id": device, "run_id": run_id, "metric": metric, "derived": False, "rep1": v[0], "rep2": v[1],
                    "rep3": v[2], "median": sorted(v)[1], "min": min(v), "max": max(v), "cv_pct": 0})
    return out


def _dir(tmp_path, name, rows):
    d = tmp_path / name
    dr.write_csv(d / "summary.csv", ["device_id", "run_id", "metric", "derived", "rep1", "rep2", "rep3", "median", "min",
                                     "max", "cv_pct"], rows)
    return d


@pytest.mark.parametrize("fp32, int8, verdict", [
    ((50.0, 51.0, 50.5), (40.0, 41.0, 40.5), "SPEED-UP"),            # min(fp32)=50 > max(int8)=41
    ((38.0, 38.5, 38.2), (48.0, 49.0, 48.5), "SLOWER"),              # max(fp32)=38.5 < min(int8)=48
    ((38.0, 41.0, 39.0), (37.0, 40.0, 38.5), "NO APPRECIABLE DIFFERENCE"),  # ranges overlap
])
def test_speedup_rule(tmp_path, fp32, int8, verdict):
    dev = "jetson_cpu_30w"
    d = _dir(tmp_path, "a", _summary_rows(dev, "vit_deit_tiny_deploy_fp32", fp32, (1, 1, 1)) +
             _summary_rows(dev, "vit_deit_tiny_deploy_per_tensor_symmetric_int8", int8, (1, 1, 1)))

    (row,) = dr.classify(dr.load_summaries([d]))

    assert row["verdict"] == verdict and row["core"] is True and row["model"] == "deit_tiny"


def test_speedup_boundary_is_strict(tmp_path):
    dev = "d"
    d = _dir(tmp_path, "a", _summary_rows(dev, "vit_deit_tiny_deploy_fp32", (40.0, 41.0, 40.0), (1, 1, 1)) +
             _summary_rows(dev, "vit_deit_tiny_deploy_per_tensor_symmetric_int8", (40.0, 40.0, 40.0), (1, 1, 1)))
    # min(fp32)/max(int8) = 40/40 = 1.0 is not > 1
    assert dr.classify(dr.load_summaries([d]))[0]["verdict"] == "NO APPRECIABLE DIFFERENCE"


def test_sibling_rule_flags_the_single_measurement_row_that_deviates(tmp_path):
    dev = "jetson_cpu_15w"
    core = "vit_deit_tiny_deploy_per_tensor_symmetric_int8"
    d = _dir(tmp_path, "a", _summary_rows(dev, "vit_deit_tiny_deploy_fp32", (74.0, 74.2, 74.4), (1.16, 1.16, 1.16)) +
             _summary_rows(dev, core, (48.5, 48.6, 49.4), (1.134, 1.137, 1.046)))
    results = tmp_path / "res.csv"
    rows = []
    for rid, lat, net in [("vit_deit_tiny_deploy_fp32", 74.2, 1.16), (core, 48.6, 1.134),
                          ("vit_deit_tiny_deploy_per_channel_symmetric_int8", 49.0, 1.136),    # ok
                          ("vit_deit_tiny_deploy_per_tensor_asymmetric_int8", 48.5, 1.583),    # power +40 %: FAILS
                          ("vit_deit_tiny_phase2_calibration_algorithm_mse_int8", 70.0, 1.2)]:  # a reference: ignored
        r = _row(rid, lat, net, net + 2, 50.0)
        r["result_id"] = f"{rid}__{dev}__rep1"
        rows.append(r)
    dr.write_csv(results, RESULT_COLUMNS, rows)

    siblings = dr.check_siblings(dr.load_summaries([d]), lambda device: results)

    by = {s["config"]: s for s in siblings}
    assert set(by) == {"per_channel_symmetric_int8", "per_tensor_asymmetric_int8"}  # only rows with no summary
    assert by["per_channel_symmetric_int8"]["meets"] is True
    assert by["per_tensor_asymmetric_int8"]["meets"] is False
    assert by["per_tensor_asymmetric_int8"]["power_dev"] == pytest.approx(1.583 / 1.134 - 1, rel=1e-3)
    text = dr.format_classification(dr.classify(dr.load_summaries([d])), siblings)
    assert "FAILS" in text and "1 of 2 single-measurement rows do not meet" in text


def test_sibling_reference_falls_back_to_the_other_rows_when_no_row_has_three_runs(tmp_path):
    dev = "d"
    d = _dir(tmp_path, "a", _summary_rows(dev, "vit_mobilevit_deploy_fp32", (200.0, 201.0, 202.0), (3, 3, 3)))
    results = tmp_path / "res.csv"
    rows = []
    for rid, lat, net in [("vit_mobilevit_deploy_per_channel_asymmetric_int8", 100.0, 2.0),
                          ("vit_mobilevit_deploy_per_tensor_asymmetric_int8", 101.0, 2.1)]:
        r = _row(rid, lat, net, net + 2, 50.0)
        r["result_id"] = f"{rid}__{dev}__rep1"
        rows.append(r)
    dr.write_csv(results, RESULT_COLUMNS, rows)

    siblings = dr.check_siblings(dr.load_summaries([d]), lambda device: results)

    assert len(siblings) == 2 and all(s["meets"] for s in siblings)  # each is compared with the other one


def test_summary_has_the_speedup_per_session_and_of_the_medians(tmp_path):
    out = tmp_path / "d"
    fp32, core = "vit_deit_tiny_deploy_fp32", "vit_deit_tiny_deploy_per_tensor_symmetric_int8"
    lat = {1: (60.0, 30.0), 2: (66.0, 30.0), 3: (63.0, 42.0)}  # rep -> (fp32 ms, int8 ms)
    for rep, (f, i) in lat.items():
        results, sampling = _setup(tmp_path, {fp32: (f, 3.0, 5.0, 100.0), core: (i, 2.0, 4.0, 60.0)})
        dr.snapshot(DEVICE, [fp32, core], rep, out, results, sampling)

    summary, _ = dr.finalize(DEVICE, [fp32, core], out, results, sampling, apply=False)

    s = {(x["run_id"], x["metric"]): x for x in summary}
    per = s[(core, "speedup_per_session")]
    assert (per["rep1"], per["rep2"], per["rep3"]) == pytest.approx((2.0, 2.2, 1.5))
    assert (per["min"], per["max"], per["median"]) == pytest.approx((1.5, 2.2, 2.0))
    assert s[(core, "speedup_of_medians")]["median"] == pytest.approx(63.0 / 30.0)  # medians 63 and 30
    assert (fp32, "speedup_per_session") not in s  # FP32 has no speed-up row
    assert [r for r in dr.read_csv(out / "summary.csv")[1] if r["metric"] == "speedup_of_medians"][0]["min"] == ""
