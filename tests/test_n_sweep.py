"""scripts/n_sweep.py with a fake run_batch (no device)."""

from __future__ import annotations

import pytest

import scripts.dispersion_rows as dr
import scripts.n_sweep as ns
from benchmarking_ondevice.runner_llm.results_io import RESULT_COLUMNS


def test_preregistered_constants_are_pinned():
    assert (ns.MEDIAN_TOL, ns.MAX_TOL, ns.COLD_MAX_CACHED) == (0.03, 0.06, 0.05)
    assert ns.STEPS == [("cold", 30), ("warmup", 10), ("n10", 10), ("n30", 30), ("n90", 90)]
    assert ns.BASE_FLAGS == ["--with-power", "--temp-cooldown", "--force"]
    assert [b[0] for b in ns.BACKENDS] == ["cpu", "gpu"] and ns.MODELS[0] == "tinyllama_1_1b"


# a synthetic world: window = FIXED + N*T_REP, net power constant P_NET (so E_net = P_NET*window - the fixed
# cost burns the same power as the timed part), idle IDLE
T_REP, FIXED, P_NET, IDLE, PP, TG = 1.0, 6.0, 5.0, 2.0, 400.0, 40.0   # 128/400 + 32/40 = 1.12 s per repetition


def _world(n, fixed=FIXED, t_rep=1.12):
    window = fixed + n * t_rep
    return window, P_NET * window


def _write_world(results, sampling, device, run_id, n, fixed=FIXED):
    window, e_net = _world(n, fixed)
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(result_id=f"{run_id}__{device}", run_id=run_id, device=device, num_inferences_measured=n,
               requested_repeats=n, throughput_prefill_tps=PP, throughput_decode_tps=TG, energy_net_j=e_net,
               energy_raw_j=(P_NET + IDLE) * window, avg_power_w_net=P_NET, avg_power_w_raw=P_NET + IDLE)
    dr.write_csv(results, RESULT_COLUMNS, [row])
    dr.write_csv(sampling, ["result_id", "run_id", "elapsed_s", "measured_at"],
                 [{"result_id": row["result_id"], "run_id": run_id, "elapsed_s": window, "measured_at": "t"}])


def _repo(tmp_path):
    for m in ns.MODELS:
        dr.write_csv(tmp_path / f"run_manifest_llm_{m}.csv", ["run_id", "repeat_id", "artifact_path"],
                     [{"run_id": f"llm_{m}_phase2_x", "repeat_id": 1, "artifact_path": "x"},
                      {"run_id": ns.llm_run_id(m), "repeat_id": "", "artifact_path": f"artifacts/llm/{ns.llm_run_id(m)}.gguf"}])
    return tmp_path


def _sweep(tmp_path, fixed=FIXED, drop=lambda: None, fincore_fn=lambda p: (100, 200)):
    repo = _repo(tmp_path)
    results = lambda d: tmp_path / f"results_{d}.csv"  # noqa: E731
    sampling = lambda d: tmp_path / f"sampling_{d}.csv"  # noqa: E731
    calls = []

    def runner(name, manifest, flags, log, device_id):
        n = int(flags[flags.index("--repeats") + 1])
        calls.append((name, device_id, tuple(flags), [r["run_id"] for r in dr.read_csv(manifest)[1]]))
        model = name.rsplit("_", 2)[0]
        _write_world(results(device_id), sampling(device_id), device_id, ns.llm_run_id(model), n, fixed)
        return 0

    runs = ns.run_sweep(out_dir=tmp_path / "out", models=ns.MODELS, backends=ns.BACKENDS, runner=runner,
                        get_temp=lambda: 45.0, fincore_fn=fincore_fn, drop_fn=drop, repo=repo,
                        results_csv_for=results, sampling_csv_for=sampling)
    return runs, calls


def test_sweep_order_flags_devices_and_one_row_manifests(tmp_path):
    runs, calls = _sweep(tmp_path)

    assert len(calls) == 30
    assert [c[0] for c in calls[:5]] == [f"tinyllama_1_1b_cpu_{s}" for s in ("cold", "warmup", "n10", "n30", "n90")]
    assert [c[1] for c in calls] == ["jetson_cpu_50w_sweep"] * 15 + ["jetson_gpu_50w_sweep"] * 15
    assert [c[0].split("_cpu_")[0] for c in calls[:15:5]] == ns.MODELS   # CPU first, models in order
    assert [int(c[2][-1]) for c in calls[:5]] == [30, 10, 10, 30, 90]
    assert all(c[2][:3] == ("--with-power", "--temp-cooldown", "--force") and c[2][-2] == "--repeats" for c in calls)
    assert all(len(c[3]) == 1 and c[3][0].endswith("_phase1_baseline_default_q4km") for c in calls)
    assert len(dr.read_csv(tmp_path / "out" / "runs.csv")[1]) == 30


def test_cold_step_drops_the_cache_after_the_first_fincore_and_before_the_run(tmp_path):
    events = []
    _sweep(tmp_path, drop=lambda: events.append("drop"), fincore_fn=lambda p: events.append("fincore") or (50, 100))
    # per combination: cold = fincore, drop, fincore, RUN; the other four = fincore, RUN
    assert events[:4] == ["fincore", "drop", "fincore", "fincore"]
    assert events.count("drop") == 6 and events.count("fincore") == 6 * (2 + 4)
    rows = dr.read_csv(tmp_path / "out" / "runs.csv")[1]
    cold = rows[0]
    assert float(cold["cached_frac_before"]) == 0.5 and float(cold["cached_frac_after_drop"]) == 0.5
    assert rows[1]["cached_frac_after_drop"] == ""


def test_a_failed_drop_stops_before_the_cold_invocation(tmp_path):
    def drop():
        raise SystemExit("sudo -n could not drop")

    with pytest.raises(SystemExit, match="could not drop"):
        _sweep(tmp_path, drop=drop)
    assert not (tmp_path / "out" / "runs.csv").exists()   # nothing measured


def test_fincore_missing_is_recorded_as_not_available(tmp_path):
    _sweep(tmp_path, fincore_fn=lambda p: None)
    assert dr.read_csv(tmp_path / "out" / "runs.csv")[1][0]["cached_frac_before"] == ""
    assert "n/a" in ns.build_report(tmp_path / "out")


def test_check_entry_requires_a_same_backend_copy():
    ref = {"device_id": "jetson_gpu_50w", "backend": "cuda", "power_mode_w": 50, "cpu_cores": 12, "n_gpu_layers": 99}
    ns.check_entry({**ref, "device_id": "x", "display_name": "t"}, ref, "gpu")
    with pytest.raises(SystemExit, match="expected the cpu"):
        ns.check_entry({**ref, "device_id": "x"}, ref, "cpu")
    with pytest.raises(SystemExit, match="differs"):
        ns.check_entry({**ref, "device_id": "x", "n_gpu_layers": 0}, ref, "gpu")


def test_fincore_parses_res_and_size_and_is_none_without_the_binary(monkeypatch):
    import types

    monkeypatch.setattr(ns.shutil, "which", lambda name: "/usr/bin/fincore")
    run = lambda cmd, **kw: types.SimpleNamespace(stdout="1048576 4194304\n")  # noqa: E731
    assert ns.fincore("x.gguf", run=run) == (1048576, 4194304)
    monkeypatch.setattr(ns.shutil, "which", lambda name: None)
    assert ns.fincore("x.gguf") is None


def test_least_squares_recovers_a_linear_relation():
    slope, intercept = ns.least_squares([(10, 70.0), (30, 190.0), (90, 550.0)])
    assert slope == pytest.approx(6.0) and intercept == pytest.approx(10.0)


def test_analysis_when_the_fixed_cost_burns_the_same_power_the_correction_is_exact(tmp_path):
    runs, _ = _sweep(tmp_path)
    steps = {r["step"]: ns.run_metrics(r) for r in runs[:5]}

    a = ns.analyze_combo(steps)

    # E_net = P_NET*(FIXED + N*t_rep): M = P_NET*t_rep, C = P_NET*FIXED, and E_corr = M exactly
    assert a["m_ls"] == pytest.approx(P_NET * 1.12) and a["m_diff"] == pytest.approx(P_NET * 1.12)
    assert a["e_corr"] == pytest.approx(a["m_ls"]) and a["dev_ls"] == pytest.approx(0, abs=1e-9)
    assert a["c_ls"] == pytest.approx(P_NET * FIXED)
    assert a["ratio_ls"] == pytest.approx(1.0)          # fixed energy share == fixed time share
    assert a["fixed_frac_time"] == pytest.approx(FIXED / (FIXED + 30 * 1.12))
    assert a["dev_cold"] == pytest.approx(0, abs=1e-9)  # same world in the cold run


def _a(dev, ec_over_m=None):
    return {"dev_ls": dev, "e_corr": 1 + dev, "m_ls": 1.0}


@pytest.mark.parametrize("devs, label", [
    ([0.01, -0.02, 0.03, 0.02, -0.01, 0.00], "VALIDADA"),
    ([0.03, 0.03, 0.03, 0.03, 0.03, 0.03], "VALIDADA"),                 # median exactly at the limit
    ([0.01, 0.02, 0.03, 0.03, 0.04, 0.06], "VALIDADA"),                 # median 3 %, max exactly 6 %
    ([0.01, 0.02, 0.01, 0.02, 0.01, 0.07], "NO VALIDADA"),              # one above 6 %
    ([0.05, 0.05, 0.05, 0.05, 0.05, 0.05], "NO VALIDADA"),              # median above 3 %
])
def test_criterion(devs, label):
    got, detail = ns.verdict([_a(d) for d in devs])
    assert got.startswith(label) and "E_corr > M in" in detail


def test_criterion_needs_all_six_combinations():
    assert ns.verdict([_a(0.0)] * 5)[0] == "INCOMPLETO"


def test_report_has_the_tables_and_the_criterion(tmp_path):
    _sweep(tmp_path)
    text = ns.build_report(tmp_path / "out")
    assert "CRITERION: VALIDADA" in text and "tinyllama_1_1b" in text and "cold: E_corr dev" in text
    assert text.count("tinyllama_1_1b") >= 5 and "E_corr > M in 6 of 6" not in text  # equal, not greater


# --------------------------------------------------------------------------
# cold3 mode
# --------------------------------------------------------------------------

def _cold3(tmp_path, events=None):
    repo = _repo(tmp_path)
    results = lambda d: tmp_path / f"results_{d}.csv"  # noqa: E731
    sampling = lambda d: tmp_path / f"sampling_{d}.csv"  # noqa: E731
    calls = []

    def runner(name, manifest, flags, log, device_id):
        calls.append((name, device_id, tuple(flags)))
        _write_world(results(device_id), sampling(device_id), device_id, ns.llm_run_id("qwen2_5_1_5b"),
                     int(flags[flags.index("--repeats") + 1]), fixed=FIXED + 0.5 * len(calls))   # a little different each time
        return 0

    ev = events if events is not None else []
    runs = ns.run_sweep(out_dir=tmp_path / "out", models=ns.COLD3_MODELS, backends=ns.BACKENDS, steps=ns.COLD3_STEPS,
                        runner=runner, get_temp=lambda: 45.0, fincore_fn=lambda p: ev.append("fincore") or (0, 100),
                        drop_fn=lambda: ev.append("drop"), repo=repo, results_csv_for=results, sampling_csv_for=sampling)
    return runs, calls, ev


def test_cold3_runs_three_cold_n30_invocations_per_backend_each_after_its_own_drop(tmp_path):
    runs, calls, events = _cold3(tmp_path)

    assert [c[0] for c in calls] == [f"qwen2_5_1_5b_{b}_cold{k}" for b in ("cpu", "gpu") for k in (1, 2, 3)]
    assert [c[1] for c in calls] == ["jetson_cpu_50w_sweep"] * 3 + ["jetson_gpu_50w_sweep"] * 3
    assert all(c[2] == ("--with-power", "--temp-cooldown", "--force", "--repeats", "30") for c in calls)
    assert events.count("drop") == 6 and events[:4] == ["fincore", "drop", "fincore", "fincore"]
    assert len(runs) == 6 and {r["step"] for r in runs} == {"cold1", "cold2", "cold3"}


def test_cold3_report_gives_per_invocation_metrics_and_range_and_cv(tmp_path):
    _cold3(tmp_path)

    text = ns.build_cold3_report(tmp_path / "out")

    assert text.count("qwen2_5_1_5b") >= 8 and "cold1" in text and "cold3" in text
    assert "range" in text and "CV" in text and "(n=3)" in text
    rows = dr.read_csv(tmp_path / "out" / "runs.csv")[1]
    st = ns.cold3_stats([r for r in rows if r["backend"] == "cpu"])
    energies = [m["e_inf"] for m in st["per_run"]]
    assert st["range"] == pytest.approx(max(energies) - min(energies))
    import statistics
    assert st["cv_pct"] == pytest.approx(100 * statistics.stdev(energies) / statistics.mean(energies))
    assert st["per_run"][0]["e_inf"] == pytest.approx(_world(30, FIXED + 0.5)[1] / 30)


def test_cold3_statistics_of_identical_runs_have_zero_dispersion(tmp_path):
    runs, _, _ = _cold3(tmp_path)
    same = [{**runs[0]}, {**runs[0]}, {**runs[0]}]
    st = ns.cold3_stats(same)
    assert st["range"] == 0 and st["cv_pct"] == 0
