"""scripts/f16_ratios.py on synthetic one-mode trees."""

from __future__ import annotations

import csv

import pytest

import scripts.f16_ratios as fr
from configs.llm_f16_reference import run_id as f16_run_id


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _res(device, run_id, pp, tg, e, window):
    """pp/tg give t_rep; window = energy_raw_j / avg_power_w_raw."""
    return {"result_id": f"{run_id}__{device}", "run_id": run_id, "throughput_prefill_tps": pp,
            "throughput_decode_tps": tg, "energy_per_inference": e, "num_inferences_measured": 30,
            "energy_raw_j": 10.0 * window, "avg_power_w_raw": 10.0}


def _ppl(run_id, value, rep=""):
    return {"run_id": run_id, "accuracy_metric_name": "perplexity", "accuracy_value": value, "repeat_id": rep}


def _setup(tmp_path, monkeypatch, model, device):
    monkeypatch.setattr(fr, "MODES", [device])
    monkeypatch.setattr(fr, "MODELS", [model])
    fid = f16_run_id(model)
    qids = [fr.q_run_id(model, s) for _, s in fr.FORMATS]
    # F16: t_rep 2 s, window 75 s -> phi 0.2; Q: t_rep 1/1.05+0.5, window = 30 t_rep / 0.9 -> phi 0.1
    t_rep_q = 128 / 134.4 + 32 / 64
    _write(tmp_path / f"results_{device}.csv",
           [_res(device, fid, 128, 32, 20.0, 75)] + [_res(device, q, 134.4, 64, 10.0, 30 * t_rep_q / 0.9) for q in qids])
    _write(tmp_path / "sizes.csv", [{"run_id": fid, "size_bytes": 300}] + [{"run_id": q, "size_bytes": 100} for q in qids])
    return fid, qids


def test_ratios_bounds_and_best_ppl_with_a_phase2_winner(tmp_path, monkeypatch):
    model, device = "qwen2_5_1_5b", "jetson_cpu_30w"
    fid, qids = _setup(tmp_path, monkeypatch, model, device)
    winner = fr.q_run_id(model, "phase2_calibration_corpus_wikitext2_q4km")
    legacy = fr.q_run_id(model, "phase2_quant_scheme_legacy_corpus_wikitext2_q4km")
    ppl = [_ppl(fid, 10.0)] + [_ppl(q, 11.0) for q in qids]
    ppl += [_ppl(winner, v, i) for i, v in enumerate((10.4, 10.6), 1)]
    ppl += [_ppl(legacy, 10.8, 1)]
    ppl += [_ppl(fr.q_run_id(model, f"phase3_combined_{b}"), 10.9, 1) for b in ("q3_k", "q2_k")]
    _write(tmp_path / "ppl.csv", ppl)
    combined = [{"run_id": fr.q_run_id(model, "phase3_combined_q3_k"), "model_name": model,
                 "calibration_corpus": "wikitext2", "calib_set_size": 200, "quant_scheme": "k_quant"}]

    rows, issues = fr.compute(tmp_path, tmp_path / "sizes.csv", [tmp_path / "ppl.csv"], tmp_path / "none.csv",
                              combined=combined)
    assert issues == [] and len(rows) == 4
    r = {x["format"]: x for x in rows}
    q4 = r["Q4_K_M"]
    assert q4["r_pp"] == pytest.approx(1.05) and q4["read_pp"] == "sin diferencia apreciable"
    assert q4["r_tg"] == pytest.approx(2.0) and q4["r_energy"] == pytest.approx(2.0)
    assert q4["fixed_frac_f16"] == pytest.approx(0.2) and q4["fixed_frac_q"] == pytest.approx(0.1)
    assert q4["r_energy_k039"] == pytest.approx(2.0 * (1 - 0.39 * 0.2) / (1 - 0.39 * 0.1))
    assert q4["energy_bias_max_pct"] == pytest.approx(100 * (2.0 / q4["r_energy_k039"] - 1))
    assert q4["r_size"] == pytest.approx(3.0) and q4["dppl"] == pytest.approx(1.0)
    assert q4["source_best"] == winner and q4["ppl_best_mean"] == pytest.approx(10.5) and q4["n_best"] == 2
    assert q4["dppl_best_pct"] == pytest.approx(5.0)
    assert r["Q4_0"]["source_best"] == legacy and r["Q4_0"]["dppl_best"] == pytest.approx(0.8)
    assert r["Q2_K"]["source_best"].endswith("phase3_combined_q2_k")


def test_no_winner_uses_baseline_and_ceiling_and_gpu_has_no_bound(tmp_path, monkeypatch):
    model, device = "ministral_3_3b", "jetson_gpu_50w"
    fid, qids = _setup(tmp_path, monkeypatch, model, device)
    legacy = fr.q_run_id(model, "phase2_quant_scheme_legacy_corpus_wikitext2_q4km")
    _write(tmp_path / "ppl.csv", [_ppl(fid, 10.0)] + [_ppl(q, 11.0) for q in qids] + [_ppl(legacy, 10.5, 1)])
    _write(tmp_path / "ceiling.csv", [{"model": model, "format": f, "repeat": i, "ppl_imatrix": v}
                                      for f in ("Q2_K", "Q3_K_M") for i, v in ((1, 10.2), (2, 10.4))])

    rows, issues = fr.compute(tmp_path, tmp_path / "sizes.csv", [tmp_path / "ppl.csv"], tmp_path / "ceiling.csv",
                              combined=[])
    assert issues == []
    r = {x["format"]: x for x in rows}
    assert "sin ganador" in r["Q4_K_M"]["source_best"] and r["Q4_K_M"]["dppl_best"] == pytest.approx(1.0)
    assert r["Q2_K"]["source_best"].startswith("ceiling") and r["Q2_K"]["ppl_best_mean"] == pytest.approx(10.3)
    assert r["Q4_K_M"]["r_energy_k039"] is None and r["Q4_K_M"]["read_energy_bounds"] == "sin corrección (GPU)"


def test_real_sources_resolve():
    """With the real generators: TinyLlama/Qwen have a Phase 2 winner, Ministral does not."""
    from configs import llm_phase3_manifest

    combined = [e for e in llm_phase3_manifest.generate()[0] if "_combined_" in e["run_id"]]
    assert fr.best_sources("tinyllama_1_1b", combined)["Q4_K_M"] == (
        "ppl", "llm_tinyllama_1_1b_phase2_calibration_corpus_wikitext2_q4km")
    assert fr.best_sources("ministral_3_3b", combined)["Q2_K"] == ("ceiling", None)


def test_read_ratio_bounds():
    assert fr.read_ratio(0.9) == fr.read_ratio(1.1) == "sin diferencia apreciable"
    assert fr.read_ratio(0.89) == "Q peor" and fr.read_ratio(1.11) == "Q mejor"
