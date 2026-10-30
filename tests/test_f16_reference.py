"""configs/llm_f16_reference.py, scripts/build_f16_manifests.py and
scripts/estimate_f16_time.py."""

from __future__ import annotations

import csv

import pytest

import scripts.artifact_sizes as az
import scripts.build_f16_manifests as bfm
import scripts.estimate_f16_time as est
from benchmarking_ondevice.runner_llm.runner import build_benchmark_command
from configs import llm_f16_reference, llm_manifest, llm_phase3_manifest
from quantization.llm_llamacpp.base_model import base_model_path
from scripts.run_batch import read_manifest


def test_run_ids_and_paths():
    assert [llm_f16_reference.run_id(m) for m in llm_manifest.MODELS] == [
        "llm_tinyllama_1_1b_reference_f16", "llm_qwen2_5_1_5b_reference_f16", "llm_ministral_3_3b_reference_f16"]
    assert all(llm_f16_reference.artifact_path(m) == base_model_path(m) for m in llm_manifest.MODELS)
    assert az.f16_run_id is llm_f16_reference.run_id


def test_no_collision_and_kept_out_of_quantization_manifests():
    f16 = {e["run_id"] for e in llm_f16_reference.generate()}
    generated = {e["run_id"] for e in llm_manifest.generate() + llm_phase3_manifest.generate()[0]}
    assert len(f16) == 3 and not f16 & generated


def test_manifest_is_accepted_by_run_batch(tmp_path):
    out = tmp_path / "run_manifest_llm_f16.csv"
    assert bfm.main(["--out", str(out)]) == 0
    with open(out, newline="") as f:
        assert list(csv.reader(f)) == [
            ["run_id", "repeat_id", "artifact_path"],
            ["llm_tinyllama_1_1b_reference_f16", "", "artifacts/llm_base/tinyllama_1_1b_f16.gguf"],
            ["llm_qwen2_5_1_5b_reference_f16", "", "artifacts/llm_base/qwen2_5_1_5b_f16.gguf"],
            ["llm_ministral_3_3b_reference_f16", "", "artifacts/llm_base/ministral_3_3b_f16.gguf"],
        ]
    rows = list(read_manifest(out, "llm"))
    assert [r[1] for r in rows] == [None, None, None]  # deterministic: no __rep suffix
    cmd = build_benchmark_command(rows[2][2], {"device_id": "d", "backend": "cuda", "cpu_cores": 12,
                                               "n_gpu_layers": 99}, repeats=30)
    assert cmd[cmd.index("-m") + 1] == "artifacts/llm_base/ministral_3_3b_f16.gguf"
    with pytest.raises(ValueError):  # never valid for ViT
        list(read_manifest(out, "vit"))


def _write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_estimate(tmp_path):
    model = "qwen2_5_1_5b"
    q = est.q_run_id(model)
    rid = f"{q}__jetson_cpu_30w"
    _write(tmp_path / "results_jetson_cpu_30w.csv",
           ["result_id", "run_id", "throughput_prefill_tps", "throughput_decode_tps", "num_inferences_measured"],
           [{"result_id": rid, "run_id": q, "throughput_prefill_tps": 128, "throughput_decode_tps": 32,
             "num_inferences_measured": 30}])  # t_rep = 1 + 1 = 2 s
    _write(tmp_path / "power_sampling_jetson_cpu_30w.csv", ["result_id", "window_s"],
           [{"result_id": rid, "window_s": 63}])  # fixed = 3 s
    sizes = tmp_path / "sizes.csv"
    _write(sizes, ["run_id", "size_bytes"], [{"run_id": q, "size_bytes": 100},
                                             {"run_id": llm_f16_reference.run_id(model), "size_bytes": 300}])
    ratios = est.size_ratios(sizes)
    assert ratios == {model: 3.0}

    rows = est.estimate(tmp_path, tmp_path, ratios)
    assert len(rows) == 1
    r = rows[0]
    assert r["t_rep_q_s"] == pytest.approx(2.0) and r["decode_share"] == pytest.approx(0.5)
    assert r["fixed_q_s"] == pytest.approx(3.0)
    assert r["t_rep_f16_s"] == pytest.approx(6.0)
    assert r["window_f16_s"] == pytest.approx(3 * 3 + 30 * 6)

    # without sizes, only an explicit assumed ratio produces an estimate, for every model present
    assert est.estimate(tmp_path, tmp_path, {}) == []
    assert est.estimate(tmp_path, tmp_path, {}, assumed_ratio=2.0)[0]["t_rep_f16_s"] == pytest.approx(4.0)
