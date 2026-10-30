"""scripts/artifact_sizes.py on a tmp tree: fake manifests and artifacts whose
paths come from the pipeline's own functions."""

from __future__ import annotations

import csv

import scripts.artifact_sizes as az
from configs import llm_manifest
from quantization.llm_llamacpp.base_model import base_model_path
from quantization.llm_llamacpp.quantize import quantized_path
from quantization.vit_brevitas import deployment

HEADER = ["run_id", "repeat_id", "artifact_path"]


def _write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def _touch(root, abs_path, size):
    p = root / az._rel(abs_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0" * size)


def _llm_rows(model):
    """Rows that exist whatever the Phase 2 selection: Q4_K_M, Q4_0, Q3_K_M (deterministic) + one stochastic."""
    det = [f"llm_{model}_phase1_baseline_default_q4km", f"llm_{model}_phase2_quant_scheme_legacy_corpus_none_q4km",
           f"llm_{model}_phase3_baseline_q3_k"]
    rows = [(r, "", az._rel(quantized_path(r, None))) for r in det]
    stoch = f"llm_{model}_phase2_calibration_corpus_wikitext2_q4km"
    rows.append((stoch, "1", az._rel(quantized_path(stoch, 1))))
    return rows


def _setup(tmp_path, *, skip_files=()):
    manifests, root = tmp_path / "manifests", tmp_path / "root"
    manifests.mkdir()
    vit_rows = []
    for i, r in enumerate(deployment.plan()):
        vit_rows.append((r["run_id"], r["repeat_id"], az._rel(r["path"])))
        if r["run_id"] not in skip_files:
            _touch(root, r["path"], 100 + i)
    _write_manifest(manifests / az.VIT_MANIFEST, vit_rows)
    for model in llm_manifest.MODELS:
        rows = _llm_rows(model)
        _write_manifest(manifests / az.llm_manifest_name(model), rows)
        for run_id, rep, rel in rows:
            if run_id not in skip_files:
                _touch(root, az._PROJECT_ROOT / rel, 2**20)
        _touch(root, base_model_path(model), 3 * 2**20)
    return manifests, root


def test_all_present(tmp_path):
    manifests, root = _setup(tmp_path)
    rows, issues = az.collect(manifests, root)
    assert issues == []
    n_models = len(llm_manifest.MODELS)
    assert len(rows) == 10 + 4 * n_models + n_models
    by_id = {r["run_id"]: r for r in rows}

    assert by_id["vit_deit_tiny_deploy_fp32"]["format"] == "fp32"
    assert by_id["vit_deit_tiny_deploy_per_tensor_symmetric_int8"]["format"] == "int8_qcdq_output_quant"
    ref = by_id["vit_mobilevit_phase2_calibration_algorithm_percentile_int8"]
    assert ref["format"] == "int8_qcdq_input_only" and ref["artifact_path"].startswith("artifacts/vit/")
    assert ref["model_name"] == "mobilevit"

    m = "qwen2_5_1_5b"
    assert by_id[f"llm_{m}_phase1_baseline_default_q4km"]["format"] == "Q4_K_M"
    assert by_id[f"llm_{m}_phase2_quant_scheme_legacy_corpus_none_q4km"]["format"] == "Q4_0"
    assert by_id[f"llm_{m}_phase3_baseline_q3_k"]["format"] == "Q3_K_M"
    stoch = by_id[f"llm_{m}_phase2_calibration_corpus_wikitext2_q4km"]
    assert stoch["artifact_path"].endswith("__rep1.gguf") and stoch["size_mib"] == "1.000"

    f16 = by_id[az.f16_run_id(m)]
    assert f16 == {"branch": "llm", "run_id": f"llm_{m}_reference_f16", "model_name": m, "format": "F16",
                   "source": "llm_base", "artifact_path": f"artifacts/llm_base/{m}_f16.gguf",
                   "size_bytes": 3 * 2**20, "size_mib": "3.000"}


def test_f16_run_id_collides_with_no_generated_run_id():
    from configs import llm_phase3_manifest

    generated = {e["run_id"] for e in llm_manifest.generate() + llm_phase3_manifest.generate()[0]}
    assert not generated & {az.f16_run_id(m) for m in llm_manifest.MODELS}


def test_missing_artifact_warns_and_has_no_row(tmp_path):
    missing = "llm_tinyllama_1_1b_phase3_baseline_q3_k"
    manifests, root = _setup(tmp_path, skip_files={missing, "vit_mobilevit_deploy_fp32"})
    rows, issues = az.collect(manifests, root)
    ids = {r["run_id"] for r in rows}
    assert missing not in ids and "vit_mobilevit_deploy_fp32" not in ids
    assert len(issues) == 2 and all("artifact does not exist" in i for i in issues)


def test_missing_manifest_and_path_mismatch_warn(tmp_path):
    manifests, root = _setup(tmp_path)
    (manifests / az.llm_manifest_name("ministral_3_3b")).unlink()
    rows = _llm_rows("qwen2_5_1_5b")
    rows[0] = (rows[0][0], "", "artifacts/llm/elsewhere.gguf")
    _write_manifest(manifests / az.llm_manifest_name("qwen2_5_1_5b"), rows)

    out_rows, issues = az.collect(manifests, root)
    assert any("manifest does not exist" in i for i in issues)
    assert any("manifest path artifacts/llm/elsewhere.gguf" in i for i in issues)
    # the pipeline's path is still the one measured; the F16 rows don't depend on the manifests
    assert any(r["run_id"] == rows[0][0] for r in out_rows)
    assert any(r["run_id"] == az.f16_run_id("ministral_3_3b") for r in out_rows)


def test_unknown_run_id_warns(tmp_path):
    manifests, root = _setup(tmp_path)
    _write_manifest(manifests / az.VIT_MANIFEST, [("vit_deit_tiny_phase2_bogus_int8", 1, "artifacts/vit/x.onnx")])
    rows, issues = az.collect(manifests, root)
    assert any("vit_deit_tiny_phase2_bogus_int8" in i for i in issues)
    assert not any(r["branch"] == "vit" for r in rows)


def test_main_writes_csv_and_log_and_exit_code(tmp_path, monkeypatch):
    manifests, root = _setup(tmp_path, skip_files={"vit_deit_tiny_deploy_fp32"})
    real_collect = az.collect
    monkeypatch.setattr(az, "collect", lambda d: real_collect(d, root))
    out = tmp_path / "out" / "artifact_sizes.csv"
    assert az.main(["--manifest-dir", str(manifests), "--out", str(out)]) == 1
    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == az.FIELDS
        assert len(list(reader)) == 9 + 5 * len(llm_manifest.MODELS)
    assert "WARNING" in out.with_suffix(".log").read_text()
