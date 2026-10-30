"""The ViT deployment set: configs/vit_deployment_manifest.py,
quantization/vit_brevitas/deployment.py and the accuracy gate's
--deployment rows."""

from __future__ import annotations

import shutil

import pytest

from configs import vit_deployment_manifest as dm
from quantization.vit_brevitas import deployment
from quantization.vit_brevitas.investigations import gate_output_quant as gate


def test_eight_rows_two_of_them_fp32():
    rows = dm.generate()
    assert len(rows) == 8
    assert [r["run_id"] for r in rows if not r["quantized"]] == ["vit_deit_tiny_deploy_fp32", "vit_mobilevit_deploy_fp32"]
    assert len({r["run_id"] for r in rows}) == 8
    assert all(r["run_id"].startswith("vit_") and "_deploy_" in r["run_id"] for r in rows)


@pytest.mark.parametrize("model, calibration, core", [
    ("deit_tiny", "mse", ("per_tensor", "symmetric")),
    ("mobilevit", "percentile", ("per_channel", "asymmetric")),  # MobileViT-S's core: per_channel + asymmetric
])
def test_core_and_one_factor_variants_on_the_models_chosen_calibration(model, calibration, core):
    rows = {r["factor_varied"]: r for r in dm.quantized_entries() if r["model_name"] == model}
    other = {"per_tensor": "per_channel", "per_channel": "per_tensor", "symmetric": "asymmetric",
             "asymmetric": "symmetric"}

    assert set(rows) == {"core", "granularity", "symmetry"}
    assert all(r["calibration_algorithm"] == calibration for r in rows.values())
    assert (rows["core"]["granularity"], rows["core"]["symmetry"]) == core
    # each variant changes exactly ONE factor with respect to the core
    assert (rows["granularity"]["granularity"], rows["granularity"]["symmetry"]) == (other[core[0]], core[1])
    assert (rows["symmetry"]["granularity"], rows["symmetry"]["symmetry"]) == (core[0], other[core[1]])
    assert all(r["bitwidth"] == "int8" and r["output_quant"] and r["fold_bn"] and r["calib_set_size"] == 128
               for r in rows.values())


def test_run_ids_name_the_configuration_not_the_role():
    for r in dm.quantized_entries():
        assert r["run_id"] == f"vit_{r['model_name']}_deploy_{r['granularity']}_{r['symmetry']}_int8"
    assert [r["run_id"] for r in dm.quantized_entries() if r["factor_varied"] == "core"] == [
        "vit_deit_tiny_deploy_per_tensor_symmetric_int8", "vit_mobilevit_deploy_per_channel_asymmetric_int8"]


def test_artifacts_live_apart_from_the_ablation_artifacts():
    paths = [r["path"] for r in deployment.plan() if not r["entry"].get("fallback_reference")]
    assert len(paths) == 8 and all(p.parent.name == "vit_deployment" for p in paths)
    assert all(p.name.endswith("__rep1.onnx") for p in paths)


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(deployment, "DEPLOYMENT_DIR", tmp_path / "vit_deployment")
    monkeypatch.setattr(deployment, "build_artifact", lambda record: pytest.fail("dry-run built an artifact"))
    monkeypatch.setattr(deployment, "check_fallback_reference", lambda record: None)

    assert deployment.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "run manifest: 10 rows (8 deployment + 2 fallback references)" in out
    assert "perf_per_watt.py included" in out and "corrected perf_per_watt: OK" in out
    assert not (tmp_path / "vit_deployment").exists()


def test_package_check_rejects_code_with_the_old_perf_per_watt(tmp_path, monkeypatch):
    import verify_edge_bundle as veb

    bench_files, dep_files = deployment._code_files()
    copies = []
    for f in bench_files + dep_files:
        dest = tmp_path / f.relative_to(veb.ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copies.append(dest)
    old = tmp_path / "benchmarking_ondevice/power_logging/perf_per_watt.py"
    old.write_text(old.read_text().replace("return TOKENS_PER_REPETITION / energy_per_inference",
                                           "return TOKENS_PER_REPETITION / energy_net_j"))
    monkeypatch.setattr(veb, "ROOT", tmp_path)

    ok, _out, err = veb.check_import_closure(copies)

    assert not ok and "old perf_per_watt formula" in err


def test_gate_deployment_rows_are_the_six_quantized_ones_with_their_own_fold_bn(monkeypatch, tmp_path):
    from evaluation_offline.vit_accuracy import accuracy as acc
    from export.vit_to_onnx import export as exp
    from quantization.vit_brevitas import quantize

    built, exported = [], []
    monkeypatch.setattr(quantize, "build_quantized_model",
                        lambda entry, repeat_id, fold_bn=False, remove_bn=False, output_quant=False:
                        built.append((fold_bn, output_quant)) or ("model", (1, 3, 224, 224)))
    monkeypatch.setattr(gate, "count_output_quant", lambda model: (50, 50))
    monkeypatch.setattr(gate, "outputs_to_q", lambda path: (2, 2))
    gelu_flags = []
    monkeypatch.setattr(exp, "export_to_onnx",
                        lambda m, s, r, rep, out_path=None, single_gelu=False:
                        exported.append(out_path) or gelu_flags.append((r, single_gelu)) or out_path)
    monkeypatch.setattr(acc, "compute_topk_accuracy_for_file", lambda model_name, path: (70.0, 90.0))

    entries = gate.select_entries(False, deployment=True)
    gate.run_gate(entries, [1], tmp_path, {})

    assert [e["run_id"] for e in entries] == [e["run_id"] for e in dm.quantized_entries()]
    assert len(entries) == 6 and built == [(True, True)] * 6
    assert {r: g for r, g in gelu_flags} == {e["run_id"]: e["model_name"] == "deit_tiny" for e in entries}
    assert all(p.name.endswith("__rep1__outq_bnfold.onnx") for p in exported)


def test_gate_deployment_run_id_selects_within_the_deployment_set():
    rows = gate.select_entries(False, run_ids=["vit_mobilevit_deploy_per_channel_asymmetric_int8"], deployment=True)
    assert [r["run_id"] for r in rows] == ["vit_mobilevit_deploy_per_channel_asymmetric_int8"]
    with pytest.raises(SystemExit):
        gate.select_entries(False, run_ids=["vit_mobilevit_deploy_fp32"], deployment=True)


def test_gate_deployment_refuses_global_bn_flags():
    with pytest.raises(SystemExit):
        gate.main(["--deployment", "--fold-bn"])


def test_deployment_builds_quantized_rows_with_output_quant_passed_explicitly(monkeypatch, tmp_path):
    from export.vit_to_onnx import export as exp
    from quantization.vit_brevitas import quantize

    calls = []
    monkeypatch.setattr(quantize, "build_quantized_model",
                        lambda entry, repeat_id, fold_bn=False, remove_bn=False, output_quant=False:
                        calls.append((entry["run_id"], repeat_id, fold_bn, output_quant)) or ("model", (1, 3, 224, 224)))
    monkeypatch.setattr(gate, "count_output_quant", lambda model: (50, 50))
    exported_with = []
    monkeypatch.setattr(exp, "export_to_onnx",
                        lambda m, s, r, rep, out_path=None, single_gelu=False: exported_with.append(single_gelu) or out_path)

    record = next(r for r in deployment.plan() if r["run_id"] == "vit_deit_tiny_deploy_per_tensor_symmetric_int8")
    deployment.build_artifact(record)

    assert calls == [("vit_deit_tiny_deploy_per_tensor_symmetric_int8", 1, True, True)]
    assert exported_with == [True]  # DeiT-Tiny's quantized rows export the GELU as one node


def test_the_package_never_ships_devices_yaml(monkeypatch, tmp_path):
    """devices.yaml on each target holds hand-filled entries this repo lacks."""
    import verify_edge_bundle as veb

    written = {}
    monkeypatch.setattr(deployment, "check_package", lambda: True)
    monkeypatch.setattr(deployment, "build_artifact", lambda record: None)
    monkeypatch.setattr(veb, "_write_package", lambda name, records, **kw: written.update(kw, name=name))

    monkeypatch.setattr(deployment, "check_fallback_reference", lambda record: None)
    assert deployment.main([]) == 0
    assert written["name"] == "code_vit" and written["include_devices_example"] is True


def test_filelists_carry_the_example_only_in_the_code_package_and_never_devices_yaml(monkeypatch, tmp_path):
    import verify_edge_bundle as veb

    monkeypatch.setattr(veb, "OUTPUT_DIR", tmp_path)
    for name, example in (("code_vit", True), ("llm_x", False)):
        path, _manifest = veb._write_package(name, [], include_devices_example=example, bench_files=[], dep_files=[])
        lines = path.read_text().split()
        assert ("configs/devices.example.yaml" in lines) is example
        assert "configs/devices.yaml" not in lines


def test_devices_example_is_valid_and_follows_the_schema():
    import yaml

    entries = yaml.safe_load(deployment.DEVICES_EXAMPLE.read_text())
    required = {"device_id", "display_name", "architecture", "backend", "power_mode_w", "ram_gb", "cpu_cores",
                "power_measurement", "added_by"}
    assert all(required <= set(e) for e in entries)
    assert all("n_gpu_layers" in e for e in entries if e["backend"] == "cuda")
    assert [e.get("min_uptime_s", 0) for e in entries] == [600, 600, 0]
    assert entries[-1]["device_id"] == "rpi_cm4"


def test_the_code_never_reads_the_devices_example():
    from benchmarking_ondevice.runner_llm import devices as llm_devices
    from benchmarking_ondevice.runner_vit import devices as vit_devices

    assert vit_devices.DEVICES_YAML_PATH.name == llm_devices.DEVICES_YAML_PATH.name == "devices.yaml"


def test_no_package_builder_ships_the_real_devices_yaml():
    import inspect

    import verify_edge_bundle as veb

    for module in (veb, deployment):
        assert "include_devices_yaml" not in inspect.getsource(module)
        assert '"configs/devices.yaml"' not in inspect.getsource(module)


def test_there_is_one_vit_package_and_verify_edge_bundle_no_longer_writes_it():
    import inspect

    import verify_edge_bundle as veb

    assert deployment.PACKAGE_NAME == "code_vit"
    assert '_write_package("code_vit"' not in inspect.getsource(veb.main)


# --------------------------------------------------------------------------
# the 2 fallback references (ablation artifacts, input-only)
# --------------------------------------------------------------------------

DEPLOYMENT_ARTIFACTS = sorted(
    f"artifacts/vit_deployment/vit_{m}_deploy_{k}__rep1.onnx"
    for m, ks in (("deit_tiny", ("fp32", "per_tensor_symmetric_int8", "per_channel_symmetric_int8",
                                 "per_tensor_asymmetric_int8")),
                  ("mobilevit", ("fp32", "per_channel_asymmetric_int8", "per_tensor_asymmetric_int8",
                                 "per_channel_symmetric_int8")))
    for k in ks
)
REFERENCE_ARTIFACTS = [
    "artifacts/vit/vit_deit_tiny_phase2_calibration_algorithm_mse_int8__rep1.onnx",
    "artifacts/vit/vit_mobilevit_phase2_calibration_algorithm_percentile_int8__rep1.onnx",
]


def test_the_package_carries_exactly_these_8_deployment_and_2_ablation_artifacts(monkeypatch, tmp_path):
    import verify_edge_bundle as veb

    monkeypatch.setattr(veb, "OUTPUT_DIR", tmp_path)
    filelist, manifest = veb._write_package("code_vit", deployment.plan(), include_devices_example=True,
                                            bench_files=[], dep_files=[])
    lines = filelist.read_text().split()

    assert sorted(l for l in lines if l.startswith("artifacts/vit_deployment/")) == DEPLOYMENT_ARTIFACTS
    assert sorted(l for l in lines if l.startswith("artifacts/vit/")) == REFERENCE_ARTIFACTS
    assert [l for l in lines if l.startswith("artifacts/")] == \
        [l for l in lines if l.startswith(("artifacts/vit_deployment/", "artifacts/vit/"))]
    rows = manifest.read_text().splitlines()
    assert len(rows) == 1 + 10 and rows[0] == "run_id,repeat_id,artifact_path"


def test_references_keep_their_ablation_run_ids_and_labels():
    refs = dm.fallback_references()

    assert [r["run_id"] for r in refs] == ["vit_deit_tiny_phase2_calibration_algorithm_mse_int8",
                                           "vit_mobilevit_phase2_calibration_algorithm_percentile_int8"]
    assert all(r["phase"] == "phase2" and r["factor_varied"] == "calibration_algorithm" for r in refs)
    assert [r["factor_level"] for r in refs] == ["mse", "percentile"]
    assert all(r["output_quant"] is False and r["fallback_reference"] for r in refs)


def test_references_are_never_built_and_never_gated():
    record = next(r for r in deployment.plan() if r["entry"].get("fallback_reference"))
    with pytest.raises(ValueError, match="never built"):
        deployment.build_artifact(record)
    assert all(not e.get("fallback_reference") for e in dm.quantized_entries())


def test_a_reference_with_output_q_or_missing_is_rejected(monkeypatch, tmp_path):
    record = {"run_id": "r", "path": tmp_path / "r.onnx"}
    with pytest.raises(FileNotFoundError):
        deployment.check_fallback_reference(record)

    record["path"].write_text("x")
    monkeypatch.setattr(gate, "outputs_to_q", lambda path: (3, 50))
    with pytest.raises(RuntimeError, match="not input-only"):
        deployment.check_fallback_reference(record)
    monkeypatch.setattr(gate, "outputs_to_q", lambda path: (0, 50))
    deployment.check_fallback_reference(record)


def test_package_check_rejects_code_with_the_old_vit_perf_per_watt(tmp_path, monkeypatch):
    import verify_edge_bundle as veb

    bench_files, dep_files = deployment._code_files()
    copies = []
    for f in bench_files + dep_files:
        dest = tmp_path / f.relative_to(veb.ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copies.append(dest)
    old = tmp_path / "benchmarking_ondevice/power_logging/perf_per_watt.py"
    old.write_text(old.read_text().replace("return num_inferences_measured / energy_net_j", "return 999.0 / 1.0"))
    monkeypatch.setattr(veb, "ROOT", tmp_path)

    ok, _out, err = veb.check_import_closure(copies)

    assert not ok and "old ViT perf_per_watt formula" in err
