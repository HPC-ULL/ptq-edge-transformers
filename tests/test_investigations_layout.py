"""Where the ViT investigations write (quantization/vit_brevitas/paths.py):
one folder per experiment kind, one run.log per run, nothing loose in the root."""

from __future__ import annotations

import sys

import pytest

from quantization.vit_brevitas import paths
from quantization.vit_brevitas.investigations import gate_output_quant as gate
from quantization.vit_brevitas.investigations import per_image_predictions as pip


def test_tee_output_prints_and_logs_stdout_and_stderr_and_restores_streams(tmp_path, capsys):
    log = tmp_path / "sub" / "run.log"
    before = (sys.stdout, sys.stderr)

    with paths.tee_output(log):
        print("to stdout")
        print("to stderr", file=sys.stderr)

    out, err = capsys.readouterr()
    assert "to stdout" in out and "to stderr" in err
    assert log.read_text() == "to stdout\nto stderr\n"
    assert (sys.stdout, sys.stderr) == before


def test_tee_output_restores_streams_and_keeps_the_log_when_the_run_crashes(tmp_path):
    log = tmp_path / "run.log"
    before = (sys.stdout, sys.stderr)

    with pytest.raises(RuntimeError):
        with paths.tee_output(log):
            print("before the crash")
            raise RuntimeError("boom")

    assert (sys.stdout, sys.stderr) == before and "before the crash" in log.read_text()


def test_investigation_dirs_live_under_results_investigations_vit():
    root = paths.INVESTIGATIONS_ROOT
    assert root.parts[-3:] == ("results", "investigations", "vit")
    assert paths.investigation_dir(paths.FUSION_PROBES, "models") == root / "fusion_probes" / "models"
    assert "raw" not in root.parts and "artifacts" not in root.parts


@pytest.mark.parametrize("argv, group, tag", [
    (["--deployment", "--run-id", "vit_mobilevit_deploy_per_channel_asymmetric_int8"], "deployment_gate", "deployment"),
    (["--deployment", "--tag", "batch2"], "deployment_gate", "batch2"),
    (["--phase3", "--fold-bn"], "output_quant_gate", "phase3_bnfold"),
    (["--with-recovery", "--tag", "first_gate"], "output_quant_gate", "first_gate"),
    ([], "output_quant_gate", "ablation_rows"),
])
def test_gate_default_output_is_a_run_folder_with_its_own_log(monkeypatch, tmp_path, argv, group, tag):
    monkeypatch.setattr(paths, "INVESTIGATIONS_ROOT", tmp_path)
    seen = {}
    monkeypatch.setattr(gate, "select_entries", lambda *a, **k: [])
    monkeypatch.setattr(gate, "_run", lambda args, entries, kind: seen.update(out=args.out_dir) or print("hello") or 0)

    assert gate.main(argv) == 0

    assert seen["out"] == tmp_path / group / tag
    assert (tmp_path / group / tag / "run.log").read_text() == "hello\n"


def test_gate_explicit_out_dir_still_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "select_entries", lambda *a, **k: [])
    monkeypatch.setattr(gate, "_run", lambda args, entries, kind: 0)

    assert gate.main(["--out-dir", str(tmp_path / "x")]) == 0
    assert (tmp_path / "x" / "run.log").exists()


def test_per_image_reads_the_gates_from_their_run_folders():
    dep = [j for j in pip.plan() if j["group"] == "deployment"]
    root = str(paths.INVESTIGATIONS_ROOT)

    assert all(str(j["artifact"]).startswith(root) for j in dep)
    refs = {j["ref"][1] for j in dep}
    assert all(r.startswith(root) for r in refs)
    assert str(paths.INVESTIGATIONS_ROOT / "deployment_gate" / "main" / "gate_deploy.log") in refs
    assert pip.investigation_dir(paths.PER_IMAGE).name == "per_image_predictions"


def test_the_fusion_probe_lives_in_scripts_and_defaults_to_its_folder(tmp_path):
    import scripts.probe_int8_kernels as probe

    assert probe.PROBES_DIR.as_posix() == "results/investigations/vit/fusion_probes"
    assert probe.PROBES_DIR == paths.investigation_dir(paths.FUSION_PROBES).relative_to(paths.PROJECT_ROOT)


def test_the_fusion_probe_ships_with_the_edge_code():
    from quantization.vit_brevitas import deployment

    bench_files, _ = deployment._code_files()
    assert "probe_int8_kernels.py" in {p.name for p in bench_files if p.parent.name == "scripts"}


def test_gate_single_gelu_reaches_the_export_of_ablation_rows_and_is_refused_with_deployment(monkeypatch, tmp_path):
    from evaluation_offline.vit_accuracy import accuracy as acc
    from export.vit_to_onnx import export as exp
    from quantization.vit_brevitas import quantize

    flags = []
    monkeypatch.setattr(quantize, "build_quantized_model",
                        lambda entry, repeat_id, fold_bn=False, remove_bn=False, output_quant=False: ("m", (1, 3, 8, 8)))
    monkeypatch.setattr(gate, "count_output_quant", lambda model: (50, 50))
    monkeypatch.setattr(gate, "outputs_to_q", lambda path: (2, 2))
    monkeypatch.setattr(exp, "export_to_onnx",
                        lambda m, s, r, rep, out_path=None, single_gelu=False: flags.append(single_gelu) or out_path)
    monkeypatch.setattr(acc, "compute_topk_accuracy_for_file", lambda model_name, path: (70.0, 90.0))

    entries = gate.select_entries(False, run_ids=["vit_deit_tiny_phase1_baseline_default_int8"])
    gate.run_gate(entries, [1], tmp_path, {}, fold_bn=True, single_gelu=True)
    assert flags == [True]

    with pytest.raises(SystemExit):
        gate.main(["--deployment", "--single-gelu"])
