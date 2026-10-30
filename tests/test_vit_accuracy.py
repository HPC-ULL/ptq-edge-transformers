"""Tests for evaluation_offline/vit_accuracy/.

Every ViT-branch manifest row is stochastic (calibration-set sampling is
always random): no deterministic/unsuffixed exception like the LLM branch,
so every row always has NUM_REPEATS=3 independent .onnx artifacts and 3
independent accuracy evaluations.

Dry-run and command-shape tests never touch real onnxruntime/timm/PIL
models or datasets. The happy-path and assert-validation tests fake
`compute_fn` (standing in for the real ONNX Runtime top-1/top-5 loop) so
they don't need real external resources either.
"""

from __future__ import annotations

import csv
from unittest.mock import Mock

import pytest

from configs import vit_manifest, vit_phase3_manifest
from evaluation_offline.vit_accuracy import accuracy, cli, results_io
from export.vit_to_onnx import export as export_module

_RUN_ID = "vit_deit_tiny_phase1_baseline_default_int8"


def _entry(run_id: str) -> dict:
    return next(e for e in vit_manifest.generate() if e["run_id"] == run_id)


# --------------------------------------------------------------------------
# manifest shape
# --------------------------------------------------------------------------

def test_manifest_has_24_rows():
    assert len(vit_manifest.generate()) == 24


# --------------------------------------------------------------------------
# evaluate_accuracy: dry-run / idempotency / happy path / invalid-value guard
# --------------------------------------------------------------------------

def test_evaluate_accuracy_dry_run_never_calls_compute_fn(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(results_io, "VIT_ACCURACY_RESULTS_CSV", tmp_path / "vit_accuracy_offline.csv")
    entry = _entry(_RUN_ID)
    compute_fn = Mock()

    result = accuracy.evaluate_accuracy(entry, 1, dry_run=True, force=False, compute_fn=compute_fn)

    assert result is None
    compute_fn.assert_not_called()
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert entry["run_id"] in out


def test_evaluate_accuracy_skips_when_already_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "VIT_ACCURACY_RESULTS_CSV", tmp_path / "vit_accuracy_offline.csv")
    entry = _entry(_RUN_ID)
    results_io.append_result(entry, 1, "top1_accuracy", 55.0)
    results_io.append_result(entry, 1, "top5_accuracy", 80.0)

    compute_fn = Mock()
    result = accuracy.evaluate_accuracy(entry, 1, dry_run=False, force=False, compute_fn=compute_fn)

    assert result is None
    compute_fn.assert_not_called()


def test_evaluate_accuracy_reruns_but_only_writes_the_missing_metric(tmp_path, monkeypatch):
    csv_path = tmp_path / "vit_accuracy_offline.csv"
    monkeypatch.setattr(results_io, "VIT_ACCURACY_RESULTS_CSV", csv_path)
    entry = _entry(_RUN_ID)
    results_io.append_result(entry, 1, "top1_accuracy", 55.0)  # top5 missing

    compute_fn = Mock(return_value=(60.0, 85.0))
    result = accuracy.evaluate_accuracy(entry, 1, dry_run=False, force=False, compute_fn=compute_fn)

    compute_fn.assert_called_once()
    assert result == (60.0, 85.0)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    # the original top1 row is untouched (still 55.0), not duplicated
    assert len(rows) == 2
    top1_rows = [r for r in rows if r["accuracy_metric_name"] == "top1_accuracy"]
    assert len(top1_rows) == 1
    assert top1_rows[0]["accuracy_value"] == "55.0"


def test_evaluate_accuracy_force_reruns_even_if_already_recorded(tmp_path, monkeypatch):
    csv_path = tmp_path / "vit_accuracy_offline.csv"
    monkeypatch.setattr(results_io, "VIT_ACCURACY_RESULTS_CSV", csv_path)
    entry = _entry(_RUN_ID)
    results_io.append_result(entry, 1, "top1_accuracy", 55.0)
    results_io.append_result(entry, 1, "top5_accuracy", 80.0)

    compute_fn = Mock(return_value=(60.0, 85.0))
    result = accuracy.evaluate_accuracy(entry, 1, dry_run=False, force=True, compute_fn=compute_fn)

    compute_fn.assert_called_once()
    assert result == (60.0, 85.0)

    # Regression check for a real bug: append_result used to always append,
    # so --force recomputing an already-recorded result left BOTH the old
    # and new values in the CSV under the identical result_id, silently
    # corrupting any aggregate (mean/stdev) computed over all matching rows.
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # exactly one row per metric, not four
    values_by_metric = {r["accuracy_metric_name"]: r["accuracy_value"] for r in rows}
    assert values_by_metric == {"top1_accuracy": "60.0", "top5_accuracy": "85.0"}


def test_evaluate_accuracy_happy_path_writes_both_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "VIT_ACCURACY_RESULTS_CSV", tmp_path / "vit_accuracy_offline.csv")
    entry = _entry(_RUN_ID)

    compute_fn = Mock(return_value=(61.25, 84.5))
    value = accuracy.evaluate_accuracy(entry, 2, dry_run=False, force=False, compute_fn=compute_fn)

    assert value == (61.25, 84.5)
    compute_fn.assert_called_once_with(entry, 2)
    assert results_io.has_result(entry["run_id"], 2, "top1_accuracy")
    assert results_io.has_result(entry["run_id"], 2, "top5_accuracy")


def test_appended_result_rows_have_device_offline(tmp_path, monkeypatch):
    csv_path = tmp_path / "vit_accuracy_offline.csv"
    monkeypatch.setattr(results_io, "VIT_ACCURACY_RESULTS_CSV", csv_path)
    entry = _entry(_RUN_ID)

    compute_fn = Mock(return_value=(61.25, 84.5))
    accuracy.evaluate_accuracy(entry, 1, dry_run=False, force=False, compute_fn=compute_fn)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(r["device"] == "offline" for r in rows)
    assert {r["accuracy_metric_name"] for r in rows} == {"top1_accuracy", "top5_accuracy"}
    assert all(r["repeat_id"] == "1" for r in rows)


def test_validate_accuracy_accepts_valid_range():
    assert accuracy.validate_accuracy(50.0, "top1_accuracy") == 50.0
    assert accuracy.validate_accuracy(0.0, "top1_accuracy") == 0.0
    assert accuracy.validate_accuracy(100.0, "top1_accuracy") == 100.0


@pytest.mark.parametrize("bad_value", [-0.1, 100.1, float("inf"), float("-inf"), float("nan")])
def test_validate_accuracy_rejects_out_of_range_or_non_finite(bad_value):
    with pytest.raises(AssertionError):
        accuracy.validate_accuracy(bad_value, "top1_accuracy")


def test_evaluate_accuracy_raises_and_never_writes_result_when_value_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "VIT_ACCURACY_RESULTS_CSV", tmp_path / "vit_accuracy_offline.csv")
    entry = _entry(_RUN_ID)

    # A value that "computes" successfully but is nonsensical (e.g. a
    # broken preprocessing/label mapping) -- must fail loudly via the
    # assert, not get written.
    compute_fn = Mock(return_value=(150.0, 85.0))
    append_result = Mock()
    monkeypatch.setattr(results_io, "append_result", append_result)

    with pytest.raises(AssertionError):
        accuracy.evaluate_accuracy(entry, 1, dry_run=False, force=False, compute_fn=compute_fn)

    append_result.assert_not_called()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_dry_run_run_id_all_repeats(tmp_path, monkeypatch, capsys):
    # evaluate_accuracy() checks results_io.has_result() BEFORE the
    # dry_run branch, so without isolating the results CSV this would hit
    # the "already recorded" skip path once real results exist for
    # _RUN_ID (which they do, post the real --all evaluation run) instead
    # of printing the dry-run line this test expects.
    monkeypatch.setattr(results_io, "VIT_ACCURACY_RESULTS_CSV", tmp_path / "vit_accuracy_offline.csv")
    rc = cli.main(["--run-id", _RUN_ID, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    for repeat_id in (1, 2, 3):
        assert f"{_RUN_ID} rep{repeat_id}: accuracy:" in out


def test_cli_dry_run_run_id_single_repeat(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(results_io, "VIT_ACCURACY_RESULTS_CSV", tmp_path / "vit_accuracy_offline.csv")
    rc = cli.main(["--run-id", _RUN_ID, "--repeat-id", "2", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"{_RUN_ID} rep2: accuracy:" in out
    assert f"{_RUN_ID} rep1: accuracy:" not in out
    assert f"{_RUN_ID} rep3: accuracy:" not in out


def test_cli_repeat_id_out_of_range_errors():
    with pytest.raises(SystemExit):
        cli.main(["--run-id", _RUN_ID, "--repeat-id", "4", "--dry-run"])


def test_cli_all_with_repeat_id_errors():
    with pytest.raises(SystemExit):
        cli.main(["--all", "--repeat-id", "1"])


def test_cli_unknown_run_id_errors():
    with pytest.raises(SystemExit):
        cli.main(["--run-id", "not_a_real_run_id", "--dry-run"])


def test_cli_dry_run_never_requires_artifact_to_exist(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    rc = cli.main(["--run-id", _RUN_ID, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[warning]" not in out


def test_cli_run_id_real_run_errors_when_artifact_missing(tmp_path, monkeypatch):
    # --run-id is a single, explicit request: unlike --all, a missing
    # artifact must fail loudly (parser.error / SystemExit), not skip
    # silently -- this is a single-request-vs-batch distinction, not a
    # deterministic-vs-stochastic one (ViT has no deterministic rows).
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["--run-id", _RUN_ID])


def test_cli_run_id_real_run_errors_when_only_one_of_several_repeats_missing(tmp_path, monkeypatch):
    # Rep 1 exists, reps 2 and 3 don't -- requesting all 3 (no --repeat-id)
    # must still fail loudly rather than silently evaluating just rep 1.
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    path = export_module.onnx_path(_RUN_ID, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake onnx")

    with pytest.raises(SystemExit):
        cli.main(["--run-id", _RUN_ID])


def test_cli_run_id_real_run_with_repeat_id_succeeds_when_that_repeat_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    monkeypatch.setattr(results_io, "VIT_ACCURACY_RESULTS_CSV", tmp_path / "vit_accuracy_offline.csv")
    path = export_module.onnx_path(_RUN_ID, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake onnx")

    evaluate = Mock()
    monkeypatch.setattr(cli.accuracy, "evaluate_accuracy", evaluate)

    rc = cli.main(["--run-id", _RUN_ID, "--repeat-id", "2"])

    assert rc == 0
    evaluate.assert_called_once()
    assert evaluate.call_args.args[1] == 2


def test_cli_all_with_no_artifacts_skips_everything_and_reports_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)

    rc = cli.main(["--all"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "72/72" in out


def test_cli_all_evaluates_only_existing_artifacts_and_lists_the_rest_as_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    monkeypatch.setattr(results_io, "VIT_ACCURACY_RESULTS_CSV", tmp_path / "vit_accuracy_offline.csv")

    entries = vit_manifest.generate()
    present = [(entries[0], 1), (entries[0], 2)]
    for entry, repeat_id in present:
        path = export_module.onnx_path(entry["run_id"], repeat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake onnx")

    evaluate = Mock()
    monkeypatch.setattr(cli.accuracy, "evaluate_accuracy", evaluate)

    rc = cli.main(["--all", "--force"])

    assert rc == 0
    assert evaluate.call_count == len(present)
    called_pairs = {(call.args[0]["run_id"], call.args[1]) for call in evaluate.call_args_list}
    expected_pairs = {(e["run_id"], r) for e, r in present}
    assert called_pairs == expected_pairs

    out = capsys.readouterr().out
    assert f"{72 - len(present)}/72" in out


# --------------------------------------------------------------------------
# --phase3: reads configs/vit_phase3_manifest.yaml instead of Phase 1-2.
# evaluation_offline never repeats per artifact -- job count here is
# configurations x NUM_REPEATS, same formula as quantization.
# --------------------------------------------------------------------------

def _vit_accuracy_row(model_name, calibration_algorithm, granularity, symmetry, calib_set_size, accuracy_value, repeat_id):
    return {
        "model_name": model_name,
        "accuracy_metric_name": "top1_accuracy",
        "calibration_algorithm": calibration_algorithm,
        "granularity": granularity,
        "symmetry": symmetry,
        "calib_set_size": str(calib_set_size),
        "bitwidth": "int8",
        "accuracy_value": str(accuracy_value),
        "repeat_id": str(repeat_id),
    }


def _synthetic_vit_accuracy_rows_no_dedup():
    """Same fixture shape as quantization/vit_brevitas's --phase3 test: all
    3 models have a factor winner (granularity=per_channel) -> 12 Phase 3
    entries, no dedup -> 12 x NUM_REPEATS(3) = 36 evaluation jobs."""
    baseline_cfg = dict(calibration_algorithm="minmax", granularity="per_tensor", symmetry="symmetric", calib_set_size=128)
    rows = []
    for model in ("deit_tiny", "mobilevit", "efficientvit_b1"):
        for rep in (1, 2, 3):
            rows.append(_vit_accuracy_row(model, **baseline_cfg, accuracy_value=73.0, repeat_id=rep))
            rows.append(_vit_accuracy_row(model, **{**baseline_cfg, "calibration_algorithm": "mse"}, accuracy_value=73.0, repeat_id=rep))
            rows.append(_vit_accuracy_row(model, **{**baseline_cfg, "granularity": "per_channel"}, accuracy_value=80.0, repeat_id=rep))
            rows.append(_vit_accuracy_row(model, **{**baseline_cfg, "symmetry": "asymmetric"}, accuracy_value=72.9, repeat_id=rep))
            rows.append(_vit_accuracy_row(model, **{**baseline_cfg, "calib_set_size": 32}, accuracy_value=72.8, repeat_id=rep))
            rows.append(_vit_accuracy_row(model, **{**baseline_cfg, "calib_set_size": 512}, accuracy_value=72.9, repeat_id=rep))
    return rows


def test_cli_phase3_all_evaluates_36_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    entries, dedup_notes = vit_phase3_manifest.generate(rows=_synthetic_vit_accuracy_rows_no_dedup())
    assert len(entries) == 12 and not dedup_notes  # sanity check on the fixture itself
    monkeypatch.setattr(cli.vit_phase3_manifest, "generate", lambda: (entries, dedup_notes))

    for entry in entries:
        for repeat_id in (1, 2, 3):
            path = export_module.onnx_path(entry["run_id"], repeat_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake onnx")

    evaluate = Mock()
    monkeypatch.setattr(cli.accuracy, "evaluate_accuracy", evaluate)

    rc = cli.main(["--all", "--phase3"])

    assert rc == 0
    assert evaluate.call_count == 36


def test_cli_phase3_flag_reads_phase3_manifest_not_phase1_2(tmp_path, monkeypatch):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    entries, _ = vit_phase3_manifest.generate(rows=_synthetic_vit_accuracy_rows_no_dedup())
    monkeypatch.setattr(cli.vit_phase3_manifest, "generate", lambda: (entries, {}))
    phase3_run_id = entries[0]["run_id"]

    rc = cli.main(["--run-id", phase3_run_id, "--phase3", "--dry-run"])
    assert rc == 0

    with pytest.raises(SystemExit):
        cli.main(["--run-id", phase3_run_id, "--dry-run"])  # no --phase3 -- unknown run_id in Phase 1-2 manifest
