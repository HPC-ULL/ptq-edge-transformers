"""Dry-run and idempotency tests for quantization/vit_brevitas/ + export/vit_to_onnx/.

Every ViT-branch manifest row is stochastic (calibration-set sampling is
always random, calibration_protocol_vit.md Section 6): 24 manifest rows x
NUM_REPEATS=3 = 72 (run_id, repeat_id) jobs, no deterministic exception
like the LLM branch.

Dry-run tests never touch real timm/Brevitas/onnxruntime: run_step's
dry-run branch returns before calling the injected pipeline_fn/action.
Several tests deliberately use the *default* pipeline_fn (not a mock) under
--dry-run -- since torch/timm/brevitas aren't installed in this dev
sandbox, that default would raise ModuleNotFoundError if dry-run ever
reached the real pipeline, so those tests passing is itself proof it
doesn't.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from configs import vit_manifest, vit_phase3_manifest
from export.vit_to_onnx import export as export_module
from quantization.vit_brevitas import cli

_BASELINE_RUN_ID = "vit_deit_tiny_phase1_baseline_default_int8"


def _entries():
    return vit_manifest.generate()


def _entry(run_id: str) -> dict:
    return next(e for e in _entries() if e["run_id"] == run_id)


# --------------------------------------------------------------------------
# manifest / job expansion
# --------------------------------------------------------------------------

def test_manifest_has_24_rows():
    assert len(_entries()) == 24


def test_iter_jobs_expands_every_row_to_3_repeats_72_total():
    jobs = list(cli.iter_jobs(_entries()))
    assert len(jobs) == 72
    for entry, repeat_id in jobs:
        assert repeat_id in (1, 2, 3)
    # every entry appears exactly 3 times
    run_ids = [e["run_id"] for e, _ in jobs]
    assert len(run_ids) == len(set(run_ids)) * 3


# --------------------------------------------------------------------------
# process_job: dry-run / idempotency / force
# --------------------------------------------------------------------------

def test_process_job_dry_run_with_default_pipeline_never_touches_real_work(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    entry = _entry(_BASELINE_RUN_ID)

    cli.process_job(entry, 1, dry_run=True, force=False)  # default pipeline_fn, on purpose

    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert entry["run_id"] in out
    assert "rep1" in out


def test_process_job_dry_run_never_calls_injected_pipeline_fn(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    entry = _entry(_BASELINE_RUN_ID)
    pipeline_fn = Mock()

    cli.process_job(entry, 1, dry_run=True, force=False, pipeline_fn=pipeline_fn)

    pipeline_fn.assert_not_called()


def test_process_job_skips_when_artifact_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    entry = _entry(_BASELINE_RUN_ID)
    out_path = export_module.onnx_path(entry["run_id"], 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"fake onnx")

    pipeline_fn = Mock()
    cli.process_job(entry, 1, dry_run=False, force=False, pipeline_fn=pipeline_fn)

    pipeline_fn.assert_not_called()


def test_process_job_runs_when_artifact_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    entry = _entry(_BASELINE_RUN_ID)

    pipeline_fn = Mock()
    cli.process_job(entry, 2, dry_run=False, force=False, pipeline_fn=pipeline_fn)

    pipeline_fn.assert_called_once_with(entry, 2)


def test_process_job_force_reruns_even_if_artifact_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    entry = _entry(_BASELINE_RUN_ID)
    out_path = export_module.onnx_path(entry["run_id"], 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"stale")

    pipeline_fn = Mock()
    cli.process_job(entry, 1, dry_run=False, force=True, pipeline_fn=pipeline_fn)

    pipeline_fn.assert_called_once_with(entry, 1)


def test_process_job_idempotency_is_per_repeat_not_per_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    entry = _entry(_BASELINE_RUN_ID)

    rep1_path = export_module.onnx_path(entry["run_id"], 1)
    rep1_path.parent.mkdir(parents=True, exist_ok=True)
    rep1_path.write_bytes(b"already there")

    pipeline_fn = Mock()
    cli.process_job(entry, 1, dry_run=False, force=False, pipeline_fn=pipeline_fn)
    pipeline_fn.assert_not_called()

    cli.process_job(entry, 2, dry_run=False, force=False, pipeline_fn=pipeline_fn)
    pipeline_fn.assert_called_once_with(entry, 2)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_all_dry_run_prints_72_quantize_export_lines(tmp_path, monkeypatch, capsys):
    # Without this, dry-run would hit run_step's output_exists branch
    # instead of the dry_run branch for any row whose real .onnx artifact
    # already exists on disk (e.g. after a completed --all run) -- exactly
    # the isolation the test_process_job_* tests above already apply.
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    rc = cli.main(["--all", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "quantize+export" in l]
    assert len(lines) == 72


def test_cli_run_id_dry_run_prints_3_lines_one_per_repeat(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    rc = cli.main(["--run-id", _BASELINE_RUN_ID, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "quantize+export" in l]
    assert len(lines) == 3
    for repeat_id in (1, 2, 3):
        assert f"{_BASELINE_RUN_ID} rep{repeat_id}:" in out


def test_cli_dry_run_describe_includes_manifest_fields(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    rc = cli.main(["--run-id", _BASELINE_RUN_ID, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "model=deit_tiny" in out
    assert "algo=minmax" in out
    assert "granularity=per_tensor" in out
    assert "symmetry=symmetric" in out
    assert "calib_set_size=128" in out
    assert "bitwidth=int8" in out


def test_cli_dry_run_variant_row_shows_its_own_factor_value(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    per_channel_run_id = next(
        e["run_id"] for e in _entries() if e["factor_varied"] == "granularity" and e["factor_level"] == "per_channel"
    )
    rc = cli.main(["--run-id", per_channel_run_id, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "granularity=per_channel" in out


def test_cli_unknown_run_id_errors():
    with pytest.raises(SystemExit):
        cli.main(["--run-id", "not_a_real_run_id", "--dry-run"])


def test_cli_all_skips_notimplementederror_rows_and_keeps_going(monkeypatch, capsys):
    """calibration_algorithm="entropy" raises NotImplementedError on
    purpose (see quantizer_config.py -- blocked due to a confirmed
    Brevitas 0.13.0 bug plus unstable results even avoiding it, not a gap
    pending implementation) -- --all must skip just those rows with a
    message and still attempt every other job, not abort the whole run at
    the first one it hits."""
    calls = []

    def fake_process_job(entry, repeat_id, *, dry_run, force):
        calls.append((entry["run_id"], repeat_id))
        if entry["calibration_algorithm"] == "entropy":
            raise NotImplementedError(f"calibration_algorithm={entry['calibration_algorithm']!r} is not wired up yet")

    monkeypatch.setattr(cli, "process_job", fake_process_job)

    rc = cli.main(["--all"])

    assert rc == 0
    assert len(calls) == 72  # every job attempted, not just those before the first failure

    out = capsys.readouterr().out
    skip_lines = [l for l in out.splitlines() if l.startswith("[skip]") and "not wired up yet" in l]
    assert len(skip_lines) == 9  # entropy x 3 models x 3 repeats


def test_cli_all_generic_exception_is_not_swallowed_and_aborts(monkeypatch):
    """Only NotImplementedError is caught and skipped. Any other exception
    (a real Brevitas bug, a bad path, a broken ONNX export) must propagate
    and abort the run -- it must not be confused with the declared,
    deliberately-blocked entropy row and silently skipped."""
    calls = []

    def fake_process_job(entry, repeat_id, *, dry_run, force):
        calls.append((entry["run_id"], repeat_id))
        if len(calls) == 5:
            raise RuntimeError("simulated real bug, e.g. a broken ONNX export")

    monkeypatch.setattr(cli, "process_job", fake_process_job)

    with pytest.raises(RuntimeError, match="simulated real bug"):
        cli.main(["--all"])

    # aborted partway through -- did not attempt all 72 jobs
    assert len(calls) == 5


def test_cli_all_summary_separates_exists_generated_and_blocked_counts(monkeypatch, capsys):
    """main()'s final summary must report exists/generated/not_implemented
    as separate counts, plus how many distinct manifest rows are blocked,
    not one combined total that hides how much is genuinely unimplemented
    vs. already done in an earlier run."""

    def fake_process_job(entry, repeat_id, *, dry_run, force):
        if entry["run_id"] == _BASELINE_RUN_ID:
            return "exists"
        if entry["calibration_algorithm"] == "entropy" and repeat_id == 1:
            raise NotImplementedError(f"calibration_algorithm={entry['calibration_algorithm']!r} not wired up")
        return "generated"

    monkeypatch.setattr(cli, "process_job", fake_process_job)

    rc = cli.main(["--all"])
    assert rc == 0

    out = capsys.readouterr().out
    # baseline's 3 repeats -> "exists"
    assert "already existed (skipped for idempotency): 3" in out
    # entropy x 3 models, only rep1 raises -> 3
    assert "blocked (calibration_algorithm not implemented): 3" in out
    # everything else: 72 total - 3 exists - 3 blocked = 66
    assert "generated in this run: 66" in out
    assert "-> 3/24 manifest rows are still genuinely blocked" in out


# --------------------------------------------------------------------------
# --phase3: reads configs/vit_phase3_manifest.yaml instead of Phase 1-2
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
    """A synthetic Phase-2-shaped results set where every one of the 3
    models has a factor winner (granularity=per_channel), so
    vit_phase3_manifest.generate() produces the full 4 rows/model x 3
    models = 12 entries -- no dedup case -- giving 12 x NUM_REPEATS(3) = 36
    quantization jobs with --phase3 --all."""
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


def test_cli_phase3_all_dry_run_prints_36_quantize_export_lines(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    entries, dedup_notes = vit_phase3_manifest.generate(rows=_synthetic_vit_accuracy_rows_no_dedup())
    assert len(entries) == 12 and not dedup_notes  # sanity check on the fixture itself
    monkeypatch.setattr(cli.vit_phase3_manifest, "generate", lambda: (entries, dedup_notes))

    rc = cli.main(["--all", "--phase3", "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "quantize+export" in l]
    assert len(lines) == 36


def test_cli_phase3_flag_reads_phase3_manifest_not_phase1_2(tmp_path, monkeypatch, capsys):
    """Distinguishes --phase3 from the default: a run_id that only exists in
    the Phase 3 manifest must resolve under --phase3 and fail under the
    default (Phase 1-2) manifest."""
    monkeypatch.setattr(export_module, "VIT_ONNX_DIR", tmp_path)
    entries, _ = vit_phase3_manifest.generate(rows=_synthetic_vit_accuracy_rows_no_dedup())
    monkeypatch.setattr(cli.vit_phase3_manifest, "generate", lambda: (entries, {}))
    phase3_run_id = entries[0]["run_id"]

    rc = cli.main(["--run-id", phase3_run_id, "--phase3", "--dry-run"])
    assert rc == 0

    with pytest.raises(SystemExit):
        cli.main(["--run-id", phase3_run_id, "--dry-run"])  # no --phase3 -- not in Phase 1-2 manifest
