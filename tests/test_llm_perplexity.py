"""Tests for evaluation_offline/llm_perplexity/.

Dry-run and command-building tests never touch real llama.cpp binaries,
datasets, or the results/ directory. The happy-path and assert-validation
tests fake `run_fn` (standing in for the llama-perplexity binary) and
`build_wikitext2_test_text_file` (standing in for the real WikiText-2 test
cache) so they don't need real external resources either.
"""

from __future__ import annotations

import csv
import subprocess
from unittest.mock import Mock

import pytest

from configs import llm_manifest, llm_phase3_manifest
from evaluation_offline.llm_perplexity import cli, perplexity, results_io
from evaluation_offline.llm_perplexity.paths import WIKITEXT2_TEST_TEXT_PATH
from quantization.llm_llamacpp import quantize as quantize_module
from quantization.llm_llamacpp.cli import iter_jobs
from quantization.llm_llamacpp.quantize import quantized_path

_DETERMINISTIC_RUN_ID = "llm_tinyllama_1_1b_phase2_quant_scheme_legacy_corpus_none_q4km"
_STOCHASTIC_RUN_ID = "llm_tinyllama_1_1b_phase2_calibration_corpus_wikitext2_q4km"


def _entry(run_id: str) -> dict:
    return next(e for e in llm_manifest.generate() if e["run_id"] == run_id)


def _fake_completed_process(stdout: str, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


# --------------------------------------------------------------------------
# build_perplexity_command
# --------------------------------------------------------------------------

def test_build_command_deterministic_row_points_at_unsuffixed_artifact():
    entry = _entry(_DETERMINISTIC_RUN_ID)
    cmd = perplexity.build_perplexity_command(entry, None)
    assert cmd[cmd.index("-m") + 1] == str(quantized_path(entry["run_id"], None))
    assert cmd[cmd.index("-f") + 1] == str(WIKITEXT2_TEST_TEXT_PATH)


def test_build_command_stochastic_row_points_at_rep_suffixed_artifact():
    entry = _entry(_STOCHASTIC_RUN_ID)
    cmd = perplexity.build_perplexity_command(entry, 2)
    assert cmd[cmd.index("-m") + 1] == str(quantized_path(entry["run_id"], 2))


def test_build_command_stochastic_row_without_repeat_id_raises():
    entry = _entry(_STOCHASTIC_RUN_ID)
    with pytest.raises(ValueError):
        perplexity.build_perplexity_command(entry, None)


def test_build_command_deterministic_row_with_repeat_id_raises():
    entry = _entry(_DETERMINISTIC_RUN_ID)
    with pytest.raises(ValueError):
        perplexity.build_perplexity_command(entry, 1)


# --------------------------------------------------------------------------
# parse_perplexity / validate_perplexity
# --------------------------------------------------------------------------

def test_parse_perplexity_extracts_the_final_estimate():
    stdout = "loading model...\n[1]4.5,[2]5.1,...\nFinal estimate: PPL = 12.3456 +/- 0.7890\n"
    assert perplexity.parse_perplexity(stdout) == 12.3456


def test_parse_perplexity_raises_when_summary_line_missing():
    with pytest.raises(ValueError, match="Could not find"):
        perplexity.parse_perplexity("no summary line in this output")


def test_parse_perplexity_finds_final_estimate_amid_realistic_multiline_output():
    """Real llama-perplexity output has per-chunk progress lines (which contain
    plain numbers, no "PPL =") and llama_print_timings lines *after* the
    "Final estimate" line -- the parser must specifically find the line that
    starts with "Final estimate", not assume it's the last line or the only
    line mentioning perplexity/PPL."""
    stdout = (
        "system_info: n_threads = 8 / 16 | AVX = 1 | AVX2 = 1 | FMA = 1 | ...\n"
        "perplexity: tokenizing the input ..\n"
        "perplexity: tokenization took 123.45 ms\n"
        "perplexity: calculating perplexity over 655 chunks, n_ctx=512, batch_size=2048\n"
        "perplexity: 10.23 seconds per pass - ETA 1.50 minutes\n"
        "[1]4.1234,[2]5.6789,[3]4.9876,[4]5.1111,[5]4.8888,\n"
        "[6]5.0000,[7]4.7777,[8]5.3333,[9]4.6666,[10]5.2222,\n"
        "...\n"
        "[655]6.0001,\n"
        "\n"
        "Final estimate: PPL = 6.5432 +/- 0.0789\n"
        "\n"
        "llama_print_timings:        load time =    512.34 ms\n"
        "llama_print_timings:      sample time =      1.23 ms /     1 runs\n"
        "llama_print_timings: prompt eval time =  98765.43 ms /  4096 tokens\n"
        "llama_print_timings:        eval time =      0.00 ms /     1 runs\n"
        "llama_print_timings:       total time = 100000.00 ms\n"
    )
    assert perplexity.parse_perplexity(stdout) == 6.5432


def test_validate_perplexity_accepts_a_normal_value():
    assert perplexity.validate_perplexity(15.2) == 15.2


@pytest.mark.parametrize("bad_value", [0.0, -3.0, float("inf"), float("-inf"), float("nan")])
def test_validate_perplexity_rejects_non_positive_or_non_finite(bad_value):
    with pytest.raises(AssertionError):
        perplexity.validate_perplexity(bad_value)


# --------------------------------------------------------------------------
# evaluate_perplexity: dry-run / idempotency / happy path / invalid-value guard
# --------------------------------------------------------------------------

def test_evaluate_perplexity_dry_run_never_calls_run_fn_or_touches_corpus(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(results_io, "LLM_PERPLEXITY_RESULTS_CSV", tmp_path / "llm_perplexity_offline.csv")
    build_corpus = Mock(side_effect=AssertionError("must not touch WikiText-2 on a dry-run"))
    monkeypatch.setattr(perplexity, "build_wikitext2_test_text_file", build_corpus)
    entry = _entry(_DETERMINISTIC_RUN_ID)

    run_fn = Mock()
    result = perplexity.evaluate_perplexity(entry, None, dry_run=True, force=False, run_fn=run_fn)

    assert result is None
    run_fn.assert_not_called()
    build_corpus.assert_not_called()
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert entry["run_id"] in out


def test_evaluate_perplexity_skips_when_already_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "LLM_PERPLEXITY_RESULTS_CSV", tmp_path / "llm_perplexity_offline.csv")
    entry = _entry(_DETERMINISTIC_RUN_ID)
    results_io.append_result(entry, None, 15.0)

    run_fn = Mock()
    result = perplexity.evaluate_perplexity(entry, None, dry_run=False, force=False, run_fn=run_fn)

    assert result is None
    run_fn.assert_not_called()


def test_evaluate_perplexity_force_reruns_even_if_already_recorded(tmp_path, monkeypatch):
    csv_path = tmp_path / "llm_perplexity_offline.csv"
    monkeypatch.setattr(results_io, "LLM_PERPLEXITY_RESULTS_CSV", csv_path)
    monkeypatch.setattr(perplexity, "build_wikitext2_test_text_file", Mock(return_value=tmp_path / "wikitext2_test.txt"))
    entry = _entry(_DETERMINISTIC_RUN_ID)
    results_io.append_result(entry, None, 15.0)

    run_fn = Mock(return_value=_fake_completed_process("Final estimate: PPL = 20.0000 +/- 1.0000\n"))
    result = perplexity.evaluate_perplexity(entry, None, dry_run=False, force=True, run_fn=run_fn)

    run_fn.assert_called_once()
    assert result == 20.0

    # Regression check mirroring the same fix in evaluation_offline/
    # vit_accuracy/results_io.py: append_result used to always append, so
    # --force left both the old and new values in the CSV under the
    # identical result_id.
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["accuracy_value"] == "20.0"


def test_evaluate_perplexity_happy_path_calls_binary_and_writes_result(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "LLM_PERPLEXITY_RESULTS_CSV", tmp_path / "llm_perplexity_offline.csv")
    monkeypatch.setattr(perplexity, "build_wikitext2_test_text_file", Mock(return_value=tmp_path / "wikitext2_test.txt"))
    entry = _entry(_STOCHASTIC_RUN_ID)

    run_fn = Mock(return_value=_fake_completed_process("Final estimate: PPL = 9.8765 +/- 0.1234\n"))
    value = perplexity.evaluate_perplexity(entry, 2, dry_run=False, force=False, run_fn=run_fn)

    assert value == 9.8765
    run_fn.assert_called_once()
    called_argv = run_fn.call_args[0][0]
    assert called_argv == perplexity.build_perplexity_command(entry, 2)
    assert run_fn.call_args.kwargs == {"check": True, "capture_output": True, "text": True}
    assert results_io.has_result(entry["run_id"], 2)


def test_evaluate_perplexity_finds_final_estimate_when_it_lands_on_stderr(tmp_path, monkeypatch):
    """Regression test: llama-perplexity (like the rest of llama.cpp's CLI
    tools) logs through stderr in practice, including the "Final estimate"
    summary line itself, while the per-chunk progress goes to stdout. A
    real run reproduced exactly this split -- stdout ending mid-chunk-list
    with no "Final estimate" line, which used to make parse_perplexity
    raise even though the run succeeded."""
    monkeypatch.setattr(results_io, "LLM_PERPLEXITY_RESULTS_CSV", tmp_path / "llm_perplexity_offline.csv")
    monkeypatch.setattr(perplexity, "build_wikitext2_test_text_file", Mock(return_value=tmp_path / "wikitext2_test.txt"))
    entry = _entry(_DETERMINISTIC_RUN_ID)

    stdout_only_progress = "[656]8.7914,[657]8.7794,[658]8.7739,\n\n"
    stderr_with_final_estimate = "Final estimate: PPL = 8.7739 +/- 0.1500\n"
    run_fn = Mock(return_value=_fake_completed_process(stdout_only_progress, stderr_with_final_estimate))

    value = perplexity.evaluate_perplexity(entry, None, dry_run=False, force=False, run_fn=run_fn)

    assert value == 8.7739
    assert results_io.has_result(entry["run_id"], None)


def test_appended_result_row_has_device_offline_not_blank(tmp_path, monkeypatch):
    """technical_architecture.md Section 4: device must be the literal
    string "offline" for accuracy/perplexity rows, not empty -- the
    results-combination script validates that every row's device is
    non-empty and in devices.yaml."""
    csv_path = tmp_path / "llm_perplexity_offline.csv"
    monkeypatch.setattr(results_io, "LLM_PERPLEXITY_RESULTS_CSV", csv_path)
    monkeypatch.setattr(perplexity, "build_wikitext2_test_text_file", Mock(return_value=tmp_path / "wikitext2_test.txt"))
    entry = _entry(_DETERMINISTIC_RUN_ID)

    run_fn = Mock(return_value=_fake_completed_process("Final estimate: PPL = 11.1111 +/- 0.2222\n"))
    perplexity.evaluate_perplexity(entry, None, dry_run=False, force=False, run_fn=run_fn)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["device"] == "offline"


def test_evaluate_perplexity_raises_and_never_writes_result_when_value_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "LLM_PERPLEXITY_RESULTS_CSV", tmp_path / "llm_perplexity_offline.csv")
    monkeypatch.setattr(perplexity, "build_wikitext2_test_text_file", Mock(return_value=tmp_path / "wikitext2_test.txt"))
    entry = _entry(_DETERMINISTIC_RUN_ID)

    # A value that parses successfully but is nonsensical (e.g. a broken/truncated
    # binary run) -- this must fail loudly via the assert, not get written.
    run_fn = Mock(return_value=_fake_completed_process("Final estimate: PPL = 0.0000 +/- 0.0000\n"))
    append_result = Mock()
    monkeypatch.setattr(results_io, "append_result", append_result)

    with pytest.raises(AssertionError):
        perplexity.evaluate_perplexity(entry, None, dry_run=False, force=False, run_fn=run_fn)

    append_result.assert_not_called()
    assert not results_io.has_result(entry["run_id"], None)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_dry_run_deterministic_row(tmp_path, monkeypatch, capsys):
    # evaluate_perplexity checks results_io.has_result() BEFORE the
    # dry_run branch, so without isolating the results CSV this would hit
    # the "already recorded" skip path once a real result exists for
    # _DETERMINISTIC_RUN_ID (it does, post the real perplexity run).
    monkeypatch.setattr(results_io, "LLM_PERPLEXITY_RESULTS_CSV", tmp_path / "llm_perplexity_offline.csv")
    rc = cli.main(["--run-id", _DETERMINISTIC_RUN_ID, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert _DETERMINISTIC_RUN_ID in out
    assert "[dry-run]" in out


def test_cli_stochastic_row_without_repeat_id_errors():
    with pytest.raises(SystemExit):
        cli.main(["--run-id", _STOCHASTIC_RUN_ID, "--dry-run"])


def test_cli_dry_run_stochastic_row_with_repeat_id(capsys):
    rc = cli.main(["--run-id", _STOCHASTIC_RUN_ID, "--repeat-id", "2", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "rep2" in out


def test_cli_deterministic_row_with_repeat_id_errors():
    with pytest.raises(SystemExit):
        cli.main(["--run-id", _DETERMINISTIC_RUN_ID, "--repeat-id", "1", "--dry-run"])


def test_cli_unknown_run_id_errors():
    with pytest.raises(SystemExit):
        cli.main(["--run-id", "not_a_real_run_id", "--dry-run"])


def test_cli_run_id_real_run_errors_when_artifact_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(quantize_module, "LLM_QUANTIZED_DIR", tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["--run-id", _DETERMINISTIC_RUN_ID])


def test_cli_run_id_dry_run_never_requires_the_artifact_to_exist(tmp_path, monkeypatch, capsys):
    # Same setup as the error test above (no artifact on disk), but --dry-run
    # must succeed anyway -- it's a pure preview. Also isolate the results
    # CSV: evaluate_perplexity checks has_result() BEFORE the dry_run branch,
    # so a real recorded result for _DETERMINISTIC_RUN_ID would otherwise
    # print "[skip] ... already recorded" instead of "[dry-run] ...".
    monkeypatch.setattr(quantize_module, "LLM_QUANTIZED_DIR", tmp_path)
    monkeypatch.setattr(results_io, "LLM_PERPLEXITY_RESULTS_CSV", tmp_path / "llm_perplexity_offline.csv")
    rc = cli.main(["--run-id", _DETERMINISTIC_RUN_ID, "--dry-run"])
    assert rc == 0
    assert "[dry-run]" in capsys.readouterr().out


# --------------------------------------------------------------------------
# CLI --all
# --------------------------------------------------------------------------

def test_cli_all_with_repeat_id_errors():
    with pytest.raises(SystemExit):
        cli.main(["--all", "--repeat-id", "1"])


def test_cli_all_with_no_artifacts_skips_everything_and_reports_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(quantize_module, "LLM_QUANTIZED_DIR", tmp_path)
    evaluate = Mock()
    monkeypatch.setattr(perplexity, "evaluate_perplexity", evaluate)

    rc = cli.main(["--all"])

    assert rc == 0
    evaluate.assert_not_called()
    out = capsys.readouterr().out
    all_jobs = list(iter_jobs(llm_manifest.generate()))
    assert f"{len(all_jobs)}/{len(all_jobs)}" in out
    assert len(all_jobs) == 51


def test_cli_all_evaluates_only_existing_artifacts_and_lists_the_rest_as_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(quantize_module, "LLM_QUANTIZED_DIR", tmp_path)
    all_jobs = list(iter_jobs(llm_manifest.generate()))
    present_jobs = all_jobs[:3]
    for entry, repeat_id in present_jobs:
        path = quantize_module.quantized_path(entry["run_id"], repeat_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake gguf")

    evaluate = Mock()
    monkeypatch.setattr(perplexity, "evaluate_perplexity", evaluate)

    rc = cli.main(["--all", "--force"])

    assert rc == 0
    assert evaluate.call_count == len(present_jobs)
    called_pairs = {(call.args[0]["run_id"], call.args[1]) for call in evaluate.call_args_list}
    expected_pairs = {(entry["run_id"], repeat_id) for entry, repeat_id in present_jobs}
    assert called_pairs == expected_pairs
    for call in evaluate.call_args_list:
        assert call.kwargs == {"dry_run": False, "force": True}

    out = capsys.readouterr().out
    assert f"{len(all_jobs) - len(present_jobs)}/{len(all_jobs)}" in out


def test_cli_all_dry_run_previews_every_job_ignoring_artifact_existence(tmp_path, monkeypatch, capsys):
    # --dry-run: no artifacts exist anywhere, but every job should still be
    # "run" (printed), none reported as missing -- pure preview semantics.
    monkeypatch.setattr(quantize_module, "LLM_QUANTIZED_DIR", tmp_path)
    evaluate = Mock()
    monkeypatch.setattr(perplexity, "evaluate_perplexity", evaluate)

    rc = cli.main(["--all", "--dry-run"])

    assert rc == 0
    all_jobs = list(iter_jobs(llm_manifest.generate()))
    assert evaluate.call_count == len(all_jobs)
    out = capsys.readouterr().out
    assert "[warning]" not in out


# --------------------------------------------------------------------------
# --phase3: reads configs/llm_phase3_manifest.yaml instead of Phase 1-2.
# evaluation_offline never repeats per artifact -- same job-count formula
# as quantization (configurations x repeats), not a different number.
# --------------------------------------------------------------------------

def _llm_phase3_source_row(model_name, calibration_corpus, quant_scheme, accuracy_value, calib_set_size=None, repeat_id=None):
    calib_set_size_field = "" if calibration_corpus == "none" else str(calib_set_size)
    return {
        "model_name": model_name,
        "calibration_corpus": calibration_corpus,
        "calib_set_size": calib_set_size_field,
        "quant_scheme": quant_scheme,
        "bitwidth": "q4km",
        "accuracy_value": str(accuracy_value),
        "repeat_id": "" if repeat_id is None else str(repeat_id),
    }


def _llm_phase2_model_rows(model_name, *, corpus_wins=False):
    rows = [
        _llm_phase3_source_row(model_name, "none", "k_quant", accuracy_value=10.0),
        _llm_phase3_source_row(model_name, "none", "legacy", accuracy_value=10.5),  # never wins
    ]
    corpus_value = 8.0 if corpus_wins else 10.2
    for rep in (1, 2, 3):
        rows.append(_llm_phase3_source_row(model_name, "wikitext2", "k_quant", accuracy_value=corpus_value, calib_set_size=200, repeat_id=rep))
        rows.append(_llm_phase3_source_row(model_name, "domain_corpus", "k_quant", accuracy_value=10.3, calib_set_size=200, repeat_id=rep))
        rows.append(_llm_phase3_source_row(model_name, "wikitext2", "k_quant", accuracy_value=10.4, calib_set_size=50, repeat_id=rep))
        rows.append(_llm_phase3_source_row(model_name, "wikitext2", "k_quant", accuracy_value=10.5, calib_set_size=1000, repeat_id=rep))
    return rows


def _synthetic_llm_phase2_results_for_18_jobs():
    """Same documented scenario as quantization/llm_llamacpp's --phase3
    test: tinyllama_1_1b dedups (2 deterministic jobs); qwen2_5_1_5b and
    ministral_3_3b each have calibration_corpus as their Phase 2 winner
    (2*1 + 2*3 = 8 jobs each). Total: 2 + 8 + 8 = 18."""
    rows = []
    rows += _llm_phase2_model_rows("tinyllama_1_1b")
    rows += _llm_phase2_model_rows("qwen2_5_1_5b", corpus_wins=True)
    rows += _llm_phase2_model_rows("ministral_3_3b", corpus_wins=True)
    return rows


def test_cli_phase3_all_dry_run_previews_18_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(quantize_module, "LLM_QUANTIZED_DIR", tmp_path)
    entries, dedup_notes = llm_phase3_manifest.generate(rows=_synthetic_llm_phase2_results_for_18_jobs())
    assert len(entries) == 10 and list(dedup_notes) == ["tinyllama_1_1b"]  # sanity check: 2 + 4 + 4 = 10 rows
    monkeypatch.setattr(cli.llm_phase3_manifest, "generate", lambda: (entries, dedup_notes))

    evaluate = Mock()
    monkeypatch.setattr(perplexity, "evaluate_perplexity", evaluate)

    rc = cli.main(["--all", "--phase3", "--dry-run"])

    assert rc == 0
    assert evaluate.call_count == 18


def test_cli_phase3_flag_reads_phase3_manifest_not_phase1_2(tmp_path, monkeypatch):
    monkeypatch.setattr(quantize_module, "LLM_QUANTIZED_DIR", tmp_path)
    entries, _ = llm_phase3_manifest.generate(rows=_synthetic_llm_phase2_results_for_18_jobs())
    monkeypatch.setattr(cli.llm_phase3_manifest, "generate", lambda: (entries, {}))
    phase3_run_id = entries[0]["run_id"]

    rc = cli.main(["--run-id", phase3_run_id, "--phase3", "--dry-run"])
    assert rc == 0

    with pytest.raises(SystemExit):
        cli.main(["--run-id", phase3_run_id, "--dry-run"])  # no --phase3 -- unknown in Phase 1-2 manifest
