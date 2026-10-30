"""Dry-run and idempotency tests for quantization/llm_llamacpp/.

Deterministic manifest rows (calibration_corpus == "none": the baseline and
the legacy_corpus_none exception row, 2 per model = 6 total) don't depend on
any sampling and get a single artifact, {run_id}.gguf. Stochastic rows (the
other 15) depend on a randomly sampled imatrix corpus and get NUM_REPEATS=3
independent artifacts each, {run_id}__rep{repeat_id}.gguf, seeded
42/43/44 -- a fixed seed across repeats would make them identical, which
defeats the purpose of repeating. Total: 6 + 15*3 = 51 artifacts.

Dry-run tests only exercise pure command-building functions (or `--dry-run`
through the CLI, which never touches subprocess/data_prep) so they run
without real llama.cpp binaries, downloaded models, or datasets.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from configs import llm_manifest, llm_phase3_manifest
from quantization.llm_llamacpp import base_model, cli, imatrix_gen, quantize
from quantization.llm_llamacpp.quant_type import resolve_for_entry


def _entries():
    return llm_manifest.generate()


def _entry(run_id: str) -> dict:
    return next(e for e in _entries() if e["run_id"] == run_id)


def _deterministic_entries():
    return [e for e in _entries() if not imatrix_gen.needs_imatrix(e)]


def _stochastic_entries():
    return [e for e in _entries() if imatrix_gen.needs_imatrix(e)]


def _jobs():
    return list(cli.iter_jobs(_entries()))


def _job_id(job) -> str:
    entry, repeat_id = job
    return entry["run_id"] + (f"__rep{repeat_id}" if repeat_id is not None else "")


_JOBS = _jobs()
_JOB_IDS = [_job_id(j) for j in _JOBS]


# --------------------------------------------------------------------------
# manifest shape: 21 rows, split 6 deterministic / 15 stochastic
# --------------------------------------------------------------------------

def test_manifest_has_21_rows():
    assert len(_entries()) == 21


def test_manifest_splits_into_6_deterministic_and_15_stochastic_rows():
    assert len(_deterministic_entries()) == 6
    assert len(_stochastic_entries()) == 15


def test_iter_jobs_yields_51_total_jobs_6_single_plus_45_repeated():
    assert len(_JOBS) == 51
    deterministic_jobs = [(e, r) for e, r in _JOBS if r is None]
    stochastic_jobs = [(e, r) for e, r in _JOBS if r is not None]
    assert len(deterministic_jobs) == 6
    assert len(stochastic_jobs) == 45


# --------------------------------------------------------------------------
# build_quantize_command: all 51 (run_id, repeat_id) jobs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("job", _JOBS, ids=_JOB_IDS)
def test_quantize_command_type_and_output_path(job):
    entry, repeat_id = job
    cmd = quantize.build_quantize_command(entry, repeat_id)
    assert cmd[-1] == resolve_for_entry(entry)
    assert cmd[-2] == str(quantize.quantized_path(entry["run_id"], repeat_id))


@pytest.mark.parametrize("job", _JOBS, ids=_JOB_IDS)
def test_quantize_command_imatrix_flag_matches_calibration_corpus(job):
    entry, repeat_id = job
    cmd = quantize.build_quantize_command(entry, repeat_id)
    has_flag = "--imatrix" in cmd
    assert has_flag == (entry["calibration_corpus"] != "none")
    if has_flag:
        idx = cmd.index("--imatrix")
        assert cmd[idx + 1] == str(imatrix_gen.imatrix_path(entry["run_id"], repeat_id))


def test_deterministic_rows_produce_a_single_command_without_rep_suffix():
    for entry in _deterministic_entries():
        out_path = quantize.quantized_path(entry["run_id"], repeat_id=None)
        assert out_path.name == f"{entry['run_id']}.gguf"
        assert "__rep" not in out_path.name
        cmd = quantize.build_quantize_command(entry, repeat_id=None)
        assert cmd[-2] == str(out_path)
        assert "--imatrix" not in cmd


def test_stochastic_rows_produce_3_commands_with_seeds_42_43_44_and_rep_suffixes():
    for entry in _stochastic_entries():
        seeds = [imatrix_gen.seed_for_repeat(r) for r in range(1, imatrix_gen.NUM_REPEATS + 1)]
        assert seeds == [42, 43, 44]
        for repeat_id in (1, 2, 3):
            out_path = quantize.quantized_path(entry["run_id"], repeat_id)
            assert out_path.name == f"{entry['run_id']}__rep{repeat_id}.gguf"
            cmd = quantize.build_quantize_command(entry, repeat_id)
            assert cmd[-2] == str(out_path)
            idx = cmd.index("--imatrix")
            assert cmd[idx + 1] == str(imatrix_gen.imatrix_path(entry["run_id"], repeat_id))


def test_build_quantize_command_rejects_mismatched_repeat_id():
    deterministic = _deterministic_entries()[0]
    stochastic = _stochastic_entries()[0]
    with pytest.raises(ValueError):
        quantize.build_quantize_command(deterministic, repeat_id=1)
    with pytest.raises(ValueError):
        quantize.build_quantize_command(stochastic, repeat_id=None)


def test_no_legacy_row_resolves_to_a_k_quant_type_in_the_command():
    for entry, repeat_id in _JOBS:
        if entry["quant_scheme"] != "legacy":
            continue
        cmd = quantize.build_quantize_command(entry, repeat_id)
        assert "_K" not in cmd[-1]


def test_quantize_command_base_model_input_is_shared_per_model_not_per_row():
    # Same model_name -> same F16 input path, regardless of which job asks for it.
    tinyllama_jobs = [(e, r) for e, r in _JOBS if e["model_name"] == "tinyllama_1_1b"]
    assert len(tinyllama_jobs) > 1
    inputs = {quantize.build_quantize_command(e, r)[-3] for e, r in tinyllama_jobs}
    assert inputs == {str(base_model.base_model_path("tinyllama_1_1b"))}


# --------------------------------------------------------------------------
# CLI --dry-run: end-to-end, no real subprocess/data_prep calls
# --------------------------------------------------------------------------

def test_cli_dry_run_all_processes_every_row_without_touching_subprocess(monkeypatch, capsys):
    monkeypatch.setattr(cli.subprocess, "run", Mock(side_effect=AssertionError("must not run for real")))
    rc = cli.main(["--all", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    for entry in _entries():
        assert entry["run_id"] in out


def test_cli_all_dry_run_prints_exactly_51_quantize_commands(tmp_path, monkeypatch, capsys):
    # quantize.prepare_quantized checks output_exists BEFORE the dry_run
    # branch (common_exec.run_step), so without isolating LLM_QUANTIZED_DIR
    # this would print "[skip] ... already exists" instead of "[dry-run]
    # ... quantize ..." for any row whose real .gguf already exists (which
    # they do, post the real --all quantization run).
    monkeypatch.setattr(quantize, "LLM_QUANTIZED_DIR", tmp_path)
    rc = cli.main(["--all", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    quantize_lines = [line for line in out.splitlines() if ": quantize" in line]
    assert len(quantize_lines) == 51


def test_cli_all_dry_run_deterministic_lines_have_no_rep_suffix_stochastic_do(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(quantize, "LLM_QUANTIZED_DIR", tmp_path)
    rc = cli.main(["--all", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    quantize_lines = [line for line in out.splitlines() if ": quantize" in line]

    for entry in _deterministic_entries():
        matches = [l for l in quantize_lines if l.startswith(f"[dry-run] {entry['run_id']}:")]
        assert len(matches) == 1, entry["run_id"]

    for entry in _stochastic_entries():
        for repeat_id in (1, 2, 3):
            matches = [l for l in quantize_lines if l.startswith(f"[dry-run] {entry['run_id']} rep{repeat_id}:")]
            assert len(matches) == 1, f"{entry['run_id']} rep{repeat_id}"


def test_cli_dry_run_single_run_id_deterministic(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(quantize, "LLM_QUANTIZED_DIR", tmp_path)
    run_id = "llm_tinyllama_1_1b_phase2_quant_scheme_legacy_corpus_none_q4km"
    rc = cli.main(["--run-id", run_id, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "Q4_0" in out  # legacy resolved correctly, not the K-quant type
    quantize_lines = [line for line in out.splitlines() if ": quantize" in line]
    assert len(quantize_lines) == 1


def test_cli_dry_run_single_run_id_stochastic_prints_3_quantize_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(quantize, "LLM_QUANTIZED_DIR", tmp_path)
    run_id = "llm_tinyllama_1_1b_phase2_calibration_corpus_wikitext2_q4km"
    rc = cli.main(["--run-id", run_id, "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    quantize_lines = [line for line in out.splitlines() if ": quantize" in line]
    assert len(quantize_lines) == 3
    for repeat_id in (1, 2, 3):
        assert f"{run_id} rep{repeat_id}: quantize" in out


def test_cli_unknown_run_id_errors():
    with pytest.raises(SystemExit):
        cli.main(["--run-id", "not_a_real_run_id", "--dry-run"])


# --------------------------------------------------------------------------
# Idempotency: existing artifact => the binary is never invoked
# --------------------------------------------------------------------------

def test_prepare_quantized_skips_binary_call_when_artifact_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(quantize, "LLM_QUANTIZED_DIR", tmp_path)
    entry = _entry("llm_tinyllama_1_1b_phase1_baseline_default_q4km")
    out_path = quantize.quantized_path(entry["run_id"], repeat_id=None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"already there")

    run_fn = Mock()
    quantize.prepare_quantized(entry, None, dry_run=False, force=False, run_fn=run_fn)

    run_fn.assert_not_called()


def test_prepare_quantized_calls_binary_when_artifact_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(quantize, "LLM_QUANTIZED_DIR", tmp_path)
    entry = _entry("llm_tinyllama_1_1b_phase1_baseline_default_q4km")

    run_fn = Mock()
    quantize.prepare_quantized(entry, None, dry_run=False, force=False, run_fn=run_fn)

    run_fn.assert_called_once()
    called_argv = run_fn.call_args[0][0]
    assert called_argv == quantize.build_quantize_command(entry, None)
    assert run_fn.call_args.kwargs == {"check": True}


def test_prepare_quantized_force_reruns_even_if_artifact_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(quantize, "LLM_QUANTIZED_DIR", tmp_path)
    entry = _entry("llm_tinyllama_1_1b_phase1_baseline_default_q4km")
    out_path = quantize.quantized_path(entry["run_id"], repeat_id=None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"stale")

    run_fn = Mock()
    quantize.prepare_quantized(entry, None, dry_run=False, force=True, run_fn=run_fn)

    run_fn.assert_called_once()


def test_prepare_quantized_idempotency_is_per_repeat_not_per_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(quantize, "LLM_QUANTIZED_DIR", tmp_path)
    entry = _entry("llm_tinyllama_1_1b_phase2_calibration_corpus_wikitext2_q4km")

    # rep1 already exists -> skipped; rep2 missing -> runs.
    rep1_path = quantize.quantized_path(entry["run_id"], repeat_id=1)
    rep1_path.parent.mkdir(parents=True, exist_ok=True)
    rep1_path.write_bytes(b"already there")

    run_fn = Mock()
    quantize.prepare_quantized(entry, 1, dry_run=False, force=False, run_fn=run_fn)
    run_fn.assert_not_called()

    quantize.prepare_quantized(entry, 2, dry_run=False, force=False, run_fn=run_fn)
    run_fn.assert_called_once()
    called_argv = run_fn.call_args[0][0]
    assert called_argv == quantize.build_quantize_command(entry, 2)


def test_prepare_imatrix_skips_binary_call_and_corpus_build_when_dat_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(imatrix_gen, "LLM_IMATRIX_DIR", tmp_path)
    entry = _entry("llm_tinyllama_1_1b_phase2_calibration_corpus_wikitext2_q4km")
    out_path = imatrix_gen.imatrix_path(entry["run_id"], repeat_id=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"already there")

    build_corpus = Mock(side_effect=AssertionError("must not touch data_prep on an idempotent hit"))
    monkeypatch.setattr(imatrix_gen, "build_corpus_text_file", build_corpus)

    run_fn = Mock()
    imatrix_gen.prepare_imatrix(entry, 1, dry_run=False, force=False, run_fn=run_fn)

    run_fn.assert_not_called()
    build_corpus.assert_not_called()


def test_prepare_base_model_skips_when_gguf_already_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(base_model, "LLM_BASE_DIR", tmp_path)
    out_path = base_model.base_model_path("tinyllama_1_1b")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"already there")
    base_model.hf_snapshot_dir("tinyllama_1_1b").mkdir(parents=True, exist_ok=True)
    (base_model.hf_snapshot_dir("tinyllama_1_1b") / "config.json").write_text("{}")

    run_fn = Mock()
    base_model.prepare_base_model("tinyllama_1_1b", dry_run=False, force=False, run_fn=run_fn)

    run_fn.assert_not_called()


def test_every_model_in_manifest_has_a_hf_repo():
    from configs import llm_manifest
    from quantization.llm_llamacpp.paths import MODEL_HF_REPOS

    assert set(llm_manifest.MODELS) == set(MODEL_HF_REPOS)


def test_ministral_downloads_the_bf16_repo_without_the_duplicate_consolidated_weights():
    cmd = base_model.build_download_command("ministral_3_3b")

    assert cmd[:3] == ["hf", "download", "mistralai/Ministral-3-3B-Instruct-2512-BF16"]
    assert "-BF16" in cmd[2]  # the default repo is FP8, which would break comparability
    assert cmd[3:5] == ["--exclude", "consolidated.safetensors"]
    assert cmd[5] == "--local-dir"  # exclude's patterns must not swallow the next flag
    assert cmd[6] == str(base_model.hf_snapshot_dir("ministral_3_3b"))


def test_other_models_download_command_is_unchanged_by_the_exclude_support():
    for model_name in ("tinyllama_1_1b", "qwen2_5_1_5b"):
        cmd = base_model.build_download_command(model_name)
        assert "--exclude" not in cmd
        assert cmd == [
            "hf", "download", base_model.hf_repo_id(model_name),
            "--local-dir", str(base_model.hf_snapshot_dir(model_name)),
        ]


def test_retired_salamandra_has_no_repo_mapping():
    import pytest

    with pytest.raises(ValueError, match="salamandra_2b"):
        base_model.hf_repo_id("salamandra_2b")


# --------------------------------------------------------------------------
# domain_corpus: now wired to data_prep.domain_corpus (databricks-dolly-15k)
# --------------------------------------------------------------------------

def test_domain_corpus_row_dry_runs_cleanly():
    entry = _entry("llm_tinyllama_1_1b_phase2_calibration_corpus_domain_corpus_q4km")
    cmd = imatrix_gen.build_imatrix_command(entry, repeat_id=1)
    assert f"{entry['run_id']}__rep1" in cmd[cmd.index("-f") + 1]


def test_domain_corpus_row_reaches_imatrix_binary_call_with_correct_corpus(tmp_path, monkeypatch):
    """No real dataset/binary needed: data_prep.domain_corpus.load_domain_corpus_train
    is faked, and run_fn is a Mock standing in for the real llama-imatrix binary --
    this only confirms the pipeline reaches and calls it, with the right corpus
    materialized on disk, instead of raising (as it used to before the loader existed).
    """
    monkeypatch.setattr(imatrix_gen, "LLM_IMATRIX_DIR", tmp_path)
    entry = _entry("llm_tinyllama_1_1b_phase2_calibration_corpus_domain_corpus_q4km")

    from data_prep import domain_corpus as domain_corpus_module

    fake_rows = [{"text": f"dolly example {i}"} for i in range(entry["calib_set_size"] + 10)]
    monkeypatch.setattr(domain_corpus_module, "load_domain_corpus_train", lambda: fake_rows)

    run_fn = Mock()
    imatrix_gen.prepare_imatrix(entry, 1, dry_run=False, force=False, run_fn=run_fn)

    run_fn.assert_called_once()
    called_argv = run_fn.call_args[0][0]
    assert called_argv == imatrix_gen.build_imatrix_command(entry, 1)

    corpus_path = imatrix_gen.corpus_text_path(entry["run_id"], 1)
    assert corpus_path.exists()
    sampled_lines = corpus_path.read_text().strip().splitlines()
    assert len(sampled_lines) == entry["calib_set_size"]
    assert all(line.startswith("dolly example ") for line in sampled_lines)


def test_cli_all_dry_run_includes_domain_corpus_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(quantize, "LLM_QUANTIZED_DIR", tmp_path)
    rc = cli.main(["--all", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "domain_corpus" in out
    quantize_lines = [line for line in out.splitlines() if ": quantize" in line]
    assert len(quantize_lines) == 51  # domain_corpus rows are no longer skipped


# --------------------------------------------------------------------------
# --phase3: reads configs/llm_phase3_manifest.yaml instead of Phase 1-2.
# --------------------------------------------------------------------------

def test_iter_jobs_generalizes_to_phase3_shaped_entries_not_assumed():
    """Explicit confirmation (not an assumption) that iter_jobs' det/stoch
    split -- driven only by imatrix_gen.needs_imatrix(), which only reads
    calibration_corpus -- works unchanged on Phase-3-shaped entries (a
    "baseline" row, deterministic, and a "combined" row whose
    calibration_corpus is itself the Phase 2 winner, stochastic), fed
    directly, bypassing manifest generation entirely."""
    phase3_baseline_entry = {
        "run_id": "llm_tinyllama_1_1b_phase3_baseline_q3_k",
        "model_name": "tinyllama_1_1b",
        "phase": "phase3",
        "factor_varied": "phase3_config",
        "factor_level": "baseline",
        "calibration_corpus": "none",
        "calib_set_size": None,
        "quant_scheme": "k_quant",
        "bitwidth": "Q3_K",
    }
    phase3_combined_entry = {
        "run_id": "llm_tinyllama_1_1b_phase3_combined_q3_k",
        "model_name": "tinyllama_1_1b",
        "phase": "phase3",
        "factor_varied": "phase3_config",
        "factor_level": "combined",
        "calibration_corpus": "wikitext2",
        "calib_set_size": 200,
        "quant_scheme": "k_quant",
        "bitwidth": "Q3_K",
    }

    jobs = list(cli.iter_jobs([phase3_baseline_entry, phase3_combined_entry]))

    assert (phase3_baseline_entry, None) in jobs  # deterministic: single job, repeat_id=None
    combined_jobs = [j for j in jobs if j[0] is phase3_combined_entry]
    assert len(combined_jobs) == 3  # stochastic: NUM_REPEATS jobs
    assert {repeat_id for _entry, repeat_id in combined_jobs} == {1, 2, 3}
    assert len(jobs) == 4  # 1 deterministic + 3 stochastic


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
    """Documented scenario: tinyllama_1_1b dedups (no factor wins -> 2
    baseline-only, deterministic rows -> 2 jobs); qwen2_5_1_5b and
    ministral_3_3b both have calibration_corpus as their Phase 2 winner
    (-> 4 rows each: 2 baseline deterministic + 2 combined stochastic).
    Total jobs: 2 + (2*1 + 2*3)*2 = 2 + 16 = 18."""
    rows = []
    rows += _llm_phase2_model_rows("tinyllama_1_1b")  # nothing wins -- dedups
    rows += _llm_phase2_model_rows("qwen2_5_1_5b", corpus_wins=True)
    rows += _llm_phase2_model_rows("ministral_3_3b", corpus_wins=True)
    return rows


def test_cli_phase3_all_dry_run_prints_18_quantize_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(quantize, "LLM_QUANTIZED_DIR", tmp_path)
    entries, dedup_notes = llm_phase3_manifest.generate(rows=_synthetic_llm_phase2_results_for_18_jobs())
    assert len(entries) == 10 and list(dedup_notes) == ["tinyllama_1_1b"]  # sanity check: 2 + 4 + 4 = 10 rows
    monkeypatch.setattr(cli.llm_phase3_manifest, "generate", lambda: (entries, dedup_notes))

    rc = cli.main(["--all", "--phase3", "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    quantize_lines = [line for line in out.splitlines() if ": quantize" in line]
    assert len(quantize_lines) == 18


def test_cli_phase3_flag_reads_phase3_manifest_not_phase1_2(tmp_path, monkeypatch):
    monkeypatch.setattr(quantize, "LLM_QUANTIZED_DIR", tmp_path)
    entries, _ = llm_phase3_manifest.generate(rows=_synthetic_llm_phase2_results_for_18_jobs())
    monkeypatch.setattr(cli.llm_phase3_manifest, "generate", lambda: (entries, {}))
    phase3_run_id = entries[0]["run_id"]

    rc = cli.main(["--run-id", phase3_run_id, "--phase3", "--dry-run"])
    assert rc == 0

    with pytest.raises(SystemExit):
        cli.main(["--run-id", phase3_run_id, "--dry-run"])  # no --phase3 -- unknown in Phase 1-2 manifest
