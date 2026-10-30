"""Tests for configs/vit_phase3_manifest.py and configs/llm_phase3_manifest.py
-- Phase 3 manifest generation from analysis/phase3_selection/'s combined
Phase 2 winners.

All tests inject synthetic rows via generate(rows=...) rather than reading
the real results CSVs -- generate()'s injectable `rows` parameter exists
specifically for this.
"""

from __future__ import annotations

import yaml

from configs import llm_phase3_manifest, vit_phase3_manifest
from configs.common import write_manifest_with_header

# --------------------------------------------------------------------------
# LLM: the explicitly requested case -- ministral_3_3b dedups (2 rows),
# tinyllama_1_1b and qwen2_5_1_5b each get their full 4 rows.
# --------------------------------------------------------------------------


def _llm_row(model_name, calibration_corpus, quant_scheme, accuracy_value, calib_set_size=None, repeat_id=None):
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


def _llm_model_rows(model_name, *, corpus_wins=False, quant_scheme_wins=False):
    """A complete Phase-2-shaped dataset for one model: global baseline,
    calibration_corpus candidates, calib_set_size candidates (comparator:
    the wikitext2/200 row), and the quant_scheme candidate
    (legacy_corpus_none). None of them win unless explicitly requested."""
    rows = [
        _llm_row(model_name, "none", "k_quant", accuracy_value=10.0),
        _llm_row(model_name, "none", "legacy",
                  accuracy_value=(9.0 if quant_scheme_wins else 10.5)),  # deterministic vs. deterministic, margin=0
    ]
    corpus_value = 8.0 if corpus_wins else 10.2  # lower PPL = better
    for rep in (1, 2, 3):
        rows.append(_llm_row(model_name, "wikitext2", "k_quant", accuracy_value=corpus_value, calib_set_size=200, repeat_id=rep))
        rows.append(_llm_row(model_name, "domain_corpus", "k_quant", accuracy_value=10.3, calib_set_size=200, repeat_id=rep))
        rows.append(_llm_row(model_name, "wikitext2", "k_quant", accuracy_value=10.4, calib_set_size=50, repeat_id=rep))
        rows.append(_llm_row(model_name, "wikitext2", "k_quant", accuracy_value=10.5, calib_set_size=1000, repeat_id=rep))
    return rows


def _synthetic_llm_rows():
    rows = []
    rows += _llm_model_rows("tinyllama_1_1b", quant_scheme_wins=True)
    rows += _llm_model_rows("qwen2_5_1_5b", corpus_wins=True)
    rows += _llm_model_rows("ministral_3_3b")  # nothing wins -- dedup case
    return rows


def test_llm_phase3_manifest_ministral_dedups_others_get_4_rows():
    entries, dedup_notes = llm_phase3_manifest.generate(rows=_synthetic_llm_rows())

    by_model = {}
    for e in entries:
        by_model.setdefault(e["model_name"], []).append(e)

    assert len(by_model["ministral_3_3b"]) == 2
    assert len(by_model["tinyllama_1_1b"]) == 4
    assert len(by_model["qwen2_5_1_5b"]) == 4
    assert list(dedup_notes.keys()) == ["ministral_3_3b"]


def test_llm_phase3_manifest_ministral_rows_are_baseline_only():
    entries, _ = llm_phase3_manifest.generate(rows=_synthetic_llm_rows())
    ministral_rows = [e for e in entries if e["model_name"] == "ministral_3_3b"]

    assert {e["factor_level"] for e in ministral_rows} == {"baseline"}
    assert {e["bitwidth"] for e in ministral_rows} == {"Q3_K", "Q2_K"}
    assert {e["run_id"] for e in ministral_rows} == {
        "llm_ministral_3_3b_phase3_baseline_q3_k",
        "llm_ministral_3_3b_phase3_baseline_q2_k",
    }


def test_llm_phase3_manifest_winning_model_has_both_baseline_and_combined():
    entries, _ = llm_phase3_manifest.generate(rows=_synthetic_llm_rows())
    tinyllama_rows = [e for e in entries if e["model_name"] == "tinyllama_1_1b"]

    kinds = {e["factor_level"] for e in tinyllama_rows}
    assert kinds == {"baseline", "combined"}
    combined_rows = [e for e in tinyllama_rows if e["factor_level"] == "combined"]
    assert all(e["quant_scheme"] == "legacy" for e in combined_rows)  # quant_scheme was the winner
    assert {e["run_id"] for e in tinyllama_rows} == {
        "llm_tinyllama_1_1b_phase3_baseline_q3_k",
        "llm_tinyllama_1_1b_phase3_baseline_q2_k",
        "llm_tinyllama_1_1b_phase3_combined_q3_k",
        "llm_tinyllama_1_1b_phase3_combined_q2_k",
    }


def test_llm_phase3_manifest_combined_calib_set_size_is_coherent_with_corpus():
    """Regression check for the selected_config() assembly bug: a winning
    combined config must never carry calib_set_size alongside
    calibration_corpus="none"."""
    entries, _ = llm_phase3_manifest.generate(rows=_synthetic_llm_rows())
    for e in entries:
        if e["calibration_corpus"] == "none":
            assert e["calib_set_size"] in (None, "", "None")


# --------------------------------------------------------------------------
# ViT: analogous non-dedup and dedup cases
# --------------------------------------------------------------------------


def _vit_row(model_name, calibration_algorithm, granularity, symmetry, calib_set_size, accuracy_value, repeat_id):
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


def _vit_model_rows(model_name, *, granularity_wins=False):
    # Every repeat uses an identical value on purpose (a controlled,
    # deterministic-looking fixture) -- but that means zero variance, so
    # margin=0 (std_baseline + std_candidate) for every comparison, and ANY
    # positive difference "wins". Non-winning candidates must therefore be
    # at or below baseline, never even slightly above it.
    baseline_cfg = dict(calibration_algorithm="minmax", granularity="per_tensor", symmetry="symmetric", calib_set_size=128)
    rows = []
    for rep in (1, 2, 3):
        rows.append(_vit_row(model_name, **baseline_cfg, accuracy_value=73.0, repeat_id=rep))
        rows.append(_vit_row(model_name, **{**baseline_cfg, "calibration_algorithm": "mse"}, accuracy_value=73.0, repeat_id=rep))
        rows.append(_vit_row(model_name, **{**baseline_cfg, "granularity": "per_channel"},
                              accuracy_value=(80.0 if granularity_wins else 72.95), repeat_id=rep))
        rows.append(_vit_row(model_name, **{**baseline_cfg, "symmetry": "asymmetric"}, accuracy_value=72.9, repeat_id=rep))
        rows.append(_vit_row(model_name, **{**baseline_cfg, "calib_set_size": 32}, accuracy_value=72.8, repeat_id=rep))
        rows.append(_vit_row(model_name, **{**baseline_cfg, "calib_set_size": 512}, accuracy_value=72.95, repeat_id=rep))
    return rows


def _synthetic_vit_rows():
    rows = []
    rows += _vit_model_rows("deit_tiny", granularity_wins=True)
    rows += _vit_model_rows("mobilevit")  # nothing wins -- dedup case
    rows += _vit_model_rows("efficientvit_b1", granularity_wins=True)
    return rows


def test_vit_phase3_manifest_dedups_when_nothing_wins():
    entries, dedup_notes = vit_phase3_manifest.generate(rows=_synthetic_vit_rows())
    by_model = {}
    for e in entries:
        by_model.setdefault(e["model_name"], []).append(e)

    assert len(by_model["mobilevit"]) == 2
    assert "mobilevit" in dedup_notes


def test_vit_phase3_manifest_generates_4_rows_when_a_factor_wins():
    entries, dedup_notes = vit_phase3_manifest.generate(rows=_synthetic_vit_rows())
    by_model = {}
    for e in entries:
        by_model.setdefault(e["model_name"], []).append(e)

    assert len(by_model["deit_tiny"]) == 4
    assert len(by_model["efficientvit_b1"]) == 4
    assert "deit_tiny" not in dedup_notes
    assert "efficientvit_b1" not in dedup_notes

    combined_rows = [e for e in by_model["deit_tiny"] if e["factor_level"] == "combined"]
    assert all(e["granularity"] == "per_channel" for e in combined_rows)


# --------------------------------------------------------------------------
# write_manifest_with_header: the dedup note actually lands in the file,
# and the file is still valid, loadable YAML
# --------------------------------------------------------------------------


def test_write_manifest_with_header_embeds_comments_and_stays_valid_yaml(tmp_path, monkeypatch):
    from configs import common as configs_common

    monkeypatch.setattr(configs_common, "CONFIGS_DIR", tmp_path)
    entries = [{"run_id": "a", "model_name": "x"}]
    header = ["Phase 3 manifest", "", "Models omitted: y (reason)"]

    path = write_manifest_with_header(entries, "test_manifest.yaml", header)
    text = path.read_text()

    assert "# Phase 3 manifest" in text
    assert "# Models omitted: y (reason)" in text
    loaded = yaml.safe_load(text)
    assert loaded == entries
