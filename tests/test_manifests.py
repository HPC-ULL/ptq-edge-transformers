"""Validation tests for configs/: manifest counts, run_id uniqueness, and the
one-factor-at-a-time (OFAT) invariant for Phase-2 entries.
"""

from __future__ import annotations

import pytest

from configs import common, llm_manifest, vit_manifest

# Fields that identify *where* an entry came from, not what it configures.
_METADATA_FIELDS = {"run_id", "model_family", "model_name", "phase", "factor_varied", "factor_level"}


def _config_diff(baseline: dict, entry: dict) -> set:
    """Field names whose value differs between `baseline` and `entry`, ignoring metadata."""
    keys = (baseline.keys() | entry.keys()) - _METADATA_FIELDS
    return {k for k in keys if baseline.get(k) != entry.get(k)}


# --------------------------------------------------------------------------
# common.py
# --------------------------------------------------------------------------

def test_check_unique_run_ids_raises_on_duplicates():
    entries = [{"run_id": "a"}, {"run_id": "b"}, {"run_id": "a"}]
    with pytest.raises(ValueError, match="duplicate run_id"):
        common.check_unique_run_ids(entries, "fake_manifest.yaml")


def test_check_unique_run_ids_passes_when_unique():
    entries = [{"run_id": "a"}, {"run_id": "b"}]
    common.check_unique_run_ids(entries, "fake_manifest.yaml")  # should not raise


# --------------------------------------------------------------------------
# ViT branch
# --------------------------------------------------------------------------

def test_vit_manifest_size():
    entries = vit_manifest.generate()
    assert len(entries) == 3 * 8 == 24


def test_vit_run_ids_unique():
    entries = vit_manifest.generate()
    run_ids = [e["run_id"] for e in entries]
    assert len(run_ids) == len(set(run_ids))


def test_vit_phase2_entries_deviate_from_baseline_in_exactly_one_field():
    entries = vit_manifest.generate()
    for entry in entries:
        if entry["phase"] != "phase2":
            continue
        diff = _config_diff(vit_manifest.BASELINE, entry)
        assert diff == {entry["factor_varied"]}, f"{entry['run_id']}: expected single deviation, got {diff}"


def test_vit_baseline_entries_match_baseline_dict():
    entries = vit_manifest.generate()
    baselines = [e for e in entries if e["phase"] == "phase1"]
    assert len(baselines) == len(vit_manifest.MODELS)
    for entry in baselines:
        assert _config_diff(vit_manifest.BASELINE, entry) == set()


# --------------------------------------------------------------------------
# LLM branch
# --------------------------------------------------------------------------

def test_llm_manifest_size():
    entries = llm_manifest.generate()
    assert len(entries) == 3 * 7 == 21


def test_llm_run_ids_unique():
    entries = llm_manifest.generate()
    run_ids = [e["run_id"] for e in entries]
    assert len(run_ids) == len(set(run_ids))


def test_llm_baseline_entries_have_no_applicable_calib_set_size():
    entries = llm_manifest.generate()
    baselines = [e for e in entries if e["phase"] == "phase1"]
    assert len(baselines) == len(llm_manifest.MODELS)
    for entry in baselines:
        assert entry["calibration_corpus"] == "none"
        assert entry["calib_set_size"] is None


def test_llm_calibration_corpus_variants_deviate_only_in_corpus():
    # Reference point: the phase-1 baseline itself (corpus is its only "fixed" factor).
    entries = llm_manifest.generate()
    variants = [e for e in entries if e["factor_varied"] == "calibration_corpus"]
    assert len(variants) == 2 * len(llm_manifest.MODELS)
    for entry in variants:
        assert entry["calib_set_size"] == llm_manifest.REFERENCE_CALIB_SET_SIZE
        diff = _config_diff(llm_manifest.BASELINE, entry)
        assert diff == {"calibration_corpus"}, f"{entry['run_id']}: expected single deviation, got {diff}"


def test_llm_calib_set_size_variants_deviate_only_in_size_from_their_own_reference():
    # This factor's reference point is "calibration_corpus=wikitext2" (fixed on the
    # factor itself, per the declared rule that size is meaningless without a corpus),
    # not the phase-1 global baseline.
    entries = llm_manifest.generate()
    variants = [e for e in entries if e["factor_varied"] == "calib_set_size"]
    assert len(variants) == 2 * len(llm_manifest.MODELS)
    reference = {**llm_manifest.BASELINE, "calibration_corpus": "wikitext2"}
    for entry in variants:
        assert entry["calibration_corpus"] == "wikitext2"
        diff = _config_diff(reference, entry)
        assert diff == {"calib_set_size"}, f"{entry['run_id']}: expected single deviation, got {diff}"


def test_llm_block_scheme_exception_rows():
    entries = llm_manifest.generate()
    per_model = [e for e in entries if e["model_name"] == llm_manifest.MODELS[0] and e["factor_varied"] == "quant_scheme"]
    assert len(per_model) == 2

    none_corpus = next(e for e in per_model if e["calibration_corpus"] == "none")
    assert none_corpus["quant_scheme"] == "legacy"
    assert none_corpus["calib_set_size"] is None
    # Single deviation from baseline (quant_scheme only) -- compare against the
    # baseline with calib_set_size cleared too, since "N/A when corpus=none"
    # applies identically to both rows and isn't itself an ablation deviation.
    cleared_baseline = {**llm_manifest.BASELINE, "calib_set_size": None}
    assert _config_diff(cleared_baseline, none_corpus) == {"quant_scheme"}

    wikitext_corpus = next(e for e in per_model if e["calibration_corpus"] == "wikitext2")
    assert wikitext_corpus["quant_scheme"] == "legacy"
    assert wikitext_corpus["calib_set_size"] == llm_manifest.REFERENCE_CALIB_SET_SIZE
    # Declared exception: two fields deviate from baseline at once.
    assert _config_diff(llm_manifest.BASELINE, wikitext_corpus) == {"quant_scheme", "calibration_corpus"}


def test_llm_block_scheme_does_not_duplicate_calibration_corpus_variant():
    entries = llm_manifest.generate()
    model = llm_manifest.MODELS[0]
    model_entries = [e for e in entries if e["model_name"] == model]

    def _config_only(entry: dict) -> tuple:
        keys = sorted(set(entry) - _METADATA_FIELDS)
        return tuple((k, entry[k]) for k in keys)

    configs = [_config_only(e) for e in model_entries]
    assert len(configs) == len(set(configs)), "two manifest rows quantize an identical config"
