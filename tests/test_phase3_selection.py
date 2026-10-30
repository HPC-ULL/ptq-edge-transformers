"""Tests for analysis/phase3_selection/ -- the Phase 3 "winner" selection
rule (technical_architecture.md Section 9.3).

select_level tests use synthetic Measurement objects directly (pure
algorithm, no CSV/manifest involved). select_phase3_config tests use
synthetic row dicts shaped like the real results CSV, to also exercise the
config-matching/exclusion plumbing around select_level.
"""

from __future__ import annotations

from configs import llm_manifest
from configs.common import Factor

from analysis.phase3_selection.selection import (
    Measurement,
    select_level,
    select_phase3_config,
)
from analysis.phase3_selection.report import (
    compare_h3_legacy_corpus_effect,
    select_llm_phase3,
    selected_config,
)


def _m(mean, std=None, n=None):
    return Measurement(n=n if n is not None else (1 if std is None else 3), mean=mean, std=std)


# --------------------------------------------------------------------------
# (a) a candidate that wins with a clear margin
# --------------------------------------------------------------------------

def test_candidate_wins_with_clear_margin():
    baseline = _m(mean=70.0, std=1.0)
    candidate = _m(mean=80.0, std=1.0)  # diff=10.0, margin=2.0 -> wins clearly

    decision = select_level("granularity", "per_tensor", baseline, [("per_channel", candidate)], higher_is_better=True)

    assert decision.selected_level == "per_channel"
    assert decision.candidates[0].beats_margin is True
    assert "beat baseline" in decision.selected_reason


# --------------------------------------------------------------------------
# (b) a candidate that does NOT beat the margin -- baseline is kept
# --------------------------------------------------------------------------

def test_candidate_within_margin_keeps_baseline():
    baseline = _m(mean=70.0, std=3.0)
    candidate = _m(mean=71.5, std=3.0)  # diff=1.5, margin=6.0 -> does not beat it

    decision = select_level("granularity", "per_tensor", baseline, [("per_channel", candidate)], higher_is_better=True)

    assert decision.selected_level == "per_tensor"  # baseline_level, unchanged
    assert decision.candidates[0].beats_margin is False
    assert "no candidate exceeded its margin" in decision.selected_reason


# --------------------------------------------------------------------------
# (c) deterministic baseline (n=1, std=None) -- margin uses candidate's std alone
# --------------------------------------------------------------------------

def test_deterministic_baseline_uses_only_candidates_std():
    baseline = _m(mean=8.50, std=None, n=1)  # e.g. LLM's calibration_corpus="none" baseline
    # margin = 0 (baseline) + 0.20 (candidate) = 0.20. Perplexity: lower is
    # better, so an improvement is baseline.mean - candidate.mean.
    winning_candidate = _m(mean=8.20, std=0.20)   # diff=0.30 > margin=0.20 -> wins
    losing_candidate = _m(mean=8.35, std=0.20)    # diff=0.15 < margin=0.20 -> baseline kept

    won = select_level("calibration_corpus", "none", baseline, [("wikitext2", winning_candidate)], higher_is_better=False)
    lost = select_level("calibration_corpus", "none", baseline, [("wikitext2", losing_candidate)], higher_is_better=False)

    assert won.selected_level == "wikitext2"
    assert won.candidates[0].margin == 0.20  # baseline's None std contributes 0, not an error
    assert lost.selected_level == "none"


# --------------------------------------------------------------------------
# (d) improvement direction is applied correctly in both senses
# --------------------------------------------------------------------------

def test_lower_is_better_direction_does_not_pick_the_higher_value():
    baseline = _m(mean=10.0, std=0.1)
    higher_candidate = _m(mean=15.0, std=0.1)  # higher value -- WORSE when lower is better
    lower_candidate = _m(mean=7.0, std=0.1)    # lower value -- BETTER when lower is better

    decision = select_level(
        "calibration_corpus", "none", baseline,
        [("worse", higher_candidate), ("better", lower_candidate)],
        higher_is_better=False,
    )

    assert decision.selected_level == "better"
    # the "worse" (higher-valued) candidate must be recorded as NOT beating its margin
    worse_result = next(c for c in decision.candidates if c.level == "worse")
    assert worse_result.beats_margin is False


def test_higher_is_better_direction_does_not_pick_the_lower_value():
    baseline = _m(mean=70.0, std=0.5)
    lower_candidate = _m(mean=60.0, std=0.5)   # lower value -- WORSE when higher is better
    higher_candidate = _m(mean=80.0, std=0.5)  # higher value -- BETTER when higher is better

    decision = select_level(
        "granularity", "per_tensor", baseline,
        [("worse", lower_candidate), ("better", higher_candidate)],
        higher_is_better=True,
    )

    assert decision.selected_level == "better"


# --------------------------------------------------------------------------
# (e) an entropy row with an artificially favorable value must not be able
# to win in ViT -- explicit exclusion, not merely absent data
# --------------------------------------------------------------------------

def _vit_row(model_name, calibration_algorithm, granularity, symmetry, calib_set_size, accuracy_value, repeat_id):
    return {
        "model_name": model_name,
        "calibration_algorithm": calibration_algorithm,
        "granularity": granularity,
        "symmetry": symmetry,
        "calib_set_size": str(calib_set_size),
        "accuracy_value": str(accuracy_value),
        "repeat_id": str(repeat_id),
    }


def test_entropy_row_with_favorable_value_cannot_win_in_vit():
    baseline_cfg = dict(calibration_algorithm="minmax", granularity="per_tensor", symmetry="symmetric", calib_set_size=128)
    factor = Factor("calibration_algorithm", ["minmax", "mse", "entropy"])

    rows = []
    for rep in (1, 2, 3):
        rows.append(_vit_row("deit_tiny", **baseline_cfg, accuracy_value=73.0, repeat_id=rep))
        rows.append(_vit_row("deit_tiny", **{**baseline_cfg, "calibration_algorithm": "mse"}, accuracy_value=73.5, repeat_id=rep))
        # Artificially favorable -- if the exclusion were missing/broken, this would trivially win.
        rows.append(_vit_row("deit_tiny", **{**baseline_cfg, "calibration_algorithm": "entropy"}, accuracy_value=99.9, repeat_id=rep))

    decisions = select_phase3_config(
        rows, [factor], baseline_cfg,
        higher_is_better=True,
        excluded_field_values={"calibration_algorithm": {"entropy"}},
    )

    decision = decisions["calibration_algorithm"]
    assert decision.selected_level != "entropy"
    assert all(c.level != "entropy" for c in decision.candidates)  # never even considered as a candidate


def test_entropy_exclusion_does_not_affect_other_factors():
    """The exclusion filter must not accidentally drop non-entropy rows for
    other factors -- it targets calibration_algorithm=='entropy' specifically."""
    baseline_cfg = dict(calibration_algorithm="minmax", granularity="per_tensor", symmetry="symmetric", calib_set_size=128)
    granularity_factor = Factor("granularity", ["per_tensor", "per_channel"])

    rows = []
    for rep in (1, 2, 3):
        rows.append(_vit_row("deit_tiny", **baseline_cfg, accuracy_value=73.0, repeat_id=rep))
        rows.append(_vit_row("deit_tiny", **{**baseline_cfg, "granularity": "per_channel"}, accuracy_value=76.0, repeat_id=rep))

    decisions = select_phase3_config(
        rows, [granularity_factor], baseline_cfg,
        higher_is_better=True,
        excluded_field_values={"calibration_algorithm": {"entropy"}},
    )

    assert decisions["granularity"].selected_level == "per_channel"


# --------------------------------------------------------------------------
# config-matching plumbing: a factor's `fixed` overrides change its
# baseline comparator (needed for LLM's calib_set_size factor)
# --------------------------------------------------------------------------

def test_factor_fixed_overrides_change_the_baseline_comparator():
    baseline_cfg = dict(calibration_corpus="none", calib_set_size="200", quant_scheme="k_quant")
    # calib_set_size is only meaningful with a corpus in use -- fixed to wikitext2.
    factor = Factor("calib_set_size", [200, 50], fixed={"calibration_corpus": "wikitext2"})

    def _llm_row(calibration_corpus, calib_set_size, accuracy_value, repeat_id):
        return {
            "model_name": "tinyllama_1_1b",
            "calibration_corpus": calibration_corpus,
            "calib_set_size": str(calib_set_size),
            "quant_scheme": "k_quant",
            "accuracy_value": str(accuracy_value),
            "repeat_id": str(repeat_id),
        }

    rows = []
    for rep in (1, 2, 3):
        # The wikitext2/200 row is calib_set_size's baseline comparator, NOT
        # the deterministic calibration_corpus="none" global baseline.
        rows.append(_llm_row("wikitext2", 200, accuracy_value=9.00, repeat_id=rep))
        rows.append(_llm_row("wikitext2", 50, accuracy_value=8.50, repeat_id=rep))  # lower PPL = better

    decisions = select_phase3_config(rows, [factor], baseline_cfg, higher_is_better=False)

    assert decisions["calib_set_size"].baseline_measurement.mean == 9.00
    assert decisions["calib_set_size"].selected_level == 50


def test_no_candidate_data_available_keeps_baseline_without_erroring():
    baseline_cfg = dict(calibration_algorithm="minmax", granularity="per_tensor", symmetry="symmetric", calib_set_size=128)
    factor = Factor("calibration_algorithm", ["minmax", "percentile"])

    rows = [_vit_row("deit_tiny", **baseline_cfg, accuracy_value=73.0, repeat_id=rep) for rep in (1, 2, 3)]
    # no "percentile" rows at all -- e.g. not evaluated yet

    decisions = select_phase3_config(rows, [factor], baseline_cfg, higher_is_better=True)

    decision = decisions["calibration_algorithm"]
    assert decision.selected_level == "minmax"
    assert decision.candidates == []
    assert decision.selected_reason == "no candidate data available"


# --------------------------------------------------------------------------
# quant_scheme as a proper factor (report.select_llm_phase3), and the H3
# diagnostic comparing legacy_corpus_none vs. legacy_corpus_wikitext2
# --------------------------------------------------------------------------

def _llm_row(model_name, calibration_corpus, quant_scheme, accuracy_value, calib_set_size=None, repeat_id=None):
    # Mirrors configs/llm_manifest.py's real behavior: calib_set_size is
    # blanked whenever calibration_corpus == "none"; bitwidth is always
    # "q4km" for Phase 1-2 and is part of every factor's comparator config
    # (it's in llm_manifest.BASELINE), so it must be present here too.
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


def _minimal_llm_dataset(model_name="tinyllama_1_1b"):
    """A complete-enough synthetic dataset for select_llm_phase3 to run
    every factor without raising (real baseline comparator rows must exist
    for calibration_corpus, calib_set_size, and quant_scheme), with only
    quant_scheme constructed to have a clear winner."""
    rows = [
        # global baseline: deterministic, n=1
        _llm_row(model_name, "none", "k_quant", accuracy_value=10.0),
        # legacy_corpus_none: quant_scheme's candidate, ALSO deterministic
        # (calibration_corpus="none") -- clearly better (lower PPL).
        _llm_row(model_name, "none", "legacy", accuracy_value=9.0),
    ]
    # calibration_corpus factor: wikitext2 is calib_set_size's baseline
    # comparator too (fixed={"calibration_corpus": "wikitext2"}) -- neither
    # candidate is constructed to win, to keep the test focused on quant_scheme.
    for rep in (1, 2, 3):
        rows.append(_llm_row(model_name, "wikitext2", "k_quant", accuracy_value=10.05, calib_set_size=200, repeat_id=rep))
        rows.append(_llm_row(model_name, "domain_corpus", "k_quant", accuracy_value=10.1, calib_set_size=200, repeat_id=rep))
        rows.append(_llm_row(model_name, "wikitext2", "k_quant", accuracy_value=10.2, calib_set_size=50, repeat_id=rep))
        rows.append(_llm_row(model_name, "wikitext2", "k_quant", accuracy_value=10.3, calib_set_size=1000, repeat_id=rep))
    return rows


def test_quant_scheme_can_contribute_a_winner_to_the_combined_config():
    rows = _minimal_llm_dataset()

    decisions = select_llm_phase3(rows, "tinyllama_1_1b")

    assert "quant_scheme" in decisions
    decision = decisions["quant_scheme"]
    assert decision.baseline_measurement.n == 1 and decision.baseline_measurement.std is None  # deterministic
    assert decision.selected_level == "legacy"
    assert selected_config(decisions)["quant_scheme"] == "legacy"


def test_quant_scheme_keeps_baseline_when_legacy_does_not_win():
    rows = _minimal_llm_dataset()
    # Replace legacy_corpus_none with a value WORSE than baseline (higher
    # PPL). Both sides are deterministic (margin=0, Section 9.3's special
    # case generalized), so this must fail to win, not just fail to clear
    # some nonzero margin.
    rows = [r for r in rows if not (r["quant_scheme"] == "legacy" and r["calibration_corpus"] == "none")]
    rows.append(_llm_row("tinyllama_1_1b", "none", "legacy", accuracy_value=10.5))

    decisions = select_llm_phase3(rows, "tinyllama_1_1b")

    assert decisions["quant_scheme"].selected_level == "k_quant"  # baseline kept


def test_h3_diagnostic_uses_legacy_corpus_none_as_comparator_not_global_baseline():
    model_name = "tinyllama_1_1b"
    rows = [
        _llm_row(model_name, "none", "k_quant", accuracy_value=10.0),   # global baseline
        _llm_row(model_name, "none", "legacy", accuracy_value=15.0),    # legacy_corpus_none: much worse than global baseline
    ]
    for rep in (1, 2, 3):
        # legacy_corpus_wikitext2: better than legacy_corpus_none (15.0) but
        # WORSE than the global baseline (10.0) -- an adversarial value that
        # only passes if H3 compares against legacy_corpus_none specifically.
        rows.append(_llm_row(model_name, "wikitext2", "legacy", accuracy_value=12.0, calib_set_size=200, repeat_id=rep))

    decision = compare_h3_legacy_corpus_effect(rows, model_name)

    assert decision is not None
    assert decision.baseline_measurement.mean == 15.0  # legacy_corpus_none, NOT the global baseline's 10.0
    assert decision.selected_level == "legacy_corpus_wikitext2"  # a real improvement relative to the correct comparator


def test_h3_diagnostic_returns_none_when_data_missing():
    rows = [_llm_row("tinyllama_1_1b", "none", "k_quant", accuracy_value=10.0)]  # only the global baseline

    assert compare_h3_legacy_corpus_effect(rows, "tinyllama_1_1b") is None


# --------------------------------------------------------------------------
# selected_config() final-assembly bug: each factor is decided independently,
# so calibration_corpus keeping its "none" baseline and calib_set_size
# independently keeping ITS OWN baseline (200) can coexist in the raw
# assembled dict, even though calib_set_size never applies without a corpus
# -- normalize_config must be reapplied after assembly, not just during
# comparator matching.
# --------------------------------------------------------------------------

def test_selected_config_clears_calib_set_size_when_corpus_baseline_is_kept():
    # _minimal_llm_dataset() already reproduces the real bug scenario: none
    # of calibration_corpus's candidates (wikitext2, domain_corpus) beat the
    # global baseline, and none of calib_set_size's candidates (50, 1000)
    # beat ITS OWN baseline (the wikitext2/200 row) -- both factors keep
    # their baseline level independently.
    rows = _minimal_llm_dataset()

    decisions = select_llm_phase3(rows, "tinyllama_1_1b")
    assert decisions["calibration_corpus"].selected_level == "none"
    assert decisions["calib_set_size"].selected_level == 200  # each factor's OWN baseline, before assembly

    raw_combined = selected_config(decisions)  # no normalize_config -- reproduces the bug
    assert raw_combined["calibration_corpus"] == "none"
    assert raw_combined["calib_set_size"] == 200  # semantically invalid: corpus=none but calib_set_size set

    fixed_combined = selected_config(decisions, normalize_config=llm_manifest._clear_inapplicable_calib_set_size)
    assert fixed_combined["calibration_corpus"] == "none"
    assert fixed_combined["calib_set_size"] is None  # cleared -- matches how the real pipeline generates this row
