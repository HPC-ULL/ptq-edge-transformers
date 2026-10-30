"""Phase 3 "winner" selection rule (technical_architecture.md
Section 9.3), shared between the ViT and LLM branches.

Pure algorithm, no file/CSV I/O -- see report.py for that. Fully unit
-testable with synthetic Measurement/row data (tests/test_phase3_selection.py).

The rule (Section 9.3):
1. For each factor, compute each level's mean accuracy_value across its
   repeat_id repetitions.
2. Select the level that improves on the baseline in the branch's
   improvement direction (higher_is_better=True for ViT accuracy,
   False for LLM perplexity -- parametrized, never hardcoded to one sense)
   ONLY IF that improvement exceeds the combined stdev of baseline and
   candidate (margin = baseline.std + candidate.std, treating a missing
   stdev -- a deterministic, n=1 row, e.g. LLM's baseline/legacy_corpus_none
   rows -- as contributing 0). This one formula naturally covers the
   explicitly-described special case (deterministic baseline -> margin =
   candidate's stdev alone) and its unstated symmetric case (both sides
   deterministic -> margin = 0, an unguarded direct comparison, since
   there's no sampling noise on either side to guard against).
3. If no level exceeds its margin, keep the baseline level for that
   factor -- "no clear winner" is a valid, final outcome, never forced.

Design decision not literally pinned down by Section 9.3, flagged here:
each factor's "baseline" comparator is `{**baseline, **factor.fixed}`, not
always the branch's single global baseline row. This matters for the LLM
branch's calib_set_size factor, which configs/llm_manifest.py deliberately
generates with calibration_corpus fixed to "wikitext2" rather than the
global baseline's "none" (a calib_set_size sweep is meaningless without a
corpus at all) -- comparing its candidates against the deterministic
global baseline would confound the calib_set_size effect with the
corpus-presence effect entirely. Using `factor.fixed` (imported directly
from the branch's own manifest module, the same source of truth
ofat_variants() itself uses to generate these rows) keeps this correct and
auditable without hardcoding branch-specific logic here.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from configs.common import Factor


@dataclass(frozen=True)
class Measurement:
    n: int
    mean: float
    std: Optional[float]  # None iff n == 1 (deterministic row)


def summarize_repeats(values: Sequence[float]) -> Measurement:
    if not values:
        raise ValueError("cannot summarize zero repeats")
    n = len(values)
    return Measurement(n=n, mean=statistics.mean(values), std=statistics.stdev(values) if n > 1 else None)


@dataclass(frozen=True)
class CandidateResult:
    level: Any
    measurement: Measurement
    diff: float  # improvement over baseline, oriented so positive = better in this branch's direction
    margin: float
    beats_margin: bool


@dataclass(frozen=True)
class FactorDecision:
    factor: str
    baseline_level: Any
    baseline_measurement: Measurement
    candidates: List[CandidateResult]
    selected_level: Any
    selected_reason: str


def _margin(baseline: Measurement, candidate: Measurement) -> float:
    return (baseline.std or 0.0) + (candidate.std or 0.0)


def select_level(
    factor_name: str,
    baseline_level: Any,
    baseline_measurement: Measurement,
    candidate_measurements: List[Tuple[Any, Measurement]],
    *,
    higher_is_better: bool,
) -> FactorDecision:
    candidates: List[CandidateResult] = []
    for level, measurement in candidate_measurements:
        raw_diff = measurement.mean - baseline_measurement.mean
        diff = raw_diff if higher_is_better else -raw_diff
        margin = _margin(baseline_measurement, measurement)
        candidates.append(CandidateResult(
            level=level, measurement=measurement, diff=diff, margin=margin, beats_margin=diff > margin,
        ))

    winners = [c for c in candidates if c.beats_margin]
    if winners:
        best = max(winners, key=lambda c: c.diff)
        return FactorDecision(
            factor=factor_name, baseline_level=baseline_level, baseline_measurement=baseline_measurement,
            candidates=candidates, selected_level=best.level,
            selected_reason=f"level {best.level!r} beat baseline by {best.diff:.4f} > margin {best.margin:.4f}",
        )
    reason = "no candidate data available" if not candidates else "no candidate exceeded its margin over baseline"
    return FactorDecision(
        factor=factor_name, baseline_level=baseline_level, baseline_measurement=baseline_measurement,
        candidates=candidates, selected_level=baseline_level, selected_reason=reason,
    )


def _row_matches(row: Dict[str, Any], target: Dict[str, Any]) -> bool:
    for key, value in target.items():
        if value is None:
            # A None target means "this field doesn't apply and was left
            # blank" (e.g. configs/llm_manifest.py's
            # _clear_inapplicable_calib_set_size blanks calib_set_size
            # whenever calibration_corpus == "none") -- match an empty
            # string, not the literal text "None".
            if row.get(key, "") != "":
                return False
        elif str(row.get(key, "")) != str(value):
            return False
    return True


def _rows_for_config(rows: List[Dict[str, Any]], target: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in rows if _row_matches(r, target)]


def select_phase3_config(
    rows: List[Dict[str, Any]],
    factors: List[Factor],
    baseline: Dict[str, Any],
    *,
    higher_is_better: bool,
    metric_column: str = "accuracy_value",
    excluded_field_values: Optional[Dict[str, Set[Any]]] = None,
    normalize_config=lambda config: config,
) -> Dict[str, FactorDecision]:
    """One FactorDecision per factor in `factors`. `rows` should already be
    filtered to one model (and, for ViT, one accuracy_metric_name -- see
    report.py) before calling this.

    `excluded_field_values`: {field_name: {values to drop}}, e.g.
    {"calibration_algorithm": {"entropy"}} for ViT. Applied as an explicit
    filter on `rows` up front -- not relied upon merely being absent from
    the data (entropy is blocked in quantizer_config.py and its results
    have been removed, but this filter must hold even if stale/re-added
    rows exist, per the same "auditable filter, not coincidence" standard
    already applied to entropy elsewhere in this project).

    `normalize_config`: applied to each comparator/candidate config dict
    right after it's built, before matching rows against it -- lets a
    branch apply its own manifest-generation-time field-clearing rules
    (kept branch-agnostic here). E.g. LLM's
    configs.llm_manifest._clear_inapplicable_calib_set_size blanks
    calib_set_size whenever calibration_corpus == "none", which the real
    generated rows (and thus the real results CSV) reflect -- any factor
    whose comparator config lands on calibration_corpus="none" (the global
    baseline itself, or e.g. a quant_scheme factor fixed to it) needs this
    applied or its comparator config won't match any real row at all.
    Identity by default (ViT needs no such rule)."""
    excluded_field_values = excluded_field_values or {}
    rows = [
        r for r in rows
        if not any(r.get(field) in values for field, values in excluded_field_values.items())
    ]

    decisions: Dict[str, FactorDecision] = {}
    for factor in factors:
        # Raw (unnormalized) base, built fresh from baseline+fixed exactly
        # like configs/common.py's ofat_variants() does -- each config
        # (comparator and every candidate) is normalized independently
        # from this same raw base, not by extending an already-normalized
        # dict, since normalize_config's rule (e.g. "blank calib_set_size
        # when calibration_corpus=='none'") only ever clears a field, it
        # never restores one, so chaining would leak a stale clear from
        # the comparator into a candidate whose own corpus differs.
        base_for_factor = {**baseline, **factor.fixed}

        comparator_config = normalize_config(dict(base_for_factor))
        baseline_rows = _rows_for_config(rows, comparator_config)
        if not baseline_rows:
            raise ValueError(f"factor {factor.name!r}: no rows match its baseline comparator config {comparator_config}")
        baseline_measurement = summarize_repeats([float(r[metric_column]) for r in baseline_rows])

        candidate_measurements: List[Tuple[Any, Measurement]] = []
        for level in factor.levels[1:]:
            candidate_config = normalize_config({**base_for_factor, factor.name: level})
            candidate_rows = _rows_for_config(rows, candidate_config)
            if not candidate_rows:
                continue  # no data for this level (e.g. excluded, or not yet evaluated)
            candidate_measurements.append(
                (level, summarize_repeats([float(r[metric_column]) for r in candidate_rows]))
            )

        decisions[factor.name] = select_level(
            factor.name, factor.levels[0], baseline_measurement, candidate_measurements,
            higher_is_better=higher_is_better,
        )

    return decisions
