"""CSV loading + branch orchestration for Phase 3 selection (see
selection.py for the pure algorithm). Reads results/raw/vit_accuracy_offline
.csv and results/raw/llm_perplexity_offline.csv, applies the selection rule
per model, and formats the result as a readable table and as a YAML dict.

Running this module (`python -m analysis.phase3_selection`) prints the full
report to stdout AND persists the identical text to
results/phase3_selection_report.txt -- the durable record of which factor
won or kept baseline for each of the 6 models, referenced from
configs/vit_phase3_manifest.py and configs/llm_phase3_manifest.py's
generated YAML headers, and meant to be citable when writing the paper's
Phase 3 section without depending on terminal scrollback.

ViT rows carry two metrics per (run_id, repeat_id) -- top1_accuracy and
top5_accuracy (results_io.py) -- so rows are filtered to top1_accuracy only
before selection; top1 is the standard primary classification metric and
Section 9.3 doesn't name a specific one, so this is a design decision, not
a literal spec requirement.

LLM's "block-scheme exception" (configs/llm_manifest.py's own docstring:
not a plain OFAT factor) is actually two separate single-factor
comparisons, not one monolithic thing:

  - legacy_corpus_none vs. the global baseline: both have
    calibration_corpus="none", differing only in quant_scheme (k_quant ->
    legacy) -- a clean, single-factor OFAT comparison. Modeled here as its
    own Factor("quant_scheme", ["k_quant", "legacy"],
    fixed={"calibration_corpus": "none"}) and fed through the SAME generic
    select_phase3_config()/Factor.fixed mechanism already used for
    calib_set_size -- no new selection code, it participates in the
    combined Phase 3 config like any other factor.
  - legacy_corpus_none vs. legacy_corpus_wikitext2: this one genuinely
    changes two fields from the GLOBAL baseline (quant_scheme AND
    calibration_corpus) at once, so comparing it against the global
    baseline would confound quant_scheme's effect with corpus's effect
    (Section 9.1's own declared OFAT-independence limitation). But
    legacy_corpus_none -> legacy_corpus_wikitext2 by itself changes only
    calibration_corpus (both are quant_scheme="legacy") -- a clean
    single-factor comparison too, just with legacy_corpus_none as ITS
    baseline instead of the global one. This is H3's actual question
    ("does a calibration corpus help the legacy scheme?"), reported
    separately by compare_h3_legacy_corpus_effect() as a diagnostic result
    -- NOT folded into the Phase 3 combined config, since it isn't a
    comparison against that config's own baseline.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from configs import llm_manifest, vit_manifest
from configs.common import Factor

from .selection import FactorDecision, select_level, select_phase3_config, summarize_repeats

VIT_EXCLUDED_FIELD_VALUES = {"calibration_algorithm": {"entropy"}}
VIT_METRIC = "top1_accuracy"

# legacy_corpus_none vs. the global baseline: a clean single-factor
# deviation (quant_scheme only, calibration_corpus="none" on both sides) --
# fed through the same Factor.fixed mechanism as calib_set_size.
LLM_QUANT_SCHEME_FACTOR = Factor("quant_scheme", ["k_quant", "legacy"], fixed={"calibration_corpus": "none"})


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def select_vit_phase3(all_rows: List[Dict[str, str]], model_name: str) -> Dict[str, FactorDecision]:
    rows = [r for r in all_rows if r["model_name"] == model_name and r["accuracy_metric_name"] == VIT_METRIC]
    return select_phase3_config(
        rows, vit_manifest.FACTORS, vit_manifest.BASELINE,
        higher_is_better=True,
        excluded_field_values=VIT_EXCLUDED_FIELD_VALUES,
    )


def select_llm_phase3(all_rows: List[Dict[str, str]], model_name: str) -> Dict[str, FactorDecision]:
    rows = [r for r in all_rows if r["model_name"] == model_name]
    factors = list(llm_manifest.FACTORS) + [LLM_QUANT_SCHEME_FACTOR]
    return select_phase3_config(
        rows, factors, llm_manifest.BASELINE,
        higher_is_better=False,
        # Real rows have calib_set_size blanked whenever
        # calibration_corpus=="none" (see module docstring); without this,
        # quant_scheme's comparator config (fixed to calibration_corpus=
        # "none") would never match the real baseline/legacy_corpus_none
        # rows at all.
        normalize_config=llm_manifest._clear_inapplicable_calib_set_size,
    )


def compare_h3_legacy_corpus_effect(all_rows: List[Dict[str, str]], model_name: str):
    """Diagnostic-only result for the H3 discussion, NOT part of the Phase 3
    combined config: does adding a calibration corpus help the legacy
    (non-K-quant) block scheme? Uses legacy_corpus_none as the comparison
    baseline -- not the global baseline, which would confound this with
    quant_scheme's own effect. Returns None if either side has no data yet."""
    rows = [r for r in all_rows if r["model_name"] == model_name]
    baseline_rows = [r for r in rows if r["quant_scheme"] == "legacy" and r["calibration_corpus"] == "none"]
    candidate_rows = [r for r in rows if r["quant_scheme"] == "legacy" and r["calibration_corpus"] == "wikitext2"]
    if not baseline_rows or not candidate_rows:
        return None

    baseline_measurement = summarize_repeats([float(r["accuracy_value"]) for r in baseline_rows])
    candidate_measurement = summarize_repeats([float(r["accuracy_value"]) for r in candidate_rows])
    return select_level(
        "quant_scheme_corpus_effect_h3", "legacy_corpus_none", baseline_measurement,
        [("legacy_corpus_wikitext2", candidate_measurement)], higher_is_better=False,
    )


def _fmt_measurement(m) -> str:
    std_str = f"{m.std:.4f}" if m.std is not None else "n/a"
    return f"n={m.n} mean={m.mean:.4f} std={std_str}"


def _format_single_decision(decision: FactorDecision) -> List[str]:
    kept = decision.selected_level == decision.baseline_level
    outcome = "baseline kept" if kept else f"WINNER: {decision.selected_level!r}"
    lines = [
        f"  factor={decision.factor} baseline={decision.baseline_level!r} "
        f"({_fmt_measurement(decision.baseline_measurement)}) -> {outcome}"
    ]
    for c in decision.candidates:
        mark = "*" if c.beats_margin else " "
        lines.append(f"    [{mark}] level={c.level!r} ({_fmt_measurement(c.measurement)}) "
                      f"diff={c.diff:.4f} margin={c.margin:.4f}")
    lines.append(f"    reason: {decision.selected_reason}")
    return lines


def format_decisions(model_name: str, decisions: Dict[str, FactorDecision]) -> List[str]:
    lines = [f"=== {model_name} ==="]
    for decision in decisions.values():
        lines.extend(_format_single_decision(decision))
    lines.append("")
    return lines


def format_h3_diagnostic(model_name: str, decision: Optional[FactorDecision]) -> List[str]:
    lines = [
        f"--- {model_name}: H3 diagnostic (legacy_corpus_none vs. legacy_corpus_wikitext2, "
        f"NOT part of the Phase 3 combined config) ---"
    ]
    if decision is None:
        lines.append("  no data for one or both sides yet")
    else:
        lines.extend(_format_single_decision(decision))
    lines.append("")
    return lines


def selected_config(decisions: Dict[str, FactorDecision], normalize_config=lambda config: config) -> Dict[str, Any]:
    """Assembles each factor's independently-selected winning level into one
    combined dict. Each factor's decision is made in isolation (Section
    9.1's own declared OFAT-independence assumption), so the raw levels
    can combine into a config that's individually valid per factor but
    jointly incoherent -- e.g. calibration_corpus keeping its "none"
    baseline while calib_set_size independently keeps ITS OWN baseline
    (200), even though calib_set_size never applies without a corpus in
    use. `normalize_config` (same hook select_phase3_config() uses during
    comparator matching -- pass llm_manifest._clear_inapplicable_calib_set
    _size for the LLM branch) is applied to the ASSEMBLED dict too, not
    just during matching, so a combination the real pipeline would never
    generate doesn't survive into the final Phase 3 config."""
    combined = {factor_name: decision.selected_level for factor_name, decision in decisions.items()}
    return normalize_config(combined)


def build_report() -> "tuple[Dict[str, Dict[str, Any]], List[str]]":
    """Returns (combined, report_lines): `combined` is
    {model_name: {factor_name: selected_level}} for all 6 models (3 ViT +
    3 LLM); `report_lines` is the complete human-readable report (per-factor
    breakdown for every model, plus each LLM model's H3 diagnostic) as
    plain text lines, identical whether printed to the console or persisted
    to disk (see write_report_file())."""
    from evaluation_offline.llm_perplexity.paths import LLM_PERPLEXITY_RESULTS_CSV
    from evaluation_offline.vit_accuracy.paths import VIT_ACCURACY_RESULTS_CSV

    combined: Dict[str, Dict[str, Any]] = {}
    lines: List[str] = []

    vit_rows = read_csv(VIT_ACCURACY_RESULTS_CSV) if VIT_ACCURACY_RESULTS_CSV.exists() else []
    for model_name in vit_manifest.MODELS:
        decisions = select_vit_phase3(vit_rows, model_name)
        lines.extend(format_decisions(model_name, decisions))
        combined[model_name] = selected_config(decisions)

    llm_rows = read_csv(LLM_PERPLEXITY_RESULTS_CSV) if LLM_PERPLEXITY_RESULTS_CSV.exists() else []
    for model_name in llm_manifest.MODELS:
        decisions = select_llm_phase3(llm_rows, model_name)
        lines.extend(format_decisions(model_name, decisions))
        combined[model_name] = selected_config(decisions, normalize_config=llm_manifest._clear_inapplicable_calib_set_size)
        lines.extend(format_h3_diagnostic(model_name, compare_h3_legacy_corpus_effect(llm_rows, model_name)))

    return combined, lines


def run() -> Dict[str, Dict[str, Any]]:
    """Kept for callers that only need the combined config (e.g.
    configs/vit_phase3_manifest.py, configs/llm_phase3_manifest.py) without
    the console/file report -- see build_report()."""
    combined, _lines = build_report()
    return combined


def write_report_file(report_lines: List[str], combined: Dict[str, Dict[str, Any]]) -> Path:
    from quantization.vit_brevitas.paths import PROJECT_ROOT

    out_path = PROJECT_ROOT / "results" / "phase3_selection_report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_block = yaml.safe_dump(combined, sort_keys=False, allow_unicode=True, default_flow_style=False)
    full_text = "\n".join(report_lines) + "\n=== Phase 3 combined configuration (YAML) ===\n" + yaml_block

    out_path.write_text(full_text)
    return out_path


def main() -> None:
    combined, lines = build_report()
    yaml_block = yaml.safe_dump(combined, sort_keys=False, allow_unicode=True, default_flow_style=False)

    print("\n".join(lines))
    print("=== Phase 3 combined configuration (YAML) ===")
    print(yaml_block)

    out_path = write_report_file(lines, combined)
    print(f"(full report persisted to {out_path})")


if __name__ == "__main__":
    main()
