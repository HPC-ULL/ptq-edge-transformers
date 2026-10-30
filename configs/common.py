"""Shared machinery for generating experiment manifests (ViT + LLM branches).

Manifests are generated programmatically: a baseline config per branch plus
one-factor-at-a-time (OFAT) variants, never hand-written as static lists.
Each variant starts from the baseline and overrides exactly one factor, so
no two ablation factors change at once within a single entry (the declared
exceptions to this rule live in the branch-specific modules, not here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent


@dataclass
class Factor:
    """One ablation factor. `levels[0]` is always the baseline/reference level.

    `fixed` holds extra baseline overrides needed for this factor's variants
    to be meaningful (e.g. an LLM calib_set_size sweep only makes sense once
    calibration_corpus != "none"). These are held constant across every
    level of this factor, so they are not themselves ablation deviations.
    """

    name: str
    levels: List[Any]
    fixed: Dict[str, Any] = field(default_factory=dict)


def _slug(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def build_run_id(family: str, model: str, phase: str, factor: str, level: Any, bitwidth: str) -> str:
    return f"{family}_{model}_{phase}_{_slug(factor)}_{_slug(level)}_{_slug(bitwidth)}"


def baseline_entry(family: str, model: str, bitwidth: str, baseline: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "run_id": build_run_id(family, model, "phase1", "baseline", "default", bitwidth),
        "model_family": family,
        "model_name": model,
        "phase": "phase1",
        "factor_varied": "baseline",
        "factor_level": "default",
    }
    entry.update(baseline)
    return entry


def ofat_variants(
    family: str, model: str, bitwidth: str, baseline: Dict[str, Any], factors: List[Factor]
) -> List[Dict[str, Any]]:
    """Generate Phase-2 OFAT variants: baseline + exactly one factor override.

    For each factor, every non-baseline level (`levels[1:]`) produces one
    entry built from `baseline`, with the factor's `fixed` overrides applied
    (if any) and then only `factor.name` itself overridden.
    """
    entries: List[Dict[str, Any]] = []
    for factor in factors:
        base_for_factor = {**baseline, **factor.fixed}
        for level in factor.levels[1:]:
            entry = {
                "run_id": build_run_id(family, model, "phase2", factor.name, level, bitwidth),
                "model_family": family,
                "model_name": model,
                "phase": "phase2",
                "factor_varied": factor.name,
                "factor_level": str(level),
            }
            entry.update(base_for_factor)
            entry[factor.name] = level
            entries.append(entry)
    return entries


def check_unique_run_ids(entries: List[Dict[str, Any]], manifest_name: str) -> None:
    seen: Dict[str, int] = {}
    duplicates: List[str] = []
    for entry in entries:
        rid = entry["run_id"]
        seen[rid] = seen.get(rid, 0) + 1
        if seen[rid] == 2:
            duplicates.append(rid)
    if duplicates:
        raise ValueError(f"{manifest_name}: duplicate run_id(s) found, refusing to write: {duplicates}")


def write_manifest(entries: List[Dict[str, Any]], filename: str) -> Path:
    """Validate run_id uniqueness, then write `entries` as a flat YAML list."""
    check_unique_run_ids(entries, filename)
    out_path = CONFIGS_DIR / filename
    with open(out_path, "w") as f:
        yaml.safe_dump(entries, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return out_path


def write_manifest_with_header(entries: List[Dict[str, Any]], filename: str, header_lines: List[str]) -> Path:
    """Like write_manifest, but prepends `header_lines` as '#'-comments
    before the YAML content -- for a manifest that needs to record its own
    generation metadata (e.g. why some expected rows were intentionally
    omitted, e.g. configs/vit_phase3_manifest.py's dedup notes) in a way
    that survives being opened directly, not just logged to console. Still
    a flat YAML list once loaded (yaml.safe_load skips '#' comment lines
    like any YAML parser), so it's read the same way as write_manifest's
    output."""
    check_unique_run_ids(entries, filename)
    out_path = CONFIGS_DIR / filename
    header = "".join((f"# {line}\n" if line else "#\n") for line in header_lines)
    body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, default_flow_style=False)
    with open(out_path, "w") as f:
        f.write(header)
        f.write("\n")
        f.write(body)
    return out_path
