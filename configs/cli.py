"""CLI entry point: `python -m configs <command>`."""

from __future__ import annotations

import argparse
import sys

from . import llm_manifest, llm_phase3_manifest, vit_manifest, vit_phase3_manifest

# Phase 3 generation recomputes the selection fresh from results/raw/ each
# time (same "generated programmatically, never cached" philosophy as every
# other manifest here) -- it does NOT read this file's contents. But its
# existence is required as an explicit precondition anyway: it's the
# tangible signal that `python -m analysis.phase3_selection` has actually
# been run against complete Phase 1-2 results, which is also exactly what
# generate() itself needs to succeed. Checking for it up front turns a
# missing/incomplete-Phase-2-results situation into one clear message
# instead of a ValueError surfacing from deep inside select_phase3_config
# with no context about *why* a comparator config has no matching rows.
_PHASE3_REPORT_RELATIVE_PATH = "results/phase3_selection_report.txt"


def _phase3_report_path():
    from quantization.vit_brevitas.paths import PROJECT_ROOT
    return PROJECT_ROOT / _PHASE3_REPORT_RELATIVE_PATH


def _require_phase3_report() -> None:
    report_path = _phase3_report_path()
    if not report_path.exists():
        raise SystemExit(
            f"Phase 3 generation requires {_PHASE3_REPORT_RELATIVE_PATH} to exist first -- run "
            f"`python -m analysis.phase3_selection` (which itself needs complete Phase 1-2 results: "
            f"results/raw/vit_accuracy_offline.csv and results/raw/llm_perplexity_offline.csv) before "
            f"generating Phase 3 manifests."
        )


def cmd_vit(_args: argparse.Namespace) -> None:
    vit_manifest.write()


def cmd_llm(_args: argparse.Namespace) -> None:
    llm_manifest.write()


def cmd_vit_phase3(_args: argparse.Namespace) -> None:
    _require_phase3_report()
    try:
        vit_phase3_manifest.write()
    except ValueError as e:
        raise SystemExit(
            f"Phase 3 generation failed -- Phase 1-2 ViT results appear incomplete or missing for at "
            f"least one model/factor: {e}\nRun evaluation_offline.vit_accuracy --all (after "
            f"quantization.vit_brevitas --all) to complete Phase 1-2, then retry."
        ) from None


def cmd_llm_phase3(_args: argparse.Namespace) -> None:
    _require_phase3_report()
    try:
        llm_phase3_manifest.write()
    except ValueError as e:
        raise SystemExit(
            f"Phase 3 generation failed -- Phase 1-2 LLM results appear incomplete or missing for at "
            f"least one model/factor: {e}\nRun evaluation_offline.llm_perplexity --all (after "
            f"quantization.llm_llamacpp --all) to complete Phase 1-2, then retry."
        ) from None


def cmd_all(_args: argparse.Namespace) -> None:
    # Phase 1-2 first, unconditionally -- Phase 3 depends on their results,
    # not the other way around.
    vit_manifest.write()
    llm_manifest.write()
    cmd_vit_phase3(_args)
    cmd_llm_phase3(_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="configs",
        description="Generate experiment manifests for the ViT and LLM ablation branches.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_vit = sub.add_parser("vit", help="Generate configs/vit_experiment_manifest.yaml (Phase 1-2)")
    p_vit.set_defaults(func=cmd_vit)

    p_llm = sub.add_parser("llm", help="Generate configs/llm_experiment_manifest.yaml (Phase 1-2)")
    p_llm.set_defaults(func=cmd_llm)

    p_vit_phase3 = sub.add_parser(
        "vit-phase3",
        help="Generate configs/vit_phase3_manifest.yaml (requires results/phase3_selection_report.txt)",
    )
    p_vit_phase3.set_defaults(func=cmd_vit_phase3)

    p_llm_phase3 = sub.add_parser(
        "llm-phase3",
        help="Generate configs/llm_phase3_manifest.yaml (requires results/phase3_selection_report.txt)",
    )
    p_llm_phase3.set_defaults(func=cmd_llm_phase3)

    p_all = sub.add_parser(
        "all",
        help="Generate all four manifests: vit, llm, vit-phase3, llm-phase3 (in that order)",
    )
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
