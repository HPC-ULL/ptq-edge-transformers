"""CLI entry point:
`python -m evaluation_offline.llm_perplexity --run-id <id> [--repeat-id <n>] [--dry-run] [--force]`
`python -m evaluation_offline.llm_perplexity --all [--dry-run] [--force]`

--repeat-id is required for stochastic rows (calibration_corpus != "none",
which get NUM_REPEATS independent GGUF artifacts -- see
quantization/llm_llamacpp/imatrix_gen.py) and forbidden for deterministic
rows (a single shared artifact, no repeat suffix); it is not valid with
--all, which expands every manifest row into its own jobs via the same
iter_jobs() used by quantization/llm_llamacpp/cli.py (51 total: 6
deterministic + 15 stochastic x 3 repeats).

--all is meant to be run incrementally, while quantization is still
producing artifacts elsewhere: jobs whose .gguf doesn't exist yet are
skipped (not a failure, exit code 0) with a clear listing at the end,
rather than requiring all 51 quantizations to finish first. --run-id
(a single, explicit request) still fails loudly if its artifact is missing.

The artifact-existence check only applies to real runs -- --dry-run is a
pure preview and never requires the artifact (or anything else) to exist
on disk, same as every other --dry-run in this project.

--phase3 reads configs/llm_phase3_manifest.yaml (via
configs.llm_phase3_manifest.generate(), which requires
results/phase3_selection_report.txt to already exist -- see configs/cli.py)
instead of the default Phase 1-2 manifest, reusing the SAME iter_jobs() as
quantization.llm_llamacpp --phase3 (imported from there, not reimplemented)
-- Phase 3 rows still split into deterministic ("baseline", always
calibration_corpus="none") and stochastic ("combined", if calibration_corpus
itself was Phase 2's winning factor for that model) exactly like Phase 1-2,
since imatrix_gen.needs_imatrix() only inspects calibration_corpus, which is
phase-independent. Job count is configurations x repeats (1 for
deterministic, NUM_REPEATS for stochastic) same as quantization -- this
module never repeats per artifact, it evaluates each .gguf exactly once.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

from configs import llm_manifest, llm_phase3_manifest
from quantization.llm_llamacpp.cli import iter_jobs
from quantization.llm_llamacpp.imatrix_gen import NUM_REPEATS, needs_imatrix
from quantization.llm_llamacpp.quantize import quantized_path

from . import perplexity


def _manifest_by_run_id(phase3: bool = False) -> Dict[str, dict]:
    if phase3:
        entries, _dedup_notes = llm_phase3_manifest.generate()
    else:
        entries = llm_manifest.generate()
    return {e["run_id"]: e for e in entries}


def _job_label(run_id: str, repeat_id) -> str:
    return run_id + (f" rep{repeat_id}" if repeat_id is not None else "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation_offline.llm_perplexity",
        description="Evaluate WikiText-2 test-split perplexity for quantized LLM-branch GGUF artifacts.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Evaluate a single manifest row (llm_experiment_manifest.yaml).")
    group.add_argument(
        "--all", action="store_true",
        help="Evaluate every (run_id, repeat_id) job in the manifest whose artifact already exists; "
        "skips the rest with a warning instead of failing.",
    )
    parser.add_argument(
        "--repeat-id", type=int, default=None,
        help=f"Which repeat's artifact to evaluate (1-{NUM_REPEATS}); only valid with --run-id on a stochastic row.",
    )
    parser.add_argument(
        "--phase3", action="store_true",
        help="Read configs/llm_phase3_manifest.yaml instead of the Phase 1-2 manifest.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the command(s) that would run, without running them or touching datasets/binaries/results.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if a result is already recorded.")
    return parser


def _run_one(entry: dict, repeat_id, *, dry_run: bool, force: bool) -> None:
    perplexity.evaluate_perplexity(entry, repeat_id, dry_run=dry_run, force=force)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = _manifest_by_run_id(args.phase3)

    if args.run_id:
        if args.run_id not in manifest:
            parser.error(f"Unknown run_id: {args.run_id!r}")
        entry = manifest[args.run_id]

        stochastic = needs_imatrix(entry)
        if stochastic and args.repeat_id is None:
            parser.error(f"{args.run_id}: stochastic row (calibration_corpus != none) requires --repeat-id (1-{NUM_REPEATS}).")
        if not stochastic and args.repeat_id is not None:
            parser.error(f"{args.run_id}: deterministic row (calibration_corpus == none) does not take --repeat-id.")

        # Dry-run is a pure preview -- it must not require the artifact (or
        # anything else) to exist on disk yet, same as every other --dry-run
        # in this project.
        if not args.dry_run:
            artifact_path = quantized_path(entry["run_id"], args.repeat_id)
            if not artifact_path.exists():
                parser.error(f"{args.run_id}: quantized artifact not found at {artifact_path} -- run quantization first.")

        _run_one(entry, args.repeat_id, dry_run=args.dry_run, force=args.force)
        return 0

    # --all
    if args.repeat_id is not None:
        parser.error("--repeat-id only applies together with --run-id, not --all.")

    jobs = list(iter_jobs(list(manifest.values())))
    missing: List[Tuple[str, str]] = []
    for entry, repeat_id in jobs:
        if not args.dry_run:
            artifact_path = quantized_path(entry["run_id"], repeat_id)
            if not artifact_path.exists():
                missing.append((_job_label(entry["run_id"], repeat_id), str(artifact_path)))
                continue
        _run_one(entry, repeat_id, dry_run=args.dry_run, force=args.force)

    if missing:
        print(f"\n[warning] {len(missing)}/{len(jobs)} artifacts not quantized yet -- evaluation skipped for:")
        for label, path in missing:
            print(f"  - {label}  ({path})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
