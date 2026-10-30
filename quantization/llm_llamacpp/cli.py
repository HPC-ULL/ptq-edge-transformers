"""CLI entry point:
`python -m quantization.llm_llamacpp --run-id <id> | --all [--phase3] [--dry-run] [--force]`.

Expands each selected manifest row into one or more jobs via `iter_jobs`:
deterministic rows (calibration_corpus == "none") yield a single job with
repeat_id=None; stochastic rows yield NUM_REPEATS jobs, one per repeat_id
(see imatrix_gen.py for why a fixed seed across repeats would defeat their
purpose). Each job orchestrates: base model prep (once per model, shared
across jobs) -> imatrix generation (stochastic jobs only) -> quantization.

--phase3 reads configs/llm_phase3_manifest.yaml (via
configs.llm_phase3_manifest.generate(), which requires
results/phase3_selection_report.txt to already exist -- see configs/cli.py)
instead of the default Phase 1-2 manifest. iter_jobs' deterministic-vs-
stochastic split is driven by imatrix_gen.needs_imatrix(), which only
inspects an entry's calibration_corpus field -- phase-independent, so Phase
3 rows (both "baseline" rows, always calibration_corpus="none", and
"combined" rows, which may or may not be, depending on whether
calibration_corpus was itself Phase 2's winning factor for that model) are
expanded exactly the same way as Phase 1-2 rows, with no special-casing
needed here (confirmed by tests/test_llm_quantize_pipeline.py's phase3
iter_jobs tests, not just assumed).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Dict, Iterator, List, Optional, Tuple

from configs import llm_manifest, llm_phase3_manifest

from . import base_model, imatrix_gen, quantize
from .common_exec import RunFn

Job = Tuple[dict, Optional[int]]


def _manifest_by_run_id(phase3: bool = False) -> Dict[str, dict]:
    if phase3:
        entries, _dedup_notes = llm_phase3_manifest.generate()
    else:
        entries = llm_manifest.generate()
    return {e["run_id"]: e for e in entries}


def iter_jobs(entries: List[dict]) -> Iterator[Job]:
    for entry in entries:
        if imatrix_gen.needs_imatrix(entry):
            for repeat_id in range(1, imatrix_gen.NUM_REPEATS + 1):
                yield entry, repeat_id
        else:
            yield entry, None


def process_job(
    entry: dict, repeat_id: Optional[int], *, dry_run: bool, force: bool, run_fn: RunFn = subprocess.run
) -> None:
    base_model.prepare_base_model(entry["model_name"], dry_run=dry_run, force=force, run_fn=run_fn)
    if imatrix_gen.needs_imatrix(entry):
        imatrix_gen.prepare_imatrix(entry, repeat_id, dry_run=dry_run, force=force, run_fn=run_fn)
    quantize.prepare_quantized(entry, repeat_id, dry_run=dry_run, force=force, run_fn=run_fn)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantization.llm_llamacpp",
        description="Produce GGUF artifacts for the LLM-branch experiment manifest (llm_experiment_manifest.yaml).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Process a single manifest row by run_id (all its repeats, if stochastic).")
    group.add_argument("--all", action="store_true", help="Process every row in the manifest.")
    parser.add_argument(
        "--phase3", action="store_true",
        help="Read configs/llm_phase3_manifest.yaml instead of the Phase 1-2 manifest.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would run, without running them or touching datasets/binaries.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate artifacts even if they already exist.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = _manifest_by_run_id(args.phase3)

    if args.run_id:
        if args.run_id not in manifest:
            parser.error(f"Unknown run_id: {args.run_id!r}")
        entries = [manifest[args.run_id]]
    else:
        entries = list(manifest.values())

    for entry, repeat_id in iter_jobs(entries):
        process_job(entry, repeat_id, dry_run=args.dry_run, force=args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())
