"""CLI entry point:
`python -m evaluation_offline.vit_accuracy --run-id <id> [--repeat-id <n>] [--phase3] [--dry-run] [--force]`
`python -m evaluation_offline.vit_accuracy --all [--phase3] [--dry-run] [--force]`

Every ViT-branch manifest row is stochastic (no deterministic exception
like the LLM branch), so --run-id without --repeat-id evaluates all
NUM_REPEATS=3 artifacts for that row; --repeat-id narrows to just one.
--all expands every row's 3 repeats: 24 rows x 3 = 72 jobs.

--phase3 reads configs/vit_phase3_manifest.yaml (via
configs.vit_phase3_manifest.generate(), which requires
results/phase3_selection_report.txt to already exist -- see configs/cli.py)
instead of the default Phase 1-2 manifest, evaluating whatever .onnx
artifacts quantization.vit_brevitas --phase3 has produced so far. Job
count is configurations x NUM_REPEATS same as quantization -- this module
never repeats per artifact, it evaluates each exported .onnx exactly once.

--run-id and --all deliberately behave differently on a missing artifact --
this is a single-request-vs-batch distinction, not a
deterministic-vs-stochastic one (ViT has no deterministic rows at all), so
it applies here exactly as it does in evaluation_offline/llm_perplexity/:
- --run-id (a single, explicit request): fails loudly (parser.error) if any
  of its requested repeat's artifact is missing. Asking for one specific
  row and getting silence back would hide a real pipeline/user error.
- --all (an incremental batch scan, meant to run while quantization/export
  is still producing artifacts elsewhere): skips jobs whose artifact
  doesn't exist yet (not a failure, exit code 0), with a clear listing at
  the end, rather than requiring all 72 exports to finish first.

The artifact-existence check only applies to real runs: --dry-run is a
pure preview and never requires the artifact (or anything else) to exist
on disk, same as every other --dry-run in this project.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

from configs import vit_manifest, vit_phase3_manifest
from export.vit_to_onnx import onnx_path
from quantization.vit_brevitas.calibration import NUM_REPEATS

from . import accuracy


def _manifest_by_run_id(phase3: bool = False) -> Dict[str, dict]:
    if phase3:
        entries, _dedup_notes = vit_phase3_manifest.generate()
    else:
        entries = vit_manifest.generate()
    return {e["run_id"]: e for e in entries}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation_offline.vit_accuracy",
        description="Evaluate ImageNet-1K top-1/top-5 accuracy for quantized ViT-branch ONNX artifacts.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Evaluate a single manifest row (vit_experiment_manifest.yaml).")
    group.add_argument(
        "--all", action="store_true",
        help="Evaluate every (run_id, repeat_id) job in the manifest whose artifact already exists; "
        "skips the rest with a warning instead of failing.",
    )
    parser.add_argument(
        "--repeat-id", type=int, default=None,
        help=f"Only evaluate this one repeat (1-{NUM_REPEATS}) of --run-id; omit to evaluate all repeats.",
    )
    parser.add_argument(
        "--phase3", action="store_true",
        help="Read configs/vit_phase3_manifest.yaml instead of the Phase 1-2 manifest.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run, without running it or touching datasets/models/results.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if a result is already recorded.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = _manifest_by_run_id(args.phase3)

    if args.run_id:
        if args.run_id not in manifest:
            parser.error(f"Unknown run_id: {args.run_id!r}")
        if args.repeat_id is not None and not (1 <= args.repeat_id <= NUM_REPEATS):
            parser.error(f"--repeat-id must be between 1 and {NUM_REPEATS}.")
        entry = manifest[args.run_id]
        repeat_ids = [args.repeat_id] if args.repeat_id is not None else list(range(1, NUM_REPEATS + 1))

        if not args.dry_run:
            missing_repeats = [r for r in repeat_ids if not onnx_path(args.run_id, r).exists()]
            if missing_repeats:
                missing_paths = ", ".join(str(onnx_path(args.run_id, r)) for r in missing_repeats)
                parser.error(
                    f"{args.run_id}: quantized artifact(s) not found for rep {missing_repeats} "
                    f"-- run quantization first ({missing_paths})."
                )

        for repeat_id in repeat_ids:
            accuracy.evaluate_accuracy(entry, repeat_id, dry_run=args.dry_run, force=args.force)
        return 0

    # --all
    if args.repeat_id is not None:
        parser.error("--repeat-id only applies together with --run-id, not --all.")

    jobs = [(entry, repeat_id) for entry in manifest.values() for repeat_id in range(1, NUM_REPEATS + 1)]

    missing: List[Tuple[str, str]] = []
    for entry, repeat_id in jobs:
        if not args.dry_run:
            path = onnx_path(entry["run_id"], repeat_id)
            if not path.exists():
                missing.append((f"{entry['run_id']} rep{repeat_id}", str(path)))
                continue
        accuracy.evaluate_accuracy(entry, repeat_id, dry_run=args.dry_run, force=args.force)

    if missing:
        print(f"\n[warning] {len(missing)}/{len(jobs)} artifacts not exported yet -- evaluation skipped for:")
        for label, path in missing:
            print(f"  - {label}  ({path})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
