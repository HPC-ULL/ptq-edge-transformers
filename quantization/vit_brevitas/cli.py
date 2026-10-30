"""CLI entry point:
`python -m quantization.vit_brevitas --run-id <id> | --all [--phase3] [--dry-run] [--force]`.

Every manifest row is stochastic (calibration-set sampling is always
random, calibration_protocol_vit.md Section 6) -- unlike the LLM branch
there is no deterministic subset, so iter_jobs always expands every row
into NUM_REPEATS=3 jobs: 24 rows x 3 = 72 total with --all.

--phase3 reads configs/vit_phase3_manifest.yaml (via
configs.vit_phase3_manifest.generate(), which requires
results/phase3_selection_report.txt to already exist -- see configs/cli.py)
instead of the default Phase 1-2 manifest. Everything else is identical:
same idempotency, --dry-run, --force, and NotImplementedError-skip
behavior, since ViT's stochastic-only NUM_REPEATS=3 expansion applies
uniformly regardless of which manifest a row came from.

Idempotent per (run_id, repeat_id): skips the whole quantize+calibrate+
export pipeline if artifacts/vit/{run_id}__rep{repeat_id}.onnx already
exists, unless --force. Quantization and ONNX export happen together, in
one in-process step (unlike the LLM branch's separate file-based stages,
imatrix .dat then .gguf) -- there is no intermediate artifact to resume
from between them, since a live Brevitas-quantized model isn't something
this project serializes to disk on its own; see quantize.py.

All 4 calibration_algorithm levels (minmax, mse, percentile, entropy) are
wired up in quantizer_config.py as of the percentile/entropy composition
added after quantization/vit_brevitas/spike_test_percentile_entropy.py --
so NotImplementedError shouldn't actually fire for any manifest row
anymore. The catch below is kept anyway, as a narrow safety net: if it ever
does fire (a future factor level added without its quantizer wired up yet,
same pattern as data_prep/domain_corpus.py before that loader existed),
--all skips just that job with a message rather than aborting the whole
run over it. Only that specific exception is caught -- any other exception
(a real Brevitas bug, a bad path, a broken ONNX export) is NOT caught here
and propagates, aborting the run. Conflating "known, declared gap" with
"unexpected failure" would hide real bugs behind the same shrug.

The final summary breaks outcomes into separate counts -- artifacts already
present (idempotency skip), artifacts generated this run, and rows blocked
on NotImplementedError -- rather than one combined total, so it's possible
to tell "these 6 rows are still genuinely unimplemented" apart from "these
were already done in an earlier run".
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Callable, Dict, Iterator, List, Tuple

from configs import vit_manifest, vit_phase3_manifest
from export.vit_to_onnx import export_to_onnx, onnx_path

from .calibration import NUM_REPEATS
from .common_exec import Outcome, run_step
from .quantize import build_quantized_model

Job = Tuple[dict, int]
PipelineFn = Callable[[dict, int], None]

_OUTCOME_LABELS = {
    "exists": "already existed (skipped for idempotency)",
    "generated": "generated in this run",
    "dry_run": "would be generated (dry-run)",
    "not_implemented": "blocked (calibration_algorithm not implemented)",
}


def _manifest_by_run_id(phase3: bool = False) -> Dict[str, dict]:
    if phase3:
        entries, _dedup_notes = vit_phase3_manifest.generate()
    else:
        entries = vit_manifest.generate()
    return {e["run_id"]: e for e in entries}


def iter_jobs(entries: List[dict]) -> Iterator[Job]:
    for entry in entries:
        for repeat_id in range(1, NUM_REPEATS + 1):
            yield entry, repeat_id


def run_quantize_and_export(entry: dict, repeat_id: int) -> None:
    quant_model, input_shape = build_quantized_model(entry, repeat_id)
    export_to_onnx(quant_model, input_shape, entry["run_id"], repeat_id)


def process_job(
    entry: dict,
    repeat_id: int,
    *,
    dry_run: bool,
    force: bool,
    pipeline_fn: PipelineFn = run_quantize_and_export,
) -> Outcome:
    out_path = onnx_path(entry["run_id"], repeat_id)
    label = f"{entry['run_id']} rep{repeat_id}"
    describe = (
        f"quantize+export model={entry['model_name']} algo={entry['calibration_algorithm']} "
        f"granularity={entry['granularity']} symmetry={entry['symmetry']} "
        f"calib_set_size={entry['calib_set_size']} bitwidth={entry['bitwidth']} -> {out_path}"
    )
    return run_step(
        label,
        describe,
        lambda: pipeline_fn(entry, repeat_id),
        output_exists=out_path.exists(),
        dry_run=dry_run,
        force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantization.vit_brevitas",
        description="Produce ONNX artifacts for the ViT-branch experiment manifest (vit_experiment_manifest.yaml).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Process a single manifest row by run_id (all its repeats).")
    group.add_argument("--all", action="store_true", help="Process every row in the manifest.")
    parser.add_argument(
        "--phase3", action="store_true",
        help="Read configs/vit_phase3_manifest.yaml instead of the Phase 1-2 manifest.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run, without running it or touching datasets/models/results.",
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

    outcome_counts: Counter = Counter()
    blocked_run_ids = set()

    for entry, repeat_id in iter_jobs(entries):
        try:
            outcome = process_job(entry, repeat_id, dry_run=args.dry_run, force=args.force)
            outcome_counts[outcome] += 1
        except NotImplementedError as e:
            # Only this specific, declared gap is swallowed -- any other
            # exception propagates and aborts the run (see module docstring).
            print(f"[skip] {entry['run_id']} rep{repeat_id}: {e}")
            outcome_counts["not_implemented"] += 1
            blocked_run_ids.add(entry["run_id"])

    total_jobs = sum(outcome_counts.values())
    print(f"\n[summary] {total_jobs} jobs processed ({len(entries)} rows x up to {NUM_REPEATS} repetitions):")
    for key in ("exists", "generated", "dry_run", "not_implemented"):
        if outcome_counts[key]:
            print(f"  {_OUTCOME_LABELS[key]}: {outcome_counts[key]}")
    if blocked_run_ids:
        print(f"  -> {len(blocked_run_ids)}/{len(entries)} manifest rows are still genuinely blocked")

    return 0


if __name__ == "__main__":
    sys.exit(main())
