"""CLI entry point:

    python -m benchmarking_ondevice.runner_llm \\
        --run-id <id> --artifact <path/to/model.gguf> --device-id <device_id> [--repeat-id <N>] \\
        [--repeats N] [--dry-run]

    python -m benchmarking_ondevice.runner_llm --self-test --device-id <device_id> [--self-test-artifact <path>]

Mirrors benchmarking_ondevice/runner_vit/cli.py -- see that module's
docstring for the reasoning behind the shape (no manifest, one job per
invocation, --repeat-id vs --repeats distinction, --self-test never
writing to results/raw/).

`--repeat-id` is OPTIONAL here, unlike runner_vit (ViT is always
stochastic). Omit it for a deterministic LLM config (baseline,
legacy_corpus_none, some Phase 3 combined rows) -- there is no manifest
here to check an artifact's determinism against, so whoever invokes this
CLI is trusted to know which case their --artifact is (the run_manifest.csv
shipped alongside each bundle lists this explicitly per artifact). Passing
--repeat-id for a deterministic artifact, or omitting it for a stochastic
one, is not validated here -- results_io.py just records whatever it's
given, same as evaluation_offline/llm_perplexity's own CLI-less callers.
"""

from __future__ import annotations

import argparse
import sys

from .runner import DEFAULT_REPEATS, run_benchmark
from .self_test import self_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarking_ondevice.runner_llm",
        description="Run llama-bench against one quantized LLM .gguf artifact on one device.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Validate output_parser against this environment's llama-bench build, using the smallest "
             "already-quantized .gguf under artifacts/llm/ instead of --artifact.",
    )
    parser.add_argument("--run-id", help="Label for this benchmark run (required unless --self-test).")
    parser.add_argument("--artifact", help="Path to the quantized .gguf artifact (required unless --self-test).")
    parser.add_argument(
        "--repeat-id", type=int,
        help="Calibration repeat that produced --artifact. Omit for a deterministic config (no calibration "
             "repeat at all) -- see run_manifest.csv for which case a given --artifact is.",
    )
    parser.add_argument("--device-id", required=True, help="device_id from configs/devices.yaml.")
    parser.add_argument(
        "--self-test-artifact",
        help="--self-test only: use this .gguf instead of auto-selecting the smallest one under artifacts/llm/.",
    )
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help=f"-r value passed to llama-bench (default {DEFAULT_REPEATS}; "
             "reduce on slow devices per technical_architecture.md Section 6).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the command that would run, without running it. Not applicable to --self-test.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        if args.run_id or args.artifact:
            parser.error("--self-test does not take --run-id/--artifact (use --self-test-artifact instead).")
        if args.dry_run:
            parser.error("--self-test cannot be combined with --dry-run -- it exists specifically to run for real.")
        self_test(args.device_id, artifact_path=args.self_test_artifact)
        return 0

    if not args.run_id or not args.artifact:
        parser.error("--run-id and --artifact are both required (unless --self-test).")
    if args.self_test_artifact:
        parser.error("--self-test-artifact only applies to --self-test.")

    run_benchmark(
        args.run_id, args.artifact, args.device_id, args.repeat_id, repeats=args.repeats, dry_run=args.dry_run
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
