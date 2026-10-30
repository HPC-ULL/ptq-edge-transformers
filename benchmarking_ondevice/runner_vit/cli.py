"""CLI entry point:

    python -m benchmarking_ondevice.runner_vit \\
        --run-id <id> --artifact <path/to/model.onnx> --device-id <device_id> --repeat-id <N> \\
        [--repeats N] [--dry-run]

    python -m benchmarking_ondevice.runner_vit --self-test --device-id <device_id> [--self-test-artifact <path>]

Unlike quantization/*'s CLIs, this runs one job per invocation against an
explicit artifact path -- there is no manifest here (each edge device
benchmarks whichever artifacts it was handed). Every real run (not
--dry-run, not --self-test) appends its parsed result to
results/raw/results_<device_id>.csv -- see results_io.py.

--repeat-id identifies which calibration repeat produced the artifact being
benchmarked (ViT is always stochastic, NUM_REPEATS=3 -- see
quantization/vit_brevitas/cli.py), required to build the common schema's
result_id (Section 4). It is NOT --repeats (the -r flag / how many times
onnxruntime_perf_test itself repeats inference internally).

--self-test is mutually exclusive with the normal --run-id/--artifact/
--repeat-id mode: it needs only --device-id (plus the optional
--self-test-artifact override for devices without the `onnx` package -- see
self_test.py), and never writes to results/raw/.
"""

from __future__ import annotations

import argparse
import sys

from .runner import DEFAULT_REPEATS, run_benchmark
from .self_test import self_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarking_ondevice.runner_vit",
        description="Run onnxruntime_perf_test against one quantized ViT .onnx artifact on one device.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Validate output_parser against this environment's onnxruntime_perf_test build, "
             "using a tiny synthetic artifact instead of --artifact.",
    )
    parser.add_argument("--run-id", help="Label for this benchmark run (required unless --self-test).")
    parser.add_argument("--artifact", help="Path to the quantized .onnx artifact (required unless --self-test).")
    parser.add_argument(
        "--repeat-id", type=int,
        help="Calibration repeat (1-3) that produced --artifact (required unless --self-test).",
    )
    parser.add_argument("--device-id", required=True, help="device_id from configs/devices.yaml.")
    parser.add_argument(
        "--self-test-artifact",
        help="--self-test only: use this .onnx instead of generating one via the `onnx` package "
             "(for devices where that package isn't installed).",
    )
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help=f"-r value passed to onnxruntime_perf_test (default {DEFAULT_REPEATS}; "
             "reduce on slow devices per technical_architecture.md Section 6).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the command that would run, without running it. Not applicable to --self-test "
             "(which exists specifically to run for real).",
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

    if not args.run_id or not args.artifact or args.repeat_id is None:
        parser.error("--run-id, --artifact and --repeat-id are all required (unless --self-test).")
    if args.self_test_artifact:
        parser.error("--self-test-artifact only applies to --self-test.")

    run_benchmark(
        args.run_id, args.artifact, args.device_id, args.repeat_id, repeats=args.repeats, dry_run=args.dry_run
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
