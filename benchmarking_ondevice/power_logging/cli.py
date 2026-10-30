"""CLI entry point:

    python -m benchmarking_ondevice.power_logging \\
        --branch {vit,llm} --run-id <id> --artifact <path> --device-id <device_id> --repeat-id <N> \\
        [--repeats N] [--dry-run]

Wraps runner_vit.run_benchmark or runner_llm.run_benchmark (selected via
--branch, always explicit -- never inferred from the artifact's file
extension, matching this project's general "don't guess, be explicit"
convention elsewhere) with jetson_powermeter.measure_and_record. Same
run_id/artifact/device_id/repeat_id/--repeats/--dry-run shape as the two
runner CLIs.

--dry-run bypasses power_logging entirely: it calls the underlying
runner's own run_benchmark(..., dry_run=True) directly, which prints the
command and returns without running anything -- no PowerMeter is ever
constructed on this path, so the ~10s idle-measurement wait never happens
for a dry-run.
"""

from __future__ import annotations

import argparse
import functools
import sys
from typing import Dict, Optional

from . import jetson_powermeter


def _vit_perf_per_watt(row: Dict, output: Dict, energy_fields: Dict) -> Optional[float]:
    """N / energy_net_j (inferences per net joule) -- formula in
    perf_per_watt.py. NOT throughput / avg_power_w_net: the throughput
    covers only onnxruntime_perf_test's timed inferences, while the power
    window also covers start-up, model load and warm-up."""
    from .perf_per_watt import vit_perf_per_watt

    return vit_perf_per_watt(energy_fields["energy_net_j"], output["num_inferences_measured"])


def _llm_perf_per_watt(row: Dict, output: Dict, energy_fields: Dict) -> Optional[float]:
    """(n_prompt + n_gen) / energy_per_inference, i.e. tokens of one
    repetition over the net energy of one repetition -- formula in
    perf_per_watt.py, shared with the offline recalculation. NOT
    throughput_decode_tps / avg_power_w_net. See jetson_powermeter.py's
    module docstring for why: decode throughput is measured over only the
    decode sub-phase, while energy covers the whole llama-bench invocation
    (prefill + decode together) -- dividing whole-run energy by a
    sub-phase-only rate would mix two different time windows.

    The tokens come from runner_llm's -p/-n constants; the ones llama-bench
    reports must match them, or the constant would be describing a run that
    did not happen."""
    from .perf_per_watt import llm_perf_per_watt
    from benchmarking_ondevice.runner_llm.runner import GEN_TOKENS, PROMPT_TOKENS

    if (output["n_prompt"], output["n_gen"]) != (PROMPT_TOKENS, GEN_TOKENS):
        raise ValueError(
            f"llama-bench reported n_prompt={output['n_prompt']}, n_gen={output['n_gen']}, but the runner's "
            f"constants are -p {PROMPT_TOKENS} -n {GEN_TOKENS}"
        )
    return llm_perf_per_watt(energy_fields["energy_net_j"], output["num_inferences_measured"])


def _branch_modules(branch: str):
    """Returns (run_benchmark, build_result_row, default_repeats,
    perf_per_watt_fn) for `branch`. The perf_per_watt formula is explicit
    here, per branch -- never inferred later from row shape (see
    jetson_powermeter.measure_and_record's docstring)."""
    if branch == "vit":
        from benchmarking_ondevice.runner_vit import results_io
        from benchmarking_ondevice.runner_vit.runner import DEFAULT_REPEATS, run_benchmark
        perf_per_watt_fn = _vit_perf_per_watt
    elif branch == "llm":
        from benchmarking_ondevice.runner_llm import results_io
        from benchmarking_ondevice.runner_llm.runner import DEFAULT_REPEATS, run_benchmark
        perf_per_watt_fn = _llm_perf_per_watt
    else:
        raise ValueError(f"Unknown branch={branch!r}. Known branches: ['vit', 'llm'].")
    return run_benchmark, results_io.build_result_row, DEFAULT_REPEATS, perf_per_watt_fn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarking_ondevice.power_logging",
        description="Run a ViT or LLM on-device benchmark under Jetson power measurement, "
                    "and write the combined performance+energy row.",
    )
    parser.add_argument("--branch", required=True, choices=["vit", "llm"], help="Which runner to wrap.")
    parser.add_argument("--run-id", required=True, help="Label for this benchmark run.")
    parser.add_argument("--artifact", required=True, help="Path to the quantized artifact (.onnx or .gguf).")
    parser.add_argument("--repeat-id", type=int, required=True, help="Calibration repeat that produced --artifact.")
    parser.add_argument("--device-id", required=True, help="device_id from configs/devices.yaml.")
    parser.add_argument(
        "--repeats", type=int, default=None,
        help="Repetitions passed to the underlying benchmark tool (-r). Defaults to that branch's own default "
             "if omitted (1000 for ViT/onnxruntime_perf_test, 5 for LLM/llama-bench).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the underlying runner's command without running it or measuring power.",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_benchmark, build_result_row, default_repeats, perf_per_watt_fn = _branch_modules(args.branch)
    repeats = args.repeats if args.repeats is not None else default_repeats

    if args.dry_run:
        # Bypass power_logging entirely -- no PowerMeter, no idle wait, no
        # measure_and_record call. See module docstring.
        run_benchmark(args.run_id, args.artifact, args.device_id, args.repeat_id, repeats=repeats, dry_run=True)
        return 0

    run_fn_bound = functools.partial(
        run_benchmark, args.run_id, args.artifact, args.device_id, args.repeat_id,
        repeats=repeats, write_result=False,
    )
    result = jetson_powermeter.measure_and_record(
        run_fn_bound, build_result_row, args.run_id, args.device_id, args.repeat_id, repeats,
        perf_per_watt_fn=perf_per_watt_fn,
    )
    print(f"{args.run_id} (device={args.device_id}): -> written to {result['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
