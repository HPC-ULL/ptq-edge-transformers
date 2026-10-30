"""Builds and runs the llama-bench command for one (artifact, device) pair.

`-p 128 -n 32` are fixed constants, not CLI-configurable -- confirmed
against technical_architecture.md Section 6.6 to match the original
paper's methodology, deliberately NOT llama-bench's own defaults (512/128).
`-o json` is likewise fixed (Section 6.6: structured output, never parse
llama-bench's human-readable default). `-t`/`-ngl` are never hardcoded or
left at llama-bench's own defaults (40 / -1 "use GPU if available") -- both
always come from devices.py's resolution of the device_id passed in, same
as runner_vit's -x/-e.

`-r <repeats>` mirrors runner_vit's `--repeats` (its own -r flag) --
exposed here too even though it wasn't in the literal fixed-flag list,
since the same "document the value used per device" (Section 6)
requirement applies, and results_io.py's `requested_repeats` column needs
a real value to persist. Default is llama-bench's own default (5), not
onnxruntime_perf_test's 1000 -- LLM inference is far slower per token, so
the same repetition count would be wildly impractical.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from . import devices as devices_module
from . import results_io
from .output_parser import parse_llama_bench_output
from .paths import llama_bench_bin

RunFn = Callable[..., subprocess.CompletedProcess]

DEFAULT_REPEATS = 5
PROMPT_TOKENS = 128
GEN_TOKENS = 32


def build_benchmark_command(
    artifact_path: Union[str, Path],
    device: Dict,
    *,
    repeats: int = DEFAULT_REPEATS,
) -> List[str]:
    threads = devices_module.resolve_threads(device)
    n_gpu_layers = devices_module.resolve_n_gpu_layers(device)
    return [
        llama_bench_bin(),
        "-m", str(artifact_path),
        "-o", "json",
        "-t", str(threads),
        "-p", str(PROMPT_TOKENS),
        "-n", str(GEN_TOKENS),
        "-ngl", str(n_gpu_layers),
        "-r", str(repeats),
    ]


def run_benchmark(
    run_id: str,
    artifact_path: Union[str, Path],
    device_id: str,
    repeat_id: Optional[int],
    *,
    repeats: int = DEFAULT_REPEATS,
    dry_run: bool = False,
    write_result: bool = True,
    run_fn: RunFn = subprocess.run,
) -> Optional[Dict[str, float]]:
    """`repeat_id` is the calibration-repeat identifier baked into the
    manifest/artifact -- see runner_vit.runner.run_benchmark's docstring
    for why this is a distinct concept from `repeats` (the -r flag).
    Unlike ViT (always stochastic), LLM has real deterministic configs
    (baseline, legacy_corpus_none, some Phase 3 combined rows) with no
    calibration-repeat concept at all -- pass None for those, mirroring
    evaluation_offline/llm_perplexity's exact Optional[int] pattern (no
    "__rep" suffix in result_id, blank repeat_id column).

    `write_result=False` (used by self_test.py) runs and parses normally
    but skips results_io.append_result.
    """
    device = devices_module.resolve_device(device_id)
    cmd = build_benchmark_command(artifact_path, device, repeats=repeats)

    if dry_run:
        print(f"[dry-run] {run_id} (device={device_id}): {' '.join(cmd)}")
        return None

    try:
        result = run_fn(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # See runner_vit.runner.run_benchmark's identical try/except for
        # why this is necessary: subprocess.run(check=True) raises before
        # any of this function's own code runs, so without this the
        # binary's actual error message is silently lost.
        print(f"[error] {run_id} (device={device_id}): llama-bench exited with status {e.returncode}")
        if e.stdout:
            print(f"--- stdout ---\n{e.stdout}")
        if e.stderr:
            print(f"--- stderr ---\n{e.stderr}")
        raise

    parsed = parse_llama_bench_output(result.stdout or "")

    print(
        f"{run_id} (device={device_id}): "
        f"throughput_prefill_tps={parsed['throughput_prefill_tps']}, "
        f"throughput_decode_tps={parsed['throughput_decode_tps']}, "
        f"num_inferences_measured={parsed['num_inferences_measured']}"
    )

    if write_result:
        out_path = results_io.append_result(run_id, device_id, repeat_id, parsed, requested_repeats=repeats)
        print(f"  -> written to {out_path}")

    return parsed
