"""Builds and runs the onnxruntime_perf_test command for one (artifact,
device) pair.

Fixed flags are exactly technical_architecture.md Section 6.6's
prescribed set: `-m times -o 99 -S 42 -z -D -y 1`, plus `-I` (added here,
beyond that literal list -- see below) and `-r <repeats>`. `-e`/`-x` are
never hardcoded: both come from devices.py's resolution of the device_id
passed in.

`-I` is not in Section 6.6's list but is required in practice: Brevitas's
QCDQ ONNX export produces a bare .onnx with no accompanying
`test_data_set_N/` input tensors, and onnxruntime_perf_test refuses to run
without either that directory or `-I` ("there is no test data for model
..." -- confirmed against a real Phase 3 artifact during setup). `-I`
tells it to synthesize random input tensors itself (free/dynamic
dimensions treated as 1), which is exactly what every artifact this runner
will ever see needs.

`-r <repeats>` is the one flag Section 6 documents as intentionally
per-device rather than fixed ("Default N=1000, reducible on slow devices...
document the value used per device") -- exposed here as
`repeats` (default 1000), not read from devices.yaml (which has no field
for it), so whoever runs a slow device passes a smaller value explicitly
and it is visible in the actual invocation, not buried in a config file.
The requested value is itself persisted to results/raw/ as
`requested_repeats` (results_io.py) -- Section 6's "document the value
used per device" means on paper/in data, not just visible in a
terminal that scrolls away.

Every real run (not --dry-run, not self_test.py's write_result=False path)
appends its parsed result to results/raw/results_<device_id>.csv via
results_io.append_result -- see that module's docstring for the CSV
schema and the replace-by-result_id write semantics.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from . import devices as devices_module
from . import results_io
from .output_parser import parse_perf_test_output
from .paths import onnxruntime_perf_test_bin

RunFn = Callable[..., subprocess.CompletedProcess]

DEFAULT_REPEATS = 1000


def build_benchmark_command(
    artifact_path: Union[str, Path],
    device: Dict,
    *,
    repeats: int = DEFAULT_REPEATS,
    output_prefix: Union[str, Path],
) -> List[str]:
    ep = devices_module.resolve_execution_provider(device)
    cores = devices_module.resolve_cpu_cores(device)
    return [
        onnxruntime_perf_test_bin(),
        "-m", "times",
        "-r", str(repeats),
        "-o", "99",
        "-S", "42",
        "-z",
        "-D",
        "-I",
        "-e", ep,
        "-x", str(cores),
        "-y", "1",
        str(artifact_path),
        str(output_prefix),
    ]


def run_benchmark(
    run_id: str,
    artifact_path: Union[str, Path],
    device_id: str,
    repeat_id: int,
    *,
    repeats: int = DEFAULT_REPEATS,
    dry_run: bool = False,
    write_result: bool = True,
    run_fn: RunFn = subprocess.run,
) -> Optional[Dict[str, float]]:
    """`repeat_id` is the calibration-repeat identifier baked into the
    manifest/artifact (ViT is always stochastic, NUM_REPEATS=3 -- see
    quantization/vit_brevitas/cli.py -- so unlike the LLM branch this is
    never None), required to build the common schema's result_id
    ({run_id}__{device}__rep{repeat_id}, Section 4). It is NOT the same
    thing as `repeats` (the -r flag, how many times onnxruntime_perf_test
    itself repeats inference internally for this one measurement).

    `write_result=False` (used by self_test.py) runs and parses normally
    but skips results_io.append_result -- self-test output must never land
    in results/raw/ as if it were a real measurement.
    """
    device = devices_module.resolve_device(device_id)

    with tempfile.TemporaryDirectory(prefix="runner_vit_") as tmp_dir:
        output_prefix = Path(tmp_dir) / "perf_test_result"
        cmd = build_benchmark_command(artifact_path, device, repeats=repeats, output_prefix=output_prefix)

        if dry_run:
            print(f"[dry-run] {run_id} (device={device_id}): {' '.join(cmd)}")
            return None

        try:
            result = run_fn(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # subprocess.run(check=True) raises before this function gets a
            # chance to print anything -- without surfacing e.stdout/e.stderr
            # here explicitly, the binary's own error message (the only
            # useful diagnostic) is silently lost, leaving just "exit status
            # N" in the traceback. Still re-raised (fail loud), just no
            # longer opaque.
            print(f"[error] {run_id} (device={device_id}): onnxruntime_perf_test exited with status {e.returncode}")
            if e.stdout:
                print(f"--- stdout ---\n{e.stdout}")
            if e.stderr:
                print(f"--- stderr ---\n{e.stderr}")
            raise
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        parsed = parse_perf_test_output(output)

    print(
        f"{run_id} (device={device_id}): "
        f"avg_latency_ms={parsed['avg_latency_ms']}, throughput_ips={parsed['throughput_ips']}, "
        f"num_inferences_measured={parsed['num_inferences_measured']}"
    )

    if write_result:
        out_path = results_io.append_result(run_id, device_id, repeat_id, parsed, requested_repeats=repeats)
        print(f"  -> written to {out_path}")

    return parsed
