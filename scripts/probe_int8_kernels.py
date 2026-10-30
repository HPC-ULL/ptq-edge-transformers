"""One-off probe (NOT part of the pipeline, NOT wired into run_batch.py): does
ONNX Runtime execute REAL integer kernels for our QCDQ INT8 models, or does
it fall back to dequantize -> float compute?

Two independent checks on the SAME device (CPU or GPU -- the execution
provider comes from the device_id's `backend`), both reusing
runner_vit's own onnxruntime_perf_test command, so the flags are exactly the
pipeline's (-m times -o 99 -S 42 -z -D -I -e cpu|cuda -x <cores> -y 1):

  1. LATENCY. The artifacts are alternated (A B A B ...) for --rounds
     rounds, WITHOUT profiling (profiling perturbs timings), and each one's
     median/min is compared with the FIRST artifact (the reference,
     normally the FP32 export). Alternating cancels drift (thermal state,
     the known post-boot performance step) that would bias back-to-back
     blocks.
  2. KERNELS. One extra run per artifact WITH `-p` (ORT profiling) and few
     repeats. The profile lists the nodes as EXECUTED, after graph
     optimisation, each with its `op_name`. They are counted and timed by
     op type and rolled up into: integer kernels / explicit Q-DQ nodes /
     float compute (Conv, MatMul, Gemm...) / other.

How to read the trace -- do not just grep for "QLinear":
  * Real integer kernels are QLinearConv, QLinearMatMul, MatMulInteger,
    ConvInteger, MatMulIntegerToFloat, DynamicQuantizeMatMul, QGemm, ...
    All of them count as integer here.
  * A fallback looks like explicit DequantizeLinear / QuantizeLinear nodes
    next to float Conv / MatMul / Gemm.
  * The ViT pipeline originally quantized only the INPUT of each Linear/Conv
    (input_quant, no output_quant -- it now quantizes the output too, see
    quantization/vit_brevitas/quantize.py). Which fusions ONNX Runtime applies to a pattern whose
    output stays float is exactly what this probe answers -- I have not
    assumed it -- so a MIXED result (e.g. integer MatMuls but float Convs)
    is a perfectly possible outcome, and the per-op table is the evidence;
    the verdict line is only a reading aid.

On a GPU device_id (-e cuda) the same table applies, and the provider
column/line matters: an op listed under CPUExecutionProvider ran on the CPU
(ORT falls back there for ops the CUDA EP lacks, inserting MemcpyToHost/
FromHost nodes), and the CUDA EP has its own, different set of integer
kernels -- so do not carry a CPU verdict over to the GPU. The probe needs the
CUDA EP to actually run on that device (it fails at the first Conv if not).

Standalone: writes no results/raw/, touches no manifest. Console output plus
one JSON file (--out, default results/investigations/vit/fusion_probes/
int8_kernel_probe_<device_id>.json) with everything printed. The .onnx files
it takes live in results/investigations/vit/fusion_probes/models/. It lives in
scripts/ so it ships with the edge code and is already on the Jetson: it uses
only benchmarking_ondevice/ (no torch, no Brevitas).

Run from the repo root on the device (needs benchmarking_ondevice/,
configs/devices.yaml, the compiled onnxruntime_perf_test; ONNXRUNTIME_DIR
set as for runner_vit), after >= 10 min of uptime and with the power mode
matching the device_id:

    M=results/investigations/vit/fusion_probes/models
    python scripts/probe_int8_kernels.py --device-id jetson_cpu_50w --threads 1 \\
        --out results/investigations/vit/fusion_probes/int8_kernel_probe_jetson_deploy_deit_tiny.json \\
        --artifact fp32=$M/vit_deit_tiny_fp32.onnx \\
        --artifact core=artifacts/vit_deployment/vit_deit_tiny_deploy_per_tensor_symmetric_int8__rep1.onnx
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# The repo root is the nearest ancestor holding benchmarking_ondevice/, so the
# probe works wherever it lives.
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarking_ondevice").is_dir())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarking_ondevice.runner_vit import devices as vit_devices  # noqa: E402
from benchmarking_ondevice.runner_vit import runner as vit_runner  # noqa: E402

# Standalone on purpose (the Jetson has no quantization/vit_brevitas): the same
# folder quantization/vit_brevitas/paths.py calls FUSION_PROBES.
PROBES_DIR = Path("results") / "investigations" / "vit" / "fusion_probes"

INTEGER_OPS = {
    "MatMulInteger", "ConvInteger", "MatMulIntegerToFloat", "DynamicQuantizeMatMul", "QGemm", "DynamicQuantizeLSTM",
}
QDQ_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"}
FLOAT_COMPUTE_OPS = {"Conv", "FusedConv", "NhwcFusedConv", "MatMul", "Gemm", "FusedMatMul"}

CATEGORIES = ("integer", "qdq", "float_compute", "other")
MIN_UPTIME_S = 600.0  # findings_llm.md: performance step ~6-7.5 min after boot / power-mode change


# --------------------------------------------------------------------------
# Profile analysis (pure -- no device needed, unit-tested)
# --------------------------------------------------------------------------

def categorize(op_name: str) -> str:
    if op_name in INTEGER_OPS or op_name.startswith("QLinear"):
        return "integer"
    if op_name in QDQ_OPS:
        return "qdq"
    if op_name in FLOAT_COMPUTE_OPS:
        return "float_compute"
    return "other"


def load_events(path: Path) -> List[dict]:
    data = json.loads(Path(path).read_text())
    return data["traceEvents"] if isinstance(data, dict) else data


def summarize_profile(events: Sequence[dict]) -> Dict[str, dict]:
    """Per op_name: unique node names, total kernel time (us), call count,
    providers -- over ORT's "<node>_kernel_time" events only (the *_fence_*
    events are bookkeeping, not execution)."""
    per_op: Dict[str, dict] = defaultdict(lambda: {"nodes": set(), "dur_us": 0.0, "calls": 0, "providers": set()})
    for event in events:
        if event.get("cat") != "Node" or not str(event.get("name", "")).endswith("_kernel_time"):
            continue
        args = event.get("args") or {}
        entry = per_op[args.get("op_name") or "?"]
        entry["nodes"].add(event["name"])
        entry["dur_us"] += float(event.get("dur", 0))
        entry["calls"] += 1
        entry["providers"].add(args.get("provider", "?"))
    return dict(per_op)


def rollup(per_op: Dict[str, dict]) -> Dict[str, dict]:
    total_us = sum(e["dur_us"] for e in per_op.values()) or 1.0
    out = {cat: {"nodes": 0, "dur_us": 0.0} for cat in CATEGORIES}
    for op, entry in per_op.items():
        cat = out[categorize(op)]
        cat["nodes"] += len(entry["nodes"])
        cat["dur_us"] += entry["dur_us"]
    for cat in out.values():
        cat["time_share"] = cat["dur_us"] / total_us
    return out


def interpret(roll: Dict[str, dict]) -> Tuple[str, str]:
    """(code, one-line reading aid). The per-op table is the evidence."""
    integer, qdq, flt = roll["integer"], roll["qdq"], roll["float_compute"]
    compute_us = integer["dur_us"] + flt["dur_us"]
    if compute_us == 0:
        return "NO_DATA", "no Conv/MatMul/Gemm or integer-kernel events found in the profile"
    if integer["nodes"] == 0 and qdq["nodes"] == 0:
        return "FLOAT", "plain float graph: no integer kernels and no Quantize/Dequantize nodes"
    if integer["nodes"] == 0:
        return "FALLBACK", (f"no integer kernels; {qdq['nodes']} explicit Quantize/Dequantize node(s) around float "
                            "compute -> dequantize -> float -> quantize, no real INT8 acceleration")
    fraction = integer["dur_us"] / compute_us
    if fraction >= 0.9:
        return "REAL_INT8", f"integer kernels carry {fraction:.0%} of the Conv/MatMul compute time"
    return "MIXED", (f"integer kernels carry {fraction:.0%} of the Conv/MatMul compute time; the rest still runs "
                     "as float compute (see the per-op table for which ops)")


def provider_shares(per_op: Dict[str, dict]) -> Dict[str, float]:
    """Share of total kernel time per execution provider (an op whose nodes
    ran on several providers has its time split evenly among them -- rare)."""
    total_us = sum(e["dur_us"] for e in per_op.values()) or 1.0
    shares: Dict[str, float] = defaultdict(float)
    for entry in per_op.values():
        providers = sorted(entry["providers"]) or ["?"]
        for provider in providers:
            shares[provider] += entry["dur_us"] / len(providers) / total_us
    return dict(shares)


def format_table(per_op: Dict[str, dict]) -> str:
    total_us = sum(e["dur_us"] for e in per_op.values()) or 1.0
    lines = [f"  {'op_name':<26}{'category':<15}{'nodes':>6}{'time %':>8}{'ms total':>11}  provider"]
    for op, e in sorted(per_op.items(), key=lambda kv: -kv[1]["dur_us"]):
        lines.append(f"  {op:<26}{categorize(op):<15}{len(e['nodes']):>6}{100 * e['dur_us'] / total_us:>7.1f}%"
                     f"{e['dur_us'] / 1000:>11.1f}  {','.join(sorted(e['providers']))}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Device side
# --------------------------------------------------------------------------

def parse_artifact(spec: str) -> Tuple[str, Path]:
    label, sep, raw = spec.partition("=")
    path = Path(raw)
    if not sep or not label or not path.is_file():
        raise SystemExit(f"--artifact must be LABEL=PATH to an existing .onnx, got {spec!r}")
    return label, path


def _run_perf_test(path: Path, device: dict, repeats: int) -> Dict[str, float]:
    """runner_vit's exact onnxruntime_perf_test command for `device` (a
    resolved devices.yaml entry, possibly with cpu_cores overridden by
    --threads), parsed with runner_vit's own parser. Never writes results."""
    from benchmarking_ondevice.runner_vit.output_parser import parse_perf_test_output

    with tempfile.TemporaryDirectory(prefix="probe_int8_") as tmp:
        cmd = vit_runner.build_benchmark_command(path, device, repeats=repeats, output_prefix=Path(tmp) / "result")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"--- stdout ---\n{e.stdout}\n--- stderr ---\n{e.stderr}")
            raise
    return parse_perf_test_output((result.stdout or "") + "\n" + (result.stderr or ""))


def measure_latency(artifacts, device: dict, rounds: int, repeats: int) -> Dict[str, List[float]]:
    samples: Dict[str, List[float]] = {label: [] for label, _ in artifacts}
    for round_idx in range(1, rounds + 1):
        print(f"\n--- latency round {round_idx}/{rounds} ---")
        for label, path in artifacts:
            latency = _run_perf_test(path, device, repeats)["avg_latency_ms"]
            print(f"  {label}: {latency:.3f} ms")
            samples[label].append(latency)
    return samples


def profile_artifact(label: str, path: Path, device: dict, repeats: int, workdir: Path) -> Path:
    prefix = workdir / f"profile_{label}"
    with tempfile.TemporaryDirectory(prefix="probe_int8_") as tmp:
        cmd = vit_runner.build_benchmark_command(path, device, repeats=repeats, output_prefix=Path(tmp) / "result")
        cmd = cmd[:-2] + ["-p", str(prefix)] + cmd[-2:]  # perf_test options precede the model path
        print(f"\n[profile:{label}] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"--- stdout ---\n{e.stdout}\n--- stderr ---\n{e.stderr}")
            raise
    found = sorted(workdir.glob(f"profile_{label}*.json"))
    if not found:
        raise SystemExit(f"onnxruntime_perf_test wrote no profile for {label!r} under {workdir} -- was -p honoured?")
    return found[-1]


def read_uptime_s() -> float:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except OSError:
        return -1.0


def report_latency(samples: Dict[str, List[float]]) -> Dict[str, dict]:
    labels = list(samples)
    ref = statistics.median(samples[labels[0]])
    summary = {}
    print(f"\n{'artifact':<12}{'median ms':>11}{'min ms':>10}{'stdev':>9}   vs {labels[0]}")
    for label in labels:
        values = samples[label]
        med = statistics.median(values)
        summary[label] = {"samples_ms": values, "median_ms": med, "min_ms": min(values),
                          "stdev_ms": statistics.stdev(values) if len(values) > 1 else None,
                          "ratio_vs_reference": med / ref}
        note = "(reference)" if label == labels[0] else (
            f"{ref / med:.2f}x faster" if med < ref else f"{med / ref:.2f}x SLOWER")
        print(f"{label:<12}{med:>11.3f}{min(values):>10.3f}{summary[label]['stdev_ms'] or 0:>9.3f}   {note}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device-id", required=True,
                        help="a device_id from configs/devices.yaml, CPU or GPU, e.g. jetson_cpu_50w / jetson_gpu_50w")
    parser.add_argument("--artifact", action="append", required=True, metavar="LABEL=PATH",
                        help="repeatable; the FIRST is the reference (normally the FP32 export)")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1000, help="perf_test -r for the latency runs")
    parser.add_argument("--profile-repeats", type=int, default=20, help="perf_test -r for the profiled runs")
    parser.add_argument("--out", type=Path, default=None,
                        help="JSON summary (default: results/investigations/vit/fusion_probes/"
                             "int8_kernel_probe_<device>.json)")
    parser.add_argument("--keep-profiles", action="store_true", help="keep the raw profile JSONs next to --out")
    parser.add_argument("--threads", type=int, default=None,
                        help="override the device_id's cpu_cores (perf_test -x) for every run, e.g. 1 to match a "
                             "single-thread measurement taken elsewhere")
    args = parser.parse_args(argv)

    artifacts = [parse_artifact(a) for a in args.artifact]
    if len(artifacts) < 2:
        raise SystemExit("need at least two --artifact (the reference and one to compare)")
    device = vit_devices.resolve_device(args.device_id)
    if device.get("backend") not in ("cpu", "cuda"):
        raise SystemExit(f"{args.device_id} has backend={device.get('backend')!r}; this probe supports cpu and cuda")
    if args.threads is not None:
        if args.threads < 1:
            raise SystemExit("--threads must be >= 1")
        device = {**device, "cpu_cores": args.threads}
    print(f"[probe] {args.device_id}: -e {device.get('backend')} -x {vit_devices.resolve_cpu_cores(device)}")
    uptime = read_uptime_s()
    if 0 <= uptime < MIN_UPTIME_S:
        print(f"[warning] uptime {uptime:.0f}s < {MIN_UPTIME_S:.0f}s: the post-boot performance step may still "
              "be in effect -- latency numbers taken now are not trustworthy")

    out = args.out or _PROJECT_ROOT / PROBES_DIR / f"int8_kernel_probe_{args.device_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    latency = report_latency(measure_latency(artifacts, device, args.rounds, args.repeats))

    profiles = {}
    with tempfile.TemporaryDirectory(prefix="probe_int8_profiles_") as tmp_dir:
        workdir = Path(out.parent if args.keep_profiles else tmp_dir)
        for label, path in artifacts:
            profile_path = profile_artifact(label, path, device, args.profile_repeats, workdir)
            per_op = summarize_profile(load_events(profile_path))
            roll = rollup(per_op)
            code, text = interpret(roll)
            profiles[label] = {
                "profile_file": str(profile_path) if args.keep_profiles else None,
                "per_op": {op: {"unique_nodes": len(e["nodes"]), "calls": e["calls"], "dur_us": e["dur_us"],
                                "providers": sorted(e["providers"]), "category": categorize(op)}
                           for op, e in per_op.items()},
                "rollup": roll, "provider_time_share": provider_shares(per_op), "verdict": code, "reading": text,
            }
            print(f"\n=== {label}: executed nodes by op type ===\n{format_table(per_op)}")
            print("  " + "  ".join(f"{cat}: {roll[cat]['nodes']} nodes / {roll[cat]['time_share']:.0%} time"
                                   for cat in CATEGORIES))
            print("  providers: " + " | ".join(f"{name} {share:.0%}" for name, share in
                                                 sorted(provider_shares(per_op).items(), key=lambda kv: -kv[1])))
            print(f"  -> {code}: {text}")

    out.write_text(json.dumps({
        "device_id": args.device_id, "threads": vit_devices.resolve_cpu_cores(device),
        "uptime_s": uptime, "rounds": args.rounds, "repeats": args.repeats,
        "profile_repeats": args.profile_repeats, "artifacts": {label: str(path) for label, path in artifacts},
        "latency": latency, "profile": profiles,
    }, indent=2) + "\n")
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
