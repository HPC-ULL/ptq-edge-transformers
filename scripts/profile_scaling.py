"""One-off diagnostic (NOT part of the pipeline): how each ONNX Runtime
operator of a model scales with the intra-op thread count, measured with
onnxruntime_perf_test's own per-node profile.

Why: on the Jetson the deployment INT8 models gain much less from more
threads than their FP32 (DeiT-Tiny's INT8 was even slower than FP32 at 12
threads). This measures, per operator and per thread count, the time per
inference over several rounds, and records per node the tensor sizes and
whether ORT dispatched the node's work to the thread pool -- so the operators
that do not scale, their weight in the total, and why, can be read off.

Two subcommands:

  run      on the device (needs only benchmarking_ondevice/ and the compiled
           onnxruntime_perf_test, like scripts/probe_int8_kernels.py). For each
           thread count, for each round, profiles every artifact once
           (artifacts alternate, never in blocks), with runner_vit's exact
           command -- `-x <threads> -y 1`, cpu_cores of the device overridden
           only -- plus `-p`. The raw profiles are reduced on the fly and
           deleted; one JSON keeps, per run, the time of every operator per
           inference, and for the first round of each thread count the per
           node detail (time, bytes read/written, parallel dispatch).
           Writes nothing under results/raw/.

  analyze  anywhere (stdlib only): the tables of the diagnosis, from that JSON
           -- per model, time per inference of each operator group at every
           thread count (median and min-max over rounds) and its 1->N scaling
           factor; the share of the time in groups that do not scale; the
           Amdahl prediction vs the measured speed-up; and at the highest
           thread count, per memory-bound op (QuantizeLinear,
           DequantizeLinear, Transpose) next to one that scales: node count,
           time and bytes per node, effective bandwidth, a fixed-cost +
           per-byte fit, and the share of nodes ORT ran in parallel.
           --graph LABEL=PATH (the graph ORT optimized, dumped on the device
           with onnxruntime_perf_test -u; needs `onnx`) adds how many
           memory-bound nodes in a row each tensor goes through -- each one is
           a full read + write pass over it.

Profiling adds a small per-node cost, which weighs most on small nodes: the
totals here are profiled totals, compared with each other, not with the
latencies of results/raw/.

    python scripts/profile_scaling.py run --device-id jetson_cpu_50w --threads 1 4 8 12 --rounds 5 \\
        --artifact deit_fp32=... --artifact deit_int8=... --out results/investigations/vit/thread_scaling/jetson.json
    python scripts/profile_scaling.py analyze results/investigations/vit/thread_scaling/jetson.json
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
from typing import Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarking_ondevice").is_dir())
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

OUT_DIR = Path("results") / "investigations" / "vit" / "thread_scaling"

# Operator groups of the diagnosis (an op not listed goes to "rest").
GROUPS = [
    ("QGemm", {"QGemm"}),
    ("QLinearConv", {"QLinearConv"}),
    ("Gemm", {"Gemm"}),
    ("Conv", {"Conv", "FusedConv", "NhwcFusedConv"}),
    ("MatMul (attention)", {"MatMul", "FusedMatMul", "MatMulIntegerToFloat"}),
    ("QuantizeLinear", {"QuantizeLinear"}),
    ("DequantizeLinear", {"DequantizeLinear"}),
    ("Transpose", {"Transpose"}),
    ("Gelu/QuickGelu", {"Gelu", "BiasGelu", "FastGelu", "QuickGelu", "Erf"}),
]
REST = "rest"
NOT_SCALING = 2.0  # a group "does not scale" if its 1 -> max-threads factor is below this
MEMORY_BOUND_OPS = ("QuantizeLinear", "DequantizeLinear", "Transpose")
DTYPE_BYTES = {"float": 4, "float32": 4, "float16": 2, "double": 8, "int8": 1, "uint8": 1, "int16": 2,
               "uint16": 2, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8, "bool": 1}


def group_of(op: str) -> str:
    for name, ops in GROUPS:
        if op in ops:
            return name
    return REST


# --------------------------------------------------------------------------
# Reducing one ORT profile (pure)
# --------------------------------------------------------------------------

def _tensor_bytes(type_shapes: Sequence[dict]) -> int:
    total = 0
    for entry in type_shapes or []:
        for dtype, shape in entry.items():
            n = 1
            for dim in shape:
                n *= int(dim)
            total += n * DTYPE_BYTES.get(dtype, 4)
    return total


def _parallel(args: dict) -> Optional[bool]:
    """True if ORT dispatched this node's work to the intra-op pool (a non-empty
    block_size), False if it ran it on the main thread only, None if the build
    does not report thread scheduling stats."""
    stats = (args.get("thread_scheduling_stats") or {}).get("main_thread")
    if stats is None or "block_size" not in stats:
        return None
    return len(stats["block_size"]) > 0


def reduce_profile(events: Sequence[dict]) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """(time per inference of each op in ms, per node detail) from ORT's
    "<node>_kernel_time" events. Per node, the time per call is its total over
    the profiled calls divided by their count; an op's time per inference is
    the sum over its nodes."""
    nodes: Dict[str, dict] = {}
    for event in events:
        if event.get("cat") != "Node" or not str(event.get("name", "")).endswith("_kernel_time"):
            continue
        args = event.get("args") or {}
        node = nodes.setdefault(event["name"], {"op": args.get("op_name") or "?", "dur_us": 0.0, "calls": 0,
                                                "in_bytes": _tensor_bytes(args.get("input_type_shape")),
                                                "out_bytes": _tensor_bytes(args.get("output_type_shape")),
                                                "parallel": _parallel(args)})
        node["dur_us"] += float(event.get("dur", 0))
        node["calls"] += 1
    per_op: Dict[str, float] = defaultdict(float)
    detail = {}
    for name, n in nodes.items():
        ms = n["dur_us"] / n["calls"] / 1000.0
        per_op[n["op"]] += ms
        detail[name] = {"op": n["op"], "ms": ms, "bytes": n["in_bytes"] + n["out_bytes"], "parallel": n["parallel"]}
    return dict(per_op), detail


# --------------------------------------------------------------------------
# run (device side)
# --------------------------------------------------------------------------

def _profile_once(path: Path, device: dict, repeats: int, workdir: Path) -> List[dict]:
    from benchmarking_ondevice.runner_vit import runner as vit_runner

    prefix = workdir / "profile"
    for old in workdir.glob("profile*.json"):
        old.unlink()
    with tempfile.TemporaryDirectory(prefix="profile_scaling_") as tmp:
        cmd = vit_runner.build_benchmark_command(path, device, repeats=repeats, output_prefix=Path(tmp) / "result")
        cmd = cmd[:-2] + ["-p", str(prefix)] + cmd[-2:]  # perf_test options precede the model path
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"--- stdout ---\n{e.stdout}\n--- stderr ---\n{e.stderr}")
            raise
    found = sorted(workdir.glob("profile*.json"))
    if not found:
        raise SystemExit(f"onnxruntime_perf_test wrote no profile for {path} -- was -p honoured?")
    data = json.loads(found[-1].read_text())
    found[-1].unlink()
    return data["traceEvents"] if isinstance(data, dict) else data


def _read_uptime_s() -> Optional[float]:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except OSError:
        return None


def run(args) -> int:
    from benchmarking_ondevice.runner_vit import devices as vit_devices

    artifacts = []
    for spec in args.artifact:
        label, sep, raw = spec.partition("=")
        if not sep or not label or not Path(raw).is_file():
            raise SystemExit(f"--artifact must be LABEL=PATH to an existing .onnx, got {spec!r}")
        artifacts.append((label, Path(raw)))
    if any(t < 1 for t in args.threads):
        raise SystemExit("--threads must be >= 1")
    base = vit_devices.resolve_device(args.device_id)
    out = args.out or _PROJECT_ROOT / OUT_DIR / f"scaling_{args.device_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    report = {"device_id": args.device_id, "uptime_s_start": _read_uptime_s(), "threads": args.threads,
              "rounds": args.rounds, "repeats": args.repeats,
              "artifacts": {label: str(path) for label, path in artifacts}, "runs": []}
    with tempfile.TemporaryDirectory(prefix="profile_scaling_work_") as work:
        workdir = Path(work)
        for threads in args.threads:
            device = {**base, "cpu_cores": threads}
            for round_idx in range(1, args.rounds + 1):
                for label, path in artifacts:
                    per_op, detail = reduce_profile(_profile_once(path, device, args.repeats, workdir))
                    total = sum(per_op.values())
                    run_entry = {"threads": threads, "round": round_idx, "label": label, "total_ms": total,
                                 "ops": per_op}
                    if round_idx == 1:
                        run_entry["nodes"] = detail
                    report["runs"].append(run_entry)
                    print(f"[threads {threads:>2} round {round_idx}] {label:<12} {total:8.2f} ms (profiled)", flush=True)
                    out.write_text(json.dumps(report))  # partial results survive an interrupted run
    report["uptime_s_end"] = _read_uptime_s()
    out.write_text(json.dumps(report))
    print(f"\nwritten {out}")
    return 0


# --------------------------------------------------------------------------
# analyze (pure)
# --------------------------------------------------------------------------

def _median_range(values: List[float]) -> Tuple[float, float, float]:
    return statistics.median(values), min(values), max(values)


def group_times(report: dict) -> Dict[str, Dict[int, Dict[str, List[float]]]]:
    """{label: {threads: {group or "TOTAL": [ms per round]}}}"""
    out: Dict[str, Dict[int, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in report["runs"]:
        groups: Dict[str, float] = defaultdict(float)
        for op, ms in r["ops"].items():
            groups[group_of(op)] += ms
        cell = out[r["label"]][r["threads"]]
        for name in [g for g, _ in GROUPS] + [REST]:
            cell[name].append(groups.get(name, 0.0))
        cell["TOTAL"].append(r["total_ms"])
    return out


def scaling_table(report: dict, label: str) -> List[dict]:
    times = group_times(report)[label]
    threads = sorted(times)
    lo, hi = threads[0], threads[-1]
    rows = []
    for name in [g for g, _ in GROUPS] + [REST, "TOTAL"]:
        med = {t: _median_range(times[t][name]) for t in threads}
        if med[lo][0] == 0 and med[hi][0] == 0:
            continue
        factor = med[lo][0] / med[hi][0] if med[hi][0] > 0 else float("inf")
        rows.append({"group": name, "by_threads": med, "factor": factor})
    return rows


def amdahl(report: dict, label: str) -> dict:
    """Groups split by their own measured scaling factor (< NOT_SCALING: serial).
    p = share of the 1-thread time in the scaling groups; Amdahl's bound for N
    threads if only the non-scaling groups were serial: 1 / ((1 - p) + p / N)."""
    rows = [r for r in scaling_table(report, label) if r["group"] != "TOTAL"]
    total = next(r for r in scaling_table(report, label) if r["group"] == "TOTAL")
    threads = sorted(total["by_threads"])
    lo, hi = threads[0], threads[-1]
    serial = [r["group"] for r in rows if r["factor"] < NOT_SCALING]
    t_lo = sum(r["by_threads"][lo][0] for r in rows)
    share = {t: sum(r["by_threads"][t][0] for r in rows if r["group"] in serial) /
             (sum(r["by_threads"][t][0] for r in rows) or 1.0) for t in threads}
    p = 1.0 - sum(r["by_threads"][lo][0] for r in rows if r["group"] in serial) / (t_lo or 1.0)
    n = hi / lo
    return {"serial_groups": serial, "serial_share": share, "parallel_fraction_1t": p,
            "amdahl_max": 1.0 / ((1.0 - p) + p / n), "measured": total["factor"], "threads": (lo, hi)}


def _fit(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Least squares ms = a + b * bytes; None with < 3 points or no spread."""
    if len(points) < 3:
        return None
    xs, ys = zip(*points)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    return my - b * mx, b


def node_stats(report: dict, label: str, threads: Optional[int] = None) -> List[dict]:
    """At `threads` (default: the highest), per memory-bound op and the largest
    op that scales: nodes, total and median per-node time, median bytes,
    effective bandwidth, fixed + per-byte fit, share of parallel-dispatched nodes."""
    runs = [r for r in report["runs"] if r["label"] == label and "nodes" in r]
    threads = threads or max(r["threads"] for r in runs)
    nodes = next(r["nodes"] for r in runs if r["threads"] == threads)
    by_op: Dict[str, List[dict]] = defaultdict(list)
    for n in nodes.values():
        by_op[n["op"]].append(n)
    scaling_ops = [op for op in ("QGemm", "QLinearConv", "Gemm", "Conv") if op in by_op]
    reference = max(scaling_ops, key=lambda op: sum(n["ms"] for n in by_op[op])) if scaling_ops else None
    rows = []
    for op in [o for o in MEMORY_BOUND_OPS if o in by_op] + ([reference] if reference else []):
        ns = by_op[op]
        total_ms = sum(n["ms"] for n in ns)
        total_bytes = sum(n["bytes"] for n in ns)
        fit = _fit([(n["bytes"], n["ms"]) for n in ns])
        flags = [n["parallel"] for n in ns if n["parallel"] is not None]
        rows.append({"op": op, "nodes": len(ns), "total_ms": total_ms,
                     "median_us": statistics.median(n["ms"] for n in ns) * 1000,
                     "median_kb": statistics.median(n["bytes"] for n in ns) / 1024,
                     "gb_per_s": (total_bytes / 1e9) / (total_ms / 1000) if total_ms else 0.0,
                     "fixed_us": fit[0] * 1000 if fit else None,
                     "fixed_share": (fit[0] * len(ns) / total_ms) if fit and total_ms else None,
                     "parallel_share": (sum(flags) / len(flags)) if flags else None,
                     "is_reference": op == reference})
    return rows


def memory_bound_runs(graph_path: Path) -> Dict[int, int]:
    """{run length: how many runs} of consecutive memory-bound nodes
    (QuantizeLinear / DequantizeLinear / Transpose) in an ONNX graph, following
    single-consumer edges: a run of k is a tensor read and rewritten k times."""
    import onnx

    graph = onnx.load(str(graph_path)).graph
    producer, consumers = {}, defaultdict(list)
    for node in graph.node:
        for name in node.output:
            producer[name] = node
        for name in node.input:
            if name:
                consumers[name].append(node)
    is_mb = lambda n: n.op_type in MEMORY_BOUND_OPS  # noqa: E731
    starts = [n for n in graph.node if is_mb(n) and not (n.input and producer.get(n.input[0]) is not None
                                                        and is_mb(producer[n.input[0]]))]
    runs: Dict[int, int] = defaultdict(int)
    for node in starts:
        length = 1
        while True:
            nxt = consumers.get(node.output[0], [])
            if len(nxt) != 1 or not is_mb(nxt[0]):
                break
            node, length = nxt[0], length + 1
        runs[length] += 1
    return dict(sorted(runs.items()))


def format_report(report: dict, graphs: Optional[Dict[str, Path]] = None) -> str:
    lines = [f"device {report['device_id']}, threads {report['threads']}, {report['rounds']} rounds x "
             f"{report['repeats']} repeats; uptime {report.get('uptime_s_start')} s -> {report.get('uptime_s_end')} s",
             "(profiled ms per inference: median [min-max] over rounds)"]
    for label in report["artifacts"]:
        lines.append(f"\n== {label}")
        rows = scaling_table(report, label)
        threads = sorted(rows[0]["by_threads"])
        lines.append(f"  {'group':<20}" + "".join(f"{f'{t} thr':>22}" for t in threads) + f"{'1->N':>7}")
        for r in rows:
            cells = "".join(f"{m:>8.2f} [{a:>5.2f}-{b:>5.2f}]" for m, a, b in (r["by_threads"][t] for t in threads))
            lines.append(f"  {r['group']:<20}{cells}{r['factor']:>6.2f}x")
        a = amdahl(report, label)
        shares = ", ".join(f"{t} thr {100 * s:.0f}%" for t, s in a["serial_share"].items())
        lines.append(f"  not scaling (< {NOT_SCALING:g}x): {', '.join(a['serial_groups']) or 'none'}; share of time: {shares}")
        lines.append(f"  Amdahl ({a['threads'][0]}->{a['threads'][1]} thr, parallel fraction "
                     f"{100 * a['parallel_fraction_1t']:.0f}%): max {a['amdahl_max']:.2f}x vs measured {a['measured']:.2f}x")
        lines.append(f"  at {max(threads)} threads, per node:")
        lines.append(f"    {'op':<18}{'nodes':>6}{'total ms':>10}{'us/node':>9}{'KB/node':>9}{'GB/s':>7}"
                     f"{'fixed us':>10}{'fixed %':>9}{'parallel':>10}")
        for s in node_stats(report, label):
            fixed = "n/a" if s["fixed_us"] is None else f"{s['fixed_us']:.1f}"
            fshare = "n/a" if s["fixed_share"] is None else f"{100 * s['fixed_share']:.0f}%"
            par = "n/a" if s["parallel_share"] is None else f"{100 * s['parallel_share']:.0f}%"
            name = s["op"] + (" (ref)" if s["is_reference"] else "")
            lines.append(f"    {name:<18}{s['nodes']:>6}{s['total_ms']:>10.2f}{s['median_us']:>9.1f}{s['median_kb']:>9.1f}"
                         f"{s['gb_per_s']:>7.1f}{fixed:>10}{fshare:>9}{par:>10}")
        if graphs and label in graphs:
            runs = memory_bound_runs(graphs[label])
            lines.append(f"  optimized graph: runs of consecutive Q/DQ/Transpose (length: count) {runs}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device-id", required=True)
    r.add_argument("--artifact", action="append", required=True, metavar="LABEL=PATH")
    r.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 12])
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--repeats", type=int, default=50, help="perf_test -r of each profiled run")
    r.add_argument("--out", type=Path, default=None)
    a = sub.add_parser("analyze")
    a.add_argument("report", type=Path)
    a.add_argument("--graph", action="append", default=[], metavar="LABEL=PATH",
                   help="ORT-optimized graph of an artifact (onnxruntime_perf_test -u)")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    graphs = {}
    for spec in args.graph:
        label, _, raw = spec.partition("=")
        graphs[label] = Path(raw)
    print(format_report(json.loads(args.report.read_text()), graphs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
