"""One-off (NOT part of the pipeline): how is the GELU laid out in an ONNX
graph -- as a single fused Gelu node, or as the loose Div/Erf/Add/Mul/Mul chain
-- and which nodes sit around it (Q/DQ pairs, extra consumers...).

Why: on the Jetson with the production thread count the INT8 DeiT-Tiny runs
slower than FP32 and its profile shows an `Erf` node that does not scale with
threads, while the FP32 profile has a fused `Gelu`. ORT's GeluFusion rewrites
the loose chain into one Gelu node when it matches; this tells whether it did
and, when it did not, what is in the way.

Two subcommands (the analysis needs only `onnx`, so it runs anywhere):

  analyze MODEL.onnx [MODEL2.onnx ...]
      op counts of the GELU family (Erf, Gelu, BiasGelu, FastGelu, QuickGelu),
      and one line per DISTINCT loose pattern around each Erf: what produces
      its input x (and how many consumers x has), the chain of ops
      Div > Erf > Add > Mul > Mul, and what consumes the result.
      Point it at the exported graph AND at the graph ORT optimized.

  variant MODEL.onnx OUT.onnx
      counterfactual, on a COPY (never an artifact): every loose GELU chain
      Div > Erf > Add > Mul > Mul becomes ONE com.microsoft::Gelu node (the
      exact erf GELU, the op ORT's own GeluFusion produces in FP32). Shows
      whether ORT keeps it fused when the quantized activation feeding it no
      longer fans out to two consumers; the copy can be timed with
      scripts/probe_int8_kernels.py like any other .onnx.

  dump MODEL.onnx OUT.onnx
      writes the graph ORT holds after loading it: onnxruntime, CPU EP,
      ORT_ENABLE_ALL -- the level onnxruntime_perf_test uses with `-o 99`
      (runner_vit passes -o 99). Needs onnxruntime. On a device without the
      Python package, get the same file from the compiled tool instead:
      onnxruntime_perf_test <the runner_vit flags> -r 1 -u OUT.onnx MODEL.onnx

    python quantization/vit_brevitas/investigations/ort_graph_gelu.py dump M.onnx M.ort_all.onnx
    python quantization/vit_brevitas/investigations/ort_graph_gelu.py analyze M.onnx M.ort_all.onnx
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

GELU_FAMILY = ("Erf", "Gelu", "BiasGelu", "FastGelu", "QuickGelu", "NewGelu")
CHAIN_AFTER_ERF = 3  # Add, Mul, Mul in the torch export of nn.GELU (exact / erf form)


def dump_optimized(model: Path, out: Path) -> Path:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(out)
    ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    return out


def single_gelu_variant(model: Path, out: Path) -> int:
    """Copy of `model` with every loose GELU chain written as one Gelu node
    (export/vit_to_onnx/single_gelu.py, the implementation the deployment
    export uses). Returns how many chains were replaced."""
    from export.vit_to_onnx.single_gelu import replace_gelu_chains

    return replace_gelu_chains(model, out)


def _index(graph):
    from export.vit_to_onnx.single_gelu import index_graph

    return index_graph(graph)


def _op(node) -> str:
    return node.op_type if not node.domain else f"{node.domain}::{node.op_type}"


def erf_patterns(graph) -> Counter:
    """Signature of every Erf node's surroundings, counted.

    "<producer of x> (x has N consumers) | Div > Erf > Add > Mul > Mul | -> <consumer>"
    where x is the input of the Div that feeds the Erf; the chain follows the
    single consumer of each output; `?` marks a step that is not the expected op."""
    producer, consumers = _index(graph)
    patterns: Counter = Counter()
    for erf in (n for n in graph.node if n.op_type == "Erf"):
        div = producer.get(erf.input[0])
        head = ["?"]
        source = "?"
        if div is not None:
            head = [_op(div)]
            x = div.input[0]
            src = producer.get(x)
            source = _op(src) if src is not None else ("initializer/input" if x else "?")
            source += f" (x has {len(consumers.get(x, []))} consumers)"
        chain, node = [_op(erf)], erf
        for _ in range(CHAIN_AFTER_ERF):
            nxt = consumers.get(node.output[0], [])
            if len(nxt) != 1:
                chain.append(f"?[{len(nxt)} consumers]")
                break
            node = nxt[0]
            chain.append(_op(node))
        after = consumers.get(node.output[0], [])
        tail = ",".join(sorted({_op(c) for c in after})) or "graph output"
        patterns[f"{source} | {' > '.join(head + chain)} | -> {tail}"] += 1
    return patterns


def analyze(path: Path) -> Dict:
    import onnx

    graph = onnx.load(str(path)).graph
    ops = Counter(_op(n) for n in graph.node)
    return {"path": str(path), "nodes": len(graph.node), "ops": ops,
            "gelu_family": {op: ops[op] for op in ops if op.split("::")[-1] in GELU_FAMILY},
            "quant": {op: ops[op] for op in ("QuantizeLinear", "DequantizeLinear") if op in ops},
            "erf_patterns": erf_patterns(graph)}


def format_report(report: Dict) -> str:
    lines = [f"== {report['path']}", f"   nodes: {report['nodes']}",
             f"   GELU family: {report['gelu_family'] or 'none'}",
             f"   Quantize/DequantizeLinear: {report['quant'] or 'none'}"]
    if report["erf_patterns"]:
        lines.append("   loose Erf patterns:")
        for signature, count in report["erf_patterns"].most_common():
            lines.append(f"     {count:>3} x  {signature}")
    else:
        lines.append("   no loose Erf node")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("models", nargs="+", type=Path)
    v = sub.add_parser("variant")
    v.add_argument("model", type=Path)
    v.add_argument("out", type=Path)
    d = sub.add_parser("dump")
    d.add_argument("model", type=Path)
    d.add_argument("out", type=Path)
    args = parser.parse_args(argv)

    if args.command == "variant":
        print(f"{single_gelu_variant(args.model, args.out)} GELU chain(s) replaced by one Gelu node -> {args.out}")
        return 0
    if args.command == "dump":
        print(f"written {dump_optimized(args.model, args.out)}")
        return 0
    for model in args.models:
        print(format_report(analyze(model)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
