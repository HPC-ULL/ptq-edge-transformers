"""One-off probe (NOT part of the pipeline): why does ONNX Runtime not fuse
the Brevitas QCDQ graph into integer kernels, and would it with ORT's own
quantizer?

Three subcommands, all writing only under results/investigations/vit/fusion_probes/models/ (never
artifacts/):

  native   Quantize the FP32 export (export_fp32_onnx.py) with ONNX
           Runtime's own onnxruntime.quantization.quantize_static: QDQ
           format, MinMax, per-tensor, symmetric int8 weights and
           activations -- mirroring the Brevitas INT8 baseline (minmax /
           per_tensor / symmetric) so the two graphs differ in STRUCTURE,
           not in the quantization recipe. Calibrated on the SAME 128
           images as the Brevitas baseline rep1 (calibration pool, train
           split, seed_for_repeat(1)). Only Conv/MatMul/Gemm are quantized,
           like Brevitas (which quantizes Linear/Conv inputs only).
             --input-only additionally sets OpTypesToExcludeOutputQuantization
           for those ops: no Q after their outputs, i.e. exactly Brevitas's
           input-only structure but with ORT's own exporter and no Clip --
           isolates "missing output Q" as a cause.

  inspect  Static structure report of any .onnx: Q/DQ/Clip counts, every
           Clip sitting between a QuantizeLinear and a DequantizeLinear (with
           its bounds, and whether it is on a weight path), and for each
           Conv/MatMul/Gemm whether its inputs come straight from a
           DequantizeLinear and whether its output feeds a QuantizeLinear.

  strip-clip  Copy of a graph without the Clips that sit right before a
           DequantizeLinear (Q -> Clip -> DQ, and int const -> Clip -> DQ,
           the form Brevitas uses for integer weights), then checks with ONNX
           Runtime that the output is unchanged (it should be exactly
           unchanged for weights quantized with a max-based scale: a
           narrow-range 8-bit Clip is then a numerical no-op).
           Isolates "Clip breaks ORT's pattern matching" as a cause.

Kernel verdicts come from scripts/probe_int8_kernels.py, run on the
resulting files; this script only builds and describes graphs.

Why Clip is suspected (checked in the Brevitas source, v0.12.1,
export/common/handler/base.py ClipMixin.int_clip_symbolic_kwargs): a Clip is
emitted when `narrow or bit_width != 8` -- so not only for INT6/INT4 but also
for 8-bit NARROW-range quantizers. The baseline's weight quantizer,
Int8WeightPerTensorFloat, is narrow (range [-127, 127]); its activation
quantizer is not. The installed Brevitas version is unpinned, so `inspect`
on the real artifact is what settles it.

Needs (cluster): onnx, onnxruntime (with onnxruntime.quantization), numpy,
and for `native`: torch, timm, Pillow and the prepared calibration pool.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantization.vit_brevitas.paths import FUSION_PROBES, investigation_dir  # noqa: E402

COMPUTE_OPS = ("Conv", "MatMul", "Gemm")
OUT_DIR = investigation_dir(FUSION_PROBES, "models")
CALIB_SET_SIZE = 128  # configs/vit_manifest.py baseline
CALIB_REPEAT_ID = 1


# --------------------------------------------------------------------------
# Graph helpers (pure onnx, unit-tested)
# --------------------------------------------------------------------------

def _constants(model) -> Dict[str, object]:
    from onnx import numpy_helper

    values = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    values[node.output[0]] = numpy_helper.to_array(attr.t)
    return values


def _index(model):
    producer = {out: node for node in model.graph.node for out in node.output}
    consumers: Dict[str, list] = defaultdict(list)
    for node in model.graph.node:
        for inp in node.input:
            if inp:
                consumers[inp].append(node)
    return producer, consumers


def find_q_clip_dq(model) -> List[dict]:
    """Every Clip sitting right before DequantizeLinear nodes (all of its
    consumers are DQ) on an integer path: its input is either a
    QuantizeLinear output ("q", Q -> Clip -> DQ) or a constant ("const",
    int const -> Clip -> DQ -- the form Brevitas uses for already-integer
    weights). An ordinary float Clip (e.g. ReLU6) never feeds a DQ."""
    producer, consumers = _index(model)
    consts = _constants(model)
    found = []
    for node in model.graph.node:
        if node.op_type != "Clip":
            continue
        src = producer.get(node.input[0])
        users = consumers.get(node.output[0], [])
        if not users or any(u.op_type != "DequantizeLinear" for u in users):
            continue
        if node.input[0] in consts:
            source = "const"
        elif src is not None and src.op_type == "QuantizeLinear":
            source = "q"
        else:
            continue
        bound_names = list(node.input[1:3]) + [""] * (2 - len(node.input[1:3]))  # opset >= 11: min/max are inputs
        low, high = (consts.get(name) if name else None for name in bound_names)
        found.append({
            "clip": node.name or node.output[0],
            "min": None if low is None else low.item(),
            "max": None if high is None else high.item(),
            "source": source,
            # a constant being quantized/clipped = a weight (or bias)
            "weight_path": source == "const" or src.input[0] in consts,
        })
    return found


def inspect_graph(model) -> dict:
    producer, consumers = _index(model)
    consts = _constants(model)
    ops = Counter(node.op_type for node in model.graph.node)
    clips = find_q_clip_dq(model)

    compute = [n for n in model.graph.node if n.op_type in COMPUTE_OPS]
    def from_dq(name: str) -> bool:
        p = producer.get(name)
        return p is not None and p.op_type == "DequantizeLinear"

    def weight_via_dq_transpose(node) -> bool:
        """Weight operand = Transpose(DQ(...)): what torch.onnx emits for a
        Linear whose weight is not a constant at export time (x @ W^T)."""
        if len(node.input) < 2:
            return False
        t = producer.get(node.input[1])
        return t is not None and t.op_type == "Transpose" and from_dq(t.input[0])

    def weight_dq_is_direct(node) -> bool:
        """The weight operand's DQ reads an int constant directly (the form
        ORT's selectors expect), rather than Q(float const) or Q -> Clip."""
        if len(node.input) < 2 or not from_dq(node.input[1]):
            return False
        return producer[node.input[1]].input[0] in consts

    per_op = defaultdict(Counter)
    for node in compute:
        c = per_op[node.op_type]
        c["total"] += 1
        c["act_input_from_dq"] += from_dq(node.input[0])
        c["weight_input_from_dq"] += len(node.input) > 1 and from_dq(node.input[1])
        c["weight_dq_reads_int_const"] += weight_dq_is_direct(node)
        c["weight_via_dq_transpose"] += weight_via_dq_transpose(node)
        users = consumers.get(node.output[0], [])
        c["output_to_q"] += bool(users) and all(u.op_type == "QuantizeLinear" for u in users)

    return {
        "op_counts": {op: ops.get(op, 0) for op in ("QuantizeLinear", "DequantizeLinear", "Clip") + COMPUTE_OPS},
        "q_clip_dq": {
            "total": len(clips),
            "weight_path": sum(c["weight_path"] for c in clips),
            "activation_path": sum(not c["weight_path"] for c in clips),
            "from_q": sum(c["source"] == "q" for c in clips),
            "from_const": sum(c["source"] == "const" for c in clips),
            "bounds": sorted(Counter((c["min"], c["max"]) for c in clips).items(), key=lambda kv: -kv[1]),
        },
        "compute_ops": {op: dict(c) for op, c in per_op.items()},
    }


def strip_q_clip_dq(model) -> int:
    """In place: rewire DQ nodes past every Clip found by find_q_clip_dq
    (Q -> Clip -> DQ and int const -> Clip -> DQ), drop those Clips.
    Returns how many were removed."""
    clips = {c["clip"] for c in find_q_clip_dq(model)}
    removed = []
    for node in list(model.graph.node):
        if node.op_type == "Clip" and (node.name or node.output[0]) in clips:
            for other in model.graph.node:
                for i, inp in enumerate(other.input):
                    if inp == node.output[0]:
                        other.input[i] = node.input[0]
            removed.append(node)
    for node in removed:
        model.graph.node.remove(node)
    return len(removed)


def quant_params(model) -> Dict[str, dict]:
    """{Q/DQ node name: {"op", "scale", "zero_point"}} (constant inputs only)."""
    consts = _constants(model)
    out = {}
    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        scale = consts.get(node.input[1]) if len(node.input) > 1 else None
        zp = consts.get(node.input[2]) if len(node.input) > 2 and node.input[2] else None
        out[node.name or node.output[0]] = {"op": node.op_type, "scale": scale, "zero_point": zp}
    return out


def compare_quant_params(a, b) -> dict:
    """Matches Q/DQ nodes BY NAME (Brevitas names them after the module path,
    so the quantized layers keep the same names across the two exports) and
    compares their scale / zero point."""
    import numpy as np

    pa, pb = quant_params(a), quant_params(b)
    common = sorted(set(pa) & set(pb))
    diffs = []
    zp_mismatch = []
    for name in common:
        sa, sb = pa[name]["scale"], pb[name]["scale"]
        if sa is None or sb is None or np.shape(sa) != np.shape(sb):
            diffs.append((name, float("inf")))
            continue
        rel = float(np.max(np.abs(np.asarray(sa, dtype=np.float64) - np.asarray(sb, dtype=np.float64))
                           / np.maximum(np.abs(np.asarray(sa, dtype=np.float64)), 1e-30)))
        diffs.append((name, rel))
        za, zb = pa[name]["zero_point"], pb[name]["zero_point"]
        if (za is None) != (zb is None) or (za is not None and not np.array_equal(za, zb)):
            zp_mismatch.append(name)
    differing = sorted([d for d in diffs if d[1] > 0], key=lambda d: -d[1])
    return {
        "only_in_a": sorted(set(pa) - set(pb)), "only_in_b": sorted(set(pb) - set(pa)),
        "matched": len(common), "scales_differing": len(differing),
        "max_rel_scale_diff": max((d[1] for d in diffs), default=0.0),
        "top_differing": differing[:8], "zero_point_mismatch": zp_mismatch,
    }


def _ort_outputs(path: Path, inputs, optimized: bool):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimized
                                        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return [session.run(None, {name: x})[0] for x in inputs]


def format_report(label: str, report: dict) -> str:
    lines = [f"=== {label} ==="]
    lines.append("  ops: " + ", ".join(f"{op} {n}" for op, n in report["op_counts"].items()))
    qcd = report["q_clip_dq"]
    lines.append(f"  Clip before DQ: {qcd['total']} (Q -> Clip -> DQ {qcd['from_q']}, int const -> Clip -> DQ "
                 f"{qcd['from_const']}; weight path {qcd['weight_path']}, activation path {qcd['activation_path']})"
                 + (f"; bounds (min, max) x count: {qcd['bounds']}" if qcd["total"] else ""))
    for op, c in sorted(report["compute_ops"].items()):
        lines.append(f"  {op:<7} total {c.get('total', 0):>3} | act input from DQ {c.get('act_input_from_dq', 0):>3} | "
                     f"weight from DQ {c.get('weight_input_from_dq', 0):>3} (DQ reads int const directly "
                     f"{c.get('weight_dq_reads_int_const', 0):>3}; via DQ -> Transpose {c.get('weight_via_dq_transpose', 0):>3})"
                     f" | output -> Q {c.get('output_to_q', 0):>3}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

class _CalibrationReader:
    """onnxruntime.quantization.CalibrationDataReader over the same images
    the Brevitas baseline rep1 calibrated on."""

    def __init__(self, model_name: str, input_name: str):
        from PIL import Image

        from data_prep.imagenet import get_calibration_batch
        from evaluation_offline.vit_accuracy.accuracy import resolve_preprocess
        from quantization.vit_brevitas.calibration import seed_for_repeat

        preprocess = resolve_preprocess(model_name)
        samples = get_calibration_batch(repetition_idx=0, batch_size=CALIB_SET_SIZE, seed=seed_for_repeat(CALIB_REPEAT_ID))
        self._items = iter(
            {input_name: preprocess(Image.open(s["path"]).convert("RGB")).unsqueeze(0).numpy()} for s in samples
        )

    def get_next(self):
        return next(self._items, None)


def cmd_native(args) -> None:
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    fp32 = Path(args.fp32 or OUT_DIR / f"vit_{args.model}_fp32.onnx")
    suffix = "ortnative_inputonly" if args.input_only else "ortnative"
    out = Path(args.out or OUT_DIR / f"vit_{args.model}_int8_{suffix}.onnx")
    input_name = onnx.load(str(fp32)).graph.input[0].name

    extra = {"ActivationSymmetric": True, "WeightSymmetric": True}
    if args.input_only:
        extra["OpTypesToExcludeOutputQuantization"] = list(COMPUTE_OPS)

    quantize_static(
        str(fp32), str(out), _CalibrationReader(args.model, input_name),
        quant_format=QuantFormat.QDQ, op_types_to_quantize=list(COMPUTE_OPS), per_channel=False,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax, extra_options=extra,
    )
    print(f"Wrote {out}  (ORT quantize_static: QDQ, MinMax, per-tensor, symmetric S8S8, "
          f"{CALIB_SET_SIZE} calibration images{', outputs NOT quantized' if args.input_only else ''})")
    print(format_report(out.name, inspect_graph(onnx.load(str(out)))))


def cmd_inspect(args) -> None:
    import onnx

    for path in args.onnx:
        print(format_report(Path(path).name, inspect_graph(onnx.load(str(path)))))


def _max_output_diff(a: Path, b: Path) -> float:
    import numpy as np
    import onnxruntime as ort

    sa = ort.InferenceSession(str(a), providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(str(b), providers=["CPUExecutionProvider"])
    inp = sa.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    return float(np.abs(sa.run(None, {inp.name: x})[0] - sb.run(None, {sb.get_inputs()[0].name: x})[0]).max())


def cmd_compare(args) -> None:
    import numpy as np
    import onnx

    a, b = Path(args.a), Path(args.b)
    report = compare_quant_params(onnx.load(str(a)), onnx.load(str(b)))
    print(f"=== quantization parameters: {a.name}  vs  {b.name} ===")
    print(f"  Q/DQ nodes matched by name: {report['matched']} (only in A: {len(report['only_in_a'])}, "
          f"only in B: {len(report['only_in_b'])})")
    print(f"  scales differing: {report['scales_differing']}, max relative difference "
          f"{report['max_rel_scale_diff']:.3e}; zero points differing: {len(report['zero_point_mismatch'])}")
    for name, rel in report["top_differing"]:
        print(f"    {rel:.3e}  {name}")
    for label, names in (("only in A", report["only_in_a"]), ("only in B", report["only_in_b"])):
        for name in names[:5]:
            print(f"    {label}: {name}")

    import onnxruntime as ort

    shape = [d if isinstance(d, int) else 1 for d in ort.InferenceSession(
        str(a), providers=["CPUExecutionProvider"]).get_inputs()[0].shape]
    rng = np.random.default_rng(0)
    inputs = [rng.standard_normal(shape).astype(np.float32) for _ in range(args.inputs)]
    print(f"\n=== ONNX Runtime outputs, A vs B, {args.inputs} random inputs ===")
    for optimized in (False, True):
        oa, ob = _ort_outputs(a, inputs, optimized), _ort_outputs(b, inputs, optimized)
        diff = max(float(np.abs(x - y).max()) for x, y in zip(oa, ob))
        agree = sum(int(np.argmax(x) == np.argmax(y)) for x, y in zip(oa, ob))
        level = "ENABLE_ALL (as evaluated)" if optimized else "DISABLE_ALL (plain Q/DQ math)"
        print(f"  graph optimizations {level:<30} max |A-B| {diff:.3e}   same top-1 {agree}/{len(inputs)}")


def cmd_strip_clip(args) -> None:
    import onnx

    src = Path(args.onnx)
    out = Path(args.out or OUT_DIR / f"{src.stem}_noclip.onnx")
    model = onnx.load(str(src))
    n = strip_q_clip_dq(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    diff = _max_output_diff(src, out)
    print(f"Removed {n} Q -> Clip -> DQ Clip node(s) -> {out}")
    print(f"max |original - stripped| on a random input: {diff:.3e}")
    if diff > 1e-5:
        print("WARNING: the Clips were NOT numerical no-ops (some quantized value hit the int8 extreme): the stripped "
              "graph is not equivalent -- use it only as a fusion test, never for accuracy.")
    print(format_report(out.name, inspect_graph(model)))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("native", help="quantize the FP32 export with onnxruntime.quantization.quantize_static")
    p.add_argument("--model", default="deit_tiny", choices=["deit_tiny", "mobilevit", "efficientvit_b1"])
    p.add_argument("--fp32", help="default: results/investigations/vit/fusion_probes/models/vit_<model>_fp32.onnx")
    p.add_argument("--out")
    p.add_argument("--input-only", action="store_true", help="do not quantize Conv/MatMul/Gemm outputs (Brevitas-like)")
    p.set_defaults(fn=cmd_native)

    p = sub.add_parser("inspect", help="structure report of one or more .onnx files")
    p.add_argument("onnx", nargs="+")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("compare", help="compare two exports: Q/DQ parameters by name, and ORT outputs with "
                                       "graph optimizations off and on")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--inputs", type=int, default=8, help="random inputs for the output comparison")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("strip-clip", help="copy without the Clips before DQ, checked for equivalence")
    p.add_argument("onnx")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_strip_clip)

    args = parser.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
