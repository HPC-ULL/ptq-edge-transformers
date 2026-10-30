"""Tests for probe_qdq_fusion.py's graph analysis on small synthetic graphs
shaped like Brevitas's QCDQ export and like ORT's native QDQ output.
Skipped where the onnx package is not installed (it is on the cluster)."""

from __future__ import annotations

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from quantization.vit_brevitas.investigations import probe_qdq_fusion as probe  # noqa: E402


def _init(name, array):
    return numpy_helper.from_array(np.asarray(array), name)


def _brevitas_like(clip_on_weight=True, clip_on_act=False):
    """x -> Q -> [Clip] -> DQ --\\
       W(float const) -> Q -> [Clip] -> DQ -> MatMul -> Add (float output, no Q)."""
    inits = [
        _init("W", np.ones((4, 4), dtype=np.float32)), _init("s", np.float32(0.1)), _init("zp", np.int8(0)),
        _init("cmin", np.int8(-127)), _init("cmax", np.int8(127)), _init("b", np.zeros(4, dtype=np.float32)),
    ]
    nodes = [helper.make_node("QuantizeLinear", ["x", "s", "zp"], ["xq"], name="qx")]
    act = "xq"
    if clip_on_act:
        nodes.append(helper.make_node("Clip", ["xq", "cmin", "cmax"], ["xqc"], name="clip_x"))
        act = "xqc"
    nodes.append(helper.make_node("DequantizeLinear", [act, "s", "zp"], ["xd"], name="dqx"))
    nodes.append(helper.make_node("QuantizeLinear", ["W", "s", "zp"], ["wq"], name="qw"))
    w = "wq"
    if clip_on_weight:
        nodes.append(helper.make_node("Clip", ["wq", "cmin", "cmax"], ["wqc"], name="clip_w"))
        w = "wqc"
    nodes += [
        helper.make_node("DequantizeLinear", [w, "s", "zp"], ["wd"], name="dqw"),
        helper.make_node("MatMul", ["xd", "wd"], ["mm"], name="mm"),
        helper.make_node("Add", ["mm", "b"], ["y"], name="add"),
    ]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _native_like():
    """x -> Q -> DQ, W(int8 const) -> DQ, MatMul -> Q -> DQ -> Add."""
    inits = [
        _init("Wq", np.ones((4, 4), dtype=np.int8)), _init("s", np.float32(0.1)), _init("zp", np.int8(0)),
        _init("b", np.zeros(4, dtype=np.float32)),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "s", "zp"], ["xq"]),
        helper.make_node("DequantizeLinear", ["xq", "s", "zp"], ["xd"]),
        helper.make_node("DequantizeLinear", ["Wq", "s", "zp"], ["wd"]),
        helper.make_node("MatMul", ["xd", "wd"], ["mm"]),
        helper.make_node("QuantizeLinear", ["mm", "s", "zp"], ["mmq"]),
        helper.make_node("DequantizeLinear", ["mmq", "s", "zp"], ["mmd"]),
        helper.make_node("Add", ["mmd", "b"], ["y"]),
    ]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_detects_the_narrow_range_weight_clip_between_q_and_dq():
    clips = probe.find_q_clip_dq(_brevitas_like())

    assert clips == [{"clip": "clip_w", "min": -127, "max": 127, "source": "q", "weight_path": True}]


def test_activation_path_clip_is_reported_as_such():
    clips = probe.find_q_clip_dq(_brevitas_like(clip_on_weight=False, clip_on_act=True))

    assert [(c["clip"], c["weight_path"]) for c in clips] == [("clip_x", False)]


def test_a_clip_not_between_q_and_dq_is_ignored():
    """A plain activation Clip (e.g. ReLU6) must not be counted."""
    model = _brevitas_like(clip_on_weight=False)
    model.graph.node.append(helper.make_node("Clip", ["y", "", ""], ["y6"], name="relu6"))

    assert probe.find_q_clip_dq(model) == []


def test_inspect_brevitas_like_structure():
    report = probe.inspect_graph(_brevitas_like())

    assert report["op_counts"]["Clip"] == 1
    assert report["q_clip_dq"]["weight_path"] == 1 and report["q_clip_dq"]["bounds"] == [((-127, 127), 1)]
    mm = report["compute_ops"]["MatMul"]
    assert mm["total"] == 1 and mm["act_input_from_dq"] == 1 and mm["weight_input_from_dq"] == 1
    assert mm["weight_dq_reads_int_const"] == 0  # DQ reads Clip(Q(float W)), not an int constant
    assert mm["output_to_q"] == 0  # input-only quantization: float output


def test_inspect_native_like_structure():
    report = probe.inspect_graph(_native_like())

    assert report["q_clip_dq"]["total"] == 0
    mm = report["compute_ops"]["MatMul"]
    assert mm["weight_dq_reads_int_const"] == 1 and mm["output_to_q"] == 1


def test_strip_clip_rewires_dq_to_the_quantize_output_and_keeps_a_valid_graph():
    model = _brevitas_like(clip_on_act=True)

    assert probe.strip_q_clip_dq(model) == 2

    assert probe.find_q_clip_dq(model) == []
    assert all(n.op_type != "Clip" for n in model.graph.node)
    dq_inputs = {n.name: n.input[0] for n in model.graph.node if n.op_type == "DequantizeLinear"}
    assert dq_inputs == {"dqx": "xq", "dqw": "wq"}
    onnx.checker.check_model(model)


def test_format_report_mentions_the_key_facts():
    text = probe.format_report("brevitas.onnx", probe.inspect_graph(_brevitas_like()))

    assert "Clip before DQ: 1 (Q -> Clip -> DQ 1, int const -> Clip -> DQ 0; weight path 1" in text
    assert "(-127, 127)" in text and "output -> Q   0" in text


def _brevitas_int_weight_like():
    """What the real Brevitas artifact looks like per the cluster inspect:
    int8 weight constant -> Clip(-127,127) -> DQ -> Transpose -> MatMul
    (no Q node for the weight, no Q after the MatMul)."""
    inits = [
        _init("Wq", np.full((4, 4), 3, dtype=np.int8)), _init("s", np.float32(0.1)), _init("zp", np.int8(0)),
        _init("cmin", np.int8(-127)), _init("cmax", np.int8(127)),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "s", "zp"], ["xq"], name="qx"),
        helper.make_node("DequantizeLinear", ["xq", "s", "zp"], ["xd"], name="dqx"),
        helper.make_node("Clip", ["Wq", "cmin", "cmax"], ["wqc"], name="clip_w"),
        helper.make_node("DequantizeLinear", ["wqc", "s", "zp"], ["wd"], name="dqw"),
        helper.make_node("Transpose", ["wd"], ["wt"], name="t", perm=[1, 0]),
        helper.make_node("MatMul", ["xd", "wt"], ["y"], name="mm"),
    ]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_detects_a_clip_on_an_int_weight_constant():
    clips = probe.find_q_clip_dq(_brevitas_int_weight_like())

    assert clips == [{"clip": "clip_w", "min": -127, "max": 127, "source": "const", "weight_path": True}]


def test_inspect_reports_the_weight_transpose_between_dq_and_matmul():
    report = probe.inspect_graph(_brevitas_int_weight_like())

    mm = report["compute_ops"]["MatMul"]
    assert mm["weight_input_from_dq"] == 0 and mm["weight_via_dq_transpose"] == 1
    assert report["q_clip_dq"]["from_const"] == 1 and report["q_clip_dq"]["from_q"] == 0


def test_strip_clip_also_removes_the_clip_on_an_int_weight_constant():
    model = _brevitas_int_weight_like()

    assert probe.strip_q_clip_dq(model) == 1

    assert all(n.op_type != "Clip" for n in model.graph.node)
    assert next(n for n in model.graph.node if n.name == "dqw").input[0] == "Wq"
    onnx.checker.check_model(model)


def _named_qdq(scale_x=0.1, scale_w=0.1, extra_node=False):
    inits = [
        _init("Wq", np.ones((4, 4), dtype=np.int8)), _init("sx", np.float32(scale_x)), _init("sw", np.float32(scale_w)),
        _init("zp", np.int8(0)),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "sx", "zp"], ["xq"], name="/layer/input_quant/QuantizeLinear"),
        helper.make_node("DequantizeLinear", ["xq", "sx", "zp"], ["xd"], name="/layer/input_quant/DequantizeLinear"),
        helper.make_node("DequantizeLinear", ["Wq", "sw", "zp"], ["wd"], name="/layer/weight_quant/DequantizeLinear"),
        helper.make_node("MatMul", ["xd", "wd"], ["y"], name="mm"),
    ]
    if extra_node:
        nodes.append(helper.make_node("QuantizeLinear", ["y", "sx", "zp"], ["yq"], name="/extra/QuantizeLinear"))
    outs = [helper.make_tensor_value_info("yq" if extra_node else "y",
                                          TensorProto.INT8 if extra_node else TensorProto.FLOAT, [1, 4])]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])], outs, inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_compare_identical_quant_params():
    report = probe.compare_quant_params(_named_qdq(), _named_qdq())

    assert report["matched"] == 3 and report["scales_differing"] == 0 and report["max_rel_scale_diff"] == 0
    assert report["only_in_a"] == [] and report["only_in_b"] == [] and report["zero_point_mismatch"] == []


def test_compare_reports_which_scale_moved_and_unmatched_nodes():
    report = probe.compare_quant_params(_named_qdq(), _named_qdq(scale_x=0.11, extra_node=True))

    assert report["scales_differing"] == 2  # the input Q and DQ share the moved scale
    assert report["max_rel_scale_diff"] == pytest.approx(0.1, rel=1e-5)
    assert {name for name, _ in report["top_differing"]} == {"/layer/input_quant/QuantizeLinear",
                                                             "/layer/input_quant/DequantizeLinear"}
    assert report["only_in_b"] == ["/extra/QuantizeLinear"]
