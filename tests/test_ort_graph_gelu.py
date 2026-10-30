"""investigations/ort_graph_gelu.py on tiny synthetic graphs (needs onnx)."""

from __future__ import annotations

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

from quantization.vit_brevitas.investigations import ort_graph_gelu as og  # noqa: E402


def _gelu_chain_model(quantized: bool):
    nodes, inits = [], [helper.make_tensor("s2", TensorProto.FLOAT, [], [1.4142135]),
                        helper.make_tensor("one", TensorProto.FLOAT, [], [1.0]),
                        helper.make_tensor("half", TensorProto.FLOAT, [], [0.5])]
    x = "x"
    if quantized:
        inits += [helper.make_tensor("sc", TensorProto.FLOAT, [], [0.1]), helper.make_tensor("zp", TensorProto.INT8, [], [0])]
        nodes.append(helper.make_node("DequantizeLinear", ["xq", "sc", "zp"], ["x"]))
    nodes += [helper.make_node("Div", [x, "s2"], ["d"]), helper.make_node("Erf", ["d"], ["e"]),
              helper.make_node("Add", ["e", "one"], ["a"]), helper.make_node("Mul", [x, "a"], ["m"]),
              helper.make_node("Mul", ["m", "half"], ["y"])]
    ins = [helper.make_tensor_value_info("xq" if quantized else "x", TensorProto.INT8 if quantized else TensorProto.FLOAT, [1, 4])]
    graph = helper.make_graph(nodes, "g", ins, [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_analyze_reports_the_loose_chain_with_its_producer_and_fan_out(tmp_path):
    path = tmp_path / "m.onnx"
    onnx.save(_gelu_chain_model(quantized=True), path)

    report = og.analyze(path)

    assert report["gelu_family"] == {"Erf": 1}
    assert list(report["erf_patterns"]) == [
        "DequantizeLinear (x has 2 consumers) | Div > Erf > Add > Mul > Mul | -> graph output"]
    assert "loose Erf patterns" in og.format_report(report)


def test_variant_replaces_the_chain_by_one_gelu_node_and_keeps_the_rest(tmp_path):
    src, out = tmp_path / "m.onnx", tmp_path / "v.onnx"
    onnx.save(_gelu_chain_model(quantized=True), src)

    assert og.single_gelu_variant(src, out) == 1

    model = onnx.load(str(out))
    assert [(n.domain, n.op_type) for n in model.graph.node] == [("", "DequantizeLinear"), ("com.microsoft", "Gelu")]
    assert any(o.domain == "com.microsoft" for o in model.opset_import)
    assert og.analyze(out)["gelu_family"] == {"com.microsoft::Gelu": 1}
    onnx.checker.check_model(model)


def test_variant_leaves_a_graph_without_the_chain_untouched(tmp_path):
    src, out = tmp_path / "m.onnx", tmp_path / "v.onnx"
    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), src)

    assert og.single_gelu_variant(src, out) == 0
    assert [n.op_type for n in onnx.load(str(out)).graph.node] == ["Relu"]
