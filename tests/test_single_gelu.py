"""export/vit_to_onnx/single_gelu.py and the deployment manifest's single_gelu flag.
The rewrite needs onnx (skipped otherwise)."""

from __future__ import annotations

import sys
import types

import pytest

from configs import vit_deployment_manifest as dm


def test_single_gelu_only_for_deit_tinys_quantized_deployment_rows():
    rows = dm.package_rows()
    flagged = [r["run_id"] for r in rows if r["single_gelu"]]

    assert flagged == ["vit_deit_tiny_deploy_per_tensor_symmetric_int8",
                       "vit_deit_tiny_deploy_per_channel_symmetric_int8",
                       "vit_deit_tiny_deploy_per_tensor_asymmetric_int8"]
    assert not any(r["single_gelu"] for r in rows if r["model_name"] == "mobilevit" or not r["quantized"]
                   or r.get("fallback_reference"))


def test_export_does_the_rewrite_only_when_asked(monkeypatch, tmp_path):
    from export.vit_to_onnx import export as export_module
    from export.vit_to_onnx import single_gelu

    fake_export = types.ModuleType("brevitas.export")
    fake_export.export_onnx_qcdq = lambda model, **kw: None
    fake_torch = types.ModuleType("torch")
    fake_torch.randn = lambda *shape: None
    monkeypatch.setitem(sys.modules, "brevitas.export", fake_export)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    calls = []
    monkeypatch.setattr(single_gelu, "fuse_gelu_in_place", lambda path: calls.append(path) or 12)

    export_module.export_to_onnx("m", (1, 3, 8, 8), "r", 1, out_path=tmp_path / "a.onnx")
    assert calls == []  # default: the ablation's export is untouched

    export_module.export_to_onnx("m", (1, 3, 8, 8), "r", 1, out_path=tmp_path / "b.onnx", single_gelu=True)
    assert calls == [tmp_path / "b.onnx"]


onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

from export.vit_to_onnx import single_gelu as sg  # noqa: E402


def _model(with_chain=True, extra_erf=False):
    nodes = [helper.make_node("Constant", [], ["s2"], value=helper.make_tensor("v", TensorProto.FLOAT, [], [1.4142135])),
             helper.make_node("Constant", [], ["one"], value=helper.make_tensor("v", TensorProto.FLOAT, [], [1.0])),
             helper.make_node("Constant", [], ["half"], value=helper.make_tensor("v", TensorProto.FLOAT, [], [0.5])),
             helper.make_node("Relu", ["in"], ["x"])]
    if with_chain:
        nodes += [helper.make_node("Div", ["x", "s2"], ["d"]), helper.make_node("Erf", ["d"], ["e"]),
                  helper.make_node("Add", ["e", "one"], ["a"]), helper.make_node("Mul", ["x", "a"], ["m"]),
                  helper.make_node("Mul", ["m", "half"], ["y"])]
    else:
        nodes.append(helper.make_node("Identity", ["x"], ["y"]))
    if extra_erf:
        nodes.append(helper.make_node("Erf", ["y"], ["z"]))
    out = "z" if extra_erf else "y"
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("in", TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info(out, TensorProto.FLOAT, [1, 4])])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_replace_writes_one_gelu_node_reading_the_same_x_and_drops_the_chains_constants(tmp_path):
    src, dst = tmp_path / "a.onnx", tmp_path / "b.onnx"
    onnx.save(_model(), src)

    assert sg.replace_gelu_chains(src, dst) == 1

    nodes = onnx.load(str(dst)).graph.node
    assert [(n.domain, n.op_type) for n in nodes] == [("", "Relu"), ("com.microsoft", "Gelu")]
    assert list(nodes[1].input) == ["x"] and list(nodes[1].output) == ["y"]


def test_fuse_in_place_raises_when_there_is_no_chain_or_an_erf_is_left(tmp_path):
    none, stray = tmp_path / "none.onnx", tmp_path / "stray.onnx"
    onnx.save(_model(with_chain=False), none)
    onnx.save(_model(extra_erf=True), stray)

    with pytest.raises(ValueError, match="does not apply"):
        sg.fuse_gelu_in_place(none)
    with pytest.raises(ValueError, match="Erf node"):
        sg.fuse_gelu_in_place(stray)


def test_fuse_in_place_rewrites_the_file(tmp_path):
    path = tmp_path / "a.onnx"
    onnx.save(_model(), path)

    assert sg.fuse_gelu_in_place(path) == 1
    assert "Erf" not in [n.op_type for n in onnx.load(str(path)).graph.node]
