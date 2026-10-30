"""scripts/profile_scaling.py with synthetic ORT profiles and a fake
onnxruntime_perf_test (no device)."""

from __future__ import annotations

import json
import types

import pytest

import scripts.profile_scaling as ps


def _ev(node, op, dur, *, block_size=(), shapes_in=None, shapes_out=None):
    stats = {"main_thread": {"block_size": list(block_size)}} if block_size is not None else {}
    return {"cat": "Node", "name": f"{node}_kernel_time", "dur": dur,
            "args": {"op_name": op, "thread_scheduling_stats": stats,
                     "input_type_shape": shapes_in or [{"float": [1, 197, 192]}],
                     "output_type_shape": shapes_out or [{"uint8": [1, 197, 192]}]}}


def test_reduce_profile_per_call_time_bytes_and_parallel_flag():
    events = [_ev("q1", "QuantizeLinear", 100), _ev("q1", "QuantizeLinear", 300),  # 2 calls -> 0.2 ms
              _ev("g1", "QGemm", 1000, block_size=[1]),
              {"cat": "Node", "name": "q1_fence_before", "dur": 999, "args": {}}]  # bookkeeping: ignored

    per_op, nodes = ps.reduce_profile(events)

    assert per_op == pytest.approx({"QuantizeLinear": 0.2, "QGemm": 1.0})
    assert nodes["q1_kernel_time"]["bytes"] == 197 * 192 * 4 + 197 * 192
    assert nodes["q1_kernel_time"]["parallel"] is False and nodes["g1_kernel_time"]["parallel"] is True


def test_parallel_flag_is_unknown_when_the_build_reports_no_thread_stats():
    _, nodes = ps.reduce_profile([_ev("t", "Transpose", 10, block_size=None)])
    assert nodes["t_kernel_time"]["parallel"] is None


def test_groups():
    assert ps.group_of("FusedConv") == "Conv" and ps.group_of("BiasGelu") == "Gelu/QuickGelu"
    assert ps.group_of("MatMulIntegerToFloat") == "MatMul (attention)" and ps.group_of("Softmax") == ps.REST


def _report(times):
    """times: {threads: {op: ms}} for one label, 3 identical rounds; nodes on round 1."""
    runs = []
    for t, ops in times.items():
        for rnd in (1, 2, 3):
            entry = {"threads": t, "round": rnd, "label": "m", "total_ms": sum(ops.values()), "ops": dict(ops)}
            if rnd == 1:
                entry["nodes"] = {f"{op}_{i}": {"op": op, "ms": ms / 4, "bytes": 1000 * (i + 1), "parallel": op == "QGemm"}
                                  for op, ms in ops.items() for i in range(4)}
            runs.append(entry)
    return {"device_id": "d", "threads": sorted(times), "rounds": 3, "repeats": 10, "artifacts": {"m": "m.onnx"},
            "runs": runs}


def test_scaling_table_and_amdahl():
    # QGemm scales 10x, QuantizeLinear does not scale at all
    report = _report({1: {"QGemm": 90.0, "QuantizeLinear": 10.0}, 12: {"QGemm": 9.0, "QuantizeLinear": 10.0}})

    rows = {r["group"]: r for r in ps.scaling_table(report, "m")}
    assert rows["QGemm"]["factor"] == pytest.approx(10.0)
    assert rows["QuantizeLinear"]["factor"] == pytest.approx(1.0)
    assert rows["TOTAL"]["factor"] == pytest.approx(100 / 19)

    a = ps.amdahl(report, "m")
    assert a["serial_groups"] == ["QuantizeLinear"]
    assert a["parallel_fraction_1t"] == pytest.approx(0.9)
    assert a["amdahl_max"] == pytest.approx(1 / (0.1 + 0.9 / 12))
    assert a["serial_share"][12] == pytest.approx(10 / 19)


def test_node_stats_reports_memory_bound_ops_next_to_the_largest_scaling_op():
    report = _report({1: {"QGemm": 90.0, "QuantizeLinear": 10.0, "Transpose": 4.0},
                      12: {"QGemm": 9.0, "QuantizeLinear": 10.0, "Transpose": 4.0}})

    rows = {s["op"]: s for s in ps.node_stats(report, "m")}

    assert set(rows) == {"QuantizeLinear", "Transpose", "QGemm"} and rows["QGemm"]["is_reference"]
    assert rows["QuantizeLinear"]["nodes"] == 4 and rows["QuantizeLinear"]["parallel_share"] == 0.0
    assert rows["QGemm"]["parallel_share"] == 1.0
    # identical per-node times regardless of size: all of it is a fixed per-node cost
    assert rows["QuantizeLinear"]["fixed_share"] == pytest.approx(1.0)
    assert "Amdahl" in ps.format_report(report)


def test_run_alternates_artifacts_uses_the_runner_command_and_writes_json(monkeypatch, tmp_path):
    from benchmarking_ondevice.runner_vit import devices as vit_devices

    a, b = tmp_path / "a.onnx", tmp_path / "b.onnx"
    a.write_text("x"), b.write_text("x")
    monkeypatch.setattr(vit_devices, "resolve_device",
                        lambda device_id: {"device_id": device_id, "backend": "cpu", "cpu_cores": 12})
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        prefix = cmd[cmd.index("-p") + 1]
        model = cmd[-2]
        json.dump([_ev("n", "QGemm" if model.endswith("a.onnx") else "Gemm", 2000)], open(prefix + "_x.json", "w"))
        return types.SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(ps.subprocess, "run", fake_run)
    out = tmp_path / "r.json"

    assert ps.main(["run", "--device-id", "jetson_cpu_50w", "--threads", "1", "4", "--rounds", "2",
                    "--artifact", f"a={a}", "--artifact", f"b={b}", "--out", str(out)]) == 0

    order = [(c[c.index("-x") + 1], c[-2].split("/")[-1]) for c in seen]
    assert order == [("1", "a.onnx"), ("1", "b.onnx")] * 2 + [("4", "a.onnx"), ("4", "b.onnx")] * 2
    assert all(c[c.index("-y") + 1] == "1" for c in seen)
    report = json.loads(out.read_text())
    assert len(report["runs"]) == 8 and sum("nodes" in r for r in report["runs"]) == 4
    assert report["runs"][0]["ops"] == {"QGemm": 2.0}


def test_memory_bound_runs_counts_consecutive_q_dq_transpose(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    sc = helper.make_tensor("s", TensorProto.FLOAT, [], [0.1])
    zp = helper.make_tensor("z", TensorProto.UINT8, [], [0])
    nodes = [helper.make_node("QuantizeLinear", ["x", "s", "z"], ["q"]),        # run of 3: Q > DQ > Transpose
             helper.make_node("DequantizeLinear", ["q", "s", "z"], ["d"]),
             helper.make_node("Transpose", ["d"], ["t"], perm=[1, 0]),
             helper.make_node("Relu", ["t"], ["r"]),
             helper.make_node("QuantizeLinear", ["r", "s", "z"], ["y"])]       # run of 1
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
                              [helper.make_tensor_value_info("y", TensorProto.UINT8, [3, 2])], [sc, zp])
    path = tmp_path / "g.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)

    assert ps.memory_bound_runs(path) == {1: 1, 3: 1}
