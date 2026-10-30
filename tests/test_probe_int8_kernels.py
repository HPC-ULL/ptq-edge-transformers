"""Tests for scripts/probe_int8_kernels.py (one-off probe): the profile analysis with
synthetic ORT profiles, and the probe's flow with a fake onnxruntime_perf_test.
"""

from __future__ import annotations

import json
import types

import pytest

import scripts.probe_int8_kernels as probe


def _ev(node, op, dur_us, provider="CPUExecutionProvider", cat="Node", suffix="_kernel_time"):
    return {"cat": cat, "name": f"{node}{suffix}", "dur": dur_us, "args": {"op_name": op, "provider": provider}}


# --------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "op, category",
    [
        ("QLinearConv", "integer"), ("QLinearMatMul", "integer"), ("QLinearAdd", "integer"),
        ("MatMulInteger", "integer"), ("ConvInteger", "integer"), ("MatMulIntegerToFloat", "integer"),
        ("DynamicQuantizeMatMul", "integer"), ("QGemm", "integer"),
        ("QuantizeLinear", "qdq"), ("DequantizeLinear", "qdq"), ("DynamicQuantizeLinear", "qdq"),
        ("Conv", "float_compute"), ("MatMul", "float_compute"), ("Gemm", "float_compute"),
        ("FusedMatMul", "float_compute"), ("NhwcFusedConv", "float_compute"),
        ("Softmax", "other"), ("LayerNormalization", "other"), ("Gelu", "other"),
    ],
)
def test_categorize(op, category):
    assert probe.categorize(op) == category


# --------------------------------------------------------------------------
# Reading the profile
# --------------------------------------------------------------------------

def test_summarize_profile_counts_only_node_kernel_events():
    events = [
        _ev("conv1", "Conv", 100), _ev("conv1", "Conv", 120),  # same node, two inference runs
        _ev("conv2", "Conv", 80),
        _ev("conv1", "Conv", 5, suffix="_fence_before"),        # bookkeeping: ignored
        _ev("session", "x", 999, cat="Session", suffix="_initialization"),
        {"cat": "Node", "name": "orphan_kernel_time", "dur": 7, "args": {}},  # no op_name -> "?"
    ]

    per_op = probe.summarize_profile(events)

    assert set(per_op) == {"Conv", "?"}
    assert per_op["Conv"]["nodes"] == {"conv1_kernel_time", "conv2_kernel_time"}
    assert per_op["Conv"]["calls"] == 3
    assert per_op["Conv"]["dur_us"] == 300.0
    assert per_op["Conv"]["providers"] == {"CPUExecutionProvider"}


def test_load_events_accepts_a_bare_list_and_a_traceevents_dict(tmp_path):
    events = [_ev("a", "Conv", 1)]
    (tmp_path / "list.json").write_text(json.dumps(events))
    (tmp_path / "dict.json").write_text(json.dumps({"traceEvents": events}))

    assert probe.load_events(tmp_path / "list.json") == events
    assert probe.load_events(tmp_path / "dict.json") == events


def _verdict(events):
    return probe.interpret(probe.rollup(probe.summarize_profile(events)))[0]


def test_verdict_real_int8_when_integer_kernels_carry_the_compute():
    events = [_ev("c", "QLinearConv", 900), _ev("m", "MatMulIntegerToFloat", 900), _ev("f", "Gemm", 100),
              _ev("q", "QuantizeLinear", 50)]

    assert _verdict(events) == "REAL_INT8"


def test_verdict_fallback_when_only_qdq_nodes_surround_float_compute():
    """DequantizeLinear -> Conv/MatMul -> QuantizeLinear, no integer kernel."""
    events = [_ev("dq", "DequantizeLinear", 200), _ev("c", "Conv", 800), _ev("m", "MatMul", 700),
              _ev("q", "QuantizeLinear", 100)]

    code, text = probe.interpret(probe.rollup(probe.summarize_profile(events)))

    assert code == "FALLBACK"
    assert "no real INT8 acceleration" in text


def test_verdict_mixed_reports_the_integer_share():
    events = [_ev("m", "MatMulIntegerToFloat", 400), _ev("c", "Conv", 600), _ev("dq", "DequantizeLinear", 100)]

    code, text = probe.interpret(probe.rollup(probe.summarize_profile(events)))

    assert code == "MIXED"
    assert "40%" in text  # 400 / (400 + 600)


def test_verdict_float_and_no_data():
    assert _verdict([_ev("c", "Conv", 500), _ev("m", "MatMul", 500)]) == "FLOAT"
    assert _verdict([_ev("s", "Softmax", 500)]) == "NO_DATA"
    assert _verdict([]) == "NO_DATA"


def test_rollup_time_shares_add_up_and_count_unique_nodes():
    events = [_ev("c1", "QLinearConv", 100), _ev("c1", "QLinearConv", 100), _ev("c2", "QLinearConv", 100),
              _ev("f", "Softmax", 100)]

    roll = probe.rollup(probe.summarize_profile(events))

    assert roll["integer"]["nodes"] == 2  # c1 and c2, however many runs
    assert roll["integer"]["time_share"] == pytest.approx(0.75)
    assert sum(c["time_share"] for c in roll.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Device side, with a fake onnxruntime_perf_test
# --------------------------------------------------------------------------

def test_parse_artifact_requires_label_and_an_existing_file(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    assert probe.parse_artifact(f"int8={model}") == ("int8", model)
    for bad in ("int8", f"={model}", f"int8={tmp_path / 'missing.onnx'}"):
        with pytest.raises(SystemExit):
            probe.parse_artifact(bad)


def test_latency_alternates_the_artifacts_instead_of_running_them_in_blocks(monkeypatch, tmp_path):
    calls = []

    def fake_run(path, device, repeats):
        calls.append((path.name, device["cpu_cores"], repeats))
        return {"avg_latency_ms": 10.0 if path.name == "a" else 5.0}

    monkeypatch.setattr(probe, "_run_perf_test", fake_run)

    samples = probe.measure_latency([("fp32", tmp_path / "a"), ("int8", tmp_path / "b")], {"cpu_cores": 1},
                                    rounds=3, repeats=10)

    assert [c[0] for c in calls] == ["a", "b"] * 3
    assert all(c[1:] == (1, 10) for c in calls)
    assert samples == {"fp32": [10.0] * 3, "int8": [5.0] * 3}


def test_run_perf_test_uses_the_runner_command_and_parser_without_writing_results(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        probe.vit_runner, "build_benchmark_command",
        lambda path, device, *, repeats, output_prefix: seen.update(device=device, repeats=repeats) or ["perf_test", str(path)],
    )
    monkeypatch.setattr(probe.subprocess, "run",
                        lambda cmd, **kw: types.SimpleNamespace(stdout="PERF", stderr=""))
    from benchmarking_ondevice.runner_vit import output_parser
    monkeypatch.setattr(output_parser, "parse_perf_test_output", lambda text: {"avg_latency_ms": 3.5, "raw": text})

    parsed = probe._run_perf_test(tmp_path / "m.onnx", {"cpu_cores": 1, "backend": "cpu"}, 50)

    assert parsed["avg_latency_ms"] == 3.5 and "PERF" in parsed["raw"]
    assert seen == {"device": {"cpu_cores": 1, "backend": "cpu"}, "repeats": 50}


def test_profile_run_puts_the_profile_flag_before_the_model_path(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        probe.vit_runner, "build_benchmark_command",
        lambda path, device, *, repeats, output_prefix: ["perf_test", "-m", "times", str(path), str(output_prefix)],
    )

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        (tmp_path / "profile_int8_2026-01-01_00-00-00.json").write_text("[]")

    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    found = probe.profile_artifact("int8", tmp_path / "model.onnx", {"backend": "cpu"}, 20, tmp_path)

    cmd = seen["cmd"]
    assert cmd[:3] == ["perf_test", "-m", "times"]
    assert cmd[3] == "-p" and cmd[4] == str(tmp_path / "profile_int8")
    assert cmd[5] == str(tmp_path / "model.onnx")  # the model path is still where perf_test expects it
    assert found.name.startswith("profile_int8_")


def test_profile_run_fails_loudly_when_no_profile_was_written(monkeypatch, tmp_path):
    monkeypatch.setattr(
        probe.vit_runner, "build_benchmark_command",
        lambda path, device, *, repeats, output_prefix: ["perf_test", str(path), str(output_prefix)],
    )
    monkeypatch.setattr(probe.subprocess, "run", lambda cmd, **kw: None)

    with pytest.raises(SystemExit, match="wrote no profile"):
        probe.profile_artifact("int8", tmp_path / "m.onnx", {}, 20, tmp_path)


def _fake_probe_environment(monkeypatch, tmp_path, *, int8_events, backend="cpu"):
    """main() end to end, with a fake perf_test: fp32 = 10 ms and float
    kernels, int8 = 4 ms and `int8_events`."""
    fp32, int8 = tmp_path / "fp32.onnx", tmp_path / "int8.onnx"
    fp32.write_bytes(b"x")
    int8.write_bytes(b"x")
    monkeypatch.setattr(probe.vit_devices, "resolve_device", lambda device_id: {"backend": backend, "cpu_cores": 12})
    monkeypatch.setattr(probe, "read_uptime_s", lambda: 5000.0)
    monkeypatch.setattr(probe, "_run_perf_test",
                        lambda path, device, repeats: {"avg_latency_ms": 10.0 if "fp32" in path.name else 4.0})

    def fake_profile(label, path, device, repeats, workdir):
        events = [_ev("c", "Conv", 900), _ev("m", "MatMul", 100)] if label == "fp32" else int8_events
        target = tmp_path / f"profile_{label}.json"
        target.write_text(json.dumps(events))
        return target

    monkeypatch.setattr(probe, "profile_artifact", fake_profile)
    return fp32, int8


def test_main_reports_latency_ratio_and_a_verdict_per_artifact(monkeypatch, tmp_path, capsys):
    fp32, int8 = _fake_probe_environment(
        monkeypatch, tmp_path, int8_events=[_ev("dq", "DequantizeLinear", 300), _ev("c", "Conv", 800)]
    )
    out = tmp_path / "probe.json"

    rc = probe.main(["--device-id", "jetson_cpu_50w", "--artifact", f"fp32={fp32}", "--artifact", f"int8={int8}",
                     "--rounds", "2", "--out", str(out)])

    assert rc == 0
    data = json.loads(out.read_text())
    assert data["latency"]["fp32"]["median_ms"] == 10.0
    assert data["latency"]["int8"]["ratio_vs_reference"] == pytest.approx(0.4)
    assert data["profile"]["fp32"]["verdict"] == "FLOAT"
    assert data["profile"]["int8"]["verdict"] == "FALLBACK"
    assert data["profile"]["int8"]["per_op"]["DequantizeLinear"]["category"] == "qdq"
    assert "2.50x faster" in capsys.readouterr().out


def test_main_recognises_real_integer_kernels(monkeypatch, tmp_path):
    fp32, int8 = _fake_probe_environment(
        monkeypatch, tmp_path, int8_events=[_ev("c", "QLinearConv", 800), _ev("m", "MatMulIntegerToFloat", 500)]
    )
    out = tmp_path / "probe.json"

    probe.main(["--device-id", "jetson_cpu_50w", "--artifact", f"fp32={fp32}", "--artifact", f"int8={int8}",
                "--rounds", "1", "--out", str(out)])

    assert json.loads(out.read_text())["profile"]["int8"]["verdict"] == "REAL_INT8"


def test_main_refuses_an_unsupported_backend_and_a_single_artifact(monkeypatch, tmp_path):
    fp32, int8 = _fake_probe_environment(monkeypatch, tmp_path, int8_events=[], backend="tpu")

    with pytest.raises(SystemExit, match="supports cpu and cuda"):
        probe.main(["--device-id", "some_tpu", "--artifact", f"fp32={fp32}", "--artifact", f"int8={int8}"])
    with pytest.raises(SystemExit, match="at least two"):
        probe.main(["--device-id", "jetson_cpu_50w", "--artifact", f"fp32={fp32}"])


def test_main_accepts_a_gpu_device_and_reports_the_provider_split(monkeypatch, tmp_path, capsys):
    cuda = "CUDAExecutionProvider"
    fp32, int8 = _fake_probe_environment(
        monkeypatch, tmp_path,
        int8_events=[_ev("c", "QLinearConv", 700, provider=cuda), _ev("dq", "DequantizeLinear", 100, provider=cuda),
                     _ev("m", "MatMul", 200, provider="CPUExecutionProvider")],
        backend="cuda",
    )
    out = tmp_path / "probe.json"

    assert probe.main(["--device-id", "jetson_gpu_50w", "--artifact", f"fp32={fp32}", "--artifact", f"int8={int8}",
                       "--rounds", "1", "--out", str(out)]) == 0

    shares = json.loads(out.read_text())["profile"]["int8"]["provider_time_share"]
    assert shares[cuda] == pytest.approx(0.8) and shares["CPUExecutionProvider"] == pytest.approx(0.2)
    assert "CUDAExecutionProvider 80%" in capsys.readouterr().out


def test_provider_shares_add_up_to_one():
    per_op = probe.summarize_profile([_ev("a", "Conv", 300, provider="CUDAExecutionProvider"),
                                      _ev("b", "MatMul", 100)])

    assert sum(probe.provider_shares(per_op).values()) == pytest.approx(1.0)


def test_threads_override_reaches_every_run_and_is_recorded(monkeypatch, tmp_path):
    fp32, int8 = _fake_probe_environment(monkeypatch, tmp_path, int8_events=[_ev("c", "QLinearConv", 900)])
    seen_threads, profiled = [], []
    monkeypatch.setattr(probe, "_run_perf_test",
                        lambda path, device, repeats: seen_threads.append(device["cpu_cores"]) or {"avg_latency_ms": 5.0})

    def fake_profile(label, path, device, repeats, workdir):
        profiled.append(device["cpu_cores"])
        target = tmp_path / f"profile_{label}.json"
        target.write_text(json.dumps([_ev("c", "QLinearConv" if label != "fp32" else "Conv", 900)]))
        return target

    monkeypatch.setattr(probe, "profile_artifact", fake_profile)
    out = tmp_path / "probe.json"

    probe.main(["--device-id", "jetson_cpu_50w", "--threads", "1", "--rounds", "2",
                "--artifact", f"fp32={fp32}", "--artifact", f"int8={int8}", "--out", str(out)])

    assert seen_threads == [1, 1, 1, 1] and profiled == [1, 1]  # 2 rounds x 2 artifacts, then 2 profiles
    assert json.loads(out.read_text())["threads"] == 1


def test_threads_must_be_positive(monkeypatch, tmp_path):
    fp32, int8 = _fake_probe_environment(monkeypatch, tmp_path, int8_events=[])

    with pytest.raises(SystemExit, match="--threads"):
        probe.main(["--device-id", "jetson_cpu_50w", "--threads", "0",
                    "--artifact", f"fp32={fp32}", "--artifact", f"int8={int8}"])


def test_probe_runs_the_exact_command_run_batch_runs_for_the_same_thread_count(monkeypatch, tmp_path):
    """--threads N must give the SAME onnxruntime_perf_test flags (-x N -y 1 and all
    the rest) as run_batch/runner_vit for a device whose cpu_cores is N: the two
    measurements are comparable except for -r (repeats) and the profile flag."""
    from benchmarking_ondevice.runner_vit import runner as vit_runner

    model = tmp_path / "m.onnx"
    model.write_text("x")
    device = {"device_id": "d", "backend": "cpu", "cpu_cores": 4}
    seen = []
    monkeypatch.setattr(probe.subprocess, "run",
                        lambda cmd, **kw: seen.append(list(cmd)) or types.SimpleNamespace(
                            stdout="Average inference time cost: 5.0 ms\nNumber of inferences per second: 200\n"
                                   "Total inference run: 1000\n", stderr=""))
    try:
        probe._run_perf_test(model, {**device, "cpu_cores": 4}, repeats=1000)
    except Exception:  # noqa: BLE001 -- only the command matters here, not parsing the fake output
        pass

    runner_cmd = vit_runner.build_benchmark_command(model, device, repeats=1000, output_prefix="OUT")
    assert seen and seen[0][:-1] == runner_cmd[:-1]  # everything but the throw-away output prefix
    assert seen[0][seen[0].index("-x") + 1] == "4" and seen[0][seen[0].index("-y") + 1] == "1"
