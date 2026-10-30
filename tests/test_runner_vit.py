"""Tests for benchmarking_ondevice/runner_vit/.

Dry-run and parsing tests never touch a real onnxruntime_perf_test binary
or a real devices.yaml -- devices tests point DEVICES_YAML_PATH at a
tmp_path fixture (same isolation pattern as the other CLI test files'
monkeypatched artifact/result paths). The self-test tests use the
artifact_path override to avoid depending on the `onnx` package being
installed in the test environment (it isn't, on this machine -- see
self_test.py's docstring for why that's also the intended fallback for
constrained edge devices, not just a testing convenience).
"""

from __future__ import annotations

import csv
import subprocess
import textwrap
from unittest.mock import Mock

import pytest
import yaml

from benchmarking_ondevice.runner_vit import devices as devices_module
from benchmarking_ondevice.runner_vit import results_io as results_io_module
from benchmarking_ondevice.runner_vit import runner, self_test as self_test_module
from benchmarking_ondevice.runner_vit.output_parser import parse_perf_test_output

# The exact, real stdout confirmed against onnxruntime_perf_test v1.23.2
# (compiled as described in the README / the ONNX Runtime installation verification).
_REAL_OUTPUT = textwrap.dedent(
    """\
    Session creation time cost: 0.194833 s
    First inference time cost: 70 ms
    Total inference time cost: 0.123647 s
    Total inference requests: 5
    Average inference time cost: 24.7294 ms
    Total inference run time: 0.123668 s
    Number of inferences per second: 40.4307
    Avg CPU usage: 102 %
    Peak working set size: 151752704 bytes
    """
)


def _fake_completed_process(stdout: str, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


def _write_devices_yaml(path, devices) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(devices, f, sort_keys=False)


def _isolate_results(tmp_path, monkeypatch):
    """Points results_io at tmp_path instead of the real results/raw/ --
    every test that runs (not --dry-run/--self-test) a real benchmark must
    call this, or it would write into the actual project results directory."""
    monkeypatch.setattr(results_io_module, "results_csv_path", lambda device_id: tmp_path / f"results_{device_id}.csv")


_CPU_DEVICE = {
    "device_id": "test_cpu_device",
    "backend": "cpu",
    "cpu_cores": 8,
}
_CUDA_DEVICE = {
    "device_id": "test_cuda_device",
    "backend": "cuda",
    "cpu_cores": 12,
}
_NO_CORES_DEVICE = {
    "device_id": "test_no_cores_device",
    "backend": "cpu",
    "cpu_cores": None,
}
_BAD_BACKEND_DEVICE = {
    "device_id": "test_bad_backend_device",
    "backend": "tpu",
    "cpu_cores": 4,
}


# --------------------------------------------------------------------------
# output_parser.parse_perf_test_output
# --------------------------------------------------------------------------

def test_parse_real_output_extracts_required_fields():
    parsed = parse_perf_test_output(_REAL_OUTPUT)
    assert parsed["num_inferences_measured"] == 5
    assert parsed["avg_latency_ms"] == pytest.approx(24.7294)
    assert parsed["throughput_ips"] == pytest.approx(40.4307)


def test_parse_real_output_extracts_optional_fields_too():
    parsed = parse_perf_test_output(_REAL_OUTPUT)
    assert parsed["session_creation_time_s"] == pytest.approx(0.194833)
    assert parsed["peak_working_set_bytes"] == 151752704
    assert parsed["avg_cpu_usage_pct"] == pytest.approx(102)


def test_num_inferences_measured_is_read_from_output_not_assumed():
    """The whole point of this field: never assume it equals -r N. Feed an
    output where 'Total inference requests' differs from any plausible -r
    value and confirm the parsed value is exactly what the text says."""
    output = _REAL_OUTPUT.replace("Total inference requests: 5", "Total inference requests: 3")
    parsed = parse_perf_test_output(output)
    assert parsed["num_inferences_measured"] == 3


@pytest.mark.parametrize("missing_line", [
    "Total inference requests: 5\n",
    "Average inference time cost: 24.7294 ms\n",
    "Number of inferences per second: 40.4307\n",
])
def test_missing_required_field_raises_explicit_error(missing_line):
    broken_output = _REAL_OUTPUT.replace(missing_line, "")
    with pytest.raises(ValueError):
        parse_perf_test_output(broken_output)


def test_missing_optional_field_does_not_raise():
    output = _REAL_OUTPUT.replace("Avg CPU usage: 102 %\n", "")
    parsed = parse_perf_test_output(output)
    assert "avg_cpu_usage_pct" not in parsed
    assert parsed["num_inferences_measured"] == 5


# --------------------------------------------------------------------------
# devices.py resolution
# --------------------------------------------------------------------------

def test_resolve_cpu_device_gives_cpu_execution_provider_and_its_own_cores(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE, _CUDA_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)

    device = devices_module.resolve_device("test_cpu_device")
    assert devices_module.resolve_execution_provider(device) == "cpu"
    assert devices_module.resolve_cpu_cores(device) == 8


def test_resolve_cuda_device_gives_cuda_execution_provider_and_its_own_cores(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE, _CUDA_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)

    device = devices_module.resolve_device("test_cuda_device")
    assert devices_module.resolve_execution_provider(device) == "cuda"
    assert devices_module.resolve_cpu_cores(device) == 12


def test_unknown_device_id_raises_with_known_ids_listed(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)

    with pytest.raises(ValueError, match="test_cpu_device"):
        devices_module.resolve_device("does_not_exist")


def test_unset_cpu_cores_raises_instead_of_guessing(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_NO_CORES_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)

    device = devices_module.resolve_device("test_no_cores_device")
    with pytest.raises(ValueError, match="cpu_cores"):
        devices_module.resolve_cpu_cores(device)


def test_unknown_backend_raises_instead_of_guessing_execution_provider(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_BAD_BACKEND_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)

    device = devices_module.resolve_device("test_bad_backend_device")
    with pytest.raises(ValueError, match="tpu"):
        devices_module.resolve_execution_provider(device)


# --------------------------------------------------------------------------
# runner.build_benchmark_command / run_benchmark
# --------------------------------------------------------------------------

def test_build_command_uses_resolved_x_and_e_not_hardcoded():
    cmd = runner.build_benchmark_command(
        "model.onnx", _CPU_DEVICE, repeats=123, output_prefix="/tmp/out_prefix"
    )
    assert cmd[cmd.index("-x") + 1] == "8"
    assert cmd[cmd.index("-e") + 1] == "cpu"
    assert cmd[cmd.index("-r") + 1] == "123"
    assert cmd[cmd.index("-y") + 1] == "1"
    assert "-I" in cmd
    assert "-z" in cmd
    assert "-D" in cmd
    assert cmd[cmd.index("-o") + 1] == "99"
    assert cmd[cmd.index("-S") + 1] == "42"
    assert cmd[-2] == "model.onnx"
    assert cmd[-1] == "/tmp/out_prefix"


def test_build_command_cuda_device_resolves_cuda_ep():
    cmd = runner.build_benchmark_command(
        "model.onnx", _CUDA_DEVICE, output_prefix="/tmp/out_prefix"
    )
    assert cmd[cmd.index("-e") + 1] == "cuda"
    assert cmd[cmd.index("-x") + 1] == "12"


def test_dry_run_never_invokes_subprocess(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    run_fn = Mock()
    result = runner.run_benchmark(
        "some_run_id", "model.onnx", "test_cpu_device", 1, dry_run=True, run_fn=run_fn
    )
    assert result is None
    run_fn.assert_not_called()


def test_run_benchmark_parses_real_run_fn_output(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    run_fn = Mock(return_value=_fake_completed_process(_REAL_OUTPUT))
    result = runner.run_benchmark("some_run_id", "model.onnx", "test_cpu_device", 1, run_fn=run_fn)

    assert result["num_inferences_measured"] == 5
    assert result["throughput_ips"] == pytest.approx(40.4307)
    run_fn.assert_called_once()


def test_run_benchmark_surfaces_captured_output_on_subprocess_failure(tmp_path, monkeypatch, capsys):
    """Regression test: subprocess.run(check=True) raises CalledProcessError
    before run_benchmark's own parsing code runs -- without explicitly
    printing e.stdout/e.stderr on that path, the binary's actual error
    message (the only useful diagnostic) was silently lost, leaving just
    'exit status N' in the traceback. Caught for real: onnxruntime_perf_test
    exit 255 against a synthetic self-test fixture, no stderr shown at all
    until this fix."""
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    error = subprocess.CalledProcessError(
        returncode=255, cmd=["onnxruntime_perf_test"], output="some stdout", stderr="the real error message"
    )
    run_fn = Mock(side_effect=error)

    with pytest.raises(subprocess.CalledProcessError):
        runner.run_benchmark("some_run_id", "model.onnx", "test_cpu_device", 1, run_fn=run_fn)

    captured = capsys.readouterr()
    assert "exited with status 255" in captured.out
    assert "the real error message" in captured.out
    assert "some stdout" in captured.out


# --------------------------------------------------------------------------
# self_test.self_test
# --------------------------------------------------------------------------

def test_self_test_prints_scope_warning(tmp_path, monkeypatch, capsys):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    fake_artifact = tmp_path / "fixture.onnx"
    fake_artifact.write_bytes(b"")  # never actually read -- run_fn is faked below
    run_fn = Mock(return_value=_fake_completed_process(_REAL_OUTPUT))

    self_test_module.self_test("test_cpu_device", artifact_path=fake_artifact, run_fn=run_fn)

    captured = capsys.readouterr()
    assert self_test_module.SCOPE_WARNING in captured.out
    assert "does NOT confirm the input flags" in captured.out


def test_self_test_with_artifact_override_does_not_require_onnx_package(tmp_path, monkeypatch):
    """Confirms the artifact_path override path never imports the `onnx`
    package -- the fallback for devices where it isn't installed."""
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    fake_artifact = tmp_path / "fixture.onnx"
    fake_artifact.write_bytes(b"")
    run_fn = Mock(return_value=_fake_completed_process(_REAL_OUTPUT))

    result = self_test_module.self_test("test_cpu_device", artifact_path=fake_artifact, run_fn=run_fn)
    assert result["num_inferences_measured"] == 5


def test_self_test_never_writes_to_results(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    fake_artifact = tmp_path / "fixture.onnx"
    fake_artifact.write_bytes(b"")
    run_fn = Mock(return_value=_fake_completed_process(_REAL_OUTPUT))

    self_test_module.self_test("test_cpu_device", artifact_path=fake_artifact, run_fn=run_fn)

    assert not (tmp_path / "results_test_cpu_device.csv").exists()


# --------------------------------------------------------------------------
# results_io: real runs write, dry-run/self-test don't, reruns append
# without clobbering
# --------------------------------------------------------------------------

def test_real_run_writes_row_with_correct_schema_and_requested_repeats(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    run_fn = Mock(return_value=_fake_completed_process(_REAL_OUTPUT))
    runner.run_benchmark("my_run_id", "model.onnx", "test_cpu_device", 2, repeats=777, run_fn=run_fn)

    out_path = tmp_path / "results_test_cpu_device.csv"
    assert out_path.exists()
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["result_id"] == "my_run_id__test_cpu_device__rep2"
    assert row["run_id"] == "my_run_id"
    assert row["device"] == "test_cpu_device"
    assert row["repeat_id"] == "2"
    assert row["latency_ms"] == "24.7294"
    assert row["throughput"] == "40.4307"
    assert row["num_inferences_measured"] == "5"
    assert row["requested_repeats"] == "777"
    # fields this CLI doesn't resolve (no manifest lookup) stay blank
    assert row["model_name"] == ""
    assert row["accuracy_value"] == ""


def test_dry_run_does_not_write_to_results(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    run_fn = Mock()
    runner.run_benchmark("my_run_id", "model.onnx", "test_cpu_device", 1, dry_run=True, run_fn=run_fn)

    assert not (tmp_path / "results_test_cpu_device.csv").exists()


def test_repeated_real_runs_append_without_clobbering_earlier_rows(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    run_fn = Mock(return_value=_fake_completed_process(_REAL_OUTPUT))
    runner.run_benchmark("run_a", "model_a.onnx", "test_cpu_device", 1, run_fn=run_fn)
    runner.run_benchmark("run_b", "model_b.onnx", "test_cpu_device", 1, run_fn=run_fn)

    out_path = tmp_path / "results_test_cpu_device.csv"
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert {r["result_id"] for r in rows} == {"run_a__test_cpu_device__rep1", "run_b__test_cpu_device__rep1"}


def test_rerun_of_same_result_id_replaces_row_instead_of_duplicating(tmp_path, monkeypatch):
    """Same fix already applied to the other two results_io.py modules
    (evaluation_offline/vit_accuracy, evaluation_offline/llm_perplexity)
    after a real duplication bug there -- a rerun against the exact same
    (run_id, device_id, repeat_id) must overwrite, not duplicate."""
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    first_output = _REAL_OUTPUT
    second_output = _REAL_OUTPUT.replace("Average inference time cost: 24.7294 ms", "Average inference time cost: 99.9999 ms")

    run_fn = Mock(side_effect=[_fake_completed_process(first_output), _fake_completed_process(second_output)])
    runner.run_benchmark("same_run_id", "model.onnx", "test_cpu_device", 1, run_fn=run_fn)
    runner.run_benchmark("same_run_id", "model.onnx", "test_cpu_device", 1, run_fn=run_fn)

    out_path = tmp_path / "results_test_cpu_device.csv"
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["latency_ms"] == "99.9999"
