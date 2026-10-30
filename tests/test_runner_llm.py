"""Tests for benchmarking_ondevice/runner_llm/.

Mirrors tests/test_runner_vit.py's isolation pattern throughout (devices.yaml
and results paths always monkeypatched to tmp_path, never touching the real
project files). The real JSON fixture below is the exact output confirmed
against the real llama-bench binary (build 10516) run on the cluster, minus
irrelevant fields trimmed for brevity -- n_prompt/n_gen/avg_ts/samples_ts
(the fields the parser actually reads) are verbatim.
"""

from __future__ import annotations

import csv
import json
import subprocess
from unittest.mock import Mock

import pytest
import yaml

from benchmarking_ondevice.runner_llm import devices as devices_module
from benchmarking_ondevice.runner_llm import results_io as results_io_module
from benchmarking_ondevice.runner_llm import runner, self_test as self_test_module
from benchmarking_ondevice.runner_llm.output_parser import parse_llama_bench_output

# The real llama-bench -o json output confirmed on the cluster (build
# 10516, qwen2_5_1_5b q4km rep3), trimmed to the fields that matter.
_PREFILL_OBJ = {
    "n_prompt": 512, "n_gen": 0,
    "avg_ns": 761691723, "avg_ts": 673.610158,
    "samples_ns": [791839514, 787678574, 722027304, 790312404, 716600819],
    "samples_ts": [646.596, 650.011, 709.114, 647.845, 714.484],
}
_DECODE_OBJ = {
    "n_prompt": 0, "n_gen": 128,
    "avg_ns": 5566494483, "avg_ts": 23.003802,
    "samples_ns": [5694388203, 5474299100, 5480503983, 5473045337, 5710235795],
    "samples_ts": [22.4783, 23.382, 23.3555, 23.3873, 22.4159],
}
_REAL_OUTPUT = json.dumps([_PREFILL_OBJ, _DECODE_OBJ])


def _fake_completed_process(stdout: str, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


def _write_devices_yaml(path, devices) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(devices, f, sort_keys=False)


def _isolate_results(tmp_path, monkeypatch):
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
    "n_gpu_layers": 33,
}
_CUDA_DEVICE_NO_LAYERS = {
    "device_id": "test_cuda_no_layers_device",
    "backend": "cuda",
    "cpu_cores": 12,
    "n_gpu_layers": None,
}
_NO_CORES_DEVICE = {
    "device_id": "test_no_cores_device",
    "backend": "cpu",
    "cpu_cores": None,
}


# --------------------------------------------------------------------------
# output_parser.parse_llama_bench_output
# --------------------------------------------------------------------------

def test_parse_real_output_extracts_prefill_and_decode():
    parsed = parse_llama_bench_output(_REAL_OUTPUT)
    assert parsed["throughput_prefill_tps"] == pytest.approx(673.610158)
    assert parsed["throughput_decode_tps"] == pytest.approx(23.003802)
    assert parsed["num_inferences_measured"] == 5


def test_parse_real_output_extracts_n_prompt_and_n_gen_from_the_right_objects():
    """n_prompt comes from the prefill object, n_gen from the decode
    object -- needed by power_logging's perf_per_watt (total tokens
    processed over the whole invocation), read from the real output, never
    assumed to equal runner.py's PROMPT_TOKENS/GEN_TOKENS constants."""
    parsed = parse_llama_bench_output(_REAL_OUTPUT)
    assert parsed["n_prompt"] == _PREFILL_OBJ["n_prompt"]
    assert parsed["n_gen"] == _DECODE_OBJ["n_gen"]


def test_prefill_and_decode_identified_by_fields_not_array_order():
    """Swap the array order -- decode first, prefill second -- and confirm
    the parser still assigns each value to the right column."""
    swapped = json.dumps([_DECODE_OBJ, _PREFILL_OBJ])
    parsed = parse_llama_bench_output(swapped)
    assert parsed["throughput_prefill_tps"] == pytest.approx(673.610158)
    assert parsed["throughput_decode_tps"] == pytest.approx(23.003802)


def test_num_inferences_measured_is_read_from_output_not_assumed():
    obj = dict(_PREFILL_OBJ, samples_ts=[1.0, 2.0, 3.0])
    decode = dict(_DECODE_OBJ, samples_ts=[1.0, 2.0, 3.0])
    parsed = parse_llama_bench_output(json.dumps([obj, decode]))
    assert parsed["num_inferences_measured"] == 3


def test_missing_decode_object_raises_explicit_error():
    with pytest.raises(ValueError, match="1 decode"):
        parse_llama_bench_output(json.dumps([_PREFILL_OBJ]))


def test_missing_prefill_object_raises_explicit_error():
    with pytest.raises(ValueError, match="1 prefill"):
        parse_llama_bench_output(json.dumps([_DECODE_OBJ]))


def test_duplicate_prefill_objects_raises_explicit_error():
    with pytest.raises(ValueError, match="1 prefill"):
        parse_llama_bench_output(json.dumps([_PREFILL_OBJ, dict(_PREFILL_OBJ), _DECODE_OBJ]))


def test_mismatched_sample_counts_between_prefill_and_decode_raises():
    decode = dict(_DECODE_OBJ, samples_ts=[1.0, 2.0])  # 2 vs prefill's 5
    with pytest.raises(ValueError, match="different repetition counts"):
        parse_llama_bench_output(json.dumps([_PREFILL_OBJ, decode]))


def test_non_json_output_raises_explicit_error():
    with pytest.raises(ValueError):
        parse_llama_bench_output("not json at all")


def test_json_object_instead_of_array_raises_explicit_error():
    with pytest.raises(ValueError, match="array"):
        parse_llama_bench_output(json.dumps({"n_prompt": 1}))


# --------------------------------------------------------------------------
# devices.py resolution
# --------------------------------------------------------------------------

def test_resolve_threads_from_cpu_cores(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)

    device = devices_module.resolve_device("test_cpu_device")
    assert devices_module.resolve_threads(device) == 8


def test_cpu_backend_resolves_n_gpu_layers_to_zero(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)

    device = devices_module.resolve_device("test_cpu_device")
    assert devices_module.resolve_n_gpu_layers(device) == 0


def test_cuda_backend_resolves_n_gpu_layers_from_devices_yaml(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CUDA_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)

    device = devices_module.resolve_device("test_cuda_device")
    assert devices_module.resolve_threads(device) == 12
    assert devices_module.resolve_n_gpu_layers(device) == 33


def test_cuda_backend_without_n_gpu_layers_raises_instead_of_default(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CUDA_DEVICE_NO_LAYERS])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)

    device = devices_module.resolve_device("test_cuda_no_layers_device")
    with pytest.raises(ValueError, match="n_gpu_layers"):
        devices_module.resolve_n_gpu_layers(device)


def test_unset_cpu_cores_raises_instead_of_guessing(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_NO_CORES_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)

    device = devices_module.resolve_device("test_no_cores_device")
    with pytest.raises(ValueError, match="cpu_cores"):
        devices_module.resolve_threads(device)


# --------------------------------------------------------------------------
# runner.build_benchmark_command
# --------------------------------------------------------------------------

def test_build_command_fixes_prompt_and_gen_tokens_not_llama_bench_defaults():
    cmd = runner.build_benchmark_command("model.gguf", _CPU_DEVICE)
    assert cmd[cmd.index("-p") + 1] == "128"
    assert cmd[cmd.index("-n") + 1] == "32"
    # llama-bench's own defaults are 512/128 -- confirm we never emit those.
    assert "512" not in cmd


def test_build_command_uses_resolved_threads_and_ngl_not_hardcoded():
    cmd = runner.build_benchmark_command("model.gguf", _CPU_DEVICE, repeats=9)
    assert cmd[cmd.index("-t") + 1] == "8"
    assert cmd[cmd.index("-ngl") + 1] == "0"
    assert cmd[cmd.index("-r") + 1] == "9"
    assert cmd[cmd.index("-o") + 1] == "json"
    assert cmd[cmd.index("-m") + 1] == "model.gguf"


def test_build_command_cuda_device_resolves_its_own_ngl():
    cmd = runner.build_benchmark_command("model.gguf", _CUDA_DEVICE)
    assert cmd[cmd.index("-t") + 1] == "12"
    assert cmd[cmd.index("-ngl") + 1] == "33"


# --------------------------------------------------------------------------
# runner.run_benchmark / results writing
# --------------------------------------------------------------------------

def test_dry_run_never_invokes_subprocess_and_does_not_write(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    run_fn = Mock()
    result = runner.run_benchmark("some_run_id", "model.gguf", "test_cpu_device", 1, dry_run=True, run_fn=run_fn)

    assert result is None
    run_fn.assert_not_called()
    assert not (tmp_path / "results_test_cpu_device.csv").exists()


def test_real_run_writes_row_with_correct_schema(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    run_fn = Mock(return_value=_fake_completed_process(_REAL_OUTPUT))
    runner.run_benchmark("my_run_id", "model.gguf", "test_cpu_device", 2, repeats=5, run_fn=run_fn)

    out_path = tmp_path / "results_test_cpu_device.csv"
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["result_id"] == "my_run_id__test_cpu_device__rep2"
    assert row["device"] == "test_cpu_device"
    assert row["repeat_id"] == "2"
    assert row["throughput_prefill_tps"] == "673.610158"
    assert row["throughput_decode_tps"] == "23.003802"
    assert row["num_inferences_measured"] == "5"
    assert row["requested_repeats"] == "5"
    # ViT-only fields stay blank
    assert row["latency_ms"] == ""
    assert row["throughput"] == ""


# --------------------------------------------------------------------------
# repeat_id: Optional[int] -- deterministic LLM configs have no calibration
# repeat at all, mirroring evaluation_offline/llm_perplexity/results_io.py's
# exact pattern (no "__rep" suffix, blank repeat_id column when None).
# --------------------------------------------------------------------------

def test_result_id_has_no_rep_suffix_when_repeat_id_is_none():
    assert results_io_module.result_id("my_run_id", "test_cpu_device", None) == "my_run_id__test_cpu_device"
    assert results_io_module.result_id("my_run_id", "test_cpu_device", 2) == "my_run_id__test_cpu_device__rep2"


def test_build_result_row_leaves_repeat_id_blank_when_none():
    parsed = {"throughput_prefill_tps": 1.0, "throughput_decode_tps": 2.0, "num_inferences_measured": 3}
    row = results_io_module.build_result_row("my_run_id", "test_cpu_device", None, parsed, requested_repeats=5)
    assert row["repeat_id"] == ""
    assert row["result_id"] == "my_run_id__test_cpu_device"


def test_real_run_with_none_repeat_id_writes_deterministic_row(tmp_path, monkeypatch):
    """A deterministic LLM artifact (baseline, legacy_corpus_none, some
    Phase 3 combined rows) has no calibration-repeat concept -- confirms
    the full run_benchmark path writes a row with no "__rep" suffix and a
    blank repeat_id column, not a fabricated repeat_id=1."""
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    run_fn = Mock(return_value=_fake_completed_process(_REAL_OUTPUT))
    runner.run_benchmark("baseline_run_id", "model.gguf", "test_cpu_device", None, repeats=5, run_fn=run_fn)

    out_path = tmp_path / "results_test_cpu_device.csv"
    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["result_id"] == "baseline_run_id__test_cpu_device"
    assert row["repeat_id"] == ""
    assert results_io_module.has_result("baseline_run_id", "test_cpu_device", None)


def test_rerun_of_same_result_id_replaces_row_instead_of_duplicating(tmp_path, monkeypatch):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    second_decode = dict(_DECODE_OBJ, avg_ts=99.9999)
    second_output = json.dumps([_PREFILL_OBJ, second_decode])

    run_fn = Mock(side_effect=[_fake_completed_process(_REAL_OUTPUT), _fake_completed_process(second_output)])
    runner.run_benchmark("same_run_id", "model.gguf", "test_cpu_device", 1, run_fn=run_fn)
    runner.run_benchmark("same_run_id", "model.gguf", "test_cpu_device", 1, run_fn=run_fn)

    with open(tmp_path / "results_test_cpu_device.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["throughput_decode_tps"] == "99.9999"


# --------------------------------------------------------------------------
# self_test.self_test
# --------------------------------------------------------------------------

def test_self_test_prints_scope_warning_and_does_not_write(tmp_path, monkeypatch, capsys):
    devices_yaml = tmp_path / "devices.yaml"
    _write_devices_yaml(devices_yaml, [_CPU_DEVICE])
    monkeypatch.setattr(devices_module, "DEVICES_YAML_PATH", devices_yaml)
    _isolate_results(tmp_path, monkeypatch)

    fake_artifact = tmp_path / "fixture.gguf"
    fake_artifact.write_bytes(b"")  # never actually read -- run_fn is faked below
    run_fn = Mock(return_value=_fake_completed_process(_REAL_OUTPUT))

    result = self_test_module.self_test("test_cpu_device", artifact_path=fake_artifact, run_fn=run_fn)

    captured = capsys.readouterr()
    assert self_test_module.SCOPE_WARNING in captured.out
    assert result["num_inferences_measured"] == 5
    assert not (tmp_path / "results_test_cpu_device.csv").exists()


def test_self_test_without_artifact_or_existing_gguf_raises_clear_error(tmp_path, monkeypatch):
    from quantization.llm_llamacpp import paths as llm_paths

    monkeypatch.setattr(llm_paths, "LLM_QUANTIZED_DIR", tmp_path / "empty_llm_dir")

    with pytest.raises(FileNotFoundError, match="self-test fixture"):
        self_test_module.self_test("test_cpu_device")
