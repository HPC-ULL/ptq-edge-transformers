"""output_quant in the ViT pipeline: off by default (the ablation's
input-only regime), on only when passed explicitly -- each Conv/Linear then
quantizes its OUTPUT with the same activation quantizer as its input -- and
the accuracy gate that always builds with it on.

torch/brevitas are not installed in this sandbox: fake modules stand in for
them, so these tests check what the pipeline PASSES to Brevitas, not Brevitas
itself.
"""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantization.vit_brevitas import quantize
from quantization.vit_brevitas.investigations import gate_output_quant as gate


class _Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.weight = SimpleNamespace(data=None)
        self.bias = SimpleNamespace(data=None)


def _fake_brevitas(monkeypatch):
    nn = types.ModuleType("brevitas.nn")
    nn.QuantLinear = type("QuantLinear", (_Recorder,), {})
    nn.QuantConv2d = type("QuantConv2d", (_Recorder,), {})
    pkg = types.ModuleType("brevitas")
    pkg.nn = nn
    monkeypatch.setitem(sys.modules, "brevitas", pkg)
    monkeypatch.setitem(sys.modules, "brevitas.nn", nn)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))


def _tensor():
    return SimpleNamespace(data=SimpleNamespace(clone=lambda: "cloned"))


def test_by_default_layers_quantize_only_their_input(monkeypatch):
    """The ablation's input-only regime is the default: nothing passes
    output_quant, so the pipeline can never silently switch regime."""
    _fake_brevitas(monkeypatch)
    linear = quantize._quantize_linear(SimpleNamespace(in_features=4, out_features=8, bias=None, weight=_tensor()),
                                       "WQ", "AQ")
    conv = quantize._quantize_conv2d(SimpleNamespace(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                                                     padding=(1, 1), dilation=(1, 1), groups=1, bias=None,
                                                     weight=_tensor()), "WQ", "AQ")

    for layer in (linear, conv):
        assert layer.kwargs["input_quant"] == "AQ" and layer.kwargs["output_quant"] is None


def test_output_quant_is_off_by_default_all_the_way_down():
    import inspect

    for fn in (quantize.build_quantized_model, quantize.quantize_module_tree,
               quantize._quantize_linear, quantize._quantize_conv2d):
        assert inspect.signature(fn).parameters["output_quant"].default is False


def test_the_ablation_pipeline_cli_never_turns_output_quant_on():
    import inspect

    from quantization.vit_brevitas import cli

    assert "output_quant" not in inspect.getsource(cli)


def test_linear_quantizes_input_and_output_with_the_same_activation_quantizer(monkeypatch):
    _fake_brevitas(monkeypatch)
    child = SimpleNamespace(in_features=4, out_features=8, bias=_tensor(), weight=_tensor())

    layer = quantize._quantize_linear(child, "WQ", "AQ", output_quant=True)

    assert layer.kwargs["input_quant"] == "AQ" and layer.kwargs["output_quant"] == "AQ"
    assert layer.kwargs["weight_quant"] == "WQ" and layer.kwargs["return_quant_tensor"] is False
    assert "bias_quant" not in layer.kwargs  # float bias: an integer bias changed nothing (probe_output_quant.py)


def test_conv_quantizes_input_and_output_with_the_same_activation_quantizer(monkeypatch):
    _fake_brevitas(monkeypatch)
    child = SimpleNamespace(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                            dilation=(1, 1), groups=1, bias=None, weight=_tensor())

    layer = quantize._quantize_conv2d(child, "WQ", "AQ", output_quant=True)

    assert layer.kwargs["input_quant"] == "AQ" and layer.kwargs["output_quant"] == "AQ"


def test_export_writes_to_the_given_path_instead_of_artifacts(monkeypatch, tmp_path):
    from export.vit_to_onnx import export as export_module

    calls = {}
    fake_export = types.ModuleType("brevitas.export")
    fake_export.export_onnx_qcdq = lambda model, **kw: calls.update(kw)
    fake_torch = types.ModuleType("torch")
    fake_torch.randn = lambda *shape: ("randn", shape)
    monkeypatch.setitem(sys.modules, "brevitas.export", fake_export)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    out = export_module.export_to_onnx("model", (1, 3, 224, 224), "run", 1, out_path=tmp_path / "sub" / "m.onnx")

    assert out == tmp_path / "sub" / "m.onnx" and (tmp_path / "sub").is_dir()
    assert calls["export_path"] == str(out)
    assert calls["opset_version"] == export_module.ONNX_OPSET and calls["keep_initializers_as_inputs"] is False
    # defaults: the pipeline artifact path, and the ablation's export untouched (no single-node GELU)
    assert export_module.export_to_onnx.__defaults__ == (None, False)


# --------------------------------------------------------------------------
# gate_output_quant
# --------------------------------------------------------------------------

def test_gate_selects_the_three_baselines_and_optionally_the_recovery_rows():
    base = [e["run_id"] for e in gate.select_entries(with_recovery=False)]
    full = [e["run_id"] for e in gate.select_entries(with_recovery=True)]

    assert base == [f"vit_{m}_phase1_baseline_default_int8" for m in ("deit_tiny", "mobilevit", "efficientvit_b1")]
    assert full == base + gate.RECOVERY_RUN_IDS
    entries = gate.select_entries(with_recovery=True)
    assert all(e["bitwidth"] == "int8" for e in entries)
    assert {(e["calibration_algorithm"], e["granularity"], e["symmetry"]) for e in entries[:3]} == {
        ("minmax", "per_tensor", "symmetric")}
    assert {e["calibration_algorithm"] for e in entries[3:]} == {"percentile"}


def test_gate_reads_the_previous_regime_values_per_run_and_repeat(tmp_path):
    path = tmp_path / "vit_accuracy_offline.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "repeat_id", "accuracy_metric_name", "accuracy_value"])
        w.writeheader()
        w.writerows([
            {"run_id": "a", "repeat_id": "1", "accuracy_metric_name": "top1_accuracy", "accuracy_value": "72.9"},
            {"run_id": "a", "repeat_id": "1", "accuracy_metric_name": "top5_accuracy", "accuracy_value": "91.0"},
            {"run_id": "a", "repeat_id": "2", "accuracy_metric_name": "top1_accuracy", "accuracy_value": "73.1"},
        ])

    prev = gate.previous_values(path)

    assert prev[("a", 1)] == {"top1_accuracy": 72.9, "top5_accuracy": 91.0}
    assert prev[("a", 2)] == {"top1_accuracy": 73.1}
    assert gate.previous_values(tmp_path / "missing.csv") == {}


def _patch_pipeline(monkeypatch, *, enabled=50, accuracy=(70.0, 90.0)):
    from evaluation_offline.vit_accuracy import accuracy as acc
    from export.vit_to_onnx import export as exp

    exported, built = [], []
    _patch_pipeline.built = built
    def fake_build(entry, repeat_id, fold_bn=False, remove_bn=False, output_quant=False):
        assert output_quant, "the gate must always build with output_quant on"
        built.append((fold_bn, remove_bn) if remove_bn else fold_bn)
        return "model", (1, 3, 224, 224)

    monkeypatch.setattr(quantize, "build_quantized_model", fake_build)
    monkeypatch.setattr(gate, "count_output_quant", lambda model: (enabled, 50))
    monkeypatch.setattr(gate, "outputs_to_q", lambda path: (2, 2))

    def fake_export(model, shape, run_id, repeat_id, out_path=None, single_gelu=False):
        exported.append(out_path)
        return out_path

    monkeypatch.setattr(exp, "export_to_onnx", fake_export)
    monkeypatch.setattr(acc, "compute_topk_accuracy_for_file", lambda model_name, path: accuracy)
    return exported


def test_gate_exports_outside_artifacts_and_pairs_new_with_previous(monkeypatch, tmp_path):
    exported = _patch_pipeline(monkeypatch, accuracy=(69.0, 89.0))
    entries = gate.select_entries(with_recovery=False)[:1]
    previous = {(entries[0]["run_id"], 1): {"top1_accuracy": 72.9, "top5_accuracy": 91.0}}

    rows = gate.run_gate(entries, [1], tmp_path, previous)

    assert exported == [tmp_path / "vit_deit_tiny_phase1_baseline_default_int8__rep1__outq.onnx"]
    assert all("artifacts" not in str(p) for p in exported)
    assert rows[0]["top1_prev"] == 72.9 and rows[0]["top1_outq"] == 69.0
    assert "-3.90" in gate.format_table(rows)


def test_gate_refuses_a_quantize_py_without_output_quant(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, enabled=0)

    with pytest.raises(SystemExit, match="NOT the output_quant version"):
        gate.run_gate(gate.select_entries(False)[:1], [1], tmp_path, {})


def test_gate_table_flags_new_and_known_collapses():
    rows = [
        {"run_id": "known", "repeat_id": 1, "top1_outq": 0.2, "top5_outq": 1.0, "top1_prev": 1.88, "top5_prev": 5.0},
        {"run_id": "new", "repeat_id": 1, "top1_outq": 0.3, "top5_outq": 1.0, "top1_prev": 68.38, "top5_prev": 88.0},
        {"run_id": "fine", "repeat_id": 1, "top1_outq": 70.0, "top5_outq": 90.0, "top1_prev": 72.9, "top5_prev": 91.0},
        {"run_id": "noprev", "repeat_id": 1, "top1_outq": 70.0, "top5_outq": 90.0, "top1_prev": None, "top5_prev": None},
    ]

    lines = {line.split()[0]: line for line in gate.format_table(rows).splitlines()[2:]}

    assert "already collapsed before" in lines["known"]
    assert "NEW" in lines["new"]
    assert "collapse" not in lines["fine"]
    assert "n/a" in lines["noprev"]


def test_gate_fold_bn_is_passed_to_the_pipeline_and_named_apart(monkeypatch, tmp_path):
    exported = _patch_pipeline(monkeypatch)
    entries = gate.select_entries(with_recovery=True, models=["mobilevit"])

    gate.run_gate(entries, [1], tmp_path, {}, fold_bn=True)

    assert [e["run_id"] for e in entries] == ["vit_mobilevit_phase1_baseline_default_int8",
                                              "vit_mobilevit_phase2_calibration_algorithm_percentile_int8"]
    assert _patch_pipeline.built == [True, True]
    assert all(p.name.endswith("__outq_bnfold.onnx") for p in exported)


def test_gate_remove_bn_is_passed_to_the_pipeline_and_named_apart(monkeypatch, tmp_path):
    exported = _patch_pipeline(monkeypatch)
    entries = gate.select_entries(with_recovery=False, models=["mobilevit"])

    gate.run_gate(entries, [1], tmp_path, {}, fold_bn=True, remove_bn=True)

    assert _patch_pipeline.built == [(True, True)]
    assert all(p.name.endswith("__outq_bnremoved.onnx") for p in exported)


def test_gate_remove_bn_without_fold_bn_is_refused():
    with pytest.raises(SystemExit):
        gate.main(["--remove-bn"])


def test_pipeline_default_keeps_folded_bns_as_identities():
    import inspect

    from quantization.vit_brevitas import bn_fold

    assert inspect.signature(quantize.build_quantized_model).parameters["remove_bn"].default is False
    assert inspect.signature(bn_fold.fold_batchnorm).parameters["remove_bn"].default is False


def test_pipeline_default_does_not_fold_bn():
    import inspect

    assert inspect.signature(quantize.build_quantized_model).parameters["fold_bn"].default is False


def test_gate_explicit_run_ids_override_the_row_selection():
    rid = "vit_mobilevit_phase2_calibration_algorithm_percentile_int8"

    assert [e["run_id"] for e in gate.select_entries(False, None, [rid])] == [rid]
    with pytest.raises(SystemExit, match="not in the ViT manifest"):
        gate.select_entries(False, None, ["vit_nope"])


def test_gate_aggregate_mean_and_sample_std_per_run():
    rows = [
        {"run_id": "a", "repeat_id": r, "top1_outq": v, "top5_outq": 70.0, "top1_prev": 68.0 + r, "top5_prev": None}
        for r, v in ((1, 40.0), (2, 42.0), (3, 44.0))
    ] + [{"run_id": "b", "repeat_id": 1, "top1_outq": 49.0, "top5_outq": 73.5, "top1_prev": None, "top5_prev": None}]

    agg = {a["run_id"]: a for a in gate.aggregate(rows)}

    assert agg["a"]["n"] == 3 and agg["a"]["repeats"] == [1, 2, 3]
    assert agg["a"]["top1_outq_mean"] == pytest.approx(42.0) and agg["a"]["top1_outq_std"] == pytest.approx(2.0)
    assert agg["a"]["top1_prev_mean"] == pytest.approx(70.0)
    assert agg["a"]["top5_prev_mean"] is None
    assert agg["b"]["top1_outq_std"] is None  # a single repeat has no std
    assert "42.00 +/- 2.00" in gate.format_aggregate(gate.aggregate(rows))


def _fake_phase3(monkeypatch):
    from configs import vit_phase3_manifest

    entries = [
        {"run_id": f"vit_{m}_phase3_{k}_{b}", "model_name": m, "bitwidth": b, "factor_level": k,
         "calibration_algorithm": "minmax", "granularity": "per_tensor", "symmetry": "symmetric", "calib_set_size": 128}
        for m in ("deit_tiny", "mobilevit", "efficientvit_b1") for k in ("baseline", "combined") for b in ("int6", "int4")
    ]
    monkeypatch.setattr(vit_phase3_manifest, "generate", lambda: (entries, {}))


def test_gate_phase3_selects_baseline_and_combined_int6_int4_per_model(monkeypatch):
    _fake_phase3(monkeypatch)

    rows = gate.select_entries(False, ["deit_tiny", "mobilevit"], phase3=True)

    assert len(rows) == 8 and {r["model_name"] for r in rows} == {"deit_tiny", "mobilevit"}
    assert {(r["factor_level"], r["bitwidth"]) for r in rows} == {
        (k, b) for k in ("baseline", "combined") for b in ("int6", "int4")}
    assert [r["run_id"] for r in gate.select_entries(False, None, ["vit_mobilevit_phase3_combined_int4"], phase3=True)] \
        == ["vit_mobilevit_phase3_combined_int4"]
    with pytest.raises(SystemExit, match="Phase 3 manifest"):
        gate.select_entries(False, None, ["vit_mobilevit_phase1_baseline_default_int8"], phase3=True)


def test_gate_skip_eval_exports_without_scoring(monkeypatch, tmp_path):
    from evaluation_offline.vit_accuracy import accuracy as acc

    exported = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(acc, "compute_topk_accuracy_for_file",
                        lambda *a: (_ for _ in ()).throw(AssertionError("must not evaluate")))

    rows = gate.run_gate(gate.select_entries(False)[:1], [1], tmp_path, {}, evaluate=False)

    assert len(exported) == 1 and rows[0]["top1_outq"] is None
    assert "not evaluated" in gate.format_table(rows)
