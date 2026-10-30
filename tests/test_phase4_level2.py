"""Phase 4 Level 2 (analysis/phase4_vit/level2.py): pre-registered constants,
references, layer orders, recovery/K_final/verdict, and the driver's K=0
gate, smoke mode and resumption -- with the build/export/evaluation replaced
by a fake (no torch needed), plus a torch-only check of the float-layer tree."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from analysis.phase4_vit import PHASE4_ROOT
from analysis.phase4_vit import level2 as l2


def test_pre_registered_constants():
    assert l2.KS == (1, 2, 4, 8) and l2.RANDOM_SEEDS == (1, 2, 3)
    assert (l2.STOP_RECOVERY, l2.MIN_RECOVERY, l2.REPEAT_ID) == (0.90, 0.10, 1)
    assert l2.BITWIDTH == {"mobilevit": "int8", "efficientvit_b1": "int8", "deit_tiny": "int6"}
    assert l2.OUT_DIR == PHASE4_ROOT / "level2"


@pytest.mark.parametrize("model, run_id", [
    ("deit_tiny", "vit_deit_tiny_phase3_baseline_int6"),
    ("mobilevit", "vit_mobilevit_phase1_baseline_default_int8"),
    ("efficientvit_b1", "vit_efficientvit_b1_phase1_baseline_default_int8"),
])
def test_reference_entries_are_the_baseline_manifest_rows(model, run_id):
    e = l2.reference_entry(model)
    assert e["run_id"] == run_id and e["model_name"] == model and e["bitwidth"] == l2.BITWIDTH[model]
    assert (e["calibration_algorithm"], e["granularity"], e["symmetry"], e["calib_set_size"]) == \
        ("minmax", "per_tensor", "symmetric", 128)


def test_recorded_accuracy_reads_rep1_only(tmp_path):
    path = tmp_path / "acc.csv"
    rows = [("r", 1, "top1_accuracy", "2.95"), ("r", 1, "top5_accuracy", "7.0"), ("r", 2, "top1_accuracy", "9.9"),
            ("x", 1, "top1_accuracy", "50")]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "repeat_id", "accuracy_metric_name", "accuracy_value"])
        w.writerows(rows)
    assert l2.recorded_accuracy("r", path) == {"top1": 2.95, "top5": 7.0}
    with pytest.raises(KeyError):
        l2.recorded_accuracy("x", path)


def test_fp32_reference_is_the_stored_q95_top1(tmp_path):
    path = tmp_path / "fp32.csv"
    path.write_text("run_id,accuracy_metric_name,accuracy_value\n"
                    "vit_mobilevit_reference_fp32,top1_accuracy,79.35\n"
                    "vit_mobilevit_reference_fp32,top5_accuracy,92.75\n"
                    "vit_mobilevit_reference_fp32_full50000,top1_accuracy,78.31\n")
    assert l2.fp32_top1("mobilevit", path) == 79.35
    with pytest.raises(KeyError):
        l2.fp32_top1("deit_tiny", path)


def test_official_fp32_csv_gives_the_values_previously_read_from_the_json():
    """Phase 4 results do not change: CSV == the stored_q95 top-1 of the JSON it replaced."""
    import json
    js = l2.INVESTIGATIONS_ROOT / "fp32_jpeg_vs_original" / "fp32_jpeg_vs_original.json"
    if not (js.exists() and l2.FP32_CSV.exists()):
        pytest.skip("fp32 reference files not present")
    models = json.loads(js.read_text())["models"]
    for m in l2.MODELS:
        assert l2.fp32_top1(m) == float(models[m]["stored_q95"]["top1"])
    assert [l2.fp32_top1(m) for m in ("deit_tiny", "mobilevit", "efficientvit_b1")] == [73.5, 79.35, 81.2]


def _layers(n=10):
    # exec order 0..n-1; R rank reverses it; local error ties between 3 and 5
    errs = [0.01 * i for i in range(n)]
    errs[5] = errs[3]
    return [{"name": f"l{i}", "exec_idx": i, "rank": n - i, "local_error_mean": errs[i]} for i in range(n)]


def test_order_by_r_follows_the_level1_rank():
    assert l2.order_by_r(_layers())[:3] == ["l9", "l8", "l7"]


def test_order_by_error_descending_with_ties_by_execution_order():
    order = l2.order_by_error(_layers())
    assert order[:3] == ["l9", "l8", "l7"] and order.index("l3") < order.index("l5")


def test_random_order_is_a_reproducible_shuffle_of_the_execution_order():
    rows = _layers()
    a = l2.order_random(list(reversed(rows)), 1)
    assert a == l2.order_random(rows, 1) and sorted(a) == sorted(r["name"] for r in rows)
    expected = [f"l{i}" for i in range(10)]
    import random
    random.Random(1).shuffle(expected)
    assert a == expected and a != l2.order_random(rows, 2)


def test_curve_orders_give_nested_prefixes():
    orders = l2.curve_orders(_layers())
    assert set(orders) == {"targeted_r", "targeted_err", "random_s1", "random_s2", "random_s3"}
    assert all(len(o) == 10 for o in orders.values())


def test_recovery_and_k_final():
    assert l2.recovery(40.0, 2.0, 78.0) == pytest.approx(0.5)
    assert l2.k_final({1: 0.2}) is None
    assert l2.k_final({1: 0.95}) == 1
    assert l2.k_final({1: 0.2, 2: 0.9}) == 2
    assert l2.k_final({1: 0.1, 2: 0.2, 4: 0.3, 8: 0.4}) == 8


def _randoms(by_k):
    return {c: dict(by_k) for c in l2.RANDOM_CURVES}


def test_verdict_clearly_better_needs_k_final_and_ceil_half():
    rnd = _randoms({1: 10.0, 2: 10.0, 4: 10.0})
    rnd["random_s2"][2] = 30.0  # max at K=2 is 30
    # K_final = 4, n = 3, need 2: beats at 1 and 4 (not 2)
    v = l2.verdict({1: 20.0, 2: 25.0, 4: 80.0}, rnd, 4, 2.0, 80.0)
    assert v["K_beating_random"] == [1, 4] and v["verdict"] == "clearly better than random"
    # beats at 1 and 2 but not at K_final -> not distinguishable
    v = l2.verdict({1: 20.0, 2: 31.0, 4: 10.0}, rnd, 4, 2.0, 80.0)
    assert v["verdict"] == "not distinguishable from random"


def test_verdict_comparison_is_strict():
    v = l2.verdict({1: 10.0}, _randoms({1: 10.0}), 1, 2.0, 80.0)
    assert v["K_beating_random"] == [] and v["verdict"] == "not distinguishable from random"


def test_no_recovery_clause_at_k8():
    rnd = _randoms({1: 3.0, 2: 3.0, 4: 3.0, 8: 9.0})  # 9 - 2 = 7 < 10 % of 78
    v = l2.verdict({1: 4.0, 2: 5.0, 4: 6.0, 8: 9.5}, rnd, 8, 2.0, 80.0)
    assert v["verdict"] == "no recovery with K <= 8"
    rnd["random_s3"][8] = 12.0  # (12 - 2) / 78 > 10 %
    assert l2.verdict({1: 4.0, 2: 5.0, 4: 6.0, 8: 9.5}, rnd, 8, 2.0, 80.0)["verdict"] != "no recovery with K <= 8"


def test_per_image_differences(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("idx,pred1\n0,5\n1,7\n")
    b.write_text("idx,pred1\n0,5\n1,8\n")
    assert l2.per_image_differences(a, b) == 1 and l2.per_image_differences(a, a) == 0
    assert l2.per_image_differences(a, tmp_path / "missing.csv") is None


# -- driver with a fake evaluation ---------------------------------------------

class FakeEval:
    """top1 = K=0 reference, + `gain` per float layer for targeted curves."""

    def __init__(self, k0=2.95, gain=1.0, random_gain=0.5):
        self.k0, self.gain, self.random_gain, self.calls = k0, gain, random_gain, []

    def __call__(self, entry, curve, k, names, out_dir):
        self.calls.append((entry["model_name"], curve, k))
        g = self.random_gain if curve.startswith("random") else self.gain
        return {"model": entry["model_name"], "bitwidth": entry["bitwidth"], "curve": curve, "K": k,
                "float_layers": ";".join(names), "top1": self.k0 + g * k, "top5": 7.0, "onnx_sha256": "x",
                "seconds": 0.0}


@pytest.fixture
def level1_dir(tmp_path):
    d = tmp_path / "level1"
    d.mkdir()
    for m in l2.MODELS:
        with open(d / f"layers_{m}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "exec_idx", "rank", "local_error_mean", "ranked"])
            w.writeheader()
            for r in _layers(20):
                w.writerow({**r, "ranked": "True"})
            w.writerow({"name": "stem", "exec_idx": 99, "rank": "", "local_error_mean": 0.5, "ranked": "False"})
    return d


def _run(tmp_path, level1_dir, fake, models=("mobilevit",), smoke=False, recorded=None):
    out = tmp_path / "out"
    runner = l2.Runner(out, evaluate=fake)
    rc = l2.run(models, out, smoke, runner, recorded=recorded or (lambda run_id: {"top1": 2.95, "top5": 7.0}),
                fp32=lambda m: 79.35, level1_dir=level1_dir)
    return rc, out


def test_k0_mismatch_stops_everything(tmp_path, level1_dir):
    fake = FakeEval()
    rc, out = _run(tmp_path, level1_dir, fake, models=("mobilevit", "deit_tiny"),
                   recorded=lambda run_id: {"top1": 3.0, "top5": 7.0})
    assert rc == 1 and all(c[1] == "k0" for c in fake.calls)
    assert "MISMATCH" in (out / "kzero_checks.csv").read_text() and not (out / "summary.csv").exists()


def test_smoke_then_full_reuses_and_runs_21_evaluations(tmp_path, level1_dir):
    fake = FakeEval(gain=0.1, random_gain=0.05)  # never reaches 90 %: K_final = 8
    rc, _ = _run(tmp_path, level1_dir, fake, smoke=True)
    assert rc == 0 and fake.calls == [("mobilevit", "k0", 0), ("mobilevit", "targeted_r", 1)]
    full = FakeEval(gain=0.1, random_gain=0.05)
    rc, out = _run(tmp_path, level1_dir, full)
    assert rc == 0 and ("mobilevit", "k0", 0) not in full.calls and ("mobilevit", "targeted_r", 1) not in full.calls
    assert len(full.calls) + 2 == 21
    rows = list(csv.DictReader(open(out / "evaluations.csv")))
    assert len(rows) == 21 and len({(r["curve"], r["K"]) for r in rows}) == 21
    summary = {r["curve"]: r for r in csv.DictReader(open(out / "summary.csv"))}
    assert summary["targeted_r"]["role"] == "primary" and summary["targeted_err"]["role"] == "exploratory"
    # recovery stays far below 10 % at K=8 for every curve -> the pre-registered clause wins
    assert summary["targeted_r"]["K_final"] == "8" and summary["targeted_r"]["verdict"] == "no recovery with K <= 8"
    assert summary["targeted_r"]["K_beating_random"] == "[1, 2, 4, 8]"


def test_stop_at_90_percent_limits_every_curve_to_k_final(tmp_path, level1_dir):
    fake = FakeEval(gain=40.0)  # K=2: 2.95 + 80 -> recovery > 90 %
    rc, out = _run(tmp_path, level1_dir, fake)
    assert rc == 0 and max(c[2] for c in fake.calls) == 2
    assert len(fake.calls) == 1 + 2 * 5  # K=0 + {1, 2} x (targeted_r, 3 random, targeted_err)
    first = next(csv.DictReader(open(out / "summary.csv")))
    assert first["K_final"] == "2" and first["verdict"] == "clearly better than random"


def test_targeted_curve_uses_the_r_prefix(tmp_path, level1_dir):
    fake = FakeEval(gain=40.0)
    _run(tmp_path, level1_dir, fake)
    rows = {(r["curve"], r["K"]): r for r in csv.DictReader(open(tmp_path / "out" / "evaluations.csv"))}
    assert rows[("targeted_r", "2")]["float_layers"] == "l19;l18"
    assert "stem" not in "".join(r["float_layers"] for r in rows.values())


def test_reuse_refuses_a_row_with_other_float_layers(tmp_path, level1_dir):
    _run(tmp_path, level1_dir, FakeEval(), smoke=True)
    path = tmp_path / "out" / "evaluations.csv"
    path.write_text(path.read_text().replace("l19", "l0"))
    with pytest.raises(RuntimeError):
        _run(tmp_path, level1_dir, FakeEval(), smoke=True)


def test_main_logs_the_exact_exception(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ValueError("export failed: mixed graph")
    monkeypatch.setattr(l2, "run", boom)
    assert l2.main(["--out-dir", str(tmp_path), "--smoke", "--models", "deit_tiny"]) == 1
    log = (tmp_path / "run.log").read_text()
    assert "ValueError: export failed: mixed graph" in log and "Traceback" in log


def test_quantize_tree_except_leaves_exactly_the_named_layers_in_float():
    torch = pytest.importorskip("torch")
    pytest.importorskip("brevitas")
    from brevitas.nn import QuantConv2d, QuantLinear

    from quantization.vit_brevitas.quantizer_config import resolve_act_quant, resolve_weight_quant

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4)))
    kept = l2.quantize_tree_except(model, resolve_weight_quant("minmax", "per_tensor", "int8"),
                                   resolve_act_quant("minmax", "symmetric", "int8"), {"1.0"})
    assert kept == ["1.0"] and type(model[1][0]) is torch.nn.Linear
    assert isinstance(model[0], QuantConv2d) and isinstance(model[1][1], QuantLinear)
