"""Sufficiency/necessity of percentile in the two MobileViT-S layers
(analysis/phase4_vit/calibration_attribution.py): S per variant, expected
counts, the criterion, the driver (control of rep1 first, exact control with
0 per-image differences, resumption), the seed per repeat, and -- with
Brevitas -- that S = none reproduces the baseline quantizers and S = all the
percentile ones."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from analysis.phase4_vit import calibration_attribution as ca
from analysis.phase4_vit import level2 as l2


ALL = [f"l{i}" for i in range(72)]
TWO = ["l3", "l9"]


def test_constants():
    assert ca.MODEL == "mobilevit" and ca.REPEATS == (1, 2, 3) and ca.N_LAYERS == 72
    assert (ca.SUFFICIENT_MIN, ca.NECESSARY_MAX) == (0.75, 0.25)
    assert ca.OUT_DIR == l2.OUT_DIR / "calibration_attribution"
    assert ca.PERCENTILE_RUN_ID == "vit_mobilevit_phase2_calibration_algorithm_percentile_int8"


@pytest.mark.parametrize("variant, n_perc", [("control", 72), ("sufficiency", 2), ("necessity", 70)])
def test_percentile_sets_and_expected_counts(variant, n_perc):
    s = ca.percentile_set(ALL, TWO, variant)
    assert len(s) == n_perc and ca.expected_counts(variant) == (n_perc, 72 - n_perc)
    assert (set(TWO) <= set(s)) == (variant != "necessity")
    assert s == [n for n in ALL if n in s]  # model order kept


def test_config_problems():
    configs = {n: ("percentile" if n in TWO else "minmax") for n in ALL}
    assert ca.config_problems(configs, TWO, "sufficiency") == []
    assert ca.config_problems(configs, TWO, "necessity")
    configs["l3"] = "minmax"
    assert any("wrong input quantizer" in p for p in ca.config_problems(configs, TWO, "sufficiency"))


@pytest.mark.parametrize("suf, nec, expected", [
    ((0.80, 0.75, 0.90), (0.10, 0.25, 0.0), (True, True)),
    ((0.80, 0.74, 0.90), (0.10, 0.20, 0.0), (False, True)),
    ((0.80, 0.80, 0.90), (0.10, 0.26, 0.0), (True, False)),
    ((0.10, 0.10, 0.10), (0.90, 0.90, 0.9), (False, False)),
])
def test_verdicts_need_all_three_repeats(suf, nec, expected):
    summaries = [{"rec_perc": 1.0, "rec_suf": s, "rec_nec": n} for s, n in zip(suf, nec)]
    v = ca.verdicts(summaries)
    assert (v["sufficient"], v["necessary"]) == expected and v["reading"]


def test_run_variant_builds_with_the_repeat_seed(tmp_path, monkeypatch):
    seen = {}

    def fake_build(two, variant, repeat_id):
        seen.update(repeat_id=repeat_id, variant=variant, two=list(two))
        return object(), (1, 3, 8, 8), {n: ("percentile" if n in two else "minmax") for n in ALL}

    import export.vit_to_onnx.export as ex

    def fake_export(model, shape, run_id, repeat_id, out_path):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"onnx")
        return out_path

    monkeypatch.setattr(ex, "export_to_onnx", fake_export)
    monkeypatch.setattr(l2, "evaluate_onnx", lambda m, p: (50.0, 70.0, []))
    row = ca.run_variant(3, "sufficiency", TWO, tmp_path, build=fake_build)
    assert seen == {"repeat_id": 3, "variant": "sufficiency", "two": TWO}
    assert row["calib_seed"] == 44 and (row["n_percentile"], row["n_minmax"]) == (2, 70)
    assert (tmp_path / "layer_configs" / "mobilevit__rep3__sufficiency.csv").exists()
    assert not (tmp_path / "onnx" / "mobilevit__rep3__sufficiency.onnx").exists() and len(row["onnx_sha256"]) == 64


# -- driver with a fake evaluation ---------------------------------------------

BASE = {1: 2.95, 2: 1.55, 3: 1.15}
PERC = {1: (68.9, 89.05), 2: (68.5, 88.85), 3: (67.75, 87.8)}


@pytest.fixture
def dirs(tmp_path):
    level1 = tmp_path / "level1"
    level1.mkdir()
    with open(level1 / "layers_mobilevit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "exec_idx", "rank", "local_error_mean", "ranked"])
        w.writeheader()
        for i, n in enumerate(["a", "b", "c"]):
            w.writerow({"name": n, "exec_idx": i, "rank": i + 1, "local_error_mean": 0.1, "ranked": "True"})
    rows = [{"model": "mobilevit", "bitwidth": "int8", "curve": "k0", "K": 0, "float_layers": "", "top1": 2.95, "top5": 7.0},
            {"model": "mobilevit", "bitwidth": "int8", "curve": "targeted_r", "K": 2, "float_layers": "a;b",
             "top1": 74.7, "top5": 90.0}]
    l2._write_csv(tmp_path / "level2" / "evaluations.csv", rows, l2.EVAL_COLUMNS)
    per_image = tmp_path / "per_image"
    for k in ca.REPEATS:
        l2._write_csv(per_image / f"{ca.PERCENTILE_RUN_ID}__rep{k}.csv", [{"idx": 0, "pred1": 5}, {"idx": 1, "pred1": 7}],
                      ["idx", "pred1"])
    return tmp_path


class FakeEval:
    def __init__(self, top1=None, control_pred=(5, 7)):
        self.top1 = top1 or {"sufficiency": 60.0, "necessity": 5.0}
        self.control_pred, self.calls = control_pred, []

    def __call__(self, repeat, variant, two, out_dir):
        self.calls.append((repeat, variant, tuple(two)))
        if variant == "control":
            top1, top5 = PERC[repeat]
            l2._write_csv(out_dir / "predictions" / f"mobilevit__rep{repeat}__control.csv",
                          [{"idx": 0, "pred1": self.control_pred[0]}, {"idx": 1, "pred1": self.control_pred[1]}],
                          ["idx", "pred1"])
        else:
            top1, top5 = self.top1[variant], 0.0
        n_perc = ca.expected_counts(variant)[0]
        return {"model": "mobilevit", "repeat": repeat, "calib_seed": 41 + repeat, "variant": variant,
                "n_percentile": n_perc, "n_minmax": 72 - n_perc, "two_layers": ";".join(two), "top1": top1,
                "top5": top5, "onnx_sha256": "x", "seconds": 0}


def _recorded(run_id, repeat_id):
    if run_id == ca.BASELINE_RUN_ID:
        return {"top1": BASE[repeat_id], "top5": 1.0}
    return {"top1": PERC[repeat_id][0], "top5": PERC[repeat_id][1]}


def _run(d, fake, control_rep1_only=False, recorded=_recorded):
    return ca.run(d / "out", False, control_rep1_only, evaluate=fake, recorded=recorded, fp32=lambda m: 79.35,
                  level2_dir=d / "level2", level1_dir=d / "level1", per_image_dir=d / "per_image")


def test_control_rep1_only_evaluates_just_that(dirs):
    fake = FakeEval()
    assert _run(dirs, fake, control_rep1_only=True) == 0
    assert fake.calls == [(1, "control", ("a", "b"))]


def test_full_run_controls_first_then_nine_evaluations_and_resume(dirs):
    _run(dirs, FakeEval(), control_rep1_only=True)
    fake = FakeEval()
    assert _run(dirs, fake) == 0
    assert [c[:2] for c in fake.calls[:2]] == [(2, "control"), (3, "control")]
    assert len(fake.calls) == 8 and all(c[2] == ("a", "b") for c in fake.calls)
    assert len(list(csv.DictReader(open(dirs / "out" / "evaluations.csv")))) == 9
    again = FakeEval()
    assert _run(dirs, again) == 0 and again.calls == []


def test_each_repeat_is_evaluated_with_its_own_repeat(dirs):
    fake = FakeEval()
    _run(dirs, fake)
    assert {(c[0], c[1]) for c in fake.calls} == {(k, v) for k in ca.REPEATS for v in ca.VARIANTS}


def test_control_with_one_per_image_difference_stops(dirs):
    fake = FakeEval(control_pred=(5, 8))
    assert _run(dirs, fake) == 1 and fake.calls == [(1, "control", ("a", "b"))]
    assert "MISMATCH" in (dirs / "out" / "controls.csv").read_text()


def test_control_with_other_top5_stops_before_other_repeats(dirs):
    def recorded(run_id, repeat_id):
        r = _recorded(run_id, repeat_id)
        return {**r, "top5": r["top5"] + 0.05} if run_id == ca.PERCENTILE_RUN_ID and repeat_id == 2 else r

    fake = FakeEval()
    assert _run(dirs, fake, recorded=recorded) == 1
    assert [c[:2] for c in fake.calls] == [(1, "control"), (2, "control")]


def test_summary_recoveries_and_verdict(dirs):
    _run(dirs, FakeEval())
    rows = list(csv.DictReader(open(dirs / "out" / "summary.csv")))
    r1 = rows[0]
    assert float(r1["rec_perc"]) == pytest.approx((68.9 - 2.95) / (79.35 - 2.95))
    assert float(r1["rec_suf"]) == pytest.approx((60.0 - 2.95) / (79.35 - 2.95))
    assert float(r1["nec_over_perc"]) == pytest.approx((5.0 - 2.95) / (68.9 - 2.95))
    assert {r["sufficient"] for r in rows} == {"True"} and {r["necessary"] for r in rows} == {"True"}


# -- Brevitas: per-layer input quantizer -------------------------------------

def _tiny(torch):
    return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.Conv2d(8, 8, 3, groups=8),
                               torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 4)))


@pytest.mark.parametrize("s_names, expected", [
    (set(), {"0": "minmax", "1": "minmax", "2.1": "minmax"}),            # S = none -> baseline
    ({"0", "1", "2.1"}, {"0": "percentile", "1": "percentile", "2.1": "percentile"}),  # S = all -> percentile
    ({"1"}, {"0": "minmax", "1": "percentile", "2.1": "minmax"}),
])
def test_mixed_tree_sets_the_input_quantizer_per_layer(s_names, expected):
    torch = pytest.importorskip("torch")
    pytest.importorskip("brevitas")
    wq, base_act, perc_act = ca.quantizers()
    model = _tiny(torch)
    ca.quantize_tree_mixed(model, wq, base_act, perc_act, s_names)
    assert ca.layer_configs(model, base_act, perc_act) == expected
    assert all(issubclass(m.weight_quant.quant_injector, wq) for m in (model[0], model[1], model[2][1]))


def test_none_and_all_match_the_ablation_quantizers():
    pytest.importorskip("brevitas")
    from quantization.vit_brevitas.quantizer_config import resolve_act_quant, resolve_weight_quant

    wq, base_act, perc_act = ca.quantizers()
    assert wq is resolve_weight_quant("percentile", "per_tensor", "int8")
    assert base_act.__name__ == resolve_act_quant("minmax", "symmetric", "int8").__name__
    assert perc_act.__name__ == resolve_act_quant("percentile", "symmetric", "int8").__name__
    assert perc_act.high_percentile_q == 99.9
