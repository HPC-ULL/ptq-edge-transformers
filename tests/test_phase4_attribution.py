"""Phase 4 weights-or-activations attribution (analysis/phase4_vit/attribution.py):
the protocol's criterion, the layer check against Level 2 and the R ranking,
the quantizer-state check, the driver (fake evaluation), and -- with Brevitas --
that each variant turns off exactly one quantizer on exactly the named layers."""

from __future__ import annotations

import csv

import pytest

from analysis.phase4_vit import attribution as at
from analysis.phase4_vit import level2 as l2


def test_constants():
    assert at.TARGET_K == {"mobilevit": 2, "efficientvit_b1": 8}
    assert at.VARIANTS == ("weights_float", "input_float") and (at.HIGH, at.LOW) == (0.75, 0.25)
    assert at.OUT_DIR == l2.OUT_DIR / "attribution"


@pytest.mark.parametrize("rec_a, rec_b, expected", [
    (0.10, 0.80, "activations"),
    (0.25, 0.75, "activations"),        # both bounds inclusive
    (0.80, 0.10, "weights"),
    (0.75, 0.25, "weights"),
    (0.26, 0.80, "mixed or interaction"),
    (0.10, 0.74, "mixed or interaction"),
    (0.50, 0.50, "mixed or interaction"),
    (0.90, 0.90, "mixed or interaction"),
])
def test_classify_follows_the_protocol(rec_a, rec_b, expected):
    assert at.classify(rec_a, rec_b, 1.0) == expected


def test_classify_scales_with_rec_full():
    # Rec_full = 0.496: activations needs Rec_b >= 0.372 and Rec_a <= 0.124
    assert at.classify(0.12, 0.38, 0.496) == "activations"
    assert at.classify(0.13, 0.38, 0.496) == "mixed or interaction"


def _level2_rows(layers_by_model):
    rows = {}
    for m, (layers, k0, full) in layers_by_model.items():
        rows[(m, l2.K0, 0)] = {"top1": k0, "float_layers": ""}
        rows[(m, l2.TARGETED_R, at.TARGET_K[m])] = {"top1": full, "float_layers": ";".join(layers)}
    return rows


def test_target_layers_must_match_the_r_ranking_prefix():
    rows = _level2_rows({"mobilevit": (["a", "b"], 2.95, 74.7)})
    assert at.target_layers("mobilevit", rows, ["a", "b", "c"]) == ["a", "b"]
    with pytest.raises(RuntimeError):
        at.target_layers("mobilevit", rows, ["b", "a", "c"])
    with pytest.raises(KeyError):
        at.target_layers("efficientvit_b1", rows, ["a"])


def test_references_and_summary_reproduce_the_level2_recoveries():
    rows = _level2_rows({"mobilevit": (["a", "b"], 2.95, 74.7), "efficientvit_b1": ([str(i) for i in range(8)], 0.2, 40.35)})
    r = at.references("mobilevit", rows, 79.35)
    assert r["rec_full"] == pytest.approx(0.939, abs=5e-4)
    assert at.references("efficientvit_b1", rows, 81.2)["rec_full"] == pytest.approx(0.496, abs=5e-4)
    s = at.summarize("mobilevit", r, top1_a=2.95 + 0.05 * 76.4, top1_b=2.95 + 0.8 * 76.4)
    assert s["rec_a"] == pytest.approx(0.05) and s["rec_b"] == pytest.approx(0.8)
    assert s["rec_b_over_full"] == pytest.approx(0.8 / r["rec_full"]) and s["classification"] == "activations"


def test_state_problems_requires_exactly_one_quantizer_off_on_the_named_layers():
    ok_a = {"x": (False, True), "y": (True, True)}
    assert at.state_problems(ok_a, ["x"], "weights_float", 2) == []
    assert at.state_problems({"x": (True, False), "y": (True, True)}, ["x"], "input_float", 2) == []
    assert at.state_problems(ok_a, ["x"], "input_float", 2)                      # wrong quantizer off
    assert at.state_problems({"x": (False, False), "y": (True, True)}, ["x"], "weights_float", 2)  # both off
    assert at.state_problems({"x": (False, True), "y": (False, True)}, ["x"], "weights_float", 2)  # extra layer
    assert at.state_problems({"y": (True, True)}, ["x"], "weights_float", 2)     # layer missing / count


# -- driver with a fake evaluation ---------------------------------------------

@pytest.fixture
def dirs(tmp_path):
    level1 = tmp_path / "level1"
    level1.mkdir()
    for m in at.TARGET_K:
        with open(level1 / f"layers_{m}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "exec_idx", "rank", "local_error_mean", "ranked"])
            w.writeheader()
            for i in range(10):
                w.writerow({"name": f"l{i}", "exec_idx": i, "rank": i + 1, "local_error_mean": 0.1, "ranked": "True"})
    level2 = tmp_path / "level2"
    rows = [{"model": "mobilevit", "bitwidth": "int8", "curve": "k0", "K": 0, "float_layers": "", "top1": 2.95,
             "top5": 7.0},
            {"model": "mobilevit", "bitwidth": "int8", "curve": "targeted_r", "K": 2, "float_layers": "l0;l1",
             "top1": 74.7, "top5": 90.0},
            {"model": "efficientvit_b1", "bitwidth": "int8", "curve": "k0", "K": 0, "float_layers": "", "top1": 0.2,
             "top5": 0.9},
            {"model": "efficientvit_b1", "bitwidth": "int8", "curve": "targeted_r", "K": 8,
             "float_layers": ";".join(f"l{i}" for i in range(8)), "top1": 40.35, "top5": 60.0}]
    l2._write_csv(level2 / "evaluations.csv", rows, l2.EVAL_COLUMNS)
    return level1, level2, tmp_path / "out"


class FakeEval:
    def __init__(self, top1):
        self.top1, self.calls = top1, []

    def __call__(self, entry, variant, names, out_dir):
        self.calls.append((entry["model_name"], variant, tuple(names)))
        return {"model": entry["model_name"], "bitwidth": "int8", "variant": variant, "K": len(names),
                "layers": ";".join(names), "top1": self.top1[variant], "top5": 0.0, "onnx_sha256": "x", "seconds": 0}


def _run(dirs, fake, models, fp32=None):
    level1, level2, out = dirs
    return at.run(models, out, False, evaluate=fake, level2_dir=level2, level1_dir=level1,
                  fp32=fp32 or {"mobilevit": 79.35, "efficientvit_b1": 81.2}.get)


def test_two_evaluations_per_model_with_the_level2_layers_then_resume(dirs):
    fake = FakeEval({"weights_float": 5.0, "input_float": 70.0})
    assert _run(dirs, fake, ["mobilevit"]) == 0
    assert fake.calls == [("mobilevit", "weights_float", ("l0", "l1")), ("mobilevit", "input_float", ("l0", "l1"))]
    again = FakeEval({"weights_float": 0.0, "input_float": 0.0})
    assert _run(dirs, again, ["mobilevit", "efficientvit_b1"]) == 0
    assert [c[0] for c in again.calls] == ["efficientvit_b1", "efficientvit_b1"]
    assert len(again.calls[0][2]) == 8
    out = dirs[2]
    summary = {r["model"]: r for r in csv.DictReader(open(out / "summary.csv"))}
    assert set(summary) == {"mobilevit", "efficientvit_b1"}
    assert summary["mobilevit"]["classification"] == "activations"
    assert len(list(csv.DictReader(open(out / "evaluations.csv")))) == 4


def test_dry_run_evaluates_nothing(dirs):
    level1, level2, out = dirs
    fake = FakeEval({})
    assert at.run(["mobilevit"], out, True, evaluate=fake, level2_dir=level2, level1_dir=level1,
                  fp32=lambda m: 79.35) == 0
    assert fake.calls == [] and not out.exists()


def test_main_logs_the_exact_exception(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("QCDQ export failed: weight_quant None")
    monkeypatch.setattr(at, "run", boom)
    assert at.main(["--out-dir", str(tmp_path), "--models", "mobilevit"]) == 1
    log = (tmp_path / "run.log").read_text()
    assert "RuntimeError: QCDQ export failed: weight_quant None" in log and "Traceback" in log


# -- Brevitas: each variant turns off exactly one quantizer ---------------------

def _tiny_model(torch):
    return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.Conv2d(8, 8, 3, groups=8),
                               torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 4), torch.nn.Linear(4, 2)))


@pytest.mark.parametrize("variant, expected", [("weights_float", (False, True)), ("input_float", (True, False))])
def test_variant_turns_off_one_quantizer_on_exactly_the_named_layers(variant, expected):
    torch = pytest.importorskip("torch")
    pytest.importorskip("brevitas")
    from quantization.vit_brevitas.quantizer_config import resolve_act_quant, resolve_weight_quant

    model = _tiny_model(torch)
    names = {"1", "2.1"}
    done = at.quantize_tree_variant(model, resolve_weight_quant("minmax", "per_tensor", "int8"),
                                    resolve_act_quant("minmax", "symmetric", "int8"), names, variant)
    assert sorted(done) == sorted(names)
    state = at.quant_state(model)
    assert len(state) == 4
    assert {n for n, s in state.items() if s == expected} == names
    assert all(s == (True, True) for n, s in state.items() if n not in names)
    assert at.state_problems(state, sorted(names), variant, 4) == []


def test_unknown_variant_is_rejected():
    torch = pytest.importorskip("torch")
    pytest.importorskip("brevitas")
    with pytest.raises(ValueError):
        at.quantize_tree_variant(_tiny_model(torch), None, None, set(), "both_float")
