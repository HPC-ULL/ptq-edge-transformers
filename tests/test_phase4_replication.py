"""Phase 4 replication with calibration repeats 2 and 3
(analysis/phase4_vit/replication.py): seeds per repeat, same float layers as
rep1, K=0 verified before anything else, recovery/ratio/verdict, resumption."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from analysis.phase4_vit import level2 as l2
from analysis.phase4_vit import replication as rp


def test_constants_and_seeds():
    assert rp.REPEATS == (2, 3) and rp.MIN_RATIO == 0.75
    assert rp.TARGET_K == {"mobilevit": 2, "efficientvit_b1": 8}
    assert [rp.calib_seed(k) for k in (1, 2, 3)] == [42, 43, 44]
    assert rp.OUT_DIR == l2.OUT_DIR / "replication"


def test_verdict_needs_both_repeats_at_or_above_three_quarters():
    assert rp.replication_verdict([0.75, 0.9]) == "replicates"
    assert rp.replication_verdict([0.74, 1.2]) == "does not replicate"
    assert rp.replication_verdict([1.0, 0.5]) == "does not replicate"


def test_run_evaluation_builds_with_the_repeat_seed(tmp_path, monkeypatch):
    seen = {}

    def fake_build(entry, names, repeat_id):
        seen.update(repeat_id=repeat_id, names=list(names))
        return object(), (1, 3, 8, 8)

    import export.vit_to_onnx.export as ex

    def fake_export(model, shape, run_id, repeat_id, out_path):
        seen["export_repeat"] = repeat_id
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"onnx")
        return out_path

    monkeypatch.setattr(ex, "export_to_onnx", fake_export)
    monkeypatch.setattr(l2, "evaluate_onnx", lambda m, p: (50.0, 70.0, [{"idx": 0, "filename": "a", "label": 1,
                                                                       "pred1": 1, "correct1": 1}]))
    entry = {"model_name": "mobilevit", "run_id": "r"}
    row = rp.run_evaluation(entry, 3, "targeted_r", ["a", "b"], tmp_path, build=fake_build)
    assert seen == {"repeat_id": 3, "names": ["a", "b"], "export_repeat": 3}
    assert row["calib_seed"] == 44 and row["repeat"] == 3 and row["float_layers"] == "a;b" and row["K"] == 2
    assert not (tmp_path / "onnx" / "mobilevit__rep3__targeted_r__K2.onnx").exists()
    assert (tmp_path / "predictions" / "mobilevit__rep3__targeted_r__K2.csv").exists() and len(row["onnx_sha256"]) == 64


# -- driver with a fake evaluation ---------------------------------------------

LAYERS = {"mobilevit": ["l0", "l1"], "efficientvit_b1": [f"l{i}" for i in range(8)]}
K0 = {("mobilevit", 2): 3.10, ("mobilevit", 3): 2.80, ("efficientvit_b1", 2): 0.15, ("efficientvit_b1", 3): 0.25}


@pytest.fixture
def dirs(tmp_path):
    level1 = tmp_path / "level1"
    level1.mkdir()
    for m in rp.TARGET_K:
        with open(level1 / f"layers_{m}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "exec_idx", "rank", "local_error_mean", "ranked"])
            w.writeheader()
            for i in range(10):
                w.writerow({"name": f"l{i}", "exec_idx": i, "rank": i + 1, "local_error_mean": 0.1, "ranked": "True"})
    rows = []
    for m, k0, full in (("mobilevit", 2.95, 74.7), ("efficientvit_b1", 0.2, 40.35)):
        rows.append({"model": m, "bitwidth": "int8", "curve": "k0", "K": 0, "float_layers": "", "top1": k0, "top5": 1.0})
        rows.append({"model": m, "bitwidth": "int8", "curve": "targeted_r", "K": rp.TARGET_K[m],
                     "float_layers": ";".join(LAYERS[m]), "top1": full, "top5": 1.0})
    l2._write_csv(tmp_path / "level2" / "evaluations.csv", rows, l2.EVAL_COLUMNS)
    return tmp_path


class FakeEval:
    def __init__(self, targeted):
        self.targeted, self.calls = targeted, []

    def __call__(self, entry, repeat, curve, names, out_dir):
        m = entry["model_name"]
        self.calls.append((m, repeat, curve, tuple(names)))
        top1 = K0[(m, repeat)] if curve == "k0" else self.targeted[(m, repeat)]
        return {"model": m, "repeat": repeat, "calib_seed": 41 + repeat, "curve": curve, "K": len(names),
                "float_layers": ";".join(names), "top1": top1, "top5": 1.0, "onnx_sha256": "x", "seconds": 0}


def _recorded(run_id, repeat_id):
    m = "mobilevit" if "mobilevit" in run_id else "efficientvit_b1"
    return {"top1": K0[(m, repeat_id)], "top5": 1.0}


def _run(d, fake, recorded=_recorded, models=("mobilevit", "efficientvit_b1")):
    out = d / "out"
    return rp.run(list(models), out, False, runner=rp.Runner(out, evaluate=fake), recorded=recorded,
                  fp32={"mobilevit": 79.35, "efficientvit_b1": 81.2}.get, level2_dir=d / "level2",
                  level1_dir=d / "level1", per_image_dir=d / "none"), out


GOOD = {("mobilevit", 2): 70.0, ("mobilevit", 3): 72.0, ("efficientvit_b1", 2): 25.0, ("efficientvit_b1", 3): 38.0}  # eff. rep2 ratio 0.62


def test_every_k0_is_verified_before_any_targeted_evaluation(dirs):
    fake = FakeEval(GOOD)
    rc, out = _run(dirs, fake)
    assert rc == 0 and len(fake.calls) == 8
    assert [c[2] for c in fake.calls[:4]] == ["k0"] * 4 and [c[2] for c in fake.calls[4:]] == ["targeted_r"] * 4
    assert {(c[1]) for c in fake.calls} == {2, 3}


def test_k0_mismatch_stops_before_any_targeted_evaluation(dirs):
    fake = FakeEval(GOOD)

    def wrong(run_id, repeat_id):
        r = _recorded(run_id, repeat_id)
        return {**r, "top5": r["top5"] + 0.05} if repeat_id == 3 and "efficientvit" in run_id else r

    rc, out = _run(dirs, fake, recorded=wrong)
    assert rc == 1 and all(c[2] == "k0" for c in fake.calls)
    assert "MISMATCH" in (out / "kzero_checks.csv").read_text() and not (out / "summary.csv").exists()


def test_targeted_float_layers_are_the_rep1_layers(dirs):
    fake = FakeEval(GOOD)
    _run(dirs, fake)
    targeted = {(c[0], c[1]): c[3] for c in fake.calls if c[2] == "targeted_r"}
    assert all(targeted[(m, k)] == tuple(LAYERS[m]) for m in LAYERS for k in (2, 3))


def test_recovery_uses_each_repeat_k0_and_the_verdict(dirs):
    _, out = _run(dirs, FakeEval(GOOD))
    s = {r["model"]: r for r in csv.DictReader(open(out / "summary.csv"))}
    mv = s["mobilevit"]
    assert float(mv["rec_rep2"]) == pytest.approx((70.0 - 3.10) / (79.35 - 3.10))
    assert float(mv["rec_rep3"]) == pytest.approx((72.0 - 2.80) / (79.35 - 2.80))
    assert float(mv["ratio_rep2"]) == pytest.approx(float(mv["rec_rep2"]) / ((74.7 - 2.95) / (79.35 - 2.95)))
    assert float(mv["threshold"]) == pytest.approx(0.704, abs=5e-4)
    assert float(s["efficientvit_b1"]["threshold"]) == pytest.approx(0.372, abs=5e-4)
    assert mv["verdict"] == "replicates" and s["efficientvit_b1"]["verdict"] == "does not replicate"


def test_resume_repeats_nothing(dirs):
    _run(dirs, FakeEval(GOOD))
    again = FakeEval(GOOD)
    rc, out = _run(dirs, again)
    assert rc == 0 and again.calls == []
    assert len(list(csv.DictReader(open(out / "evaluations.csv")))) == 8


def test_level2_layers_must_match_the_r_ranking(dirs):
    path = dirs / "level2" / "evaluations.csv"
    path.write_text(path.read_text().replace("l0;l1", "l1;l0"))
    with pytest.raises(RuntimeError):
        _run(dirs, FakeEval(GOOD))


def test_main_logs_the_exact_exception(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("export failed")
    monkeypatch.setattr(rp, "run", boom)
    assert rp.main(["--out-dir", str(tmp_path)]) == 1
    assert "RuntimeError: export failed" in (tmp_path / "run.log").read_text()
