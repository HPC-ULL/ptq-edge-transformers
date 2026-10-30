"""Random control of the replication (analysis/phase4_vit/replication_random.py):
rep1's random float layers (8 per seed) reused in repeats 2 and 3, each repeat
with its own calibration seed, K=0 and targeted reused from the replication,
strict comparison with the max of the 3 seeds, resumption."""

from __future__ import annotations

import csv

import pytest

from analysis.phase4_vit import level2 as l2
from analysis.phase4_vit import replication as rp
from analysis.phase4_vit import replication_random as rr


def test_constants():
    assert (rr.MODEL, rr.K, rr.REPEATS, rr.SEEDS) == ("efficientvit_b1", 8, (2, 3), (1, 2, 3))
    assert [rp.calib_seed(k) for k in rr.REPEATS] == [43, 44]


RANKED = [{"name": f"l{i}", "exec_idx": i, "rank": i + 1, "local_error_mean": 0.1} for i in range(20)]
TARGETED = [f"l{i}" for i in range(8)]


def _random_layers(seed):
    return l2.order_random(RANKED, seed)[:8]


@pytest.fixture
def dirs(tmp_path):
    level1 = tmp_path / "level1"
    level1.mkdir()
    with open(level1 / "layers_efficientvit_b1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "exec_idx", "rank", "local_error_mean", "ranked"])
        w.writeheader()
        for r in RANKED:
            w.writerow({**r, "ranked": "True"})
    rows = [{"model": "efficientvit_b1", "bitwidth": "int8", "curve": "k0", "K": 0, "float_layers": "",
             "top1": 0.2, "top5": 0.9},
            {"model": "efficientvit_b1", "bitwidth": "int8", "curve": "targeted_r", "K": 8,
             "float_layers": ";".join(TARGETED), "top1": 40.35, "top5": 60.0}]
    for s in (1, 2, 3):
        rows.append({"model": "efficientvit_b1", "bitwidth": "int8", "curve": f"random_s{s}", "K": 8,
                     "float_layers": ";".join(_random_layers(s)), "top1": 1.0, "top5": 2.0})
    l2._write_csv(tmp_path / "level2" / "evaluations.csv", rows, l2.EVAL_COLUMNS)
    out = tmp_path / "replication"
    rep_rows = []
    for k, k0, tg in ((2, 0.1, 34.25), (3, 0.1, 29.15)):
        rep_rows.append({"model": "efficientvit_b1", "repeat": k, "calib_seed": 41 + k, "curve": "k0", "K": 0,
                         "float_layers": "", "top1": k0, "top5": 0.6})
        rep_rows.append({"model": "efficientvit_b1", "repeat": k, "calib_seed": 41 + k, "curve": "targeted_r",
                         "K": 8, "float_layers": ";".join(TARGETED), "top1": tg, "top5": 55.0})
    l2._write_csv(out / "evaluations.csv", rep_rows, rp.EVAL_COLUMNS)
    l2._write_csv(out / "kzero_checks.csv", [{"model": "efficientvit_b1", "repeat": k, "status": "OK"} for k in (2, 3)],
                  ["model", "repeat", "status"])
    return tmp_path


class FakeEval:
    def __init__(self, top1):
        self.top1, self.calls = top1, []

    def __call__(self, entry, repeat, curve, names, out_dir):
        self.calls.append((repeat, curve, tuple(names)))
        return {"model": entry["model_name"], "repeat": repeat, "calib_seed": rp.calib_seed(repeat), "curve": curve,
                "K": len(names), "float_layers": ";".join(names), "top1": self.top1[(repeat, curve)], "top5": 1.0,
                "onnx_sha256": "x", "seconds": 0}


LOW = {(k, f"random_s{s}"): 5.0 + s for k in (2, 3) for s in (1, 2, 3)}


def _run(d, fake, dry_run=False):
    out = d / "replication"
    return rr.run(out, dry_run, runner=rp.Runner(out, evaluate=fake), fp32=lambda m: 81.2,
                  level2_dir=d / "level2", level1_dir=d / "level1")


def test_six_evaluations_with_rep1_layers_eight_each_and_the_repeat_seed(dirs):
    fake = FakeEval(LOW)
    assert _run(dirs, fake) == 0
    assert len(fake.calls) == 6
    for repeat, curve, names in fake.calls:
        assert repeat in (2, 3) and len(names) == 8
        assert list(names) == _random_layers(int(curve[-1]))
    rows = list(csv.DictReader(open(dirs / "replication" / "evaluations.csv")))
    assert len(rows) == 10  # 4 reused + 6 new
    assert {(r["repeat"], r["calib_seed"]) for r in rows if r["curve"].startswith("random")} == {("2", "43"), ("3", "44")}


def test_k0_and_targeted_are_reused_not_reevaluated(dirs):
    fake = FakeEval(LOW)
    _run(dirs, fake)
    assert all(c[1].startswith("random") for c in fake.calls)


def test_rep1_random_layers_must_match_order_random(dirs):
    path = dirs / "level2" / "evaluations.csv"
    first = _random_layers(2)
    path.write_text(path.read_text().replace(";".join(first), ";".join(reversed(first))))
    fake = FakeEval(LOW)
    with pytest.raises(RuntimeError):
        _run(dirs, fake)
    assert fake.calls == []


def test_missing_replication_targeted_stops_before_evaluating(dirs):
    path = dirs / "replication" / "evaluations.csv"
    kept = [r for r in csv.DictReader(open(path)) if not (r["repeat"] == "3" and r["curve"] == "targeted_r")]
    l2._write_csv(path, kept, rp.EVAL_COLUMNS)
    fake = FakeEval(LOW)
    with pytest.raises(KeyError):
        _run(dirs, fake)
    assert fake.calls == []


def test_k0_check_not_ok_stops_before_evaluating(dirs):
    l2._write_csv(dirs / "replication" / "kzero_checks.csv",
                  [{"model": "efficientvit_b1", "repeat": 2, "status": "OK"},
                   {"model": "efficientvit_b1", "repeat": 3, "status": "MISMATCH"}], ["model", "repeat", "status"])
    fake = FakeEval(LOW)
    with pytest.raises(RuntimeError):
        _run(dirs, fake)
    assert fake.calls == []


def test_verdict_holds_when_targeted_beats_the_max_in_both_repeats(dirs):
    _run(dirs, FakeEval(LOW))
    rows = list(csv.DictReader(open(dirs / "replication" / "random_summary.csv")))
    assert [r["repeat"] for r in rows] == ["2", "3"] and {r["verdict"] for r in rows} == {"superiority holds"}
    r2 = rows[0]
    assert float(r2["random_max_top1"]) == 8.0
    assert float(r2["targeted_recovery"]) == pytest.approx((34.25 - 0.1) / (81.2 - 0.1))
    assert float(r2["random_s3_recovery"]) == pytest.approx((8.0 - 0.1) / (81.2 - 0.1))


def test_comparison_is_strict_and_needs_both_repeats(dirs):
    tie = dict(LOW)
    tie[(3, "random_s2")] = 29.15  # equals the targeted of rep3
    _run(dirs, FakeEval(tie))
    rows = list(csv.DictReader(open(dirs / "replication" / "random_summary.csv")))
    assert rows[0]["targeted_beats_max"] == "True" and rows[1]["targeted_beats_max"] == "False"
    assert {r["verdict"] for r in rows} == {"superiority does not hold"}


def test_resume_repeats_nothing(dirs):
    _run(dirs, FakeEval(LOW))
    again = FakeEval(LOW)
    assert _run(dirs, again) == 0 and again.calls == []


def test_dry_run_evaluates_nothing(dirs):
    fake = FakeEval(LOW)
    assert _run(dirs, fake, dry_run=True) == 0 and fake.calls == []
