"""DeiT-Tiny INT6 replication (analysis/phase4_vit/replication_deit.py): rep1's
float layers per curve (8 each) reused in repeats 2 and 3, each repeat with its
own calibration seed, both K=0 verified (exact, 0 per-image differences) before
anything else, the verdict on the R curve only, resumption."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from analysis.phase4_vit import level2 as l2
from analysis.phase4_vit import replication as rp
from analysis.phase4_vit import replication_deit as rd

RANKED = [{"name": f"l{i}", "exec_idx": i, "rank": 20 - i, "local_error_mean": 0.01 * ((7 * i) % 20)}
          for i in range(20)]
RUN_ID = "vit_deit_tiny_phase3_baseline_int6"
K0 = {2: 58.45, 3: 56.15}


def test_constants():
    assert (rd.MODEL, rd.K, rd.REPEATS) == ("deit_tiny", 8, (2, 3))
    assert rd.CURVES == ("targeted_r", "random_s1", "random_s2", "random_s3", "targeted_err")
    assert l2.reference_entry("deit_tiny")["run_id"] == RUN_ID


@pytest.fixture
def dirs(tmp_path):
    level1 = tmp_path / "level1"
    level1.mkdir()
    with open(level1 / "layers_deit_tiny.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "exec_idx", "rank", "local_error_mean", "ranked"])
        w.writeheader()
        for r in RANKED:
            w.writerow({**r, "ranked": "True"})
    expected = rd.expected_layers(RANKED)
    rows = [{"model": "deit_tiny", "bitwidth": "int6", "curve": "k0", "K": 0, "float_layers": "", "top1": 58.8,
             "top5": 81.55}]
    rows += [{"model": "deit_tiny", "bitwidth": "int6", "curve": c, "K": 8, "float_layers": ";".join(expected[c]),
              "top1": 60.0, "top5": 80.0} for c in rd.CURVES]
    l2._write_csv(tmp_path / "level2" / "evaluations.csv", rows, l2.EVAL_COLUMNS)
    for k in rd.REPEATS:
        l2._write_csv(tmp_path / "per_image" / f"{RUN_ID}__rep{k}.csv", [{"idx": 0, "pred1": 1}, {"idx": 1, "pred1": 2}],
                      ["idx", "pred1"])
    return tmp_path


class FakeEval:
    def __init__(self, top1, k0_pred=(1, 2)):
        self.top1, self.k0_pred, self.calls = top1, k0_pred, []

    def __call__(self, entry, repeat, curve, names, out_dir):
        self.calls.append((repeat, curve, tuple(names), rp.calib_seed(repeat)))
        if curve == "k0":
            top1, top5 = K0[repeat], {2: 81.35, 3: 80.1}[repeat]
            l2._write_csv(out_dir / "predictions" / f"deit_tiny__rep{repeat}__k0__K0.csv",
                          [{"idx": 0, "pred1": self.k0_pred[0]}, {"idx": 1, "pred1": self.k0_pred[1]}], ["idx", "pred1"])
        else:
            top1, top5 = self.top1[(repeat, curve)], 0.0
        return {"model": "deit_tiny", "repeat": repeat, "calib_seed": rp.calib_seed(repeat), "curve": curve,
                "K": len(names), "float_layers": ";".join(names), "top1": top1, "top5": top5, "onnx_sha256": "x",
                "seconds": 0}


def _recorded(run_id, repeat_id):
    assert run_id == RUN_ID
    return {"top1": K0[repeat_id], "top5": {2: 81.35, 3: 80.1}[repeat_id]}


def _top1(targeted, randoms=(59.0, 60.0, 61.0), err=62.0):
    out = {}
    for k in (2, 3):
        out[(k, "targeted_r")] = targeted[k]
        out[(k, "targeted_err")] = err
        for c, v in zip(l2.RANDOM_CURVES, randoms):
            out[(k, c)] = v
    return out


def _run(d, fake, recorded=_recorded, dry_run=False):
    out = d / "replication"
    return rd.run(out, dry_run, runner=rp.Runner(out, evaluate=fake), recorded=recorded, fp32=lambda m: 73.5,
                  level2_dir=d / "level2", level1_dir=d / "level1", per_image_dir=d / "per_image"), out


def test_twelve_evaluations_k0_first_rep1_layers_eight_each_and_the_repeat_seed(dirs):
    fake = FakeEval(_top1({2: 60.5, 3: 60.0}))
    rc, out = _run(dirs, fake)
    assert rc == 0 and len(fake.calls) == 12
    assert [c[1] for c in fake.calls[:2]] == ["k0", "k0"]
    expected = rd.expected_layers(RANKED)
    for repeat, curve, names, seed in fake.calls[2:]:
        assert list(names) == expected[curve] and len(names) == 8
        assert seed == {2: 43, 3: 44}[repeat]
    rows = list(csv.DictReader(open(out / "evaluations.csv")))
    assert {(r["repeat"], r["calib_seed"]) for r in rows} == {("2", "43"), ("3", "44")}


def test_rep1_layers_must_match_the_level2_orders(dirs):
    path = dirs / "level2" / "evaluations.csv"
    exp = rd.expected_layers(RANKED)["random_s3"]
    path.write_text(path.read_text().replace(";".join(exp), ";".join(exp[1:] + exp[:1])))
    fake = FakeEval(_top1({2: 60.0, 3: 60.0}))
    with pytest.raises(RuntimeError):
        _run(dirs, fake)
    assert fake.calls == []


def test_k0_with_a_per_image_difference_stops_before_anything_else(dirs):
    fake = FakeEval(_top1({2: 60.0, 3: 60.0}), k0_pred=(1, 3))
    rc, out = _run(dirs, fake)
    assert rc == 1 and [c[1] for c in fake.calls] == ["k0"]
    assert "MISMATCH" in (out / "deit_kzero_checks.csv").read_text()


def test_k0_with_other_top5_in_rep3_stops_before_any_curve(dirs):
    def recorded(run_id, repeat_id):
        r = _recorded(run_id, repeat_id)
        return {**r, "top5": r["top5"] + 0.05} if repeat_id == 3 else r

    fake = FakeEval(_top1({2: 60.0, 3: 60.0}))
    rc, _ = _run(dirs, fake, recorded=recorded)
    assert rc == 1 and [c[1] for c in fake.calls] == ["k0", "k0"]


def test_negative_result_replicates_when_r_never_beats_the_max(dirs):
    _, out = _run(dirs, FakeEval(_top1({2: 61.0, 3: 59.5})))  # 61.0 ties the max: not strictly greater
    rows = list(csv.DictReader(open(out / "deit_summary.csv")))
    assert {r["verdict"] for r in rows} == {"negative result replicates"}
    assert [r["targeted_r_beats_max"] for r in rows] == ["False", "False"]
    assert [r["targeted_err_beats_max"] for r in rows] == ["True", "True"]  # reported, outside the verdict
    assert float(rows[0]["targeted_r_recovery"]) == pytest.approx((61.0 - 58.45) / (73.5 - 58.45))


def test_negative_result_does_not_replicate_if_r_beats_the_max_in_one_repeat(dirs):
    _, out = _run(dirs, FakeEval(_top1({2: 60.0, 3: 61.5})))
    rows = list(csv.DictReader(open(out / "deit_summary.csv")))
    assert {r["verdict"] for r in rows} == {"negative result does not replicate"}


def test_resume_repeats_nothing(dirs):
    _run(dirs, FakeEval(_top1({2: 60.0, 3: 60.0})))
    again = FakeEval(_top1({2: 60.0, 3: 60.0}))
    rc, _ = _run(dirs, again)
    assert rc == 0 and again.calls == []


def test_dry_run_evaluates_nothing(dirs):
    fake = FakeEval({})
    rc, out = _run(dirs, fake, dry_run=True)
    assert rc == 0 and fake.calls == [] and not (out / "evaluations.csv").exists()
