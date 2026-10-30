"""scripts/mcnemar_vit.py: exact McNemar, Holm, pairing and the alignment check, on known cases."""

from __future__ import annotations

import csv
from math import comb

import pytest

import scripts.mcnemar_vit as mc


def test_identical_vectors_give_p_one():
    v = [1, 0, 1, 1, 0] * 20
    assert mc.mcnemar_exact(v, v) == (0, 0, 1.0)


def test_exact_p_matches_hand_computation():
    a = [1] * 9 + [0] * 1 + [1] * 10
    b = [0] * 9 + [1] * 1 + [1] * 10
    n10, n01, p = mc.mcnemar_exact(a, b)
    assert (n10, n01) == (9, 1)
    assert p == pytest.approx(2 * (1 + 10) / 2**10)  # 0.021484375


def test_balanced_discordants_give_p_one_and_p_is_capped():
    a, b = [1, 0] * 5, [0, 1] * 5
    assert mc.mcnemar_exact(a, b)[2] == 1.0


def test_all_discordant_one_direction_is_two_over_two_to_n():
    n = 12
    assert mc.mcnemar_exact([1] * n, [0] * n)[2] == pytest.approx(2 / 2**n)


def test_holm_known_values():
    assert mc.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert mc.holm([0.5]) == [0.5]
    assert mc.holm([0.9, 0.8]) == [1.0, 1.0]  # 2*0.8 capped at 1, then monotone


def test_plan_has_the_comparisons_of_section_6_1():
    pairs = mc.build_pairs()
    assert mc.families(pairs) == {"H1": 6, "H2": 12, "H3": 6, "RQ3b": 2}
    assert all(p[0] != "H4" for p in pairs)
    dep = {n for p in pairs for n, g in ((p[2], p[4]), (p[3], p[5])) if g == "deployment"}
    assert {n for n in dep if n.startswith("deploy_deit_tiny")} and all(n.endswith("_gelu1") for n in dep if n.startswith("deploy_deit_tiny"))
    assert not any("asymmetric" in n and n.startswith("deploy_deit_tiny") for n in dep)


def _write(path, correct, labels=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "filename", "label", "pred1", "top5", "correct1", "correct5"])
        for i, c in enumerate(correct):
            w.writerow([i, f"{i}.jpg", (labels or [0] * len(correct))[i], 0, "", c, c])


def test_alignment_detects_order_and_label_mismatch(tmp_path):
    _write(tmp_path / "a.csv", [1, 0, 1])
    _write(tmp_path / "b.csv", [1, 0, 1])
    _write(tmp_path / "c.csv", [1, 0, 1], labels=[0, 5, 0])
    _write(tmp_path / "d.csv", [1, 0])
    a = mc.read_preds(tmp_path / "a.csv")
    assert mc.check_alignment(a, mc.read_preds(tmp_path / "b.csv")) == ""
    assert "row 1" in mc.check_alignment(a, mc.read_preds(tmp_path / "c.csv"))
    assert "length" in mc.check_alignment(a, mc.read_preds(tmp_path / "d.csv"))


def test_run_tests_identical_pair_is_not_significant_and_clear_effect_is(tmp_path, monkeypatch):
    pairs = [("HX", "m", "A", "B", "ablation", "ablation"), ("HX", "m", "A", "C", "ablation", "ablation")]
    for k in mc.REPS:
        _write(tmp_path / "ablation" / f"A__rep{k}.csv", [1] * 100)
        _write(tmp_path / "ablation" / f"B__rep{k}.csv", [1] * 100)           # identical to A
        _write(tmp_path / "ablation" / f"C__rep{k}.csv", [0] * 40 + [1] * 60)  # 40 discordant, one direction
    rows = mc.run_tests(pairs, tmp_path)
    ab = [r for r in rows if r["B"] == "B"]
    ac = [r for r in rows if r["B"] == "C"]
    assert all(r["p"] == 1.0 and r["significant"] == 0 and r["delta_accuracy_pp"] == 0 for r in ab)
    assert all(r["only_A_correct"] == 40 and r["only_B_correct"] == 0 and r["significant"] == 1 for r in ac)
    assert all(r["p_holm"] == pytest.approx(min(1.0, 2 * 2 / 2**40)) for r in ac)  # rank 0 of m=2
    s = {x["B"]: x for x in mc.summarize(rows)}
    assert s["B"]["claimed"] == 0 and s["C"]["claimed"] == 1 and s["C"]["n_significant_reps"] == 3


def test_run_tests_refuses_misaligned_pair(tmp_path):
    for k in mc.REPS:
        _write(tmp_path / "ablation" / f"A__rep{k}.csv", [1, 0, 1])
        _write(tmp_path / "ablation" / f"B__rep{k}.csv", [1, 0, 1], labels=[0, 9, 0])
    with pytest.raises(ValueError, match="row 1"):
        mc.run_tests([("HX", "m", "A", "B", "ablation", "ablation")], tmp_path)


def test_tiny_p_is_reported_as_bound_not_zero():
    n10, n01, p = mc.mcnemar_exact_frac([1] * 1500, [0] * 1500)
    assert float(p) == 0.0 and p > 0 and mc.fmt_p(p) == "<1e-300"
    assert mc.fmt_p(mc.mcnemar_exact_frac([1] * 12, [0] * 12)[2]) == pytest.approx(2 / 2**12)
