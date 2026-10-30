"""Tensor sets for the per-tensor imatrix intervention (analysis/phase4_llm/sets.py)."""

from __future__ import annotations

import random

import pytest

from analysis.phase4_llm import sets as st


def test_ranking_descending_with_ties_by_execution_order():
    assert st.ranked_blocks({0: 2.0, 1: 5.0, 2: 5.0, 3: 1.0}) == [1, 2, 0, 3]


def _rho(n, seed=0):
    rng = random.Random(seed)
    return {t: {b: rng.random() for b in range(n)} for t in st.TYPE_ORDER}


def test_h_l_halves_per_type_and_complementary():
    rho = _rho(22)
    sets = st.block_sets(rho, 22)
    for t in st.TYPE_ORDER:
        top = st.ranked_blocks(rho[t])[:11]
        assert sets["H"][t] == sorted(top) and sorted(sets["H"][t] + sets["L"][t]) == list(range(22))


def test_random_sets_follow_the_protocol_procedure():
    sets = st.block_sets(_rho(28), 28)
    for s in (1, 2, 3):
        rng = random.Random(s)
        for t in ("qkv", "attn_output", "gate_up", "down"):  # one rng across the types, in this order
            b = list(range(28))
            rng.shuffle(b)
            assert sets[f"R{s}"][t] == sorted(b[:14])
    assert sets["R1"] != sets["R2"] != sets["R3"]


def test_tensor_expansion_and_counts():
    names = st.tensor_names(st.block_sets(_rho(22), 22)["H"])
    assert len(names) == 77 and len(set(names)) == 77
    assert sum(".attn_k." in n for n in names) == 11 and sum(".ffn_up." in n for n in names) == 11
    assert len(st.all_tensor_names(28)) == 196 and len(st.all_tensor_names(22)) == 154


def test_check_catches_wrong_sizes_missing_names_and_overlap():
    rho = _rho(22)
    sets = st.block_sets(rho, 22)
    names = {s: st.tensor_names(b) for s, b in sets.items()}
    available = [set(st.all_tensor_names(22))] * 3
    assert st.check("tinyllama_1_1b", sets, names, available) == []
    bad = {**sets, "R1": {**sets["R1"], "down": sets["R1"]["down"][:-1]}}
    assert st.check("tinyllama_1_1b", bad, {s: st.tensor_names(b) for s, b in bad.items()}, available)
    assert st.check("tinyllama_1_1b", sets, names, [available[0] - {names["H"][0]}] * 3)
    overlap = {**sets, "L": sets["H"]}
    assert any("disjoint" in p for p in st.check("tinyllama_1_1b", overlap,
                                                {s: st.tensor_names(b) for s, b in overlap.items()}, available))


def test_mean_rho_requires_all_blocks_and_repeats():
    rows = [{"model": "tinyllama_1_1b", "type": t, "block": b, "rho": 1.0, "repeat": k}
            for t in st.TYPE_ORDER for b in range(22) for k in (1, 2, 3)]
    assert st.mean_rho(rows, "tinyllama_1_1b")["down"][5] == 1.0
    with pytest.raises(RuntimeError):
        st.mean_rho(rows[:-1], "tinyllama_1_1b")


def test_depth_set_e_matches_the_protocol():
    assert st.depth_blocks(22) == [0, 1, 2, 3, 4, 5, 17, 18, 19, 20, 21]  # tie 5 vs 16 -> 5
    assert st.depth_blocks(28) == list(range(0, 7)) + list(range(21, 28))
    sets = st.block_sets(_rho(22), 22)
    assert all(sets["E"][t] == st.depth_blocks(22) for t in st.TYPE_ORDER)
    assert len(st.tensor_names(sets["E"])) == 77
    names = {s: st.tensor_names(b) for s, b in sets.items()}
    assert st.check("tinyllama_1_1b", sets, names, [set(st.all_tensor_names(22))] * 3) == []


def test_extra_random_sets_follow_the_same_procedure_and_leave_r1_r3_unchanged():
    sets = st.block_sets(_rho(28), 28)
    assert [f"R{s}" for s in st.EXTRA_SEEDS] == [f"R{s}" for s in range(4, 21)]
    for s in range(1, 21):
        rng = random.Random(s)
        for t in st.TYPE_ORDER:
            b = list(range(28))
            rng.shuffle(b)
            assert sets[f"R{s}"][t] == sorted(b[:14])
        assert len(st.tensor_names(sets[f"R{s}"])) == 98
    assert st.random_blocks(28, 2) == sets["R2"]
