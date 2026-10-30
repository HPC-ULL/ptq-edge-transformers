"""Phase 4 R-ranking stability (analysis/phase4_vit/stability.py): constants
and the comparison of the two rankings."""

from __future__ import annotations

from analysis.phase4_vit import PHASE4_ROOT
from analysis.phase4_vit import stability as st


def test_constants():
    assert st.SECOND_SEED == 1 and st.TOP == 8 and st.OUT_DIR == PHASE4_ROOT / "stability"


def _rows(order, extra_unranked=()):
    rows = [{"name": n, "R": str(100 - i), "rank": str(i + 1), "ranked": "True"} for i, n in enumerate(order)]
    return rows + [{"name": n, "R": "1", "rank": "", "ranked": "False"} for n in extra_unranked]


def test_compare_counts_common_ranked_layers_and_top8_overlap_ignoring_order():
    primary = _rows([f"l{i}" for i in range(12)], extra_unranked=["stem"])
    # second set: top 8 reshuffled, l7 swapped with l10, l11 not ranked
    second_order = ["l1", "l0", "l3", "l2", "l4", "l5", "l6", "l10", "l7", "l8", "l9"]
    second = _rows(second_order, extra_unranked=["stem", "l11"])
    seen = {}

    def fake_spearman(x, y):
        seen["n"] = len(x)
        return 0.9, None

    c = st.compare(primary, second, fake_spearman)
    assert c["n_common"] == 11 and seen["n"] == 11 and c["spearman_R"] == 0.9
    assert c["top8_overlap"] == 7
    assert c["top8_primary"].split(";") == [f"l{i}" for i in range(8)]
    assert c["top8_second"].split(";") == second_order[:8]


def test_compare_accepts_booleans_from_a_fresh_pass():
    primary = _rows(["a", "b"])
    second = [{"name": "a", "R": 5.0, "rank": 1, "ranked": True}, {"name": "b", "R": 4.0, "rank": 2, "ranked": True}]
    c = st.compare(primary, second, lambda x, y: (1.0, None))
    assert c["n_common"] == 2 and c["top8_overlap"] == 2
