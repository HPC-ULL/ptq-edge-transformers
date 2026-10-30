"""plots/scripts/fig_vit_phase4_recovery.py on synthetic CSVs and (smoke) the real ones."""

from __future__ import annotations

import pandas as pd
import pytest

import plots.scripts.fig_vit_phase4_recovery as fig
from plots.scripts.fig_vit_phase4_recovery import LEVEL2

MODELS = ["deit_tiny", "mobilevit", "efficientvit_b1"]


def _ev(rows):
    return pd.DataFrame(rows, columns=["model", "bitwidth", "curve", "K", "top1"])


def _data():
    ev, rep = [], []
    for m in MODELS:
        b = fig.BITWIDTH[m]
        ev += [(m, b, "k0", 0, 10.0), (m, b, "targeted_r", 1, 40.0), (m, b, "targeted_r", 2, 55.0),
               (m, b, "targeted_err", 2, 99.0), (m, b, "random_s1", 2, 20.0)]
        rep += [(m, 2, "k0", 0, 20.0), (m, 2, "targeted_r", 2, 60.0)]
    ev = _ev(ev)
    rep = pd.DataFrame(rep, columns=["model", "repeat", "curve", "K", "top1"])
    fp = pd.DataFrame([dict(model_name=m, accuracy_metric_name="top1_accuracy", factor_level="fp32",
                            accuracy_value=110.0) for m in MODELS]
                      + [dict(model_name=m, accuracy_metric_name="top1_accuracy",
                              factor_level="fp32_full50000", accuracy_value=1.0) for m in MODELS])
    return ev, rep, fp


def test_recovered_fraction_pairs_k0_per_repeat_and_drops_targeted_err():
    t = fig.build_table(*_data())
    assert "targeted_err" not in set(t.curve) and "k0" not in set(t.curve)
    r1 = t[(t.model == "deit_tiny") & (t["repeat"] == 1) & (t.curve == "targeted_r") & (t.K == 2)]
    assert r1.recovered_pct.iloc[0] == pytest.approx(100 * (55 - 10) / (110 - 10))
    r2 = t[(t.model == "deit_tiny") & (t["repeat"] == 2)]
    assert r2.recovered_pct.iloc[0] == pytest.approx(100 * (60 - 20) / (110 - 20))   # k0 of rep 2


def test_wrong_bitwidth_is_rejected():
    ev, rep, fp = _data()
    ev.loc[ev.model == "deit_tiny", "bitwidth"] = "int8"
    with pytest.raises(AssertionError, match="bitwidth"):
        fig.build_table(ev, rep, fp)


def test_sanity_check_stops_when_expected_row_missing():
    with pytest.raises(SystemExit, match="SANITY"):
        fig.sanity_check(fig.build_table(*_data()))


@pytest.mark.skipif(not (LEVEL2 / "evaluations.csv").exists(), reason="real CSVs not in this checkout")
def test_smoke_real_csvs(tmp_path):
    fig.main(["--out-dir", str(tmp_path)])
    d = tmp_path / fig.NAME
    assert (d / f"{fig.NAME}.pdf").stat().st_size > 0 and (d / f"{fig.NAME}.png").stat().st_size > 0
    data = pd.read_csv(d / f"{fig.NAME}_data.csv")
    assert len(data) == 58
    mv = data[data.model == "mobilevit"]
    assert set(mv.K) == {1, 2}                                # no K = 4 or 8 invented
    assert set(mv[mv["repeat"] > 1].curve) == {"targeted_r"}  # no random replicas
