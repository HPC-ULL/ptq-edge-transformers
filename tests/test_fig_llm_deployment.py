"""plots/scripts/fig_llm_deployment.py on a synthetic summary and (smoke) the real one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import plots.scripts.fig_llm_deployment as fig
from plots import paper_labels as L
from plots.common import OUT

REAL = OUT / "llm_deployment_summary.csv"


def _summary(energy=2.0):
    rows = []
    for dev in L.LLM_DEPLOY_DEVICES:
        for m in L.LLM_MODEL_ORDER:
            for cfg in list(L.LLM_DEPLOY_FORMATS) + ["Q4_K_M imatrix 50", "phase3_combined_q2_k"]:
                rows.append(dict(device=dev, model=m, config=cfg, reference="F16", r_energy=energy, r_tg=3.0))
    d = pd.DataFrame(rows)
    d = d[~((d.model == "ministral_3_3b") & d.device.isin(["kube01", "visionfive_2"]))]   # not measured
    bad = (d.device == "n150") & (d.model == "ministral_3_3b") & d.config.isin(L.LLM_DEPLOY_FORMATS)
    d.loc[bad, ["reference", "r_energy", "r_tg"]] = ["F16 (excluida)", np.nan, np.nan]      # excluded F16
    d.loc[(d.device == "rk1-02") & (d.model == "qwen2_5_1_5b") & (d.config == "Q2_K"), "r_energy"] = 1.30
    return d


def test_grid_filters_imatrix_and_marks_excluded_and_missing_cells():
    t = fig.build_table(_summary())
    assert len(t) == 3 * 4 * 10 and set(t.config) == set(L.LLM_DEPLOY_FORMATS)
    assert t.r_energy.notna().sum() == 108 and t.r_tg.notna().sum() == 108
    m = t[t.model == "ministral_3_3b"]
    assert m[m.device.isin(["n150", "kube01", "visionfive_2"])].r_energy.isna().all()
    assert m[m.device == "rk1-02"].r_energy.notna().all()


def test_checks_pass_and_fail():
    fig.checks(fig.build_table(_summary()))
    d = _summary()
    d.loc[(d.device == "rk1-02") & (d.model == "qwen2_5_1_5b") & (d.config == "Q2_K"), "r_energy"] = 1.10
    with pytest.raises(AssertionError, match="minimum energy"):
        fig.checks(fig.build_table(d))
    with pytest.raises(AssertionError, match="108"):
        fig.checks(fig.build_table(_summary().iloc[1:]))


def test_duplicate_row_is_rejected():
    d = _summary()
    with pytest.raises(AssertionError, match="more than one row"):
        fig.build_table(pd.concat([d, d.iloc[[0]]], ignore_index=True))


def test_labels():
    assert list(L.LLM_DEPLOY_FORMATS.values()) == ["Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]
    assert L.LLM_DEPLOY_DEVICES["kube01"] == "CM4" and len(L.LLM_DEPLOY_DEVICES) == 10


@pytest.mark.skipif(not REAL.exists(), reason="real summary CSV not in this checkout")
def test_smoke_real_csv(tmp_path):
    fig.main(["--out-dir", str(tmp_path)])
    d = tmp_path / fig.NAME
    assert (d / f"{fig.NAME}.pdf").stat().st_size > 0 and (d / f"{fig.NAME}.png").stat().st_size > 0
    data = pd.read_csv(d / f"{fig.NAME}_data.csv")
    assert len(data) == 120 and data.r_energy.notna().sum() == 108
