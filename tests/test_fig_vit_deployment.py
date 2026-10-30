"""plots/scripts/fig_vit_deployment.py on a synthetic summary and (smoke) the real one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import plots.scripts.fig_vit_deployment as fig
from plots import paper_labels as L
from plots.common import OUT

REAL = OUT / "vit_deployment_summary.csv"


def _summary():
    rows = []
    for dev in L.DEPLOY_DEVICE_ORDER:
        jet = dev.startswith("jetson")
        for m in L.DEPLOY_MODEL_ORDER:
            for cfg in ("core", "fallback"):
                core_jet = jet and cfg == "core"
                rows.append(dict(
                    device=dev, backend="cpu", model=m, config=cfg, run_id=f"{m}_{cfg}_whatever",
                    reference="fp32", speedup=1.2, energy_ratio=1.4,
                    sess_lat_lo=1.1 if core_jet else np.nan, sess_lat_hi=1.3 if core_jet else np.nan,
                    sess_ene_lo=1.3 if core_jet else np.nan, sess_ene_hi=1.5 if core_jet else np.nan,
                    n_sessions_ref=3 if jet else 1, n_sessions_x=3 if core_jet else 1))
    d = pd.DataFrame(rows)
    # rows the figure must ignore: GPU, other references, other configs
    extra = d.iloc[:3].copy()
    extra["backend"] = "gpu"
    extra2 = d.iloc[:3].copy()
    extra2["config"] = "var_symmetry"
    return pd.concat([d, extra, extra2], ignore_index=True)


def test_filters_by_config_and_keeps_bars_only_for_three_sessions():
    t = fig.build_table(_summary())
    assert len(t) == 28 and set(t.config) == {"core", "fallback"}
    assert t[t.has_bars].device.str.startswith("jetson").all() and (t[t.has_bars].config == "core").all()
    assert t[~t.has_bars][["sess_lat_lo", "sess_ene_hi"]].isna().all().all()


def test_seven_devices_expected_with_cm4_before_riscv():
    assert len(L.DEPLOY_DEVICE_ORDER) == 7
    assert L.DEPLOY_DEVICE_ORDER[-2:] == ("kube01", "visionfive_2") and L.DEVICES["kube01"] == "CM4"


def test_duplicate_row_is_rejected():
    d = _summary()
    d = pd.concat([d, d.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="more than one row"):
        fig.build_table(d)


def test_missing_device_is_rejected():
    d = _summary()
    with pytest.raises(AssertionError, match="devices"):
        fig.build_table(d[d.device != "n150"])


def test_value_outside_session_range_is_rejected():
    d = _summary()
    d.loc[(d.device == "jetson_cpu_15w") & (d.config == "core"), "speedup"] = 5.0
    with pytest.raises(AssertionError, match="outside"):
        fig.build_table(d)


def test_sanity_check_stops_on_mismatch():
    with pytest.raises(SystemExit, match="SANITY"):
        fig.sanity_check(fig.build_table(_summary()))


def _real_has_cm4():
    return REAL.exists() and "kube01" in set(pd.read_csv(REAL, usecols=["device"]).device)


@pytest.mark.skipif(not _real_has_cm4(), reason="real summary CSV missing or not regenerated with kube01 (CM4)")
def test_smoke_real_csv(tmp_path):
    fig.main(["--out-dir", str(tmp_path)])
    d = tmp_path / fig.NAME
    assert (d / f"{fig.NAME}.pdf").stat().st_size > 0 and (d / f"{fig.NAME}.png").stat().st_size > 0
    data = pd.read_csv(d / f"{fig.NAME}_data.csv")
    assert len(data) == 28 and data.has_bars.sum() == 6
