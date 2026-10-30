"""Paper Section IV-C: ViT INT8 vs FP32 on CPU, per device and power mode.
Top row: latency speedup; bottom row: energy ratio (both FP32 / INT8, > 1 favours
INT8). Integer-kernel INT8 (deployment core) vs floating-point fallback INT8
(ablation regime). CPU only.

Reads plots/out/vit_deployment_summary.csv (read-only). Writes
plots/out/vit_deployment_cpu/ with the PDF, a 300 dpi PNG and the CSV of the
plotted values.

    python plots/scripts/fig_vit_deployment.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plots import paper_labels as L
from plots.common import OUT, load_csv, save_figure
from plots.paper_style import DOUBLE_COLUMN_IN, apply_style

NAME = "vit_deployment_cpu"
CONFIGS = ("core", "fallback")
BAND = (0.85, 1.15)
BAR_W = 0.38
TOL = 0.005
N_SESSIONS = 3

# Sanity values given in the figure spec: (model, config, device, column) -> value.
EXPECTED = {
    ("deit_tiny", "core", "jetson_cpu_15w", "speedup"): 1.528,
    ("deit_tiny", "core", "jetson_cpu_15w", "energy_ratio"): 1.549,
    ("deit_tiny", "core", "visionfive_2", "speedup"): 0.670,
    ("mobilevit_s", "core", "rk1-02", "speedup"): 1.01,
    ("deit_tiny", "fallback", "jetson_cpu_30w", "speedup"): 0.65,
}


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    # filter by config, never by run_id (the DeiT-Tiny core run_id does not say "mse")
    t = df[(df.backend == "cpu") & (df.reference == "fp32") & df.config.isin(CONFIGS)].copy()
    assert not t.duplicated(["model", "device", "config"]).any(), "more than one row per model/device/config"
    assert set(t.device) == set(L.DEPLOY_DEVICE_ORDER), (
        f"devices {sorted(set(t.device))}, expected {sorted(L.DEPLOY_DEVICE_ORDER)}")
    assert set(t.model) == set(L.DEPLOY_MODEL_ORDER), f"models {sorted(set(t.model))}"
    assert len(t) == len(L.DEPLOY_DEVICE_ORDER) * len(L.DEPLOY_MODEL_ORDER) * len(CONFIGS), (
        f"{len(t)} rows, expected one per model, device and config")
    t["has_bars"] = (t.n_sessions_ref == N_SESSIONS) & (t.n_sessions_x == N_SESSIONS)
    bars = t[t.has_bars]
    assert not bars[["sess_lat_lo", "sess_lat_hi", "sess_ene_lo", "sess_ene_hi"]].isna().any().any(), (
        "n_sessions == 3 but session bounds missing")
    for v, lo, hi in (("speedup", "sess_lat_lo", "sess_lat_hi"), ("energy_ratio", "sess_ene_lo", "sess_ene_hi")):
        assert ((bars[lo] <= bars[v]) & (bars[v] <= bars[hi])).all(), f"{v} outside its session range"
    t.loc[~t.has_bars, ["sess_lat_lo", "sess_lat_hi", "sess_ene_lo", "sess_ene_hi"]] = np.nan
    return t[["model", "device", "config", "speedup", "energy_ratio", "sess_lat_lo", "sess_lat_hi",
              "sess_ene_lo", "sess_ene_hi", "n_sessions_ref", "n_sessions_x", "has_bars"]]


def sanity_check(t: pd.DataFrame) -> None:
    for (m, c, dev, col), want in EXPECTED.items():
        g = t[(t.model == m) & (t.config == c) & (t.device == dev)]
        got = g[col].iloc[0]
        if abs(got - want) > TOL:
            sys.exit(f"SANITY CHECK FAILED: {m}/{c}/{dev}/{col}: got {got:.4f}, expected {want} "
                     f"(tol {TOL}). Figure not written.")


def plot(t: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey="row", figsize=(DOUBLE_COLUMN_IN, 3.6))
    devices = L.DEPLOY_DEVICE_ORDER
    rows = (("speedup", "sess_lat_lo", "sess_lat_hi", "speedup"),
            ("energy_ratio", "sess_ene_lo", "sess_ene_hi", "energy_ratio"))
    for j, m in enumerate(L.DEPLOY_MODEL_ORDER):
        for i, (col, lo, hi, ylab) in enumerate(rows):
            ax = axes[i, j]
            ax.axhspan(*BAND, color=L.COLORS["band"], lw=0, zorder=0)
            ax.axhline(1, color="black", lw=0.6, zorder=1)
            for k, cfg in enumerate(CONFIGS):
                g = t[(t.model == m) & (t.config == cfg)].set_index("device").loc[list(devices)]
                x = np.arange(len(devices)) + (k - 0.5) * BAR_W
                ax.bar(x, g[col], BAR_W, color=L.COLORS[cfg], zorder=2)
                b = g[g.has_bars]
                if len(b):
                    xb = x[[devices.index(d) for d in b.index]]
                    ax.errorbar(xb, b[col], yerr=[b[col] - b[lo], b[hi] - b[col]], fmt="none",
                                ecolor="0.15", elinewidth=0.8, capsize=1.5, capthick=0.8, zorder=3)
            ax.set_ylim(0, 1.8)
            ax.set_yticks([0, 0.5, 1, 1.5], ["0", "0.5", "1.0", "1.5"])
            ax.set_xlim(-0.6, len(devices) - 0.4)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, color="0.85", lw=0.5)
            ax.xaxis.grid(False)
            if j == 0:
                ax.set_ylabel(L.DEPLOY_LABELS[ylab])
        axes[0, j].set_title(L.MODELS[m])
        axes[1, j].set_xticks(range(len(devices)), [L.DEVICE_TICK_LABELS[d] for d in devices])
    handles = [Patch(color=L.COLORS["core"]), Patch(color=L.COLORS["fallback"]),
               Patch(color=L.COLORS["band"])]
    labels = [L.DEPLOY_LABELS[k] for k in ("core", "fallback", "band")]
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.88, wspace=0.06, hspace=0.12)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    return fig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--summary-csv", default=OUT / "vit_deployment_summary.csv")
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)

    t = build_table(load_csv(a.summary_csv))
    sanity_check(t)
    font = apply_style()
    fig = plot(t)
    d = save_figure(fig, NAME, a.out_dir)

    order = {k: i for i, k in enumerate(L.DEPLOY_DEVICE_ORDER)}
    t = t.sort_values(["model", "device", "config"], key=lambda s: s.map(order) if s.name == "device" else s)
    t.assign(model_label=t.model.map(L.MODELS), device_label=t.device.map(L.DEVICES)).to_csv(
        d / f"{NAME}_data.csv", index=False, float_format="%.4f")
    show = t.assign(model=t.model.map(L.MODELS), device=t.device.map(L.DEVICES))
    print(show.drop(columns="has_bars").to_string(index=False, float_format=lambda x: f"{x:.3f}", na_rep=""))
    print(f"font: {font}\nwrote: {d}")


if __name__ == "__main__":
    main()
