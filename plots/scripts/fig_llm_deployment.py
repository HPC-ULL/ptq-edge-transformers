"""Paper Section IV-C: LLM energy per request and decode speed of the four
non-imatrix quantized formats vs F16, on every device (two heatmaps).

Both ratios are > 1 in favour of the quantized format. Colour: sequential
(Blues), logarithmic from 1 to the data maximum, shared by both heatmaps; cells within
[0.9, 1.1] are white with a diagonal hatch; cells not measured or excluded are grey and empty.

Reads plots/out/llm_deployment_summary.csv (read-only). Writes
plots/out/llm_deployment_heatmap/ with the PDF, a 300 dpi PNG and the CSV of
the plotted values.

    python plots/scripts/fig_llm_deployment.py
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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.patches import Patch, Rectangle

from plots import paper_labels as L
from plots.common import OUT, load_csv, save_figure
from plots.paper_style import DOUBLE_COLUMN_IN, apply_style

NAME = "llm_deployment_heatmap"
METRICS = (("energy", "r_energy"), ("decode", "r_tg"))
NEUTRAL = (0.9, 1.1)
GROUP_GAP = 0.25
HEADER_H = 0.95   # row units reserved above each model for its name
N_VALID = 108
MIN_ENERGY, TOL = 1.30, 0.005
CMAP = "Blues"  # sequential, 1 -> light, data maximum -> dark
CELL_FS = 6.5
HATCH = "////"   # "within 0.9 to 1.1" cells: white with a fine grey diagonal hatch


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, config, device) of the 3 x 4 x 10 grid; NaN ratios where not
    measured or excluded."""
    q = df[df.config.isin(L.LLM_DEPLOY_FORMATS)]          # filter by config: no imatrix, no phase3_combined_*
    assert not q.duplicated(["device", "model", "config"]).any(), "more than one row per device/model/config"
    assert set(q.device) == set(L.LLM_DEPLOY_DEVICES), f"devices {sorted(set(q.device))}"
    q = q.assign(valid=q.reference == "F16")               # "F16 (excluida)" rows are not valid
    q.loc[~q.valid, ["r_energy", "r_tg"]] = np.nan
    # full grid: combinations with no row at all (not measured) become empty cells
    grid = pd.MultiIndex.from_product([L.LLM_MODEL_ORDER, L.LLM_DEPLOY_FORMATS, L.LLM_DEPLOY_DEVICES],
                                      names=["model", "config", "device"])
    assert set(q.model) <= set(L.LLM_MODEL_ORDER), f"models {sorted(set(q.model))}"
    q = q.set_index(["model", "config", "device"]).reindex(grid)
    return q[["reference", "r_energy", "r_tg"]].reset_index()


def checks(t: pd.DataFrame) -> None:
    for _, col in METRICS:
        n = int(t[col].notna().sum())
        assert n == N_VALID, f"{col}: {n} cells with a value, expected {N_VALID}"
    low = float(t[["r_energy", "r_tg"]].stack().min())
    assert low >= NEUTRAL[0], f"ratio {low:.3f} < {NEUTRAL[0]}: the 1-to-max colour scale cannot show it"
    lo = t.r_energy.min()
    assert abs(lo - MIN_ENERGY) <= TOL, f"minimum energy ratio {lo:.4f}, expected {MIN_ENERGY} (tol {TOL})"


def _luminance(rgb):
    """WCAG relative luminance of an sRGB colour."""
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(rgb, other):
    a, b = sorted((_luminance(rgb), _luminance(other)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def _text_color(rgb):
    """White or black, whichever contrasts more with the cell colour."""
    return "white" if _contrast(rgb, (1, 1, 1)) > _contrast(rgb, (0, 0, 0)) else "black"


def _cell_color(v, norm, cmap):
    if np.isnan(v):
        return to_rgb(L.COLORS["cell_missing"])
    if NEUTRAL[0] <= v <= NEUTRAL[1]:
        return (1.0, 1.0, 1.0)             # white; the hatch is drawn over it in plot()
    return cmap(norm(v))[:3]


def _row_layout():
    """[(model, config, y_top)] and the y_top of each model's header line."""
    rows, headers, y = [], {}, 0.0
    for m in L.LLM_MODEL_ORDER:
        headers[m] = y
        y += HEADER_H
        for c in L.LLM_DEPLOY_FORMATS:
            rows.append((m, c, y))
            y += 1.0
        y += GROUP_GAP
    return rows, headers, y - GROUP_GAP


def plot(t: pd.DataFrame):
    devices = list(L.LLM_DEPLOY_DEVICES)
    plt.rcParams["hatch.linewidth"] = 0.5
    rows, headers, y_end = _row_layout()
    vmax = float(t[["r_energy", "r_tg"]].stack().max())
    norm, cmap = LogNorm(1, vmax), plt.get_cmap(CMAP)           # log scale from 1 to the data maximum
    lut = t.set_index(["model", "config", "device"])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_IN, 3.2))
    for k, (ax, (key, col)) in enumerate(zip(axes, METRICS)):
        for m, c, y in rows:
            for j, dev in enumerate(devices):
                v = lut.loc[(m, c, dev), col]
                rgb = _cell_color(v, norm, cmap)
                ax.add_patch(Rectangle((j, y), 1, 1, facecolor=rgb, edgecolor="white", lw=0.6))
                if not np.isnan(v) and NEUTRAL[0] <= v <= NEUTRAL[1]:
                    ax.add_patch(Rectangle((j, y), 1, 1, fill=False, hatch=HATCH, lw=0,
                                           edgecolor=L.COLORS["cell_neutral_hatch"]))
                if not np.isnan(v):
                    txt = ax.text(j + 0.5, y + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=CELL_FS,
                                  color=_text_color(rgb))
                    if NEUTRAL[0] <= v <= NEUTRAL[1]:   # thin white halo so the hatch does not cut the digits
                        txt.set_path_effects([pe.withStroke(linewidth=1.6, foreground="white")])
        ax.set_xlim(0, len(devices))
        ax.set_ylim(y_end, 0)
        ax.set_xticks(np.arange(len(devices)) + 0.5, [L.LLM_DEPLOY_DEVICES[d] for d in devices],
                      rotation=45, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="both", length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(L.LLM_DEPLOY_LABELS[key])
        if k == 0:
            ax.set_yticks([y + 0.5 for _, _, y in rows], [L.LLM_DEPLOY_FORMATS[c] for _, c, _ in rows])
            for m, yh in headers.items():
                ax.text(0, yh + HEADER_H * 0.55, L.MODELS[m], ha="left", va="center", fontweight="bold",
                        clip_on=False)
        else:
            ax.set_yticks([])

    fig.subplots_adjust(left=0.082, right=0.912, bottom=0.3, top=0.93, wspace=0.05)
    cax = fig.add_axes([0.93, 0.34, 0.012, 0.55])
    cb = fig.colorbar(ScalarMappable(norm, cmap), cax=cax, ticks=[1, 2, 4, 6])
    cb.ax.set_yticklabels(["1", "2", "4", "6"])
    cb.ax.minorticks_off()
    cb.outline.set_linewidth(0.4)
    cb.set_label(L.LLM_DEPLOY_LABELS["colorbar"], fontsize=8)
    fig.legend([Patch(facecolor="white", edgecolor=L.COLORS["cell_neutral_hatch"], hatch=HATCH, lw=0.4),
                Patch(facecolor=L.COLORS["cell_missing"], edgecolor="0.6", lw=0.4)],
               [L.LLM_DEPLOY_LABELS["neutral"], L.LLM_DEPLOY_LABELS["missing"]],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    return fig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--summary-csv", default=OUT / "llm_deployment_summary.csv")
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)

    t = build_table(load_csv(a.summary_csv))
    checks(t)
    font = apply_style()
    fig = plot(t)
    d = save_figure(fig, NAME, a.out_dir)

    mo = {k: i for i, k in enumerate(L.LLM_MODEL_ORDER)}
    fo = {k: i for i, k in enumerate(L.LLM_DEPLOY_FORMATS)}
    do = {k: i for i, k in enumerate(L.LLM_DEPLOY_DEVICES)}
    t = t.assign(_k=list(zip(t.model.map(mo), t.config.map(fo), t.device.map(do)))).sort_values("_k").drop(columns="_k")
    t.assign(model_label=t.model.map(L.MODELS), format_label=t.config.map(L.LLM_DEPLOY_FORMATS),
             device_label=t.device.map(L.LLM_DEPLOY_DEVICES)).to_csv(d / f"{NAME}_data.csv", index=False,
                                                                     float_format="%.4f")
    show = t.assign(model=t.model.map(L.MODELS), config=t.config.map(L.LLM_DEPLOY_FORMATS),
                    device=t.device.map(L.LLM_DEPLOY_DEVICES))
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}", na_rep="-"))
    print(f"cells with value: energy {t.r_energy.notna().sum()}, decode {t.r_tg.notna().sum()}; "
          f"empty: {t.r_energy.isna().sum()}")
    print(f"font: {font}\nwrote: {d}")


if __name__ == "__main__":
    main()
