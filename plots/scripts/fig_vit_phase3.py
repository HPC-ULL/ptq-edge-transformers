"""Paper Section IV-A: ViT top-1 accuracy vs bit-width, baseline vs selected
calibration (Phase 3), one panel per model.

Reads results/raw/vit_accuracy_offline.csv and vit_accuracy_fp32_reference.csv
(read-only) and writes plots/out/vit_phase3_bitwidth/ with the PDF, a 300 dpi
PNG preview and the CSV of the plotted values.

    python plots/scripts/fig_vit_phase3.py
    python plots/scripts/fig_vit_phase3.py --out-dir /some/dir
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
import pandas as pd
from matplotlib.lines import Line2D

from plots import paper_labels as L
from plots.common import OUT, RAW, load_csv, save_figure
from plots.paper_style import DOUBLE_COLUMN_IN, apply_style

NAME = "vit_phase3_bitwidth"
BITS = ("int8", "int6", "int4")
OFFSET = {"baseline": -0.08, "selected": 0.08, "best_single": 0.08}
MARKER = {"baseline": "o", "selected": "s", "best_single": "s"}
N_REPEATS = 3
TOL = 0.01
# Phase 2 level the code must select as best single factor (spec check).
EXPECTED_BEST_LEVEL = {"deit_tiny": "mse", "mobilevit": "percentile", "efficientvit_b1": "percentile"}

# Sanity values given in the figure spec: (model, series, bitwidth) -> mean.
EXPECTED = {
    ("deit_tiny", "baseline", "int8"): 72.90,
    ("deit_tiny", "selected", "int6"): 70.32,
    ("deit_tiny", "selected", "int4"): 34.33,
    ("mobilevit", "baseline", "int8"): 1.88,
    ("mobilevit", "selected", "int6"): 22.07,
    ("efficientvit_b1", "baseline", "int8"): 0.13,
    ("deit_tiny", "best_single", "int8"): 73.43,
    ("mobilevit", "best_single", "int8"): 68.38,
    ("efficientvit_b1", "best_single", "int8"): 8.15,
    ("deit_tiny", "fp32", "fp32"): 73.50,
    ("mobilevit", "fp32", "fp32"): 79.35,
    ("efficientvit_b1", "fp32", "fp32"): 81.20,
}


def _select(df: pd.DataFrame, series: str, bit: str) -> pd.DataFrame:
    if series == "baseline" and bit == "int8":
        m = (df.phase == "phase1") & (df.factor_varied == "baseline")
    elif series == "baseline":
        m = (df.phase == "phase3") & (df.factor_level == "baseline")
    else:
        m = (df.phase == "phase3") & (df.factor_level == "combined")
    return df[m & (df.bitwidth == bit)]


def _best_single(d: pd.DataFrame, model: str) -> dict:
    """Phase 2 level (one run_id) with the highest INT8 mean top-1."""
    d = d[(d.phase == "phase2") & (d.bitwidth == "int8")]
    cands = []
    for _, g in d.groupby("run_id"):
        reps = sorted(g.repeat_id.astype(int))
        assert reps == list(range(1, N_REPEATS + 1)), (
            f"{model}/phase2/{g.run_id.iloc[0]}: repeat_id {reps}, expected 1..{N_REPEATS}")
        cands.append((g.accuracy_value.mean(), g))
    cands.sort(key=lambda c: c[0], reverse=True)
    assert len(cands) > 1 and cands[0][0] != cands[1][0], f"{model}: tie for best single factor"
    g = cands[0][1]
    v = g.accuracy_value
    return dict(model=model, series="best_single", bitwidth="int8", mean=v.mean(),
                std=v.std(ddof=1), n=len(v),
                level=f"{g.factor_varied.iloc[0]}={g.factor_level.iloc[0]}")


def build_table(acc: pd.DataFrame, fp32: pd.DataFrame) -> pd.DataFrame:
    acc = acc[acc.accuracy_metric_name == "top1_accuracy"]
    # factor_level "fp32" = the same 2000-image subset as the quantized rows;
    # "fp32_full50000" (full validation set) is not comparable and is excluded.
    fp32 = fp32[(fp32.accuracy_metric_name == "top1_accuracy") & (fp32.factor_level == "fp32")]
    rows = []
    for model in L.VIT_MODEL_ORDER:
        d = acc[acc.model_name == model]
        for series, bits in (("baseline", BITS), ("selected", ("int6", "int4"))):
            for bit in bits:
                g = _select(d, series, bit)
                reps = sorted(g.repeat_id.astype(int))
                assert reps == list(range(1, N_REPEATS + 1)), (
                    f"{model}/{series}/{bit}: repeat_id {reps}, expected 1..{N_REPEATS}")
                v = g.accuracy_value
                rows.append(dict(model=model, series=series, bitwidth=bit,
                                 mean=v.mean(), std=v.std(ddof=1), n=len(v)))
        rows.append(_best_single(d, model))
        r = fp32[fp32.model_name == model]
        assert len(r) == 1, f"{model}: {len(r)} FP32 reference rows, expected 1"
        rows.append(dict(model=model, series="fp32", bitwidth="fp32",
                         mean=float(r.accuracy_value.iloc[0]), std=float("nan"), n=1))
    return pd.DataFrame(rows)


def sanity_check(t: pd.DataFrame) -> None:
    for model, want in EXPECTED_BEST_LEVEL.items():
        got = t[(t.model == model) & (t.series == "best_single")]["level"].iloc[0]
        if not got.endswith(f"={want}"):
            sys.exit(f"SANITY CHECK FAILED: {model}: best single factor is {got}, "
                     f"expected level {want}. Figure not written.")
    for (model, series, bit), want in EXPECTED.items():
        got = t[(t.model == model) & (t.series == series) & (t.bitwidth == bit)]["mean"].iloc[0]
        if abs(got - want) > TOL:
            sys.exit(f"SANITY CHECK FAILED: {model}/{series}/{bit}: got {got:.4f}, "
                     f"expected {want:.2f} (tol {TOL}). Figure not written.")


def plot(t: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(DOUBLE_COLUMN_IN, 2.1))
    for ax, model in zip(axes, L.VIT_MODEL_ORDER):
        d = t[t.model == model]
        for series in ("baseline", "selected"):
            s = d[d.series == series].set_index("bitwidth")
            xs = [BITS.index(b) + OFFSET[series] for b in BITS if b in s.index]
            ys = [s.loc[b, "mean"] for b in BITS if b in s.index]
            es = [s.loc[b, "std"] for b in BITS if b in s.index]
            ax.errorbar(xs, ys, yerr=es, color=L.COLORS[series], marker=MARKER[series],
                        ms=4, lw=1, capsize=2, capthick=0.8, elinewidth=0.8)
        b = d[d.series == "best_single"].iloc[0]
        # bars first, white-filled marker on top so the bar does not show through it
        ax.errorbar([OFFSET["best_single"]], [b["mean"]], yerr=[b["std"]], color=L.COLORS["selected"],
                    fmt="none", capsize=2, capthick=0.8, elinewidth=0.8, zorder=2)
        ax.plot([OFFSET["best_single"]], [b["mean"]], color=L.COLORS["selected"],
                marker=MARKER["best_single"], mfc="white", ls="none", ms=4, mew=1, zorder=3)
        ax.axhline(d[d.series == "fp32"]["mean"].iloc[0], color=L.COLORS["fp32"],
                   ls="--", lw=0.8)
        ax.set_title(L.MODELS[model])
        ax.set_xticks(range(len(BITS)), [L.BITWIDTHS_VIT[b] for b in BITS])
        ax.set_xlim(-0.5, len(BITS) - 0.5)
        ax.set_ylim(-2, 85)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.85", lw=0.5)
        ax.xaxis.grid(False)
    axes[0].set_ylabel(L.AXIS_LABELS["top1"])
    axes[1].set_xlabel(L.AXIS_LABELS["bitwidth"])
    handles = [
        Line2D([], [], color=L.COLORS["baseline"], marker="o", ms=4, lw=1),
        Line2D([], [], color=L.COLORS["selected"], marker="s", ms=4, lw=1),
        Line2D([], [], color=L.COLORS["selected"], marker="s", mfc="white", mew=1, ms=4, ls="none"),
        Line2D([], [], color=L.COLORS["fp32"], ls="--", lw=0.8),
    ]
    labels = [L.SERIES_LABELS[k] for k in ("baseline", "selected", "best_single", "fp32")]
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.2, top=0.76, wspace=0.06)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, frameon=False)
    return fig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--accuracy-csv", default=RAW / "vit_accuracy_offline.csv")
    ap.add_argument("--fp32-csv", default=RAW / "vit_accuracy_fp32_reference.csv")
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)

    t = build_table(load_csv(a.accuracy_csv), load_csv(a.fp32_csv))
    sanity_check(t)
    font = apply_style()
    fig = plot(t)
    d = save_figure(fig, NAME, a.out_dir)

    t.assign(model_label=t.model.map(L.MODELS)).to_csv(
        d / f"{NAME}_data.csv", index=False, float_format="%.4f")
    show = t.assign(model=t.model.map(L.MODELS), bitwidth=t.bitwidth.map(L.BITWIDTHS_VIT)).fillna({"level": ""})
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"font: {font}\nwrote: {d}")


if __name__ == "__main__":
    main()
