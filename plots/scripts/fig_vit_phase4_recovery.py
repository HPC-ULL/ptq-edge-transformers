"""Paper Section IV-B: fraction of the accuracy gap recovered by keeping the K
layers with highest outlier ratio R in float, vs K random layers (Phase 4,
level 2), with the replicas on the other two calibration samples.

Reads (read-only) results/investigations/vit/phase4/level2/evaluations.csv
(repetition 1), .../replication/evaluations.csv (repetitions 2-3) and
results/raw/vit_accuracy_fp32_reference.csv. Writes
plots/out/vit_phase4_recovery/ with the PDF, a 300 dpi PNG and the CSV of the
plotted values.

    python plots/scripts/fig_vit_phase4_recovery.py
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
from matplotlib.ticker import NullLocator

from plots import paper_labels as L
from plots.common import OUT, RAW, ROOT, load_csv, save_figure
from plots.paper_style import DOUBLE_COLUMN_IN, apply_style

NAME = "vit_phase4_recovery"
LEVEL2 = ROOT / "results" / "investigations" / "vit" / "phase4" / "level2"
CURVES = ("targeted_r", "random_s1", "random_s2", "random_s3")  # targeted_err excluded (supplementary)
BITWIDTH = {"deit_tiny": "int6", "mobilevit": "int8", "efficientvit_b1": "int8"}
STOP_PCT = 90
REP_SHIFT = 1.06
TOL = 0.1

# Sanity values given in the figure spec: (model, curve, K, repeat) -> recovered %.
EXPECTED = {
    ("mobilevit", "targeted_r", 2, 1): 93.9,
    ("mobilevit", "targeted_r", 2, 2): 93.8,
    ("mobilevit", "targeted_r", 2, 3): 91.4,
    ("efficientvit_b1", "targeted_r", 8, 1): 49.6,
    ("efficientvit_b1", "targeted_r", 8, 2): 42.1,
    ("efficientvit_b1", "targeted_r", 8, 3): 35.8,
    ("deit_tiny", "targeted_r", 8, 1): 21.1,
    ("deit_tiny", "random_s3", 8, 1): 33.7,
}


def build_table(ev: pd.DataFrame, rep: pd.DataFrame, fp32: pd.DataFrame) -> pd.DataFrame:
    for m, bit in BITWIDTH.items():
        got = set(ev[ev.model == m].bitwidth)
        assert got == {bit}, f"{m}: bitwidth {got}, expected {{'{bit}'}}"
    fp32 = fp32[(fp32.accuracy_metric_name == "top1_accuracy") & (fp32.factor_level == "fp32")]
    ev = ev.assign(repeat=1)
    allr = pd.concat([ev, rep], ignore_index=True)
    rows = []
    for m in L.VIT_MODEL_ORDER:
        ref = fp32[fp32.model_name == m]
        assert len(ref) == 1, f"{m}: {len(ref)} FP32 reference rows, expected 1"
        top1_fp32 = float(ref.accuracy_value.iloc[0])
        d = allr[allr.model == m]
        for r in sorted(d["repeat"].unique()):
            dr = d[d["repeat"] == r]
            k0 = dr[dr.curve == "k0"]
            assert len(k0) == 1, f"{m} rep {r}: {len(k0)} k0 rows, expected 1"
            top1_k0 = float(k0.top1.iloc[0])
            for _, x in dr[dr.curve.isin(CURVES)].iterrows():
                rows.append(dict(model=m, bitwidth=BITWIDTH[m], repeat=int(r), curve=x.curve,
                                 K=int(x.K), top1=x.top1, top1_k0=top1_k0, top1_fp32=top1_fp32,
                                 recovered_pct=100 * (x.top1 - top1_k0) / (top1_fp32 - top1_k0)))
    return pd.DataFrame(rows)


def sanity_check(t: pd.DataFrame) -> None:
    for (m, c, k, r), want in EXPECTED.items():
        g = t[(t.model == m) & (t.curve == c) & (t.K == k) & (t["repeat"] == r)]
        if len(g) != 1 or abs(g.recovered_pct.iloc[0] - want) > TOL:
            got = g.recovered_pct.iloc[0] if len(g) == 1 else f"{len(g)} rows"
            sys.exit(f"SANITY CHECK FAILED: {m}/{c}/K={k}/rep {r}: got {got}, expected "
                     f"{want} (tol {TOL}). Figure not written.")


def plot(t: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(DOUBLE_COLUMN_IN, 2.1))
    blue, grey, light = L.COLORS["selected"], L.COLORS["random"], L.COLORS["random_light"]
    for ax, m in zip(axes, L.VIT_MODEL_ORDER):
        d = t[t.model == m]
        r1, rr = d[d["repeat"] == 1], d[d["repeat"] > 1]
        for c in CURVES[1:]:
            g = r1[r1.curve == c].sort_values("K")
            ax.plot(g.K, g.recovered_pct, color=grey, lw=0.8, marker="x", ms=3, mew=0.8, zorder=2)
        g = r1[r1.curve == "targeted_r"].sort_values("K")
        ax.plot(g.K, g.recovered_pct, color=blue, lw=1, marker="o", ms=4, zorder=3)
        g = rr[rr.curve == "targeted_r"]
        ax.plot(g.K / REP_SHIFT, g.recovered_pct, ls="none", color=blue, marker="o", mfc="white",
                ms=4, mew=1, zorder=4)
        g = rr[rr.curve != "targeted_r"]
        ax.plot(g.K * REP_SHIFT, g.recovered_pct, ls="none", color=light, marker="x", ms=4,
                mew=1, zorder=2)
        ax.axhline(STOP_PCT, color=L.COLORS["stop"], ls=":", lw=0.8, zorder=1)
        ax.set_title(f"{L.MODELS[m]} ({L.BITWIDTHS_VIT[BITWIDTH[m]]})")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.8, 10)
        ax.set_ylim(-5, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="0.85", lw=0.5)
        ax.xaxis.grid(False)
    axes[0].set_ylabel(L.AXIS_LABELS["recovered_gap"])
    axes[1].set_xlabel(L.AXIS_LABELS["k_float"])
    handles = [
        Line2D([], [], color=blue, marker="o", ms=4, lw=1),
        Line2D([], [], color=grey, marker="x", ms=3, mew=0.8, lw=0.8),
        Line2D([], [], color=blue, marker="o", mfc="white", mew=1, ms=4, ls="none"),
        Line2D([], [], color=light, marker="x", mew=1, ms=4, ls="none"),
        Line2D([], [], color=L.COLORS["stop"], ls=":", lw=0.8),
    ]
    labels = [L.RECOVERY_LABELS[k] for k in ("targeted", "random", "targeted_rep", "random_rep", "stop")]
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.2, top=0.76, wspace=0.06)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5,
               frameon=False, columnspacing=1.2, handletextpad=0.5)
    return fig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--evaluations-csv", default=LEVEL2 / "evaluations.csv")
    ap.add_argument("--replication-csv", default=LEVEL2 / "replication" / "evaluations.csv")
    ap.add_argument("--fp32-csv", default=RAW / "vit_accuracy_fp32_reference.csv")
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)

    t = build_table(load_csv(a.evaluations_csv), load_csv(a.replication_csv), load_csv(a.fp32_csv))
    sanity_check(t)
    font = apply_style()
    fig = plot(t)
    d = save_figure(fig, NAME, a.out_dir)

    t.assign(model_label=t.model.map(L.MODELS)).to_csv(d / f"{NAME}_data.csv", index=False,
                                                       float_format="%.4f")
    show = t.assign(model=t.model.map(L.MODELS), bitwidth=t.bitwidth.map(L.BITWIDTHS_VIT))
    print(show[["model", "bitwidth", "repeat", "curve", "K", "top1", "recovered_pct"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"font: {font}\nwrote: {d}")


if __name__ == "__main__":
    main()
