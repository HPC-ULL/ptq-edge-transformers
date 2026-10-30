"""Paper Introduction: why calibration decides accuracy. Histogram of the input
activations of the MobileViT-S layer with the highest outlier ratio R, with the
symmetric INT8 range that MinMax and the 99.9 percentile give, and the step
each implies.

Reads plots/out/calibration_concept_data.npz (written on the cluster by
capture_calibration_concept.py). The histogram is drawn from the stored
histogram; ranges, steps and the clipped fraction are the exact values the
capture computed on the full tensor. The stored sample is only used for checks.
Only the clipped tail with data (x > +range) is shaded. The console also reports the
minimum value and the exact fraction of negative values. Writes
plots/out/calibration_concept/ with the PDF, a 300 dpi PNG and a CSV with the numbers.

    python plots/scripts/fig_calibration_concept.py
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

from plots import paper_labels as L
from plots.common import OUT, save_figure
from plots.paper_style import apply_style

NAME = "calibration_concept"
SUB_TOL = 0.10   # the sample's percentile must be within 10 % of the exact one (sample noise ~ 3 %)


def load(path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def checks(d: dict) -> None:
    n = int(d["n_values"])
    assert int(d["hist_counts"].sum()) == n, "histogram does not hold every value"
    assert abs(d["hist_edges"][0] + d["max_abs"]) < 1e-6 * d["max_abs"] and \
        abs(d["hist_edges"][-1] - d["max_abs"]) < 1e-6 * d["max_abs"], "histogram does not span +-max|x|"
    assert len(d["hist_counts"]) % 2 == 0, "0 should be a bin edge"
    assert d["range_minmax"] >= d["range_percentile"] > 0
    assert abs(d["step_minmax"] - d["range_minmax"] / 127) < 1e-9 * d["range_minmax"]
    assert abs(d["step_percentile"] - d["range_percentile"] / 127) < 1e-9 * d["range_percentile"]
    sub = np.abs(d["subsample"])
    assert len(sub) < n and sub.max() <= d["max_abs"] * (1 + 1e-6), "sample inconsistent with the full tensor"
    est = float(np.percentile(sub, d["percentile_q"]))
    assert abs(est - d["percentile_exact_ref"]) <= SUB_TOL * d["percentile_exact_ref"], (
        f"sample percentile {est:.4g} vs exact {d['percentile_exact_ref']:.4g}")


def negative_stats(d: dict) -> dict:
    """Exact minimum and fraction of negative values, from the stored full-tensor data.

    0 is a bin edge (even number of bins, symmetric edges), so the counts of the
    bins below it are exactly the values < 0; the sample is only a cross-check."""
    counts, edges, n = d["hist_counts"], d["hist_edges"], int(d["n_values"])
    half = len(counts) // 2
    assert abs(edges[half]) <= 1e-12 * d["max_abs"], "the middle bin edge is not 0"
    frac = float(counts[:half].sum()) / n
    sub = float((d["subsample"] < 0).mean())
    assert abs(frac - sub) < 0.01, f"negative fraction: histogram {frac:.4f} vs sample {sub:.4f}"
    return dict(vmin=float(d["vmin"]), vmin_over_max_abs=float(d["vmin"]) / float(d["max_abs"]),
                vmin_in_steps_minmax=-float(d["vmin"]) / float(d["step_minmax"]),
                vmin_in_steps_percentile=-float(d["vmin"]) / float(d["step_percentile"]),
                negative_fraction=frac, negative_fraction_sample=sub)


def plot(d: dict):
    edges, counts = d["hist_edges"], d["hist_counts"]
    m, p = float(d["range_minmax"]), float(d["range_percentile"])
    c_mm, c_p, c_h = L.COLORS["baseline"], L.COLORS["selected"], L.COLORS["hist"]
    fig, ax = plt.subplots(figsize=(3.5, 1.75))
    fig.subplots_adjust(left=0.15, right=0.995, top=0.78, bottom=0.2)
    ax.axvspan(p, edges[-1], color=L.COLORS["band"], lw=0, zorder=0)   # the clipped tail that has data (x > +range)
    ax.stairs(np.maximum(counts, 0.5), edges, baseline=0.5, fill=True, facecolor=c_h, edgecolor=c_h,
              lw=0.4, zorder=1)   # thin edge so single-count bins in the tails stay visible
    ax.axvline(-m, color=c_mm, ls="--", lw=1.0, zorder=3)
    ax.axvline(m, color=c_mm, ls="--", lw=1.0, zorder=3)
    ax.axvline(-p, color=c_p, ls="--", lw=1.0, zorder=3)
    ax.axvline(p, color=c_p, ls="--", lw=1.0, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.5, counts.max() * 3)
    pad = 0.04 * edges[-1]
    ax.set_xlim(edges[0] - pad, edges[-1] + pad)
    ax.set_xlabel(L.CONCEPT_LABELS["xlabel"])
    ax.set_ylabel(L.CONCEPT_LABELS["ylabel"])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="0.85", lw=0.5)
    labels = [f"{L.CALIBRATION['minmax']}: step = {d['step_minmax']:.3g}",
              f"{L.CALIBRATION['percentile']}: step = {d['step_percentile']:.3g}, "
              f"{100 * d['clipped_fraction']:.2f}% clipped"]
    handles = [Line2D([], [], color=c_mm, ls="--", lw=1.0), Line2D([], [], color=c_p, ls="--", lw=1.0)]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False,
              handlelength=2.2)
    return fig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--npz", default=OUT / "calibration_concept_data.npz")
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)

    d = load(a.npz)
    d["percentile_exact_ref"] = d["p_linear"]
    checks(d)
    neg = negative_stats(d)
    font = apply_style()
    fig = plot(d)
    folder = save_figure(fig, NAME, a.out_dir)

    rows = [(k, d[k]) for k in ("layer", "n_values", "max_abs", "percentile_q", "range_minmax",
                                "range_minmax_source", "range_percentile", "range_percentile_source",
                                "step_minmax", "step_percentile", "step_ratio", "clipped_fraction",
                                "brevitas_range_minmax", "brevitas_range_percentile", "p_linear", "p_nearest",
                                "calib_seed", "calib_set_size", "images_sha1", "versions")]
    rows += list(neg.items())
    pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(folder / f"{NAME}_data.csv", index=False)
    print(f"layer: {d['layer']}\nvalues: {int(d['n_values'])}")
    print(f"MinMax:     range {d['range_minmax']:.6g}  step {d['step_minmax']:.6g}   [{d['range_minmax_source']}]")
    print(f"Percentile: range {d['range_percentile']:.6g}  step {d['step_percentile']:.6g}   "
          f"[{d['range_percentile_source']}]")
    print(f"step ratio MinMax / Percentile: {d['step_ratio']:.3f}; clipped fraction: "
          f"{100 * d['clipped_fraction']:.4f}% (percentile {d['percentile_q']})")
    print(f"minimum value: {neg['vmin']:.6g} ({100 * abs(neg['vmin_over_max_abs']):.2f}% of max|x|; "
          f"{neg['vmin_in_steps_minmax']:.2f} MinMax steps, {neg['vmin_in_steps_percentile']:.2f} percentile steps)")
    print(f"negative values: {100 * neg['negative_fraction']:.4f}% (exact, from the histogram; "
          f"sample: {100 * neg['negative_fraction_sample']:.2f}%)")
    print(f"font: {font}\nwrote: {folder}")


if __name__ == "__main__":
    main()
