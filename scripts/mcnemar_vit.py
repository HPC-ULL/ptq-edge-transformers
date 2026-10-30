"""McNemar contrasts of calibration_protocol_vit.md, Section 6.1 (ViT only).

Input: results/investigations/vit/per_image_predictions/{ablation,deployment}/<name>__rep<k>.csv
Test:  exact McNemar (two-sided binomial on the discordant pairs, p0 = 0.5) on top-1
       correct/incorrect, one test per repetition k (rep k of A against rep k of B).
Holm:  within each hypothesis family, alpha = 0.05, separately for each repetition.
       A difference is claimed only if significant after Holm in all 3 repetitions.
H4 is not tested. Standard library only.

  --list   build the pairs, check images/order match between A and B, print the plan; no test is computed
  --run    the above, then compute and write <out-dir>/mcnemar_vit.csv and mcnemar_vit_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "results/investigations/vit/per_image_predictions"
OUT_DIR = ROOT / "results/investigations/vit/mcnemar"
REPS = (1, 2, 3)
ALPHA = Fraction(5, 100)
ABL_MODELS = {"deit_tiny": "vit_deit_tiny", "mobilevit": "vit_mobilevit", "efficientvit_b1": "vit_efficientvit_b1"}


def _abl(model, suffix):
    return "ablation", f"{ABL_MODELS[model]}_{suffix}"


def _dep(name):
    return "deployment", name


def build_pairs():
    """(hypothesis, model, A, B, group_A, group_B) for every comparison of Section 6.1."""
    pairs = []

    def add(h, model, a, b):
        pairs.append((h, model, a[1], b[1], a[0], b[0]))

    base = "phase1_baseline_default_int8"
    for m in ABL_MODELS:
        add("H1", m, _abl(m, "phase2_granularity_per_channel_int8"), _abl(m, base))
    add("H1", "mobilevit", _dep("deploy_mobilevit_percentile_per_channel_symmetric"), _dep("deploy_mobilevit_percentile_per_tensor_symmetric"))
    add("H1", "mobilevit", _dep("deploy_mobilevit_percentile_per_channel_asymmetric"), _dep("deploy_mobilevit_percentile_per_tensor_asymmetric"))
    add("H1", "deit_tiny", _dep("deploy_deit_tiny_mse_per_channel_symmetric_gelu1"), _dep("deploy_deit_tiny_mse_per_tensor_symmetric_gelu1"))
    for m in ABL_MODELS:
        for alg in ("percentile", "mse"):
            add("H2", m, _abl(m, f"phase2_calibration_algorithm_{alg}_int8"), _abl(m, base))
        for bits in ("int6", "int4"):
            add("H2", m, _abl(m, f"phase3_combined_{bits}"), _abl(m, f"phase3_baseline_{bits}"))
    for m in ABL_MODELS:
        for n in (32, 512):
            add("H3", m, _abl(m, f"phase2_calib_set_size_{n}_int8"), _abl(m, base))  # the baseline is the 128-sample set
    add("RQ3b", "deit_tiny", _dep("deploy_deit_tiny_mse_per_tensor_symmetric_gelu1"), _dep("deploy_deit_tiny_minmax_per_tensor_symmetric_gelu1"))
    add("RQ3b", "mobilevit", _dep("deploy_mobilevit_percentile_per_channel_asymmetric"), _dep("deploy_mobilevit_minmax_per_tensor_symmetric"))
    return pairs


def csv_path(group, name, rep, pred_dir=PRED_DIR):
    return Path(pred_dir) / group / f"{name}__rep{rep}.csv"


def read_preds(path):
    """[(idx, filename, label, correct1)] in file order."""
    with open(path, newline="") as f:
        return [(r["idx"], r["filename"], r["label"], int(r["correct1"])) for r in csv.DictReader(f)]


def check_alignment(a, b):
    """Empty string if A and B list the same images with the same labels in the same order, else the reason."""
    if len(a) != len(b):
        return f"length {len(a)} != {len(b)}"
    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra[:3] != rb[:3]:
            return f"row {i}: {ra[:3]} != {rb[:3]}"
    return ""


def mcnemar_exact_frac(a_ok, b_ok):
    """(only A correct, only B correct, two-sided exact p as a Fraction) for paired 0/1 vectors."""
    n10 = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    n01 = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n, k = n10 + n01, min(n10, n01)
    if n == 0:
        return n10, n01, Fraction(1)
    return n10, n01, min(Fraction(1), Fraction(2 * sum(comb(n, i) for i in range(k + 1)), 2**n))


def mcnemar_exact(a_ok, b_ok):
    n10, n01, p = mcnemar_exact_frac(a_ok, b_ok)
    return n10, n01, float(p)


def fmt_p(p):
    """float, or '<1e-300' when a float would underflow (exact Fractions are kept until here)."""
    return "<1e-300" if 0 < p < Fraction(1, 10**300) else float(p)


def _g(v):
    return v if isinstance(v, str) else f"{v:.3g}"


def holm(pvals):
    """Holm-adjusted p-values (step-down, monotone), same order as the input (floats or Fractions)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0
    for rank, i in enumerate(order):
        running = max(running, min(1, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def check_pairs(pairs, pred_dir=PRED_DIR):
    """Per (pair, rep): files and alignment result. Returns [(pair, rep, path_a, path_b, problem)]."""
    out = []
    for p in pairs:
        for k in REPS:
            pa, pb = csv_path(p[4], p[2], k, pred_dir), csv_path(p[5], p[3], k, pred_dir)
            if not (pa.exists() and pb.exists()):
                out.append((p, k, pa, pb, "missing file"))
            else:
                out.append((p, k, pa, pb, check_alignment(read_preds(pa), read_preds(pb))))
    return out


def run_tests(pairs, pred_dir=PRED_DIR):
    """Rows for the output CSV, with Holm per (hypothesis, repetition)."""
    rows = []
    for p in pairs:
        h, model, a, b, ga, gb = p
        for k in REPS:
            ra, rb = read_preds(csv_path(ga, a, k, pred_dir)), read_preds(csv_path(gb, b, k, pred_dir))
            if (why := check_alignment(ra, rb)):
                raise ValueError(f"{a} vs {b} rep{k}: {why}")
            xa, xb = [r[3] for r in ra], [r[3] for r in rb]
            n10, n01, pv = mcnemar_exact_frac(xa, xb)
            rows.append({"hypothesis": h, "model": model, "A": a, "B": b, "repetition": k, "n_images": len(xa),
                         "only_A_correct": n10, "only_B_correct": n01,
                         "delta_accuracy_pp": round(100 * (sum(xa) - sum(xb)) / len(xa), 4), "p": fmt_p(pv), "_p": pv})
    for h in {r["hypothesis"] for r in rows}:
        for k in REPS:
            fam = [r for r in rows if r["hypothesis"] == h and r["repetition"] == k]
            for r, adj in zip(fam, holm([r["_p"] for r in fam])):
                r["p_holm"], r["significant"] = fmt_p(adj), int(adj < ALPHA)
    for r in rows:
        del r["_p"]
    return rows


def summarize(rows):
    out = []
    for key in dict.fromkeys((r["hypothesis"], r["model"], r["A"], r["B"]) for r in rows):
        rs = sorted((r for r in rows if (r["hypothesis"], r["model"], r["A"], r["B"]) == key), key=lambda r: r["repetition"])
        nsig = sum(r["significant"] for r in rs)
        out.append({"hypothesis": key[0], "model": key[1], "A": key[2], "B": key[3], "n_significant_reps": nsig,
                    "claimed": int(nsig == len(REPS)), "varies_across_reps": int(0 < nsig < len(REPS)),
                    "delta_accuracy_pp": "/".join(f"{r['delta_accuracy_pp']:+.2f}" for r in rs),
                    "p": "/".join(_g(r['p']) for r in rs), "p_holm": "/".join(_g(r['p_holm']) for r in rs)})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def families(pairs):
    fam = {}
    for p in pairs:
        fam[p[0]] = fam.get(p[0], 0) + 1
    return fam


def print_plan(pairs, checked):
    print(f"{len(pairs)} pairs x {len(REPS)} repetitions; Holm families (hypothesis -> tests per repetition): {families(pairs)}\n")
    for i, p in enumerate(pairs, 1):
        bad = [(k, why) for q, k, _, _, why in checked if q == p and why]
        print(f"{i:2d}. {p[0]:5s} {p[1]:16s} A={p[4]}/{p[2]}__rep{{1,2,3}}.csv\n"
              f"    {'':5s} {'':16s} B={p[5]}/{p[3]}__rep{{1,2,3}}.csv   alignment: {'OK (3/3)' if not bad else 'PROBLEM ' + str(bad)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="plan + alignment check only")
    ap.add_argument("--run", action="store_true", help="compute and write the CSVs")
    ap.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    if args.list == args.run:
        ap.error("pass exactly one of --list / --run")
    pairs = build_pairs()
    checked = check_pairs(pairs, args.pred_dir)
    print_plan(pairs, checked)
    if any(c[4] for c in checked):
        sys.exit("alignment/missing-file problems above; nothing computed")
    if args.list:
        return
    rows = run_tests(pairs, args.pred_dir)
    summ = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "mcnemar_vit.csv", rows)
    write_csv(args.out_dir / "mcnemar_vit_summary.csv", summ)
    print("\nhypothesis model            A  vs  B | reps signif | delta_pp(rep1/2/3) | p(rep1/2/3) | p_Holm(rep1/2/3)")
    for s in summ:
        print(f"{s['hypothesis']:5s} {s['model']:16s} {s['A']} vs {s['B']} | {s['n_significant_reps']}/3 "
              f"{'CLAIMED' if s['claimed'] else '-'}{' VARIES' if s['varies_across_reps'] else ''} | {s['delta_accuracy_pp']} | {s['p']} | {s['p_holm']}")


if __name__ == "__main__":
    main()
