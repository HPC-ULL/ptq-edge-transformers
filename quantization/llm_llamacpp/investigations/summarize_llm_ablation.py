"""One-off reporting script (NOT part of the pipeline): summarize each
LLM-branch model's full ablation from results/raw/llm_perplexity_offline.csv
-- mean/stdev of accuracy_value (perplexity) per configuration, for each of
tinyllama_1_1b, qwen2_5_1_5b, ministral_3_3b.

Unlike the ViT branch, rows are NOT uniform: 5 stochastic configs per model
(calibration_corpus != "none") get 3 independent repeats each; 2
deterministic configs per model (baseline, legacy_corpus_none) get exactly
one artifact/value (n=1, no stdev -- that's the row's design, not missing
data). Sorted by mean perplexity ascending (lower is better), with
bitwidth/quant_scheme/calibration_corpus printed alongside each row so an
outlier can be checked for an obvious explanation (e.g. a legacy quant
scheme naturally landing elsewhere) rather than guessed at.

Flags: any stochastic row's stdev more than 3x the median stdev among the
model's other stochastic rows; any row's mean perplexity deviating more
than 20% (relative) from the model's own median mean perplexity across all
7 configs -- flagged for review against the printed bitwidth/quant_scheme/
calibration_corpus columns, not asserted as a bug.

Run directly:

    python quantization/llm_llamacpp/investigations/summarize_llm_ablation.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MODEL_NAMES = ["tinyllama_1_1b", "qwen2_5_1_5b", "ministral_3_3b"]
STD_OUTLIER_FACTOR = 3.0
MEAN_OUTLIER_RELATIVE = 0.20  # flag if a row's mean deviates >20% from the model's median mean


def main() -> None:
    from evaluation_offline.llm_perplexity.paths import LLM_PERPLEXITY_RESULTS_CSV

    if not LLM_PERPLEXITY_RESULTS_CSV.exists():
        print(f"No results file found at {LLM_PERPLEXITY_RESULTS_CSV}")
        return

    with open(LLM_PERPLEXITY_RESULTS_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))

    for model_name in MODEL_NAMES:
        print(f"=== {model_name} ===")
        model_rows = [r for r in all_rows if r["model_name"] == model_name]
        if not model_rows:
            print("  no rows found\n")
            continue

        # values[run_id] = list of accuracy_value across repeats (1 for
        # deterministic rows, up to 3 for stochastic)
        values: dict = defaultdict(list)
        meta: dict = {}
        for row in model_rows:
            values[row["run_id"]].append(float(row["accuracy_value"]))
            meta[row["run_id"]] = row

        summary = []
        for run_id, vals in values.items():
            row = meta[run_id]
            summary.append({
                "run_id": run_id,
                "label": f"{row['factor_varied']}={row['factor_level']}",
                "bitwidth": row["bitwidth"],
                "quant_scheme": row.get("quant_scheme", ""),
                "calibration_corpus": row.get("calibration_corpus", ""),
                "n": len(vals),
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else None,
            })
        summary.sort(key=lambda r: r["mean"])  # ascending: lower perplexity first (better)

        stochastic_stds = [r["std"] for r in summary if r["std"] is not None]
        median_std = statistics.median(stochastic_stds) if stochastic_stds else 0.0
        all_means = [r["mean"] for r in summary]
        median_mean = statistics.median(all_means)

        header = (f"{'config':<32} {'bitwidth':>8} {'quant_scheme':>14} {'calib_corpus':>14} "
                  f"{'n':>2} {'mean PPL':>10} {'std':>8}")
        print(header)
        print("-" * len(header))

        flags = []
        for r in summary:
            std_str = f"{r['std']:.4f}" if r["std"] is not None else "n/a"
            print(f"{r['label']:<32} {r['bitwidth']:>8} {r['quant_scheme']:>14} "
                  f"{r['calibration_corpus']:>14} {r['n']:>2} {r['mean']:>10.4f} {std_str:>8}")

            if median_std > 0 and r["std"] is not None and r["std"] > STD_OUTLIER_FACTOR * median_std:
                flags.append(f"  ANOMALOUS std: {r['label']} ({r['std']:.4f} vs. median {median_std:.4f})")

            relative_dev = abs(r["mean"] - median_mean) / median_mean if median_mean else 0.0
            if relative_dev > MEAN_OUTLIER_RELATIVE:
                flags.append(
                    f"  OUTLIER mean PPL: {r['label']} ({r['mean']:.4f} vs. model median "
                    f"{median_mean:.4f}, {relative_dev:.0%} off) -- check bitwidth="
                    f"{r['bitwidth']!r}/quant_scheme={r['quant_scheme']!r}/"
                    f"calibration_corpus={r['calibration_corpus']!r} for an explanation"
                )

        if flags:
            print("Flags:")
            for flag in flags:
                print(flag)
        else:
            print("No flags.")
        print()


if __name__ == "__main__":
    main()
