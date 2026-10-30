"""One-off reporting script (NOT part of the pipeline): summarize each
LLM-branch model's Phase 3 ablation from results/raw/llm_perplexity_offline.csv
-- mean/stdev of accuracy_value (perplexity) per Phase 3 configuration, for
each of tinyllama_1_1b, qwen2_5_1_5b, ministral_3_3b.

Mirrors quantization/vit_brevitas/investigations/summarize_phase3_ablation.py
(same per-model grouping, same 3x-median-stdev anomaly criterion) but adapted
for LLM's schema differences:

  - A single `perplexity` metric per row (not top1/top5 pairs), so there is
    one value list per run_id instead of two.
  - Up to 4 configs per model (baseline x {Q3_K,Q2_K} + combined x
    {Q3_K,Q2_K}), but only 2 for a model whose combined config deduped
    against baseline (see configs/llm_phase3_manifest.py) -- the case for
    the retired salamandra_2b; whether ministral_3_3b dedups is only known
    once its Phase 2 results exist (results/phase3_selection_report.txt).
  - Rows are NOT uniformly stochastic like ViT Phase 3: baseline rows are
    always deterministic (calibration_corpus="none" -> n=1, no stdev, exactly
    like Phase 1-2's global baseline), while combined rows are stochastic
    (n=3) only if that model's Phase 2 winner for calibration_corpus is
    itself non-"none" -- otherwise combined is ALSO deterministic (n=1).
    This is exactly the same graceful n=1/no-stdev handling already used in
    quantization/llm_llamacpp/investigations/summarize_llm_ablation.py, not
    new logic.
  - Sorted by mean perplexity ascending (lower is better), matching that
    same Phase 1-2 script's convention (the opposite of ViT's
    descending-accuracy convention, since the two metrics run in opposite
    directions).

Flags: any stochastic row's stdev more than 3x the median stdev among the
model's own OTHER stochastic configs (a model with 0 or 1 stochastic
configs -- e.g. a model that deduped against baseline, or any model whose combined row
is also deterministic -- has no stdev basis for comparison and is skipped
for this check, printed as "n/a").

Run directly:

    python quantization/llm_llamacpp/investigations/summarize_llm_phase3_ablation.py
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


def _summarize_model(model_rows: list) -> list:
    # values[run_id] = list of accuracy_value (perplexity) across repeats
    # (1 for deterministic rows, up to 3 for stochastic).
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
            "label": f"{row['factor_level']}_{row['bitwidth']}",
            "bitwidth": row["bitwidth"],
            "quant_scheme": row.get("quant_scheme", ""),
            "calibration_corpus": row.get("calibration_corpus", ""),
            "n": len(vals),
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else None,
        })
    summary.sort(key=lambda r: r["mean"])  # ascending: lower perplexity first (better)
    return summary


def main() -> None:
    from evaluation_offline.llm_perplexity.paths import LLM_PERPLEXITY_RESULTS_CSV

    if not LLM_PERPLEXITY_RESULTS_CSV.exists():
        print(f"No results file found at {LLM_PERPLEXITY_RESULTS_CSV}")
        return

    with open(LLM_PERPLEXITY_RESULTS_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))

    phase3_rows = [r for r in all_rows if "phase3" in r["run_id"]]
    if not phase3_rows:
        print("No Phase 3 rows found (looked for 'phase3' in run_id).")
        return

    any_anomalies = False

    for model_name in MODEL_NAMES:
        print(f"=== {model_name} ===")
        model_rows = [r for r in phase3_rows if r["model_name"] == model_name]
        if not model_rows:
            print("  no Phase 3 rows found\n")
            continue

        summary = _summarize_model(model_rows)

        stochastic_stds = [r["std"] for r in summary if r["std"] is not None]
        median_std = statistics.median(stochastic_stds) if stochastic_stds else None

        header = (f"{'config':<24} {'bitwidth':>8} {'quant_scheme':>14} {'calib_corpus':>14} "
                  f"{'n':>2} {'mean PPL':>10} {'std':>8}")
        print(header)
        print("-" * len(header))

        flags = []
        for r in summary:
            std_str = f"{r['std']:.4f}" if r["std"] is not None else "n/a"
            print(f"{r['label']:<24} {r['bitwidth']:>8} {r['quant_scheme']:>14} "
                  f"{r['calibration_corpus']:>14} {r['n']:>2} {r['mean']:>10.4f} {std_str:>8}")

            if median_std is not None and median_std > 0 and r["std"] is not None \
                    and r["std"] > STD_OUTLIER_FACTOR * median_std:
                flags.append(f"  ANOMALOUS std: {r['label']} ({r['std']:.4f} vs. median {median_std:.4f})")

        if median_std is None:
            print("  (no stochastic configs for this model -- std comparison skipped)")

        if flags:
            any_anomalies = True
            print("Flags:")
            for flag in flags:
                print(flag)
        else:
            print("No flags.")
        print()

    if not any_anomalies:
        print("Overall: no anomalies detected across any model's Phase 3 configs.")


if __name__ == "__main__":
    main()
