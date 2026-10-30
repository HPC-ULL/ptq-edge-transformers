"""Estimates the wall time of each F16 reference row on the Jetson
(technical_architecture.md Section 6.7), per model and power mode,
from the already-measured Q4_K_M baseline row (phase1_baseline_default_q4km,
no imatrix) and the F16 / Q4_K_M size ratio -- to decide whether N=30 is
viable before measuring. An ESTIMATE under stated assumptions, not a
measurement.

Per (mode, model), from results/raw/results_<mode>.csv and, when present,
results/characterization/power_sampling_<mode>.csv:

  t_rep_q   = 128 / throughput_prefill_tps + 32 / throughput_decode_tps
              (one llama-bench repetition of pp128 + tg32)
  fixed_q   = window_s - N * t_rep_q   (load + llama-bench warm-up; only
              where the sampling record exists)
  ratio     = size(F16) / size(Q4_K_M), from results/artifact_sizes.csv, or
              --assumed-ratio when that CSV is not available yet
  t_rep_f16 = ratio * t_rep_q
  window_f16 = ratio * fixed_q + N * t_rep_f16

Assumptions: (a) the whole repetition scales with the bytes read -- close
for decode (memory-bound), pessimistic for prefill where F16 avoids
dequantization (on GPU especially); (b) the fixed cost scales with file
size (loading). Cooldown between rows is not included.

    python3 scripts/estimate_f16_time.py --sizes results/artifact_sizes.csv
    python3 scripts/estimate_f16_time.py --assumed-ratio 3.3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.llm_f16_reference import run_id as f16_run_id  # noqa: E402
from configs.llm_manifest import MODELS  # noqa: E402

MODES = [f"jetson_{b}_{w}w" for b in ("cpu", "gpu") for w in (15, 30, 50)]
PROMPT_TOKENS, GEN_TOKENS = 128, 32
N_F16 = 30
FIELDS = ["device", "model", "pp_tps_q", "tg_tps_q", "t_rep_q_s", "decode_share", "window_q_s", "fixed_q_s",
          "ratio", "t_rep_f16_s", "window_f16_s"]


def q_run_id(model: str) -> str:
    return f"llm_{model}_phase1_baseline_default_q4km"


def _read(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def size_ratios(sizes_csv: Path) -> Dict[str, float]:
    by_id = {r["run_id"]: int(r["size_bytes"]) for r in _read(sizes_csv)}
    return {m: by_id[f16_run_id(m)] / by_id[q_run_id(m)] for m in MODELS
            if f16_run_id(m) in by_id and q_run_id(m) in by_id}


def estimate(results_dir: Path, sampling_dir: Path, ratios: Dict[str, float],
             assumed_ratio: Optional[float] = None) -> List[Dict]:
    out = []
    for device in MODES:
        results = {r["run_id"]: r for r in _read(results_dir / f"results_{device}.csv")}
        windows = {r["result_id"]: float(r["window_s"]) for r in _read(sampling_dir / f"power_sampling_{device}.csv")}
        for model in MODELS:
            row = results.get(q_run_id(model))
            ratio = ratios.get(model, assumed_ratio)
            if row is None or ratio is None:
                continue
            pp, tg = float(row["throughput_prefill_tps"]), float(row["throughput_decode_tps"])
            n = int(row["num_inferences_measured"])
            t_rep = PROMPT_TOKENS / pp + GEN_TOKENS / tg
            window = windows.get(row["result_id"])
            fixed = window - n * t_rep if window is not None else None
            t_rep_f16 = ratio * t_rep
            out.append({
                "device": device, "model": model, "pp_tps_q": pp, "tg_tps_q": tg, "t_rep_q_s": t_rep,
                "decode_share": (GEN_TOKENS / tg) / t_rep, "window_q_s": window, "fixed_q_s": fixed,
                "ratio": ratio, "t_rep_f16_s": t_rep_f16,
                "window_f16_s": (ratio * fixed if fixed is not None else 0.0) + N_F16 * t_rep_f16,
            })
    return out


def _fmt(v) -> str:
    if v is None:
        return ""
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", type=Path, default=_PROJECT_ROOT / "results" / "raw")
    parser.add_argument("--sampling-dir", type=Path, default=_PROJECT_ROOT / "results" / "characterization")
    parser.add_argument("--sizes", type=Path, default=_PROJECT_ROOT / "results" / "artifact_sizes.csv")
    parser.add_argument("--assumed-ratio", type=float, default=None,
                        help="F16/Q4_K_M size ratio for models missing from --sizes (an assumption)")
    parser.add_argument("--out", type=Path, default=None, help="optional CSV (under results/investigations/)")
    args = parser.parse_args(argv)

    ratios = size_ratios(args.sizes)
    for m in MODELS:
        src = f"{ratios[m]:.3f} (measured, {args.sizes.name})" if m in ratios else (
            f"{args.assumed_ratio} (ASSUMED)" if args.assumed_ratio else "no data")
        print(f"size ratio {m:<16}: {src}")
    rows = estimate(args.results_dir, args.sampling_dir, ratios, args.assumed_ratio)
    if not rows:
        print("no rows: results or size ratios are missing")
        return 1

    print(f"\n{'mode':<16} {'model':<16} {'t_rep_Q':>8} {'%dec':>5} {'fixed_Q':>7} {'t_rep_F16':>9} {'row_F16':>9}")
    for r in rows:
        fixed = f"{r['fixed_q_s']:.1f}" if r["fixed_q_s"] is not None else "-"
        print(f"{r['device']:<16} {r['model']:<16} {r['t_rep_q_s']:>8.2f} {100 * r['decode_share']:>5.0f} "
              f"{fixed:>7} {r['t_rep_f16_s']:>9.2f} {r['window_f16_s']:>9.1f}")
    print("\nseconds; row_F16 = window estimated with N=30 (without the fixed part if there is no sampling record; without cooldown)")
    for device in MODES:
        total = sum(r["window_f16_s"] for r in rows if r["device"] == device)
        if total:
            print(f"  {device:<16} 3 F16 rows: {total / 60:.1f} min of measurement")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({k: _fmt(v) for k, v in r.items()} for r in rows)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
