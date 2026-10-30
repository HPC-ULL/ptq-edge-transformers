"""Per-model, per-type summaries shared by Parts A and B (calibration_protocol_llm.md
Section 3.0.1): median across blocks, maximum and the block where it occurs,
medians by thirds of relative depth (block i of n -> third floor(3 i / n)), and
a 95 % percentile bootstrap interval of the median (10 000 resamples of blocks
with replacement, seed 0). Descriptive only: no test between models."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

N_BOOT = 10_000
BOOT_SEED = 0


def third(i: int, n: int) -> int:
    return (3 * i) // n


def bootstrap_median_ci(values: Sequence[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def describe(blocks: Sequence[int], values: Sequence[float], n_blocks: int) -> Dict:
    """Summary of one metric over the blocks of one model and type."""
    b = np.asarray(blocks)
    x = np.asarray(values, dtype=np.float64)
    i_max = int(np.argmax(x))
    lo, hi = bootstrap_median_ci(x)
    out = {"n_blocks": len(x), "median": float(np.median(x)), "ci_low": lo, "ci_high": hi,
           "max": float(x[i_max]), "max_block": int(b[i_max])}
    thirds = np.array([third(int(i), n_blocks) for i in b])
    for t in range(3):
        out[f"third{t + 1}_median"] = float(np.median(x[thirds == t]))
    return out


def ministral_criterion(medians: Dict[str, Dict[str, float]], types: Sequence[str]) -> Dict:
    """medians[model][type] -> 'coherent' if Ministral's median is lower than
    TinyLlama's AND Qwen2.5's in at least 3 of the 4 types."""
    lower = [t for t in types if medians["ministral_3_3b"][t] < medians["tinyllama_1_1b"][t]
             and medians["ministral_3_3b"][t] < medians["qwen2_5_1_5b"][t]]
    return {"types_lower": lower, "n_lower": len(lower),
            "verdict": "coherent" if len(lower) >= 3 else "not explained by this metric"}


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def rows_to_table(rows: List[Dict], columns: Sequence[str]) -> str:
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)
