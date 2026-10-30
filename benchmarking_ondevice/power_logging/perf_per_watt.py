"""The perf_per_watt formulas of both branches, in ONE place: the power
wrapper (cli.py) uses them when a row is measured, and
scripts/recompute_llm_perf_per_watt.py (and any results-combination step)
re-derives the LLM one from a stored row.

perf_per_watt (ViT) = N / energy_net_j          (inferences per net joule)

with N = num_inferences_measured. NOT throughput / avg_power_w_net: the
throughput onnxruntime_perf_test reports covers only its timed inferences,
while the power window (and so avg_power_w_net) also covers process start,
model load and warm-up -- a rate from one window over a power from another.
N and energy_net_j come from the same measurement, so their ratio is
1 / energy_per_inference, consistent with the LLM formula below.

perf_per_watt (LLM) = tokens per repetition / energy_per_inference
                    = (PROMPT_TOKENS + GEN_TOKENS) * N / energy_net_j

with N = num_inferences_measured. energy_net_j is measured over the WHOLE
llama-bench invocation, i.e. over all N repetitions (each one a pp128 and a
tg32 test), so it has to be divided by N before it is compared with the
tokens of ONE repetition. The formula used until this fix,
(n_prompt + n_gen) / energy_net_j, divided one repetition's tokens by N
repetitions' energy: tokens/J scaled down by a factor N.

Tokens per repetition come from runner_llm's own -p/-n constants (the
values every LLM row was benchmarked with), not from anything stored in
the row -- the results CSV has no n_prompt/n_gen columns.
"""

from __future__ import annotations

from typing import Dict, Optional

from benchmarking_ondevice.runner_llm.runner import GEN_TOKENS, PROMPT_TOKENS

TOKENS_PER_REPETITION = PROMPT_TOKENS + GEN_TOKENS


def llm_perf_per_watt(energy_net_j: float, num_inferences_measured: int) -> Optional[float]:
    """Tokens per net joule. None (never 0.0) when there is no usable energy
    or repetition count -- same convention as the rest of power_logging."""
    if not energy_net_j or energy_net_j <= 0 or not num_inferences_measured or num_inferences_measured <= 0:
        return None
    energy_per_inference = energy_net_j / num_inferences_measured
    return TOKENS_PER_REPETITION / energy_per_inference


def vit_perf_per_watt(energy_net_j: float, num_inferences_measured: int) -> Optional[float]:
    """Inferences per net joule, i.e. 1 / energy_per_inference. None (never
    0.0) when there is no usable energy or inference count."""
    if not energy_net_j or energy_net_j <= 0 or not num_inferences_measured or num_inferences_measured <= 0:
        return None
    return num_inferences_measured / energy_net_j


def is_llm_row(row: Dict) -> bool:
    """LLM rows are told apart by run_id's family prefix (the per-device
    results CSV holds both branches -- see runner_llm/paths.py)."""
    return str(row.get("run_id", "")).startswith("llm_")


def llm_perf_per_watt_from_row(row: Dict) -> Optional[float]:
    """Re-derives perf_per_watt from a stored results row (CSV strings),
    ignoring whatever perf_per_watt it holds. None if the row has no
    energy measurement."""
    energy = str(row.get("energy_net_j", "")).strip()
    n = str(row.get("num_inferences_measured", "")).strip()
    if not energy or not n:
        return None
    return llm_perf_per_watt(float(energy), int(float(n)))
