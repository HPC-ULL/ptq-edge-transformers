"""Parses llama-bench's `-o json` stdout.

Format confirmed against the real binary compiled as described in the README (a real
run against artifacts/llm/llm_qwen2_5_1_5b_phase2_calib_set_size_1000_q4km
__rep3.gguf): a JSON array of per-test objects. With -p/-n both given
(never llama-bench's own defaults, see runner.py), exactly one test runs
"prefill" (n_prompt > 0, n_gen == 0) and one runs "decode" (n_prompt == 0,
n_gen > 0) -- identified by those field values, NEVER by array position
(nothing in llama-bench's docs guarantees ordering).

Each object's `avg_ts` (tokens/sec) is the throughput for that test;
`samples_ts` is the raw per-repetition array -- `num_inferences_measured`
is its length, read from the actual output same as runner_vit's
`num_inferences_measured`, never assumed to equal the `-r` value passed in.
Both objects come from the same invocation (same `-r`), so their sample
counts are expected to match; if they don't, something is wrong enough
that reporting either one silently would be misleading -- fail loud
instead of picking one arbitrarily.

`n_prompt` (from the prefill object) and `n_gen` (from the decode object)
are also returned, read from the real objects rather than assumed to equal
runner.py's PROMPT_TOKENS/GEN_TOKENS constants -- benchmarking_ondevice/
power_logging/ needs the actual total token count (n_prompt + n_gen) for
perf_per_watt, since throughput_decode_tps alone is measured over only the
decode sub-phase while the energy figure it would otherwise be divided by
covers the WHOLE llama-bench invocation (prefill + decode together) --
dividing a whole-run energy by a sub-phase-only rate mixes two different
time windows. See power_logging/jetson_powermeter.py's perf_per_watt_fn
for where this is actually used.

Real example (2-object array, prompt/gen defaults 512/128 in this
example -- runner.py always overrides them to 128/32):

    [
      {"n_prompt": 512, "n_gen": 0, "avg_ts": 673.610158,
       "samples_ts": [646.596, 650.011, 709.114, 647.845, 714.484], ...},
      {"n_prompt": 0, "n_gen": 128, "avg_ts": 23.003802,
       "samples_ts": [22.4783, 23.382, 23.3555, 23.3873, 22.4159], ...}
    ]
"""

from __future__ import annotations

import json
from typing import Dict, List


def _extract_json_array(text: str) -> List[dict]:
    # Defensive against stray non-JSON text around the array (progress
    # indicators, warnings) -- llama-bench's docs don't guarantee -o json
    # stdout is ONLY the array with nothing else ever printed alongside it.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find a JSON array in llama-bench output:\n{text}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"llama-bench output looked like JSON but failed to parse ({e}):\n{text}") from None
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array from llama-bench, got {type(data).__name__}:\n{text}")
    return data


def parse_llama_bench_output(output: str) -> Dict[str, float]:
    data = _extract_json_array(output)

    prefill_objs = [obj for obj in data if obj.get("n_prompt", 0) > 0 and obj.get("n_gen", 0) == 0]
    decode_objs = [obj for obj in data if obj.get("n_prompt", 0) == 0 and obj.get("n_gen", 0) > 0]

    if len(prefill_objs) != 1 or len(decode_objs) != 1:
        raise ValueError(
            f"Expected exactly 1 prefill object (n_prompt>0, n_gen==0) and 1 decode object "
            f"(n_prompt==0, n_gen>0) in llama-bench output; found {len(prefill_objs)} prefill and "
            f"{len(decode_objs)} decode candidates among {len(data)} object(s). Full output:\n{output}"
        )

    prefill, decode = prefill_objs[0], decode_objs[0]

    for label, obj in (("prefill", prefill), ("decode", decode)):
        if "avg_ts" not in obj or "samples_ts" not in obj:
            raise ValueError(f"llama-bench {label} object is missing avg_ts/samples_ts. Full output:\n{output}")

    prefill_n = len(prefill["samples_ts"])
    decode_n = len(decode["samples_ts"])
    if prefill_n != decode_n:
        raise ValueError(
            f"prefill and decode objects report different repetition counts "
            f"(prefill: {prefill_n} samples, decode: {decode_n} samples) from the same llama-bench "
            f"invocation -- refusing to pick one arbitrarily as num_inferences_measured. Full output:\n{output}"
        )

    return {
        "throughput_prefill_tps": float(prefill["avg_ts"]),
        "throughput_decode_tps": float(decode["avg_ts"]),
        "num_inferences_measured": prefill_n,
        "n_prompt": prefill["n_prompt"],
        "n_gen": decode["n_gen"],
    }
