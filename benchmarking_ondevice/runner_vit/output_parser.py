"""Parses onnxruntime_perf_test's fixed-format stdout summary.

Format confirmed against the real binary compiled as described in the README (ONNX
Runtime v1.23.2) -- there is no structured (JSON/CSV) output mode
(technical_architecture.md Section 6.6), so this is necessarily a
text parse, with no stability guarantee across onnxruntime versions. This
is exactly what --self-test (self_test.py) exists to catch early per
device/environment, before any real measurement is trusted there.

Example input (one real run, 5 repeats):

    Session creation time cost: 0.194833 s
    First inference time cost: 70 ms
    Total inference time cost: 0.123647 s
    Total inference requests: 5
    Average inference time cost: 24.7294 ms
    Total inference run time: 0.123668 s
    Number of inferences per second: 40.4307
    Avg CPU usage: 102 %
    Peak working set size: 151752704 bytes

`num_inferences_measured` is always read from "Total inference requests"
in the actual output -- never assumed to equal the `-r N` value passed on
the command line (the tool's own repeat-count bookkeeping, e.g. around
warm-up, is not something this parser should second-guess).
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# (result dict key, label, regex group pattern, caster) -- required fields:
# missing any one of these is a parse failure, not a partial result.
_REQUIRED_FIELDS = [
    ("num_inferences_measured", "Total inference requests", r"(\d+)", int),
    ("avg_latency_ms", "Average inference time cost", r"([0-9.]+)\s*ms", float),
    ("throughput_ips", "Number of inferences per second", r"([0-9.]+)", float),
]

# Best-effort extras: useful context if present, but not guaranteed stable
# across onnxruntime versions the way the 3 fields above are (confirmed by
# --self-test) -- absence of any of these does not fail the parse.
_OPTIONAL_FIELDS = [
    ("session_creation_time_s", "Session creation time cost", r"([0-9.]+)\s*s", float),
    ("first_inference_time_ms", "First inference time cost", r"([0-9.]+)\s*ms", float),
    ("total_inference_time_s", "Total inference time cost", r"([0-9.]+)\s*s", float),
    ("total_inference_run_time_s", "Total inference run time", r"([0-9.]+)\s*s", float),
    ("avg_cpu_usage_pct", "Avg CPU usage", r"([0-9.]+)\s*%", float),
    ("peak_working_set_bytes", "Peak working set size", r"(\d+)\s*bytes", int),
]


def _find(label: str, group_pattern: str, text: str) -> Optional[str]:
    match = re.search(rf"{re.escape(label)}:\s*{group_pattern}", text)
    return match.group(1) if match else None


def parse_perf_test_output(output: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    missing = []

    for key, label, pattern, cast in _REQUIRED_FIELDS:
        raw = _find(label, pattern, output)
        if raw is None:
            missing.append(label)
        else:
            result[key] = cast(raw)

    if missing:
        raise ValueError(
            f"onnxruntime_perf_test output is missing required field(s) {missing} -- "
            f"cannot trust this result. Full output:\n{output}"
        )

    for key, label, pattern, cast in _OPTIONAL_FIELDS:
        raw = _find(label, pattern, output)
        if raw is not None:
            result[key] = cast(raw)

    return result
