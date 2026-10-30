"""Reference/dummy interface for the Section 6.4/6.4.1 power-measurement
contract -- NO real sensor implementation. Extracted from
jetson_powermeter.py (the one real, working implementation) so a future
hardware module (e.g. power_logging/accelpowercape.py, for the
AccelPowerCape/INA219-based devices) has a direct template: copy this
file's shape, replace DummyPowerMeter with a real sensor class, and reuse
everything else in this file as-is (the merging/translation/write logic
below is already hardware-agnostic -- it only calls the sensor object
through the two methods documented on DummyPowerMeter).

Nothing in this module touches real hardware or does real I/O -- every
function/method here raises NotImplementedError. Its purpose is the
signature and the docstrings, not behavior.

------------------------------------------------------------------------
Sensor-level contract (what a real hardware class -- orin_power.PowerMeter
today, a future AccelPowerCape wrapper tomorrow -- must implement):
------------------------------------------------------------------------

__init__(self, idle_load_duration, idle_load_samples, sampling_rate, ...):
    Discover the sensor's channels/rails and measure an idle (background)
    power baseline ONCE here, storing it for later subtraction. Two
    requirements are NOT optional per implementation (Section 6.4.1):

      1. Sum ALL rails/components the sensor exposes, never just the one
         "active" for the current execution backend (CPU vs GPU) -- a rail
         that looks idle still reflects real orchestration/launch cost.
      2. Measure idle over a window that uses the SAME per-tick
         methodology as the load measurement (see estimate_fn_power below)
         -- e.g. if multiple rails exist, sample them together each tick,
         not one rail's whole window followed by another's (orin_power.py's
         own _estimate_standby_load had exactly this bug once; see its
         module history / tests/test_orin_power.py for the regression
         coverage that would have caught it here too).

estimate_fn_power(self, fn) -> Dict:
    Run `fn()` (a zero-argument callable -- the caller has already bound
    every real argument via functools.partial before this point, this
    layer never sees run_id/artifact_path/etc. directly) while sampling
    power at the sensor's own rate, integrating energy over the actual
    elapsed wall-clock time (not a nominal fixed interval -- protects
    against scheduling jitter). Must return a dict with AT LEAST:

        {
            "elapsed": <float, seconds>,
            "total": {
                "energy_total_j": <float>,   # raw, idle NOT subtracted
                "energy_net_j": <float>,     # idle subtracted, clamped >= 0
            },
            "output": <whatever fn() returned, untouched>,
        }

    ("total" mirrors orin_power.PowerMeter's own key, meaning "summed
    across every rail" -- Section 6.4.1's first mandatory requirement.
    Per-rail sub-dicts, if the sensor has more than one, are useful for
    debugging but not required by anything below this layer.)

------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable, Dict, Optional


class DummyPowerMeter:
    """Template for a real hardware sensor class. See the module docstring
    for exactly what __init__ and estimate_fn_power must do -- neither is
    implemented here; both raise NotImplementedError.
    """

    def __init__(
        self,
        idle_load_duration: float = 10.0,
        idle_load_samples: int = 10,
        sampling_rate: float = 30.0,
    ) -> None:
        raise NotImplementedError(
            "DummyPowerMeter is a reference interface, not a real sensor -- "
            "implement __init__ against real hardware (discover channels, "
            "measure idle baseline) in a concrete subclass/module instead."
        )

    def estimate_fn_power(self, fn: Callable[[], Dict]) -> Dict:
        raise NotImplementedError(
            "DummyPowerMeter is a reference interface, not a real sensor -- "
            "implement estimate_fn_power against real hardware in a "
            "concrete subclass/module instead. Must return "
            "{'elapsed': float, 'total': {'energy_total_j': float, "
            "'energy_net_j': float}, 'output': <fn()'s return value>}."
        )


_dummy_power_meter: Optional[DummyPowerMeter] = None


def get_power_meter(**power_meter_kwargs) -> DummyPowerMeter:
    """Constructed exactly once per process and reused for every
    subsequent call -- same singleton pattern as
    jetson_powermeter.get_power_meter(). Idle is measured once, during
    that first construction; re-measuring it before every repeat would pay
    its real-world wait every time and let the baseline drift between
    repeats of the same device/session."""
    global _dummy_power_meter
    if _dummy_power_meter is None:
        _dummy_power_meter = DummyPowerMeter(**power_meter_kwargs)
    return _dummy_power_meter


def measure_and_record(
    run_fn_bound: Callable[[], Dict],
    build_result_row: Callable[[str, str, Optional[int], Dict, int], Dict],
    run_id: str,
    device_id: str,
    repeat_id: int,
    requested_repeats: int,
    *,
    perf_per_watt_fn: Callable[[Dict, Dict, Dict], Optional[float]],
    power_meter_kwargs: Optional[Dict] = None,
) -> Dict:
    """The hardware-agnostic pipeline every power_logging module
    implements identically -- see jetson_powermeter.py for the real,
    working version this was extracted from. A new hardware module only
    ever needs its own DummyPowerMeter-shaped sensor class (above); this
    function's logic does not change per device.

    Parameters
    ----------
    run_fn_bound
        A zero-argument callable -- typically
        `functools.partial(runner_vit.run_benchmark, run_id, artifact_path,
        device_id, repeat_id, repeats=..., write_result=False)` (or the
        same for runner_llm). write_result=False is mandatory: the runner
        must never write its own row, since this function writes the
        COMBINED row instead.
    build_result_row
        The wrapped branch's own row builder -- runner_vit.results_io.
        build_result_row or runner_llm.results_io.build_result_row. Reused
        as-is so the performance-field mapping (latency_ms/throughput vs.
        throughput_prefill_tps/throughput_decode_tps) is never
        reimplemented here.
    perf_per_watt_fn(row, output, energy_fields) -> Optional[float]
        Computes perf_per_watt -- the FORMULA, not just which field feeds
        it, is branch-specific (a benchmark that measures a sub-phase
        throughput over a shorter window than the energy figure covers
        cannot just divide the two -- see step 8 below), so this is passed
        in explicitly by the caller rather than inferred from row shape.

    Pipeline (must be followed in this order by any real implementation):

      1. power_meter = get_power_meter(**power_meter_kwargs) -- reuse the
         cached sensor, do not reconstruct it per call.
      2. power_results = power_meter.estimate_fn_power(run_fn_bound).
      3. output = power_results["output"] -- the wrapped runner's own
         parsed result. This is the ONLY place it is captured; never
         re-parse or reconstruct it another way.
      4. Translate field names to the project's homogeneous schema
         (Section 6.4.1) -- this translation lives in the power_logging
         layer, never in the sensor class itself:
           energy_raw_j       = power_results["total"]["energy_total_j"]
           energy_net_j       = power_results["total"]["energy_net_j"]
           avg_power_w_raw    = energy_raw_j / power_results["elapsed"]
           avg_power_w_net    = energy_net_j / power_results["elapsed"]
      5. row = build_result_row(run_id, device_id, repeat_id, output,
         requested_repeats) -- the performance portion.
      6. row.update(<the 4 translated energy fields from step 4>).
      7. row["energy_per_inference"] = energy_net_j /
         output["num_inferences_measured"] -- NET, per Section 6.4.1
         ("...net is the mandatory metric for all comparisons
         of the ablation"); read num_inferences_measured from the real
         parsed output, never assume it equals requested_repeats.
      8. perf_per_watt = perf_per_watt_fn(row, output, energy_fields); if
         not None, row["perf_per_watt"] = perf_per_watt. The formula
         itself must divide over a SINGLE, CONSISTENT time window -- if
         the wrapped benchmark reports a throughput measured over only a
         sub-phase of the run (e.g. llama-bench's decode-only tps, while
         the run_fn_bound() call it came from covers prefill+decode
         together), dividing that sub-phase rate by the whole-run energy
         mixes two different windows and must NOT be done; use a
         quantity that spans the same window the energy figure does
         instead (e.g. total tokens processed / total net energy). A
         benchmark whose reported throughput already spans its whole
         execution (like onnxruntime_perf_test's) has no such mismatch and
         can divide directly. perf_per_watt_fn returns None (not 0.0) when
         its own denominator is zero -- a real, if unusual, outcome for a
         very fast/light benchmark; not a ZeroDivisionError to raise.
      9. Write `row` via this module's own results_io.append_result
         (replace-by-result_id, never a duplicate append) -- the wrapped
         runner never writes, since it was called with write_result=False.

    Not implemented here -- raises NotImplementedError.
    """
    raise NotImplementedError(
        "measure_and_record is a reference interface, not a real "
        "implementation -- see jetson_powermeter.measure_and_record for "
        "the real, working version, and this function's docstring for the "
        "exact pipeline a new hardware module must follow."
    )
