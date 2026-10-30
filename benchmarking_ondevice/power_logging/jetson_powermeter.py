"""Section 6.4/6.4.1 common-contract wrapper around orin_power.PowerMeter.

get_power_meter() constructs the underlying orin_power.PowerMeter EXACTLY
ONCE per process and caches it at module scope -- idle power is measured
once, during that first construction, and reused for every subsequent
measurement in this process. Re-measuring idle before every single repeat
would (a) pay its ~10s wait every time for no benefit, and (b) let the
idle baseline itself drift between repeats of the same device/session,
undermining the very consistency orin_power.py's own idle-window fix
(interleaved per-tick sampling, see that module) was about.

measure_and_record() is the actual contract implementation:

  0. Verifies the Jetson's live nvpmodel power mode matches the device_id's
     declared power_mode_w (_verify_power_mode) BEFORE anything else --
     mismatch raises PowerModeMismatchError and nothing is measured;
     failed query warns and continues; match / no declared mode is silent.
  1. Wraps the caller-provided `run_fn_bound` (a zero-arg callable --
     typically `functools.partial(runner_vit.run_benchmark, ...,
     write_result=False)` or the same for runner_llm -- see cli.py) with
     power_meter.estimate_fn_power(), which returns a dict whose `output`
     key is exactly the wrapped runner's own parsed result (latency/
     throughput for ViT, prefill/decode tps for LLM) -- this is the ONLY
     place that result is captured; it is never re-parsed or reconstructed
     here.
  2. Translates orin_power's own field names to this project's homogeneous
     schema (Section 6.4.1) -- energy_total_j -> energy_raw_j (renamed),
     energy_net_j (kept as-is), plus avg_power_w_raw/avg_power_w_net
     derived from energy/duration. This translation lives HERE, not in
     orin_power.py, which keeps its own internal naming untouched.
  3. Combines the wrapped runner's performance row (built via the SAME
     branch-specific `build_result_row` the plain runner itself uses --
     runner_vit.results_io.build_result_row or
     runner_llm.results_io.build_result_row, passed in by the caller) with
     the translated energy fields into ONE row, and writes it via
     results_io.append_result -- the runner itself never writes (it was
     called with write_result=False), so this is the only write for that
     result_id.

`energy_per_inference` (Section 4's existing common column) is populated
with the NET value (energy_net_j / num_inferences_measured) -- Section
6.4.1 mandates net as "the mandatory metric for all comparisons
of the ablation", so that's what the one shared column holds; the raw energy
figure remains available separately via energy_raw_j/avg_power_w_raw for
deployment-cost interpretation, just not duplicated into its own
energy_per_inference_raw column (not asked for).

`perf_per_watt`'s FORMULA is branch-dependent, not just which field feeds
it, so the caller (cli.py) passes a `perf_per_watt_fn(row, output,
energy_fields) -> Optional[float]` callable rather than a single field
name -- explicit per branch, never guessed from row shape (same
"don't infer, be explicit" convention as cli.py's --branch flag):

  - ViT: N / energy_net_j, i.e. 1 / energy_per_inference
    (perf_per_watt.py) -- NOT throughput / avg_power_w_net: the throughput
    covers only onnxruntime_perf_test's timed inferences, while the power
    window (hence avg_power_w_net) also covers start-up, model load and
    warm-up, so that ratio mixed two windows.
  - LLM: (n_prompt + n_gen) / energy_per_inference, i.e.
    (n_prompt + n_gen) * N / energy_net_j (perf_per_watt.py). energy_net_j
    covers all N repetitions, so it is divided by N before being set
    against ONE repetition's tokens -- the earlier (n_prompt + n_gen) /
    energy_net_j was tokens/J divided by N. NOT
    throughput_decode_tps / avg_power_w_net. llama-bench's decode
    throughput is measured only over the decode sub-phase of one
    invocation, while avg_power_w_net/energy_net_j cover the WHOLE
    invocation (prefill + decode together, since both run inside the same
    run_fn_bound() call power_meter.estimate_fn_power wraps) -- dividing a
    whole-run energy figure by a sub-phase-only rate would mix two
    different time windows. Total tokens processed (n_prompt + n_gen, both
    read from the real llama-bench output -- see runner_llm/
    output_parser.py) divided by total net energy sidesteps that mismatch
    entirely and stays inside one consistent window. Confirmed via a real
    review of this exact inconsistency before landing this version.

Either function returns None (not a hardcoded 0.0) when its denominator is
zero (energy_net_j rounding to zero -- a real, if
unusual, outcome for a very fast/light benchmark whose power draw never
exceeded idle) -- measure_and_record leaves `perf_per_watt` unset in that
case rather than raising ZeroDivisionError or writing a nonsense value.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from . import orin_power, sampling_characterization
from .results_io import append_result

_power_meter: Optional[orin_power.PowerMeter] = None


class PowerModeMismatchError(RuntimeError):
    """The Jetson's active nvpmodel power mode does not match the
    power_mode_w declared for the device_id being measured -- measuring
    anyway would file the result under the wrong device_id."""


def _resolve_power_mode_w(device_id: str) -> Optional[int]:
    """The `power_mode_w` declared for `device_id` in configs/devices.yaml,
    or None if it declares none (non-Jetson devices, dev entries). Raises
    if `device_id` is not in the catalog at all -- a real run against an
    uncatalogued device_id is a mistake worth stopping for. Read here
    directly (not via runner_vit/runner_llm's resolve_device) to keep this
    Jetson-specific check self-contained in this file."""
    import yaml

    from configs.common import CONFIGS_DIR

    devices_yaml = CONFIGS_DIR / "devices.yaml"
    with open(devices_yaml) as f:
        devices = yaml.safe_load(f) or []
    for device in devices:
        if device.get("device_id") == device_id:
            return device.get("power_mode_w")
    known = sorted(d.get("device_id", "?") for d in devices)
    raise ValueError(f"Unknown device_id={device_id!r} in {devices_yaml}. Known device_ids: {known}.")


def _verify_power_mode(device_id: str) -> None:
    """Pre-flight check run before any measurement. Jetson-specific (not
    part of the hardware-agnostic contract -- nvpmodel is NVIDIA-only), and
    resolves the expected mode itself from the `device_id` it receives.
    Four outcomes:

      - device declares no expected mode (power_mode_w is None): nothing to
        check against, return silently -- get_power_mode() is not even
        queried.
      - get_power_mode() cannot answer (nvpmodel missing, permission error,
        unparseable output -> returns None): print a warning and continue.
        A check we could not perform is a documented gap, not a reason to
        block a measurement that may well be fine.
      - get_power_mode() answers and it does NOT match the declared
        power_mode_w: raise PowerModeMismatchError WITHOUT measuring -- no
        point spending the run (and the idle wait) on a result already
        known to be mislabeled.
      - it matches: return silently, no message.
    """
    expected_power_mode_w = _resolve_power_mode_w(device_id)
    if expected_power_mode_w is None:
        return

    actual = orin_power.get_power_mode()
    if actual is None:
        print(
            "[power] could not verify the active power mode "
            "(nvpmodel unavailable / permission error / unexpected output) "
            "-- continuing the measurement without verifying"
        )
        return

    try:
        actual_w: Optional[int] = int(str(actual).strip().rstrip("Ww"))
    except ValueError:
        actual_w = None  # e.g. "MAXN" -- an uncapped mode, not a watt value

    if actual_w == int(expected_power_mode_w):
        return

    raise PowerModeMismatchError(
        f"Active Jetson power mode is {actual!r} (parsed as {actual_w}W) but the device_id being "
        f"measured declares power_mode_w={expected_power_mode_w}W in configs/devices.yaml. "
        f"Refusing to measure -- the result would be mislabeled. Set the correct mode first "
        f"(e.g. `sudo nvpmodel -m <mode-id>` for {expected_power_mode_w}W) and retry."
    )


def get_power_meter(**power_meter_kwargs) -> orin_power.PowerMeter:
    global _power_meter
    if _power_meter is None:
        _power_meter = orin_power.PowerMeter(**power_meter_kwargs)
    return _power_meter


def _translate_energy_fields(power_results: Dict) -> Dict[str, float]:
    total = power_results["total"]
    duration_s = power_results["elapsed"]
    energy_raw_j = total["energy_total_j"]
    energy_net_j = total["energy_net_j"]
    return {
        "energy_raw_j": energy_raw_j,
        "energy_net_j": energy_net_j,
        "avg_power_w_raw": (energy_raw_j / duration_s) if duration_s else 0.0,
        "avg_power_w_net": (energy_net_j / duration_s) if duration_s else 0.0,
    }


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
    """Runs `run_fn_bound` under power measurement and writes the combined
    row. Returns {"row": ..., "path": ..., "power_results": ...,
    "sampling_path": ...} (power_results is orin_power's own raw results
    dict, useful for debugging/logging beyond what lands in the CSV;
    sampling_path is the per-device sampling-characterization CSV this
    window was also recorded in, see sampling_characterization.py).

    `perf_per_watt_fn(row, output, energy_fields) -> Optional[float]`
    computes perf_per_watt -- branch-specific formula, passed in by the
    caller (cli.py). See module docstring for why the formula itself
    (not just which field feeds it) differs between branches, and why it
    is passed explicitly rather than inferred here.

    Before anything else, _verify_power_mode(device_id) checks the Jetson's
    live nvpmodel state against the power_mode_w this device_id declares in
    configs/devices.yaml (resolved inside _verify_power_mode itself): a
    mismatch raises PowerModeMismatchError and nothing is measured or
    written; a failed query warns and continues; no declared mode / a
    match is silent."""
    _verify_power_mode(device_id)

    power_meter = get_power_meter(**(power_meter_kwargs or {}))
    power_results = power_meter.estimate_fn_power(run_fn_bound)

    output = power_results["output"]  # the wrapped runner's own parsed result -- captured here only
    energy_fields = _translate_energy_fields(power_results)

    row = build_result_row(run_id, device_id, repeat_id, output, requested_repeats)
    row.update(energy_fields)
    row["energy_per_inference"] = energy_fields["energy_net_j"] / output["num_inferences_measured"]

    perf_per_watt = perf_per_watt_fn(row, output, energy_fields)
    if perf_per_watt is not None:
        row["perf_per_watt"] = perf_per_watt

    out_path = append_result(device_id, row)

    # Sampling-window characterization (sampling_characterization.py): a
    # separate file, written AFTER the results row so a failure here can
    # never lose the measurement itself. Observational only -- nothing
    # above depends on it.
    sampling = power_results["sampling"]
    record = sampling_characterization.build_sampling_record(
        row, device_id, sampling_characterization.resolve_device_entry(device_id), sampling, power_results["elapsed"],
    )
    sampling_path = sampling_characterization.append_sampling_record(device_id, record)
    hz = record["effective_hz"]
    print(
        f"[power] sampling: {sampling['n_samples']} samples in {sampling['window_s']:.2f}s "
        f"({(f'{hz:.1f}' if hz != '' else 'n/a')} Hz effective, configured {sampling['configured_hz']:g} Hz)"
    )
    if not record["meets_threshold"]:
        print(
            f"[power] WARNING: window with {sampling['n_samples']} samples < pre-registered threshold "
            f"({sampling_characterization.MIN_SAMPLES_PER_WINDOW}) -- {run_id}"
        )
    return {"row": row, "path": out_path, "power_results": power_results, "sampling_path": sampling_path}
