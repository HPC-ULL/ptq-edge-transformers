"""Batch-drives runner_vit/runner_llm over a run_manifest_<package>.csv
(as written by verify_edge_bundle.py: columns run_id,repeat_id,
artifact_path), one row per invocation of the branch's own run_benchmark()
-- WITHOUT touching runner_vit/runner_llm's one-artifact-per-invocation CLI
design at all. This lives outside those packages entirely and is a second,
independent consumer of run_benchmark(), same pattern already used by
benchmarking_ondevice/power_logging/cli.py (which also calls run_benchmark
directly rather than shelling out to the runner's own CLI).

Resumable: before each row, checks the branch's own results_io.has_result
(run_id, device_id, repeat_id) against the REAL results/raw/
results_<device_id>.csv -- if that result_id is already recorded, the row
is skipped with an explicit printed notice (never silently), so relaunching
this script after an interruption (a crash, a killed process, a device
reboot) picks up exactly where it left off instead of repeating finished
work. Neither runner_vit nor runner_llm's run_benchmark() has this
skip-if-recorded check built in (they always run and overwrite via
append_result's replace-by-result_id) -- that resumability logic belongs
here, in the batch driver, not in the single-artifact CLIs.

With --with-power, "already done" means more than "the result_id exists":
_resumability_status also requires energy_per_inference to be populated. A
row recorded by an earlier plain (non--with-power) run of the same
manifest has a result_id but no energy data -- it is NOT skipped in that
case, it is re-run (printed as "[re-run] ... recorded without energy
data") so the row gets completed instead of permanently missing energy
columns. Without --with-power, behavior is unchanged: only the result_id
is checked, exactly as before.

--dry-run still performs the real has_result check (so it previews exactly
which rows a real run would skip vs. actually attempt) but never invokes
anything real -- it delegates to run_benchmark(..., dry_run=True), which
already knows how to print the command it would run without running it;
this script does not duplicate that logic.

--force overrides the "already done" outcome only -- a row that would
otherwise be "skip" is run anyway (printed as "[force-rerun] ..."), and
overwrites the existing row in place (results_io.append_result already
replaces by result_id, see runner_llm/results_io.py's module docstring --
no need to delete rows from the CSV first). Rows that are genuinely
missing, or --with-power rows still missing energy data, behave exactly
the same with or without --force -- it only matters for rows that were
already complete. Meant for redoing an already-measured batch under a
different --repeats (e.g. raising the LLM energy branch's N after a
convergence check showed the previous N was insufficient), not for
routine use.

Minimum uptime: before the first row that is actually going to run in this
invocation (never a skipped row), waits until the system has been up for
the device's `min_uptime_s` (configs/devices.yaml, optional per device;
ABSENT means 0 = no wait), reading /proc/uptime. It is a property of the
hardware, not of the batch -- the Jetson entries set 600, every other device
leaves it out -- so there is no command-line override to forget.
Reason: on the Jetson, decode throughput (tg) jumps by 13-24% at a fixed
time after boot, so rows measured before that point come out low -- this
was the "first rows of the batch are slow" pattern seen after every reboot
(changing the power mode reboots the board). Measured with continuous load
from minute 1 on Qwen, same row, one boot per power mode:
30W ~6.5 min (+17%), 15W ~7.3 min (+13%), 50W ~6.8 min (+24%); it stays
flat afterwards (checked up to 37 min). Confirmed on a second model: TinyLlama,
50W, step at ~6.1 min (+33%, tg 37.0 -> 49.2; prefill unchanged). Raw data and
captures in results/diagnostics/.
Ruled out along the way: temperature (rises smoothly, 45-52C, across the
step); idle time (180 s of idle does not reproduce it); warm-up load
(continuous load from minute 1 does not bring the step earlier, and a
system left idle until minute 11.5 already shows the high level on its
first measurement); and frequency scaling -- tegrastats shows CPU and
memory (EMC) frequencies constant across the step, and with `jetson_clocks`
(CPU/GPU/EMC pinned at maximum, CPU idle states off) the step is still
there (+20-21%, twice). What rises at the step is memory utilisation and
CPU-rail power at unchanged clocks; what changes at that time is NOT known.
`jetson_clocks` is deliberately NOT used as a fix: it pins the clocks to the
maximum and would invalidate measuring the power modes as such. 600 s is the
largest step time seen (7.3 min) plus ~2.7 min of margin, from a single
boot per mode: on a different board or L4T version, re-measure it with
scripts/uptime_probe.py before trusting it. A system that has been up
longer than the threshold (any long-running host) never waits.

A row with a blank repeat_id is valid for --branch llm (a deterministic
config -- baseline, legacy_corpus_none, etc. -- has no calibration-repeat
concept, see runner_llm/results_io.py) but never for --branch vit (ViT is
always stochastic); a blank repeat_id in a vit manifest is treated as a
real data error and raises, rather than being silently passed through.

Cooldown between consecutive runs -- pluggable strategy, not a fixed
mechanism, mirroring power_logging's own "common contract, per-hardware
implementation" pattern (a cooldown strategy is just a zero-arg callable;
build_parser/main pick which one to construct, nothing about the batch
loop itself changes per strategy):

  - Default: make_fixed_cooldown(seconds) -- sleeps a fixed
    --cooldown-seconds between runs. Works anywhere, no hardware
    assumptions.
  - --temp-cooldown: make_temperature_cooldown(...) -- polls a
    --temp-cooldown-impl module's get_cpu_temperature() -> float (the
    fixed contract, see resolve_temp_impl_module) until it drops below
    --temp-threshold-c, capped at --cooldown-max-seconds of total wait so
    a CPU that never cools back down (bad threshold, broken sensor, hot
    ambient) can't block the driver forever. --temp-cooldown-impl
    defaults to "jetson" (backed by power_logging/orin_power.py, which
    already exposes get_cpu_temperature()) -- any dotted module path
    exposing the same zero-argument, Celsius-returning contract works
    too, e.g. a different board's own thermal reader once it exists.
    resolve_temp_impl_module() resolves and imports the module BEFORE the
    batch loop starts (same fail-fast timing as resolve_power_impl_module)
    -- a bad name or an unimportable/incomplete module raises immediately,
    before any benchmark has run, never silently falls back to jetson.
  - --jetson-cooldown: DEPRECATED alias for --temp-cooldown (with
    --temp-cooldown-impl left at its "jetson" default) -- kept only for
    backward compatibility with scripts/instructions already written
    against it; prints a one-line deprecation notice and behaves
    identically to --temp-cooldown otherwise. New usage should prefer
    --temp-cooldown.

Neither strategy runs during --dry-run (nothing real happened, nothing to
cool down from) or after a skipped row (the device was never touched).
Both DO run after the last processed row, even though nothing follows it
-- a deliberate simplification: skipping that one only requires knowing
whether more real work follows, and the added lookahead isn't worth it for
one wasted wait at the very end of a batch.

--temp-log: purely observational, independent of --temp-cooldown --
prints CPU temperature (same --temp-cooldown-impl contract) right before
and right after each row that is actually run, so a long batch (e.g. many
LLM-on-GPU rows, which hold sustained load far longer than ViT's short
runs) leaves a real record of how temperature evolves -- useful evidence
for picking/validating a --temp-threshold-c before trusting it, without
being tied to --temp-cooldown itself being on. Same skip/--dry-run
exclusions as the cooldown strategies (never logs for a skipped row, never
during --dry-run); never alters cooldown timing or any other behavior.

--with-power: measures energy per row instead of only performance --
substitutes the plain run_benchmark() call with a measure_and_record()
call, reusing power_logging/cli.py's own branch resolution
(build_result_row, perf_per_watt_fn) rather than reimplementing it here --
there is exactly one place that picks build_result_row/perf_per_watt_fn
per branch (power_logging/cli.py's _branch_modules); this script just
calls into it.

--power-impl selects WHICH measure_and_record implementation is used --
"jetson" (the default, backed by power_logging/jetson_powermeter.py) is
built into POWER_IMPL_REGISTRY, but any dotted module path exposing a
measure_and_record with the exact signature
power_logging/dummy_powermeter.py defines works too, e.g.
--power-impl benchmarking_ondevice.power_logging.accelpowercape_powermeter
for a colleague's own hardware once they've built it following that same
contract. resolve_power_impl_module() resolves and imports the module
BEFORE the batch loop starts (same fail-fast timing as --jetson-cooldown's
orin_power check) -- a bad name or an unimportable/incomplete module
raises immediately, never silently falls back to jetson.

Still in-process, still one Python interpreter for the whole batch -- this
matters concretely for --with-power regardless of --power-impl: a
correctly-written implementation's own get_power_meter()-style cache (or
equivalent) is naturally shared across every row of the batch, not reset
per row, because this driver never shells out to a new `python -m ...`
per row. Confirmed for the jetson implementation by a real test
(test_with_power_reuses_the_same_power_meter_across_batch_rows), not
assumed. A subprocess-per-row design would silently defeat this (a fresh
interpreter re-measures idle on every single row).

--with-power has no effect during --dry-run (a dry run must never
construct a PowerMeter -- same bypass power_logging/cli.py's own --dry-run
already does) and combines freely with --temp-cooldown/--jetson-cooldown
(both active at once, energy measurement AND temperature-gated cooldown
between runs, is the expected real usage on the Jetson).

Usage:
    python scripts/run_batch.py --branch {vit,llm} \\
        --manifest run_manifest_<package>.csv --device-id <device_id> \\
        [--repeats N] [--dry-run] [--with-power [--power-impl jetson|<module>]] \\
        [--cooldown-seconds N | --temp-cooldown [--temp-cooldown-impl jetson|<module>]
            [--temp-threshold-c N] [--cooldown-max-seconds N]]
"""

from __future__ import annotations

import argparse
import csv
import functools
import importlib
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_COOLDOWN_SECONDS = 30.0
DEFAULT_TEMP_THRESHOLD_C = 55.0
DEFAULT_COOLDOWN_MAX_SECONDS = 120.0
DEFAULT_TEMP_POLL_INTERVAL_S = 5.0

DEFAULT_POWER_IMPL = "jetson"
DEFAULT_TEMP_COOLDOWN_IMPL = "jetson"
UPTIME_POLL_SECONDS = 30.0

# name -> dotted module path, for the --power-impl registry. Any dotted
# module path not listed here also works (see resolve_power_impl_module) --
# this registry only exists so the common/built-in case ("jetson") can be
# selected by a short name instead of the full path.
POWER_IMPL_REGISTRY = {
    "jetson": "benchmarking_ondevice.power_logging.jetson_powermeter",
}

# name -> dotted module path, for the --temp-cooldown-impl registry (same
# convention as POWER_IMPL_REGISTRY). The registered/resolved module must
# expose get_cpu_temperature() -> float (Celsius, callable with zero
# arguments) -- the fixed contract, matching what
# power_logging/orin_power.py already provides for the Jetson.
TEMP_COOLDOWN_IMPL_REGISTRY = {
    "jetson": "benchmarking_ondevice.power_logging.orin_power",
}

CooldownFn = Callable[[], None]


def _branch_modules(branch: str):
    """Returns (run_benchmark, results_io_module, default_repeats) for
    `branch`. Imported lazily and branch-conditionally -- same pattern as
    power_logging/cli.py's own _branch_modules. results_io_module itself
    (not just its has_result) is returned so _resumability_status can also
    reach result_id/results_csv_path for the --with-power completeness
    check, without a second, separate branch dispatch."""
    if branch == "vit":
        from benchmarking_ondevice.runner_vit import results_io
        from benchmarking_ondevice.runner_vit.runner import DEFAULT_REPEATS, run_benchmark
    elif branch == "llm":
        from benchmarking_ondevice.runner_llm import results_io
        from benchmarking_ondevice.runner_llm.runner import DEFAULT_REPEATS, run_benchmark
    else:
        raise ValueError(f"Unknown branch={branch!r}. Known branches: ['vit', 'llm'].")
    return run_benchmark, results_io, DEFAULT_REPEATS


def _resumability_status(
    results_io_module, run_id: str, device_id: str, repeat_id: Optional[int], *, require_energy: bool,
    force: bool = False,
) -> str:
    """Returns "skip" (already done, don't run), "incomplete" (recorded,
    but --with-power needs energy data that isn't there yet -- run it),
    "forced" (already complete, but --force says run it anyway), or "new"
    (never recorded -- run it).

    Without --with-power (require_energy=False): identical to
    results_io_module.has_result -- only whether the result_id exists,
    unchanged from before.

    With --with-power (require_energy=True): a result_id that exists but
    has no energy_per_inference populated (e.g. recorded by an earlier
    plain, non--with-power run of the same manifest) is treated as NOT
    done -- it gets re-run to complete it, rather than skipped forever
    with missing energy columns.

    --force (force=True) only changes what would otherwise be "skip":
    a genuinely-missing row is still "new" and an --with-power row missing
    energy is still "incomplete" either way -- --force's only job is
    overriding the "already complete, leave it alone" case.
    """
    if not require_energy:
        if not results_io_module.has_result(run_id, device_id, repeat_id):
            return "new"
        return "forced" if force else "skip"

    target_id = results_io_module.result_id(run_id, device_id, repeat_id)
    path = results_io_module.results_csv_path(device_id)
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["result_id"] == target_id:
                    has_energy = bool((row.get("energy_per_inference") or "").strip())
                    if not has_energy:
                        return "incomplete"
                    return "forced" if force else "skip"
    return "new"


def resolve_min_uptime_s(device_id: str) -> float:
    """The device's optional `min_uptime_s` from configs/devices.yaml; 0.0
    (no wait) when the entry does not have it. Anything but a number >= 0
    raises -- a typo must not silently disable the wait."""
    from benchmarking_ondevice.runner_vit.devices import resolve_device

    value = resolve_device(device_id).get("min_uptime_s")
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"device_id={device_id!r}: min_uptime_s must be a number >= 0, got {value!r}")
    return float(value)


def read_uptime_seconds() -> float:
    with open("/proc/uptime") as f:
        return float(f.read().split()[0])


def wait_for_min_uptime(
    min_uptime_s: float, *, dry_run: bool = False,
    read_uptime: Callable[[], float] = read_uptime_seconds, sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Blocks until the system uptime reaches `min_uptime_s` (see the module
    docstring for why). Returns the seconds waited. Never waits under
    dry_run (only reports what it would do). If /proc/uptime cannot be read
    (non-Linux host), warns and continues rather than blocking the batch."""
    try:
        uptime = read_uptime()
    except OSError as e:
        print(f"[uptime] could not read the uptime ({e}) -- continuing without waiting")
        return 0.0
    if uptime >= min_uptime_s:
        return 0.0
    remaining = min_uptime_s - uptime
    verb = "would wait" if dry_run else "waiting"
    print(
        f"[uptime] {uptime:.0f}s since boot < {min_uptime_s:g}s: {verb} {remaining:.0f}s "
        f"before the first row (see docstring)"
    )
    if dry_run:
        return 0.0
    waited = 0.0
    while True:
        remaining = min_uptime_s - read_uptime()
        if remaining <= 0:
            break
        step = min(remaining, UPTIME_POLL_SECONDS)
        sleep(step)
        waited += step
    print(f"[uptime] ready ({read_uptime():.0f}s since boot)")
    return waited


def resolve_power_impl_module(power_impl: str):
    """Resolves --power-impl to an actual imported Python module exposing
    measure_and_record(...) with the exact signature
    power_logging/dummy_powermeter.py defines. `power_impl` is either:

      - a name registered in POWER_IMPL_REGISTRY (built-in: "jetson"), or
      - any dotted module path -- a hardware implementation that follows
        the same dummy_powermeter.py contract but isn't (yet) registered
        by a short name, e.g.
        "benchmarking_ondevice.power_logging.accelpowercape_powermeter".

    Called BEFORE the batch loop starts (same fail-fast timing as
    resolve_temp_impl_module's own check) -- an unknown name or an
    unimportable/incomplete module raises immediately here, never
    silently falls back to jetson, and never fails mid-batch after some
    rows already ran.
    """
    module_path = POWER_IMPL_REGISTRY.get(power_impl, power_impl)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise RuntimeError(
            f"--power-impl {power_impl!r} (resolved to module {module_path!r}) could not be imported: {e}. "
            "Use the default 'jetson' implementation instead, or ship this module alongside this script."
        ) from e
    if not hasattr(module, "measure_and_record"):
        raise RuntimeError(
            f"--power-impl {power_impl!r} (resolved to module {module_path!r}) has no "
            "measure_and_record(...) -- it must follow the exact contract "
            "power_logging/dummy_powermeter.py defines."
        )
    return module


def _build_power_context(branch: str, power_impl_module):
    """Returns (measure_and_record, build_result_row, perf_per_watt_fn) for
    --with-power. Reuses power_logging/cli.py's own _branch_modules for
    build_result_row/perf_per_watt_fn (hardware-agnostic, branch-specific)
    instead of reimplementing that dispatch here -- power_impl_module is
    the ONLY hardware-specific piece, already resolved and imported by the
    caller (main(), via resolve_power_impl_module), never hardcoded to
    jetson_powermeter here."""
    from benchmarking_ondevice.power_logging import cli as power_cli

    _run_benchmark, build_result_row, _default_repeats, perf_per_watt_fn = power_cli._branch_modules(branch)
    return power_impl_module.measure_and_record, build_result_row, perf_per_watt_fn


def resolve_temp_impl_module(temp_impl: str):
    """Resolves --temp-cooldown-impl to an actual imported Python module
    exposing get_cpu_temperature() -> float (Celsius, callable with zero
    arguments) -- the fixed contract for temperature-based cooldown,
    matching what power_logging/orin_power.py already provides for the
    Jetson (its own `zone` parameter defaults to 0, so it's already
    callable with zero arguments). `temp_impl` is either:

      - a name registered in TEMP_COOLDOWN_IMPL_REGISTRY (built-in:
        "jetson"), or
      - any dotted module path -- a different board's own temperature
        reader that exposes the same get_cpu_temperature() -> float
        contract but isn't (yet) registered by a short name.

    Called BEFORE the batch loop starts (same fail-fast timing as
    resolve_power_impl_module) -- an unknown name or an
    unimportable/incomplete module raises immediately here, never
    silently falls back to jetson, and never fails mid-batch after some
    rows already ran.
    """
    module_path = TEMP_COOLDOWN_IMPL_REGISTRY.get(temp_impl, temp_impl)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise RuntimeError(
            f"--temp-cooldown-impl {temp_impl!r} (resolved to module {module_path!r}) could not be "
            f"imported: {e}. Use the default fixed --cooldown-seconds strategy instead, or ship this "
            "module alongside this script."
        ) from e
    if not hasattr(module, "get_cpu_temperature"):
        raise RuntimeError(
            f"--temp-cooldown-impl {temp_impl!r} (resolved to module {module_path!r}) has no "
            "get_cpu_temperature() -- it must expose a zero-argument callable returning the "
            "current CPU temperature in Celsius."
        )
    return module


def read_manifest(path: Path, branch: str) -> Iterator[Tuple[str, Optional[int], str]]:
    """Yields (run_id, repeat_id, artifact_path) per row. repeat_id is None
    for a blank column -- valid only for branch == "llm" (a deterministic
    config); a blank repeat_id for branch == "vit" raises, since ViT is
    always stochastic and a blank there means the manifest doesn't match
    the branch it's being run against."""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            run_id = row["run_id"]
            raw_repeat_id = row["repeat_id"].strip()
            if raw_repeat_id == "":
                if branch == "vit":
                    raise ValueError(
                        f"{path}: row for run_id={run_id!r} has a blank repeat_id, but ViT is always "
                        "stochastic -- this manifest looks corrupted, or built for the wrong branch."
                    )
                repeat_id = None
            else:
                repeat_id = int(raw_repeat_id)
            yield run_id, repeat_id, row["artifact_path"]


def make_fixed_cooldown(seconds: float) -> CooldownFn:
    """Default cooldown strategy: sleep a fixed duration. Works on any
    device, no hardware assumptions."""

    def _cooldown() -> None:
        if seconds > 0:
            print(f"[cooldown] waiting {seconds:g}s (fixed strategy)...")
            time.sleep(seconds)

    return _cooldown


def make_temperature_cooldown(
    threshold_c: float,
    max_wait_seconds: float,
    temp_impl_module,
    poll_interval_s: float = DEFAULT_TEMP_POLL_INTERVAL_S,
) -> CooldownFn:
    """Temperature-based cooldown strategy: polls temp_impl_module's
    get_cpu_temperature() -> float (the fixed contract, see
    resolve_temp_impl_module) until it drops below `threshold_c`, capped
    at `max_wait_seconds` of total waiting so a temperature that never
    drops (bad threshold, broken sensor, hot ambient) can't block the
    driver forever.

    temp_impl_module is already resolved and imported by the caller
    (main(), via resolve_temp_impl_module) -- never hardcoded to
    orin_power here, so any module following the same contract works
    identically.
    """
    get_temperature_fn = temp_impl_module.get_cpu_temperature

    def _cooldown() -> None:
        start = time.monotonic()
        temp = get_temperature_fn()
        while temp >= threshold_c:
            elapsed = time.monotonic() - start
            if elapsed >= max_wait_seconds:
                print(
                    f"[cooldown] safety limit ({max_wait_seconds:g}s) reached -- temperature "
                    f"still at {temp:g}C (threshold {threshold_c:g}C), continuing anyway"
                )
                return
            print(f"[cooldown] temperature {temp:g}C >= threshold {threshold_c:g}C, waiting...")
            time.sleep(poll_interval_s)
            temp = get_temperature_fn()

    return _cooldown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.run_batch",
        description="Batch-drive runner_vit/runner_llm over a run_manifest_<package>.csv, "
                    "resumable across interruptions (skips rows already in results/raw/).",
    )
    parser.add_argument("--branch", required=True, choices=["vit", "llm"], help="Which runner to drive.")
    parser.add_argument("--manifest", required=True, help="Path to run_manifest_<package>.csv.")
    parser.add_argument(
        "--device-id", required=True,
        help="device_id from configs/devices.yaml -- fixed for the whole batch.",
    )
    parser.add_argument(
        "--repeats", type=int, default=None,
        help="Passed through to run_benchmark (-r). Defaults to the branch's own default if omitted.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run per row (still checking real results/raw/ for skips), "
             "without invoking anything real.",
    )
    parser.add_argument(
        "--with-power", action="store_true",
        help="Measure energy per row instead of only performance -- writes the combined "
             "performance+energy row, using the --power-impl implementation. No effect on "
             "--dry-run (never constructs a PowerMeter there).",
    )
    parser.add_argument(
        "--power-impl", default=DEFAULT_POWER_IMPL,
        help=f"--with-power only: which measure_and_record implementation to use -- a name "
             f"registered in POWER_IMPL_REGISTRY (built-in: {sorted(POWER_IMPL_REGISTRY)}) or any "
             f"dotted module path exposing the same contract as power_logging/dummy_powermeter.py "
             f"(e.g. a colleague's own accelpowercape implementation). "
             f"Default {DEFAULT_POWER_IMPL!r}.",
    )
    parser.add_argument(
        "--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN_SECONDS,
        help=f"Fixed-wait cooldown between consecutive runs (default strategy, ignored if "
             f"--temp-cooldown/--jetson-cooldown is set). Default {DEFAULT_COOLDOWN_SECONDS:g}s.",
    )
    parser.add_argument(
        "--temp-cooldown", action="store_true",
        help="Use a temperature-based cooldown strategy (--temp-cooldown-impl) instead of the "
             "fixed wait -- requires that implementation's module to be importable, fails "
             "immediately if not.",
    )
    parser.add_argument(
        "--temp-cooldown-impl", default=DEFAULT_TEMP_COOLDOWN_IMPL,
        help=f"--temp-cooldown only: which get_cpu_temperature() implementation to use -- a name "
             f"registered in TEMP_COOLDOWN_IMPL_REGISTRY (built-in: {sorted(TEMP_COOLDOWN_IMPL_REGISTRY)}) "
             f"or any dotted module path exposing get_cpu_temperature() -> float. "
             f"Default {DEFAULT_TEMP_COOLDOWN_IMPL!r}.",
    )
    parser.add_argument(
        "--jetson-cooldown", action="store_true",
        help="DEPRECATED: alias for --temp-cooldown (--temp-cooldown-impl already defaults to "
             "'jetson') -- kept only for backward compatibility, prints a deprecation notice. "
             "Use --temp-cooldown instead.",
    )
    parser.add_argument(
        "--temp-threshold-c", type=float, default=DEFAULT_TEMP_THRESHOLD_C,
        help=f"--temp-cooldown/--jetson-cooldown only: wait until CPU temperature drops below this "
             f"(Celsius). Default {DEFAULT_TEMP_THRESHOLD_C:g}C.",
    )
    parser.add_argument(
        "--cooldown-max-seconds", type=float, default=DEFAULT_COOLDOWN_MAX_SECONDS,
        help=f"--temp-cooldown/--jetson-cooldown only: safety cap -- stop waiting for the "
             f"temperature to drop after this many seconds regardless, so the driver never blocks "
             f"forever. Default {DEFAULT_COOLDOWN_MAX_SECONDS:g}s.",
    )
    parser.add_argument(
        "--temp-log", action="store_true",
        help="Print CPU temperature (--temp-cooldown-impl) before and after each row that is "
             "actually run (never for a skipped row, never during --dry-run) -- purely "
             "observational, independent of --temp-cooldown: does not gate or alter cooldown "
             "timing/behavior, just adds the two log lines. Useful to build a real record of how "
             "temperature evolves over a long batch (e.g. sustained LLM-on-GPU rows) before "
             "trusting a --temp-threshold-c value.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run every row in the manifest even if results/raw/ already has a complete result "
             "for it (energy_per_inference populated, or just the result_id when not --with-power) "
             "-- e.g. redoing a whole batch at a different --repeats. Does not require deleting rows "
             "from the CSV first: append_result already replaces any existing row with the same "
             "result_id in place, so a forced re-run overwrites just that row. --dry-run with "
             "--force previews forced rows as '[force-rerun]' instead of running them.",
    )
    return parser


def _build_cooldown(args) -> CooldownFn:
    if args.jetson_cooldown:
        print("[deprecation] --jetson-cooldown is obsolete, use --temp-cooldown "
              "(with --temp-cooldown-impl jetson, which is already the default)")
    if args.temp_cooldown or args.jetson_cooldown:
        temp_impl_module = resolve_temp_impl_module(args.temp_cooldown_impl)  # fails fast, before the loop
        return make_temperature_cooldown(args.temp_threshold_c, args.cooldown_max_seconds, temp_impl_module)
    return make_fixed_cooldown(args.cooldown_seconds)


def _resolve_temp_log_module(args):
    """--temp-log only: resolves --temp-cooldown-impl to a get_cpu_temperature()
    module for logging, independently of whether --temp-cooldown/--jetson-cooldown
    is also set -- reuses the same resolve_temp_impl_module() (and thus the same
    TEMP_COOLDOWN_IMPL_REGISTRY/dotted-path contract, same fail-fast-before-the-loop
    timing) rather than a second flag, since it is the same kind of module either
    way. Returns None if --temp-log wasn't passed, so the loop can skip logging
    entirely without importing anything device-specific."""
    if not args.temp_log:
        return None
    return resolve_temp_impl_module(args.temp_cooldown_impl)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_benchmark, results_io_module, default_repeats = _branch_modules(args.branch)
    repeats = args.repeats if args.repeats is not None else default_repeats
    cooldown = _build_cooldown(args)  # resolved (and, for --jetson-cooldown, validated) before the loop starts
    temp_log_module = _resolve_temp_log_module(args)  # None unless --temp-log; resolved before the loop too

    power_ctx = None
    if args.with_power:
        power_impl_module = resolve_power_impl_module(args.power_impl)  # resolved before the loop, fails fast
        power_ctx = _build_power_context(args.branch, power_impl_module)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        parser.error(f"--manifest not found: {manifest_path}")
    min_uptime_s = resolve_min_uptime_s(args.device_id)  # devices.yaml; resolved before the loop, fails fast

    total = run_count = skip_count = 0
    uptime_checked = False
    for run_id, repeat_id, artifact_path in read_manifest(manifest_path, args.branch):
        total += 1
        label = run_id + (f" rep{repeat_id}" if repeat_id is not None else "")

        status = _resumability_status(
            results_io_module, run_id, args.device_id, repeat_id, require_energy=args.with_power,
            force=args.force,
        )
        if status == "skip":
            print(f"[skip] {label}: already in results/raw/results_{args.device_id}.csv")
            skip_count += 1
            continue
        if status == "incomplete":
            print(f"[re-run] {label}: recorded without energy data -- re-running to complete it")
        if status == "forced":
            print(f"[force-rerun] {label}: a complete result already existed -- re-running because of --force")

        if not uptime_checked and min_uptime_s > 0:
            wait_for_min_uptime(min_uptime_s, dry_run=args.dry_run)
        uptime_checked = True

        if temp_log_module is not None and not args.dry_run:
            print(f"[temp] {label}: start {temp_log_module.get_cpu_temperature():g}C")

        if args.dry_run:
            # --with-power never applies to --dry-run: no PowerMeter is
            # ever constructed here, same bypass power_logging/cli.py's own
            # --dry-run already does. The plain run_benchmark's own
            # dry-run print is reused, not duplicated.
            run_benchmark(run_id, artifact_path, args.device_id, repeat_id, repeats=repeats, dry_run=True)
        elif power_ctx is not None:
            measure_and_record, build_result_row, perf_per_watt_fn = power_ctx
            run_fn_bound = functools.partial(
                run_benchmark, run_id, artifact_path, args.device_id, repeat_id,
                repeats=repeats, write_result=False,
            )
            measure_and_record(
                run_fn_bound, build_result_row, run_id, args.device_id, repeat_id, repeats,
                perf_per_watt_fn=perf_per_watt_fn,
            )
        else:
            run_benchmark(run_id, artifact_path, args.device_id, repeat_id, repeats=repeats, dry_run=False)

        if temp_log_module is not None and not args.dry_run:
            print(f"[temp] {label}: end {temp_log_module.get_cpu_temperature():g}C")

        run_count += 1
        if not args.dry_run:
            cooldown()

    verb = "would run (dry-run)" if args.dry_run else "run"
    print(f"\n[summary] {total} manifest row(s) -- {skip_count} already existed (skipped), {run_count} {verb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
