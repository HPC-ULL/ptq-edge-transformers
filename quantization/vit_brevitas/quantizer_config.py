"""Resolve (calibration_algorithm, granularity, symmetry, bitwidth) manifest
fields into concrete Brevitas weight/activation quantizer classes.

bitwidth was NOT actually consumed here for a while after Phase 3's
manifest/CLI plumbing was built (this docstring used to say "only int8 is
wired up... Phase 3's INT6/INT4 is a separate module built once Phase 2
results exist" -- that module was never actually built, silently leaving
every Phase 3 "int6"/"int4" row to fall through to the resolvers' hardcoded
INT8 presets). Caught only once real Phase 3 accuracy results for INT6 and
INT4 came back byte-identical to each other and to the known Phase 1 INT8
baseline -- see _bit_width_override()'s docstring and
investigations/probe_qcdq_bitwidth_support.sh for the export-path
viability check that preceded actually wiring this up.

Three design decisions this module makes, not literally pinned down by
calibration_protocol_vit.md's factor names, flagged here so they can be
revisited if any reading turns out wrong:

- `granularity` (per_tensor/per_channel) controls WEIGHT quantization only;
  activations stay per-tensor regardless. Per-channel *activation*
  quantization is rare/impractical in PTQ (activation statistics vary per
  input, not just per weight tensor) -- this follows the standard
  convention (and Brevitas's own preset naming) of applying "per-channel"
  to weights.
- `symmetry` (symmetric/asymmetric) controls ACTIVATION quantization only;
  weights stay symmetric regardless. Weight distributions are close to
  zero-centered in practice and are almost universally quantized
  symmetrically; symmetric-vs-asymmetric is the axis actually varied for
  *activations* in PTQ literature (an asymmetric/unsigned range captures a
  post-ReLU/GELU activation distribution better than a symmetric one).
- `calibration_algorithm=percentile`/`entropy` change ACTIVATION
  quantization only; weight quantization stays at baseline (minmax) for
  both. Confirmed by reading the installed Brevitas source
  (brevitas/core/stats/stats_op.py, brevitas/quant/base.py): Brevitas
  itself only ships percentile/entropy machinery wired for *runtime*
  (activation) stats -- there is no weight-percentile or weight-entropy
  preset anywhere in the package, unlike MSE, which Brevitas explicitly
  supports for both weights and activations
  (Int8WeightPerTensorFloatMSE + Int8ActPerTensorFloatMSE both exist).
  This mirrors standard PTQ practice too: percentile/entropy calibration is
  an activation-outlier-clipping technique (this is exactly how NVIDIA's
  TensorRT calibrators use it), not something applied to static weight
  tensors.

Because the manifest is generated OFAT (configs/common.py's ofat_variants:
never more than one factor away from the baseline), Phase 1-2 only ever
needs these combinations -- e.g. never calibration_algorithm=mse together
with granularity=per_channel -- so the lookup tables below only cover what
the manifest can actually produce, not the full cross product.

Confidence levels:
- minmax / per_tensor (weights): CONFIRMED (quantization/vit_brevitas/
  spike_test.py ran this for real on all 3 models; Int8WeightPerTensorFloat
  is genuinely MaxStatsScaling-based, a literal per-tensor abs-max, unlike
  the activation quantizer below).
- minmax / symmetric (activations): Int8ActPerTensorFloatMinMax, added
  after discovering Brevitas's own "standard" Int8ActPerTensorFloat is
  *not* literal minmax -- it's built on ParamFromRuntimePercentileScaling
  internally (high_percentile_q=99.999), confirmed by reading the installed
  source. Int8ActPerTensorFloatMinMax swaps in ParamFromRuntimeMinMaxScaling
  (StatsOp.MIN_MAX) instead, composed the same way Int8ActPerTensorFloat
  itself is (same IntQuant + PerTensorFloatScaling8bit + ActQuantSolver
  pieces) -- see _minmax_act_quant()'s docstring. NOT yet run for real:
  switching this invalidates the 4 deit_tiny rows already confirmed under
  the old (percentile-based) "minmax", which need regenerating with
  --force and re-checking before trusting them again.
- per_channel, mse: were CONFIRMED for deit_tiny under the OLD
  percentile-based "minmax" baseline (72.85-73.70% / 90.95-91.20% top1/top5,
  vs. that baseline's 73.25%/91.20%) -- both change weight quantization
  only, so those numbers still describe the same weight-side behavior.
  Still, re-verify under the new baseline before trusting them, since
  "baseline" itself changed.
- asymmetric (activations): added after discovering Brevitas's own
  "standard" asymmetric preset (ShiftedUint8ActPerTensorFloat) is *also
  not* literal minmax -- mirroring the symmetric discovery exactly, it's
  built on ParamFromRuntimePercentileIntervalScaling internally
  (high_percentile_q=99.999, low_percentile_q=0.001), confirmed by reading
  its real MRO on the installed source. Discovered via a real ValueError
  when Phase 3's combined config needed (percentile, asymmetric) and
  exposed that (minmax, asymmetric) was never actually verified as literal
  minmax either. Per explicit user decision, given the same treatment as
  the symmetric case: this invalidates the symmetry=asymmetric Phase 1-2
  row for all 3 models, which need regenerating (including re-running
  Phase 3 selection, since it depends on these results) before trusting
  them again.

  FIRST attempt composed a new class from smaller pieces (ShiftedMinUintQuant
  + ParamFromRuntimeMinMaxScaling, mirroring exactly how the symmetric case
  was fixed) -- this resolved fine at the class-attribute level (unit tests
  passed) but FAILED for real at quantization time with `DependencyError:
  ... can not resolve attribute 'scaling_stats_input_concat_dim'`, a real
  Brevitas dependency the shifted/asymmetric solver path needs that neither
  piece alone provides.

  SECOND attempt subclassed ShiftedUint8ActPerTensorFloat instead
  (_shifted_percentile_interval_act_quant), pushing percentile bounds to
  100.0/0.0 -- resolved AND passed unit tests, but ALSO failed for real,
  this time with `RuntimeError: kthvalue(): selected number k out of range
  for dimension 0`. Confirmed against the real source
  (brevitas/core/stats/stats_op.py's PercentileInterval.forward): percentile
  bounds are converted to a 1-indexed kthvalue() rank via
  `low_k = ceil(0.01*low_q*x.numel())`, and low_q=0.0 gives low_k=0, which
  kthvalue() rejects. high_q=100.0 has no such issue (its formula rounds to
  exactly x.numel(), a valid rank). Fixed by using _LOW_Q_EPSILON (1e-9)
  instead of literal 0.0 for low_q -- still mathematically EXACT, not an
  approximation: ceil() rounds any value in (0, 100/N] up to exactly 1, and
  kthvalue(x, 1) IS the literal minimum, so this is exact for any
  realistic tensor size (safe up to roughly 10^11 elements). See
  _minmax_asymmetric_act_quant()'s and _LOW_Q_EPSILON's own comments.

  Two real, different runtime failures in a row on the same fix are a
  reminder that class-attribute-level unit tests here (checking identity,
  scaling_stats_op, percentile bounds) confirm composition, not full
  resolvability or numerical validity at real quantization time -- both
  gaps only surfaced by actually running quantization for real.
- percentile: composed as a plain subclass of Int8ActPerTensorFloat (the
  percentile-based class Brevitas itself calls "standard"), overriding just
  high_percentile_q to 99.9 -- see _percentile_act_quant()'s docstring.
  Spike-tested OK (quantization/vit_brevitas/spike_test_percentile_entropy.py);
  confirmed for real on deit_tiny (70.95%/89.75% under the corrected minmax
  baseline).
- percentile, asymmetric: needed once Phase 3 combined deit_tiny's
  (or another model's) independently-winning calibration_algorithm=
  percentile and symmetry=asymmetric factors into one config (Section 9.1's
  declared OFAT-independence limitation surfacing for real) -- Phase 1-2
  never needs this combination, so it was never wired up until Phase 3 hit
  it. Composed as a subclass of ShiftedUint8ActPerTensorFloat (Brevitas's
  own asymmetric "standard", percentile-interval-based -- see the
  asymmetric confidence entry above) overriding both percentile bounds --
  see _percentile_asymmetric_act_quant()'s docstring. Not yet run for real.
- deployment-set combinations (configs/vit_deployment_manifest.py: the
  granularity/symmetry factors applied on top of each model's chosen
  calibration, INT8 with output_quant) -- never needed by the ablation:
  - weights mse + per_channel: Brevitas's own Int8WeightPerChannelFloatMSE,
    the per-channel twin of the Int8WeightPerTensorFloatMSE already used.
  - weights percentile + per_channel: Int8WeightPerChannelFloat, the same
    plain per-channel weights as minmax -- percentile only touches
    activations (see above), exactly as percentile + per_tensor maps to the
    plain per-tensor weights.
  - activations mse + asymmetric: Brevitas's own
    ShiftedUint8ActPerTensorFloatMSE, the asymmetric twin of
    Int8ActPerTensorFloatMSE -- a stock preset, not recomposed from pieces
    (see the asymmetric entry above for why that matters), with dtype/device
    declared: the stock class fails for real without them
    (_mse_asymmetric_act_quant()'s docstring).
  The two per_channel ones ran for real in the deployment gate (DeiT-Tiny);
  mse + asymmetric passed the spike, not yet the gate.
- entropy: BLOCKED (raises NotImplementedError), for all 3 models. Initially
  composed by setting scaling_stats_impl directly to
  brevitas.core.stats.stats_op.KLMinimizerThreshold (Brevitas's own
  histogram + KL-divergence threshold search, based on the MXNet
  quantization calibrator), bypassing the scaling_stats_op enum dispatch
  since StatsOp has no ENTROPY member to select it through -- this part
  works and is not itself the problem. Blocked after a real ablation run
  showed severe, consistent collapse on deit_tiny specifically (24-27% top1
  in 2/3 repeats, 12.85% in the third, vs. ~73% in every other
  configuration -- deit_tiny is otherwise stable across all 7 other
  configs, so this pointed at the implementation, not a real architecture
  finding). Root-caused via investigations/diagnose_entropy_wiring.py and
  investigations/diagnose_entropy_num_bins_*.py to two independent,
  confirmed issues in the installed Brevitas 0.13.0, not this project's
  wiring (calibration data does reach the quantizer; the real pipeline's
  resolved scale matches a standalone reproduction closely):
  1. A genuine off-by-one bug in KLMinimizerThreshold.forward() (brevitas/
     core/stats/stats_op.py): the histogram bin-merging loop's last
     iteration computes p_bin_idx_stop = num_bins//2 + num_bins//2 + 1,
     which equals num_bins+1 (one past hist_edges' last valid index) when
     num_bins is EVEN, but equals num_bins (valid) when num_bins is ODD due
     to floor-division truncation -- an IndexError for every even num_bins
     tested (256/384/512/750/1500/2048/4096), success only for the odd
     default (1001). Confirmed against the actual Brevitas source; full
     writeup and a suggested fix in
     investigations/brevitas_issue_klminimizerthreshold_parity_bug.md.
  2. Independent of that crash: sweeping ODD num_bins values that avoid the
     bug (501/1001/2001/3001/5001/8001) on the same real captured tensor
     (deit_tiny, blocks.0.attn.qkv) gave threshold/absmax ranging 14.89%-
     34.22% with no convergence pattern -- the resulting threshold is
     itself unstable, not just inaccessible for most num_bins values. This
     means num_bins=1001 "working" was never evidence of a validated,
     good default; it only avoided finding (1) by chance (being odd).
  Excluded from the ablation for lack of reliability of the method in this
  Brevitas version, not because of the crash itself -- see
  investigations/diagnose_entropy_num_bins_odd_sweep.py for the
  reproduction. Revisit if a future Brevitas release fixes finding (1) and
  demonstrates stable behavior across num_bins for (2).
"""

from __future__ import annotations

from typing import Tuple


_PERCENTILE_HIGH_Q = 99.9

# Phase 3's aggressive bit-widths, added once the QCDQ export path was
# confirmed to support bit_width <= 8 via clipping within the same int8/
# uint8 ONNX container (no opset bump, no native ONNX INT4 needed -- see
# investigations/probe_qcdq_bitwidth_support.sh's findings, cross-checked
# against the real installed Brevitas source, not assumed). "int8" maps to
# every existing Phase 1-2 quantizer unchanged (see _bit_width_override).
_BITWIDTH_VALUES = {"int8": 8, "int6": 6, "int4": 4}


def _bit_width_override(base_class, bw: int):
    """Returns base_class unchanged for bw==8 (every quantizer in this
    module's own native default); otherwise a subclass overriding just
    bit_width, following the pattern an isolated check already confirmed
    works with no MRO conflicts or missing dependencies for bit_width=6/4
    on Brevitas's own presets (see
    investigations/probe_qcdq_bitwidth_support.sh's QuantLinear check).

    The attribute MUST be set inside the class body, not via a
    post-creation assignment: a first attempt built the subclass via
    type(name, (base_class,), {}) and then did override.bit_width = bw
    afterward, specifically to dodge the OTHER bug below -- but Brevitas's
    ExtendedInjector metaclass explicitly forbids modifying an
    Injector-derived class after it's defined (`DependencyError: 'Injector'
    modification is not allowed`), confirmed by hitting it for real.

    The parameter is named `bw`, not `bit_width`, specifically to avoid
    THAT other bug already hit elsewhere in this module (see
    _shifted_percentile_interval_act_quant's history): a class-body
    assignment `bit_width = bit_width` shadows itself and raises NameError,
    since Python treats an assigned name as local to the class body's own
    namespace for the entire body, including the RHS of that same
    assignment. Renaming the parameter sidesteps that collision while still
    satisfying Brevitas's "set it in the body" requirement.
    """
    if bw == 8:
        return base_class

    class _BitWidthOverride(base_class):
        bit_width = bw

    return _BitWidthOverride


def _resolve_bitwidth(bitwidth: str) -> int:
    try:
        return _BITWIDTH_VALUES[bitwidth]
    except KeyError:
        raise ValueError(
            f"Unknown bitwidth={bitwidth!r}. Known values: {sorted(_BITWIDTH_VALUES)}."
        ) from None

# See _minmax_asymmetric_act_quant()'s docstring: Brevitas's real
# PercentileInterval.forward() computes low_k = ceil(0.01*low_q*N) as a
# 1-indexed kthvalue() rank, so low_q=0.0 gives low_k=0 (invalid --
# confirmed by hitting RuntimeError: "selected number k out of range" for
# real). Any low_q in (0, 100/N] gives low_k=1 exactly (the literal
# minimum), so this epsilon is exact, not an approximation, for any
# realistic tensor size (safe up to roughly 10^11 elements).
_LOW_Q_EPSILON = 1e-9


def _minmax_act_quant():
    """INT8, per-tensor, symmetric activation quantizer using a literal
    runtime min/max stat (StatsOp.MIN_MAX) -- unlike Brevitas's own
    Int8ActPerTensorFloat, which is percentile-based internally (see
    module docstring).

    Composed the same way Int8ActPerTensorFloat itself is composed (same
    IntQuant + PerTensorFloatScaling8bit + ActQuantSolver base pieces,
    confirmed by reading the installed source), swapping only the
    scale-stats mixin: ParamFromRuntimeMinMaxScaling is structurally
    identical to ParamFromRuntimePercentileScaling (same scaling_impl_type,
    collect_stats_steps=300, scaling_min_val=1e-10), just
    scaling_stats_op=StatsOp.MIN_MAX instead of PERCENTILE. Unlike
    percentile, this mixin is *not* already embedded in any of the other
    three base pieces, so this doesn't hit the "duplicate base at a
    different MRO position" conflict percentile did.

    (Int8ActPerTensorFloatMinMaxInit, a same-named-sounding preset already
    in brevitas.quant, was considered and ruled out: its docstring says its
    scale is "initialized from user-defined min and max values" -- it needs
    min_val/max_val passed in by hand at construction time, meant for
    initializing a *trainable* QAT parameter, not for computing a scale
    from observed calibration data the way PTQ needs.)
    """
    from brevitas.quant.base import IntQuant, ParamFromRuntimeMinMaxScaling, PerTensorFloatScaling8bit
    from brevitas.quant.solver.act import ActQuantSolver

    class Int8ActPerTensorFloatMinMax(
        IntQuant, ParamFromRuntimeMinMaxScaling, PerTensorFloatScaling8bit, ActQuantSolver
    ):
        pass

    return Int8ActPerTensorFloatMinMax


def _percentile_act_quant():
    """INT8, per-tensor, symmetric activation quantizer -- a subclass of
    Int8ActPerTensorFloat (Brevitas's own percentile-based "standard" INT8
    activation quantizer, see module docstring) with a more aggressive
    percentile clip than its default.

    Int8ActPerTensorFloat is *already* built on ParamFromRuntimePercentileScaling
    internally, with high_percentile_q=99.999 (i.e. it already clips only
    the most extreme 0.001% of observed values; in practice this is very
    close to true absolute max, but it is technically percentile-based, not
    literal minmax -- see _minmax_act_quant(), added once this was
    discovered, for the literal-minmax baseline). Mixing
    ParamFromRuntimePercentileScaling into it a second time at a different
    MRO position is what originally raised "Cannot create a consistent
    MRO" here -- the class was already there.

    99.9 clips visibly more than Int8ActPerTensorFloat's default 99.999
    while still being a commonly-cited "safe" percentile calibration value
    in the PTQ literature (not aggressive enough to clip a large fraction
    of the real activation range). Adjust _PERCENTILE_HIGH_Q above if a
    different comparison point is wanted.
    """
    from brevitas.quant.scaled_int import Int8ActPerTensorFloat

    class Int8ActPerTensorFloatPercentile(Int8ActPerTensorFloat):
        high_percentile_q = _PERCENTILE_HIGH_Q

    return Int8ActPerTensorFloatPercentile


def _shifted_percentile_interval_act_quant(high_q: float, low_q: float):
    """Shared builder for both asymmetric activation quantizers below --
    subclasses ShiftedUint8ActPerTensorFloat (Brevitas's own asymmetric
    "standard" INT8 activation quantizer) overriding just its percentile
    bounds, reusing 100% of its internal dependency-injection plumbing
    rather than recomposing from smaller pieces.

    That "recompose from pieces" approach was tried first for the minmax
    case (ShiftedMinUintQuant + ParamFromRuntimeMinMaxScaling, mirroring
    exactly how _minmax_act_quant() fixed the symmetric case) and FAILED at
    real quantization time with `DependencyError: 'X' can not resolve
    attribute 'scaling_stats_input_concat_dim'` -- a real Brevitas
    dependency the shifted/asymmetric solver path needs that neither of
    those two pieces alone provides (unlike the symmetric case, where
    IntQuant + ParamFromRuntimeMinMaxScaling resolves cleanly). Class
    -attribute-level unit tests (checking e.g. scaling_stats_op) did NOT
    catch this, since that specific dependency is only resolved when
    Brevitas builds the full graph for a real forward pass -- a reminder
    that those tests confirm identity/composition, not full resolvability.

    Subclassing the real preset instead sidesteps the issue entirely, and
    for the minmax case specifically it's not an approximation:
    high_percentile_q=100.0/low_percentile_q=0.0 is mathematically EXACT
    literal min/max, not a close approximation -- Brevitas's percentile
    machinery here uses PARAMETER_FROM_STATS (torch.quantile applied
    directly to observed values across calibration steps, not a
    histogram-binned method like KLMinimizerThreshold), and
    torch.quantile(x, q=1.0)/q=0.0 are exactly max(x)/min(x), with no
    interpolation error to worry about.
    """
    from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat

    class ShiftedUint8ActPerTensorFloatCustomInterval(ShiftedUint8ActPerTensorFloat):
        high_percentile_q = high_q
        low_percentile_q = low_q

    return ShiftedUint8ActPerTensorFloatCustomInterval


def _minmax_asymmetric_act_quant():
    """INT8, per-tensor, ASYMMETRIC activation quantizer that is literal
    min/max -- unlike Brevitas's own ShiftedUint8ActPerTensorFloat, which
    turns out to be percentile-interval-based internally
    (high_percentile_q=99.999, low_percentile_q=0.001), mirroring the
    earlier Int8ActPerTensorFloat discovery for the symmetric case (see
    module docstring). Achieved via
    _shifted_percentile_interval_act_quant(100.0, _LOW_Q_EPSILON) -- see
    that function's docstring for why this isn't built from
    ParamFromRuntimeMinMaxScaling directly (that approach resolves at the
    class-attribute level but fails with a DependencyError at real
    quantization time).

    low_q is NOT literally 0.0 -- confirmed against the real Brevitas
    source (brevitas/core/stats/stats_op.py's PercentileInterval.forward),
    the percentile bounds are converted to a 1-indexed kthvalue() rank via
    `low_k = ceil(0.01 * low_q * x.numel())`; at low_q=0.0 this gives
    low_k=0, which kthvalue() rejects (RuntimeError: "selected number k out
    of range"), confirmed by hitting this for real. high_q=100.0 has no
    such issue: `high_k = floor(0.01 * high_q * x.numel() + 0.5)` gives
    exactly x.numel() for integer sizes, a valid rank corresponding to the
    true maximum, so it's left as literal 100.0. _LOW_Q_EPSILON keeps
    low_k exact, not approximate: ceil() rounds ANY value in (0, 100/N] up
    to exactly 1 -- kthvalue(x, 1) IS the literal minimum -- so as long as
    _LOW_Q_EPSILON stays below 100/N (true up to roughly 10^11 elements,
    far beyond any real activation tensor here), this is still exactly
    min/max, not a close approximation of it.
    """
    return _shifted_percentile_interval_act_quant(high_q=100.0, low_q=_LOW_Q_EPSILON)


def _percentile_asymmetric_act_quant():
    """INT8, per-tensor, ASYMMETRIC activation quantizer -- a more
    aggressive percentile clip than ShiftedUint8ActPerTensorFloat's default
    (99.999/0.001), via _shifted_percentile_interval_act_quant().

    Reuses _PERCENTILE_HIGH_Q (99.9, vs. the default's 99.999) for the high
    end, and its complement (100 - _PERCENTILE_HIGH_Q = 0.1, vs. the
    default's 0.001) for the low end -- keeping the interval symmetric in
    percentile space, the same convention Brevitas's own default
    (99.999/0.001) already uses, just more aggressive on both ends by the
    same margin _PERCENTILE_HIGH_Q already establishes for the symmetric
    case, rather than introducing an unrelated second magic number.
    """
    return _shifted_percentile_interval_act_quant(
        high_q=_PERCENTILE_HIGH_Q, low_q=100.0 - _PERCENTILE_HIGH_Q
    )


def _mse_asymmetric_act_quant():
    """Brevitas's own asymmetric MSE preset (deployment set only -- see the
    module docstring's deployment-set entry), with `dtype` and `device`
    declared on it.

    The stock ShiftedUint8ActPerTensorFloatMSE fails for real in the
    installed Brevitas 0.13.0 with `DependencyError: ... can not resolve
    attribute 'dtype'` -- its MSE zero-point search needs dtype/device, which
    nothing provides to an activation quantizer (weight quantizers get them
    from their parameter). Declaring them (float32, CPU: where this project
    quantizes and calibrates) is the only change; confirmed through
    calibration, eval forward and QCDQ export by
    investigations/spike_mse_asymmetric.py, where the stock preset fails."""
    import torch
    from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE

    class ShiftedUint8ActPerTensorFloatMSEWithDtype(ShiftedUint8ActPerTensorFloatMSE):
        dtype = torch.float32
        device = torch.device("cpu")

    return ShiftedUint8ActPerTensorFloatMSEWithDtype


def _entropy_act_quant():
    """Blocked -- see module docstring's "entropy" confidence-level entry
    for the full investigation. Two independent, confirmed issues in the
    installed Brevitas 0.13.0's KLMinimizerThreshold, not this project's
    quantizer wiring:

    1. A parity-dependent off-by-one bug in KLMinimizerThreshold.forward()
       crashes with IndexError for every EVEN num_bins; only odd num_bins
       (e.g. the library default, 1001) avoid it, by chance of parity, not
       by validation.
    2. Independent of that crash, sweeping odd num_bins values that DO run
       (501-8001) on the same real activation tensor gave threshold/absmax
       ranging 14.89%-34.22% with no convergence -- the resulting
       threshold is itself unstable, not just inaccessible for most
       num_bins choices.

    Raises NotImplementedError for lack of reliability of the method in
    this Brevitas version, not because of the crash alone -- see
    investigations/diagnose_entropy_wiring.py,
    investigations/diagnose_entropy_num_bins_odd_sweep.py, and
    investigations/brevitas_issue_klminimizerthreshold_parity_bug.md.
    """
    raise NotImplementedError(
        "calibration_algorithm='entropy' is blocked: Brevitas 0.13.0's "
        "KLMinimizerThreshold has a parity-dependent off-by-one bug (crashes "
        "for even num_bins) AND, independent of that, gives unstable "
        "thresholds (14.89%-34.22% of absmax, no convergence) across odd "
        "num_bins values that do run. See quantizer_config.py's module "
        "docstring and investigations/ for the full investigation."
    )


def resolve_weight_quant(calibration_algorithm: str, granularity: str, bitwidth: str = "int8"):
    import brevitas.quant as bq

    weight_quant = {
        ("minmax", "per_tensor"): bq.Int8WeightPerTensorFloat,
        ("minmax", "per_channel"): bq.Int8WeightPerChannelFloat,
        ("mse", "per_channel"): bq.Int8WeightPerChannelFloatMSE,
        ("percentile", "per_channel"): bq.Int8WeightPerChannelFloat,
        ("mse", "per_tensor"): bq.Int8WeightPerTensorFloatMSE,
        # percentile/entropy don't touch weight quantization -- see module docstring.
        ("percentile", "per_tensor"): bq.Int8WeightPerTensorFloat,
        ("entropy", "per_tensor"): bq.Int8WeightPerTensorFloat,
    }
    key: Tuple[str, str] = (calibration_algorithm, granularity)
    try:
        base_class = weight_quant[key]
    except KeyError:
        raise ValueError(
            f"No weight quantizer mapped for calibration_algorithm={calibration_algorithm!r}, "
            f"granularity={granularity!r}. Known combinations: {sorted(weight_quant)}."
        ) from None

    return _bit_width_override(base_class, _resolve_bitwidth(bitwidth))


def resolve_act_quant(calibration_algorithm: str, symmetry: str, bitwidth: str = "int8"):
    import brevitas.quant as bq

    # Values are zero-arg callables, not classes directly: a dict literal
    # evaluates every value eagerly at construction time, so
    # _percentile_act_quant()/_entropy_act_quant() (which each define a new
    # class on every call) would run on *every* resolve_act_quant() call
    # regardless of which key is actually looked up -- wrapping them (and,
    # for consistency, the plain preset classes too) in callables makes only
    # the selected one actually execute.
    act_quant = {
        ("minmax", "symmetric"): _minmax_act_quant,
        ("minmax", "asymmetric"): _minmax_asymmetric_act_quant,
        ("mse", "symmetric"): lambda: bq.Int8ActPerTensorFloatMSE,
        ("mse", "asymmetric"): _mse_asymmetric_act_quant,
        ("percentile", "symmetric"): _percentile_act_quant,
        ("percentile", "asymmetric"): _percentile_asymmetric_act_quant,
        ("entropy", "symmetric"): _entropy_act_quant,
    }
    key: Tuple[str, str] = (calibration_algorithm, symmetry)
    try:
        base_class = act_quant[key]()
    except KeyError:
        raise ValueError(
            f"No activation quantizer mapped for calibration_algorithm={calibration_algorithm!r}, "
            f"symmetry={symmetry!r}. Known combinations: {sorted(act_quant)}."
        ) from None

    return _bit_width_override(base_class, _resolve_bitwidth(bitwidth))
