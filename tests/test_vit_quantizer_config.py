"""Tests for quantization/vit_brevitas/quantizer_config.py's resolvers.

Unlike the rest of this project's tests (which mock/inject around Brevitas
entirely, since it isn't installed in every dev environment), these
actually import and call into real Brevitas classes -- there is no other
way to verify class identity/inheritance/attributes meaningfully.
pytest.importorskip means these are skipped automatically wherever brevitas
isn't installed (leaving the rest of the suite unaffected); run them for
real on the cluster, where Brevitas is actually installed, to get an actual
pass/fail.
"""

from __future__ import annotations

import pytest

pytest.importorskip("brevitas")

from quantization.vit_brevitas import quantizer_config  # noqa: E402


def test_minmax_resolves_to_literal_minmax_not_brevitas_percentile_default():
    """Brevitas's own "standard" INT8 activation quantizer
    (Int8ActPerTensorFloat) turns out to be percentile-based internally
    (ParamFromRuntimePercentileScaling, high_percentile_q=99.999) -- see
    quantizer_config.py's module docstring. calibration_algorithm="minmax"
    must resolve to our own Int8ActPerTensorFloatMinMax (literal
    StatsOp.MIN_MAX), not to Int8ActPerTensorFloat, or the "minmax"
    baseline wouldn't actually be minmax."""
    import brevitas.quant as bq

    resolved = quantizer_config.resolve_act_quant("minmax", "symmetric")

    assert resolved is not bq.Int8ActPerTensorFloat
    assert resolved.__name__ == "Int8ActPerTensorFloatMinMax"


def test_minmax_act_quant_uses_min_max_stats_op():
    from brevitas.inject.enum import StatsOp

    resolved = quantizer_config.resolve_act_quant("minmax", "symmetric")

    assert resolved.scaling_stats_op == StatsOp.MIN_MAX


def test_percentile_act_quant_uses_percentile_stats_op_not_min_max():
    from brevitas.inject.enum import StatsOp

    resolved = quantizer_config.resolve_act_quant("percentile", "symmetric")

    assert resolved.scaling_stats_op == StatsOp.PERCENTILE
    assert resolved.high_percentile_q == quantizer_config._PERCENTILE_HIGH_Q


def test_minmax_and_percentile_resolve_to_distinct_classes():
    minmax = quantizer_config.resolve_act_quant("minmax", "symmetric")
    percentile = quantizer_config.resolve_act_quant("percentile", "symmetric")

    assert minmax is not percentile


def test_weight_quant_unaffected_by_the_activation_minmax_change():
    """resolve_weight_quant never touched Int8ActPerTensorFloat at all --
    weight quantization was already literal minmax (MaxStatsScaling) before
    and after this change, so it should be untouched."""
    import brevitas.quant as bq

    resolved = quantizer_config.resolve_weight_quant("minmax", "per_tensor")

    assert resolved is bq.Int8WeightPerTensorFloat


def test_minmax_asymmetric_resolves_to_literal_minmax_not_brevitas_percentile_interval_default():
    """Brevitas's own "standard" asymmetric INT8 activation quantizer
    (ShiftedUint8ActPerTensorFloat) turns out to ALSO be percentile-based
    internally (ParamFromRuntimePercentileIntervalScaling,
    high_percentile_q=99.999, low_percentile_q=0.001) -- mirroring the
    symmetric discovery exactly. calibration_algorithm="minmax",
    symmetry="asymmetric" must resolve to a subclass with percentile bounds
    pushed to the exact extremes (100.0/_LOW_Q_EPSILON -- mathematically
    exact literal min/max, not the default 99.999/0.001), not to
    ShiftedUint8ActPerTensorFloat itself, or the "minmax" baseline wouldn't
    actually be minmax for the asymmetric OFAT row either.

    low_percentile_q is NOT literally 0.0: Brevitas's real
    PercentileInterval.forward() converts it to a 1-indexed kthvalue() rank
    via ceil(0.01*low_q*x.numel()), and 0.0 gives rank 0, which kthvalue()
    rejects with a RuntimeError -- confirmed by hitting it for real on the
    cluster. _LOW_Q_EPSILON (1e-9) still gives rank 1 exactly (the literal
    minimum) for any realistic tensor size, so this isn't an approximation,
    just working around this specific implementation's boundary handling.

    (Not checked via scaling_stats_op: it stays StatsOp.PERCENTILE_INTERVAL
    even in the fixed version -- literal min/max is achieved by pushing the
    percentile bounds to the exact extremes, not by changing the
    underlying stats-op enum. An earlier attempt built a class with a
    genuinely different scaling_stats_op (StatsOp.MIN_MAX, via
    ParamFromRuntimeMinMaxScaling) but that failed for real at quantization
    time with a DependencyError -- see quantizer_config.py's module
    docstring.)"""
    import brevitas.quant as bq

    resolved = quantizer_config.resolve_act_quant("minmax", "asymmetric")

    assert resolved is not bq.ShiftedUint8ActPerTensorFloat
    assert resolved.high_percentile_q == 100.0
    assert resolved.low_percentile_q == quantizer_config._LOW_Q_EPSILON


def test_percentile_asymmetric_act_quant_uses_percentile_interval_stats_op():
    from brevitas.inject.enum import StatsOp

    resolved = quantizer_config.resolve_act_quant("percentile", "asymmetric")

    assert resolved.scaling_stats_op == StatsOp.PERCENTILE_INTERVAL
    assert resolved.high_percentile_q == quantizer_config._PERCENTILE_HIGH_Q
    assert resolved.low_percentile_q == 100.0 - quantizer_config._PERCENTILE_HIGH_Q


def test_minmax_and_percentile_asymmetric_resolve_to_distinct_classes():
    minmax = quantizer_config.resolve_act_quant("minmax", "asymmetric")
    percentile = quantizer_config.resolve_act_quant("percentile", "asymmetric")

    assert minmax is not percentile


def test_entropy_act_quant_is_blocked():
    """calibration_algorithm="entropy" must raise NotImplementedError, not
    resolve to a working quantizer -- blocked after a real ablation run
    showed severe, deit_tiny-specific collapse (24-27%/12.85% top1 vs. ~73%
    everywhere else), root-caused to two confirmed issues in the installed
    Brevitas 0.13.0's KLMinimizerThreshold (a parity-dependent off-by-one
    crash for even num_bins, and unstable thresholds -- 14.89%-34.22% of
    absmax, no convergence -- across odd num_bins values that do run), not
    this project's wiring. See quantizer_config.py's module docstring and
    investigations/ for the full investigation."""
    with pytest.raises(NotImplementedError):
        quantizer_config.resolve_act_quant("entropy", "symmetric")


# --------------------------------------------------------------------------
# bitwidth (Phase 3's INT6/INT4): resolve_weight_quant/resolve_act_quant
# previously ignored this field entirely, silently quantizing every Phase 3
# "int6"/"int4" row identically to INT8 -- caught only after real Phase 3
# accuracy came back byte-identical across bit-widths. See
# _bit_width_override()'s docstring.
# --------------------------------------------------------------------------

def test_bitwidth_int8_default_returns_class_unchanged():
    """bitwidth="int8" (the default, and every Phase 1-2 row's literal
    value) must behave exactly as before this fix. Checked two different
    ways deliberately: resolve_weight_quant's non-bitwidth resolution
    returns a fixed class reference straight out of a dict (no factory), so
    `is` identity holds and is the strongest check available. Several of
    resolve_act_quant's entries (e.g. minmax, percentile) are factory
    functions that build a FRESH class on every single call regardless of
    bitwidth -- true before this fix too, by design (see the eager
    -evaluation comment in resolve_act_quant) -- so `is` identity was never
    guaranteed there; bit_width and the base classes are checked instead."""
    weight_with_default = quantizer_config.resolve_weight_quant("minmax", "per_tensor")
    weight_with_explicit_int8 = quantizer_config.resolve_weight_quant("minmax", "per_tensor", "int8")
    assert weight_with_default is weight_with_explicit_int8
    assert weight_with_default.bit_width == 8

    act_with_default = quantizer_config.resolve_act_quant("minmax", "symmetric")
    act_with_explicit_int8 = quantizer_config.resolve_act_quant("minmax", "symmetric", "int8")
    assert act_with_default.bit_width == act_with_explicit_int8.bit_width == 8
    assert act_with_default.__bases__ == act_with_explicit_int8.__bases__


def test_bitwidth_int6_and_int4_override_weight_quant_bit_width():
    for bitwidth, expected_bits in (("int6", 6), ("int4", 4)):
        resolved = quantizer_config.resolve_weight_quant("minmax", "per_tensor", bitwidth)
        assert resolved.bit_width == expected_bits
        assert resolved is not quantizer_config.resolve_weight_quant("minmax", "per_tensor", "int8")


def test_bitwidth_int6_and_int4_override_act_quant_bit_width():
    for bitwidth, expected_bits in (("int6", 6), ("int4", 4)):
        resolved = quantizer_config.resolve_act_quant("minmax", "symmetric", bitwidth)
        assert resolved.bit_width == expected_bits


def test_unknown_bitwidth_raises_value_error():
    with pytest.raises(ValueError):
        quantizer_config.resolve_weight_quant("minmax", "per_tensor", "int3")
    with pytest.raises(ValueError):
        quantizer_config.resolve_act_quant("minmax", "symmetric", "int3")


def test_bit6_and_bit4_quantizers_keep_round_to_nearest_rounding_mode():
    """The QCDQ export path falls back to a slower, less-faithful
    export_fake_quantized=True mode if rounding_mode isn't exactly 'ROUND'
    (confirmed by reading brevitas/export/onnx/standard/qcdq/handler.py --
    see investigations/probe_qcdq_bitwidth_support.sh's findings).
    _bit_width_override only overrides bit_width, so rounding_mode SHOULD
    be inherited unchanged from the base (already-working, ROUND-mode)
    quantizer -- but per this project's own track record (percentile MRO,
    a DependencyError, a kthvalue boundary crash, all from "should be
    fine" compositions), this is checked directly against a real
    instantiated layer, not assumed from class composition alone."""
    import torch
    from brevitas.nn import QuantLinear

    for bitwidth in ("int6", "int4"):
        weight_quant = quantizer_config.resolve_weight_quant("minmax", "per_tensor", bitwidth)
        act_quant = quantizer_config.resolve_act_quant("minmax", "symmetric", bitwidth)

        layer = QuantLinear(
            4, 4, bias=False, weight_quant=weight_quant, input_quant=act_quant, return_quant_tensor=False
        )
        layer.eval()
        with torch.no_grad():
            layer(torch.randn(1, 4))

        assert layer.weight_quant.rounding_mode.upper() == "ROUND", bitwidth
        assert layer.input_quant.rounding_mode.upper() == "ROUND", bitwidth


# --------------------------------------------------------------------------
# deployment-set combinations (configs/vit_deployment_manifest.py)
# --------------------------------------------------------------------------

def test_mse_per_channel_weights_are_brevitas_per_channel_mse():
    import brevitas.quant as bq

    assert quantizer_config.resolve_weight_quant("mse", "per_channel") is bq.Int8WeightPerChannelFloatMSE


def test_percentile_per_channel_weights_are_the_plain_per_channel_weights():
    """percentile only touches activations: same weights as minmax per_channel."""
    assert quantizer_config.resolve_weight_quant("percentile", "per_channel") is \
        quantizer_config.resolve_weight_quant("minmax", "per_channel")


def test_mse_asymmetric_activations_are_brevitas_shifted_mse_with_dtype_declared():
    """The stock preset fails for real without dtype/device (DependencyError)."""
    import torch
    from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE

    resolved = quantizer_config.resolve_act_quant("mse", "asymmetric")

    assert issubclass(resolved, ShiftedUint8ActPerTensorFloatMSE)
    assert resolved.dtype is torch.float32 and resolved.device == torch.device("cpu")
