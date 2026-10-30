"""plots/scripts/capture_calibration_concept.py (numpy part) and fig_calibration_concept.py
on synthetic data. No torch needed."""

from __future__ import annotations

import numpy as np
import pytest

import plots.scripts.capture_calibration_concept as cap
import plots.scripts.fig_calibration_concept as fig


def _x(n=400_000, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.laplace(0, 0.6, n), rng.uniform(-50, 50, 40)]).astype(np.float32)


def test_summarize_exact_stats_and_histogram_span():
    x = _x()
    s = cap.summarize(x, n_bins=200, n_sub=10_000)
    a = np.abs(x)
    assert s["n_values"] == x.size and s["max_abs"] == pytest.approx(float(a.max()))
    assert s["p_linear"] == pytest.approx(float(np.percentile(a.astype(np.float64), 99.9)), rel=1e-9)
    assert s["p_nearest"] == float(np.sort(a)[int(np.ceil(0.999 * x.size)) - 1])
    assert s["hist_counts"].sum() == x.size and len(s["hist_counts"]) == 200
    assert s["hist_edges"][0] == pytest.approx(-s["max_abs"]) and s["hist_edges"][-1] == pytest.approx(s["max_abs"])
    assert len(s["subsample"]) == 10_000 and np.abs(s["subsample"]).max() <= s["max_abs"]


def test_fraction_above_is_exact_on_the_full_array():
    x = _x()
    thr = float(np.percentile(np.abs(x), 99.9))
    assert cap.fraction_above(x, thr, chunk=100_000) == pytest.approx(float((np.abs(x) > thr).mean()))


def test_choose_range_prefers_brevitas_only_when_it_differs():
    assert cap.choose_range(10.0, float("nan"))[0] == 10.0
    assert cap.choose_range(10.0, 10.05)[0] == 10.0 and "matches" in cap.choose_range(10.0, 10.05)[1]
    assert cap.choose_range(10.0, 12.0) == (12.0, "Brevitas (differs from the exact value)")


def _npz(tmp_path):
    x = _x()
    s = cap.summarize(x, n_bins=200, n_sub=10_000)
    r_p, src = cap.choose_range(s["p_linear"], float("nan"))
    r_m = s["max_abs"]
    clip = cap.fraction_above(x, r_p)
    path = tmp_path / "d.npz"
    np.savez_compressed(path, layer="L", model="m", repeat_id=1, calib_seed=42, calib_set_size=128,
                        percentile_q=99.9, images_sha1="x", subsample_seed=0, versions="t",
                        brevitas_range_minmax=np.nan, brevitas_range_percentile=np.nan, range_minmax=r_m,
                        range_minmax_source="e", range_percentile=r_p, range_percentile_source=src,
                        step_minmax=r_m / 127, step_percentile=r_p / 127, step_ratio=r_m / r_p,
                        clipped_fraction=clip, clipped_fraction_linear=clip, clipped_fraction_nearest=clip, **s)
    return path


def test_figure_from_npz_writes_outputs_and_values_come_from_the_file(tmp_path):
    fig.main(["--npz", str(_npz(tmp_path)), "--out-dir", str(tmp_path)])
    d = tmp_path / fig.NAME
    assert (d / f"{fig.NAME}.pdf").stat().st_size > 0 and (d / f"{fig.NAME}.png").stat().st_size > 0
    assert "step_ratio" in (d / f"{fig.NAME}_data.csv").read_text()


def test_checks_reject_a_histogram_that_does_not_hold_every_value(tmp_path):
    d = fig.load(_npz(tmp_path))
    d["percentile_exact_ref"] = d["p_linear"]
    d["hist_counts"] = d["hist_counts"].copy()
    d["hist_counts"][0] += 1
    with pytest.raises(AssertionError, match="every value"):
        fig.checks(d)


def test_negative_stats_are_exact_from_the_histogram(tmp_path):
    d = fig.load(_npz(tmp_path))
    st = fig.negative_stats(d)
    x = _x()
    assert st["negative_fraction"] == pytest.approx(float((x < 0).mean()), abs=1e-12)
    assert st["vmin"] == pytest.approx(float(x.min())) and st["vmin"] < 0
