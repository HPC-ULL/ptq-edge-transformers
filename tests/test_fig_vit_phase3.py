"""plots/scripts/fig_vit_phase3.py on synthetic CSVs and (smoke) the real ones."""

from __future__ import annotations

import pandas as pd
import pytest

import plots.scripts.fig_vit_phase3 as fig
from plots.common import RAW

COLS = ["run_id", "phase", "factor_varied", "factor_level", "model_name", "bitwidth",
        "repeat_id", "accuracy_metric_name", "accuracy_value"]


def _acc(rows):
    return pd.DataFrame(rows, columns=COLS)


def _fp32(model, v, level="fp32"):
    return dict(run_id=f"{model}_ref_{level}", phase="reference", factor_varied="reference", factor_level=level, model_name=model,
                bitwidth="fp32", repeat_id=None, accuracy_metric_name="top1_accuracy", accuracy_value=v)


def _reps(phase, varied, level, model, bit, vals):
    return [dict(run_id=f"{model}_{phase}_{varied}_{level}_{bit}", phase=phase, factor_varied=varied, factor_level=level, model_name=model, bitwidth=bit,
                 repeat_id=i + 1, accuracy_metric_name="top1_accuracy", accuracy_value=v)
            for i, v in enumerate(vals)]


def _synthetic(model="deit_tiny"):
    rows = _reps("phase1", "baseline", "default", model, "int8", [70, 72, 74])
    for bit in ("int6", "int4"):
        rows += _reps("phase3", "phase3_config", "baseline", model, bit, [10, 20, 30])
        rows += _reps("phase3", "phase3_config", "combined", model, bit, [40, 50, 60])
    rows += _reps("phase2", "calibration_algorithm", "mse", model, "int8", [60, 62, 64])
    rows += _reps("phase2", "granularity", "per_channel", model, "int8", [80, 82, 84])
    return rows


def test_build_table_mean_std_n_and_series():
    acc = _acc(sum((_synthetic(m) for m in fig.L.VIT_MODEL_ORDER), []))
    ref = _acc([r for m in fig.L.VIT_MODEL_ORDER for r in (_fp32(m, 80.0), _fp32(m, 1.0, "fp32_full50000"))])
    t = fig.build_table(acc, ref)
    sel = t[(t.model == "deit_tiny") & (t.series == "selected")]
    assert list(sel.bitwidth) == ["int6", "int4"]          # no INT8 selected point
    row = t[(t.model == "deit_tiny") & (t.series == "baseline") & (t.bitwidth == "int8")].iloc[0]
    assert row["mean"] == 72 and row["std"] == pytest.approx(2.0) and row["n"] == 3
    best = t[(t.model == "deit_tiny") & (t.series == "best_single")].iloc[0]
    assert best.level == "granularity=per_channel" and best["mean"] == 82 and best.bitwidth == "int8"
    assert (t[t.series == "fp32"]["mean"] == 80.0).all()    # full50000 reference ignored


def test_build_table_requires_three_repeats():
    acc = _acc(_synthetic()[:-1])                             # drop one repeat
    with pytest.raises(AssertionError):
        fig.build_table(acc, _acc([_fp32("deit_tiny", 80.0)]))


def _best_rows(level):
    return [dict(model=m, series="best_single", bitwidth="int8", mean=1.0, std=0, n=3,
                 level=f"calibration_algorithm={level}") for m in fig.EXPECTED_BEST_LEVEL]


def test_sanity_check_stops_on_wrong_best_level():
    with pytest.raises(SystemExit, match="best single factor"):
        fig.sanity_check(pd.DataFrame(_best_rows("minmax")))


def test_sanity_check_stops_on_mean_mismatch():
    rows = [dict(model=m, series="best_single", bitwidth="int8", mean=1.0, std=0, n=3,
                 level=f"calibration_algorithm={fig.EXPECTED_BEST_LEVEL[m]}")
            for m in fig.EXPECTED_BEST_LEVEL]
    with pytest.raises(SystemExit, match="got"):
        fig.sanity_check(pd.DataFrame(rows + [dict(model="deit_tiny", series="baseline", bitwidth="int8",
                                                  mean=50.0, std=0, n=3, level=None)]))


@pytest.mark.skipif(not (RAW / "vit_accuracy_offline.csv").exists(), reason="real CSVs not in this checkout")
def test_smoke_real_csvs(tmp_path):
    fig.main(["--out-dir", str(tmp_path)])
    d = tmp_path / fig.NAME
    assert (d / f"{fig.NAME}.pdf").stat().st_size > 0
    assert (d / f"{fig.NAME}.png").stat().st_size > 0
    data = pd.read_csv(d / f"{fig.NAME}_data.csv")
    assert len(data) == 21 and set(data.model_label) == {"DeiT-Tiny", "MobileViT-S", "EfficientViT-B1"}
