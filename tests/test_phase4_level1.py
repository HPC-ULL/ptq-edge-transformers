"""Phase 4 Level 1 (analysis/phase4_vit/level1.py): image selection, moments,
outlier ratio, ranking, degeneracy, Holm and the correlation table. The
torch/scipy parts are skipped where those are not installed."""

from __future__ import annotations

import math
import random

import pytest

from analysis.phase4_vit import PHASE4_ROOT
from analysis.phase4_vit import level1 as l1


def test_pre_registered_constants():
    assert (l1.N_IMAGES, l1.IMAGE_SEED, l1.EXCLUDED_CALIB_SEEDS, l1.EXCLUDED_CALIB_SIZE) == (512, 0, (42, 43, 44), 512)
    assert (l1.EPS, l1.MIN_MEDIAN, l1.MIN_RANKED, l1.MIN_TYPE_N, l1.HOLM_ALPHA) == (1e-8, 1e-6, 12, 10, 0.05)
    assert l1.BASELINE_INT8 == {"calibration_algorithm": "minmax", "granularity": "per_tensor",
                                "symmetry": "symmetric", "calib_set_size": 128, "bitwidth": "int8"}
    assert l1.REPEAT_ID == 1 and l1.OUT_DIR == PHASE4_ROOT / "level1"


def test_image_selection_is_disjoint_distinct_sorted_and_reproducible():
    excluded = set(range(0, 10_000, 7))
    a = l1.select_image_indices(10_000, excluded)
    assert a == l1.select_image_indices(10_000, excluded)
    assert len(a) == len(set(a)) == 512 and a == sorted(a) and not set(a) & excluded
    assert a != l1.select_image_indices(10_000, excluded, seed=1)


def test_image_selection_fails_when_the_pool_is_too_small():
    with pytest.raises(ValueError):
        l1.select_image_indices(600, set(range(100)))


def _moments(xs):
    n = len(xs)
    m = sum(xs) / n
    return n, m, sum((x - m) ** 2 for x in xs), sum((x - m) ** 3 for x in xs), sum((x - m) ** 4 for x in xs)


def test_merged_moments_equal_the_moments_of_the_pooled_sample():
    rng = random.Random(3)
    xs = [rng.lognormvariate(0, 1) for _ in range(300)]
    merged = (0, 0.0, 0.0, 0.0, 0.0)
    for chunk in (xs[:17], xs[17:150], xs[150:151], xs[151:]):
        merged = l1.merge_moments(merged, _moments(chunk))
    for got, want in zip(merged, _moments(xs)):
        assert got == pytest.approx(want, rel=1e-9)


def test_excess_kurtosis():
    assert l1.excess_kurtosis(_moments([-1.0, 1.0] * 50)) == pytest.approx(-2.0)
    assert math.isnan(l1.excess_kurtosis(_moments([2.0] * 10)))


def test_outlier_ratio_formula():
    assert l1.outlier_ratio(10.0, 2.0) == pytest.approx(10.0 / (2.0 + 1e-8))
    assert l1.outlier_ratio(1.0, 0.0) == pytest.approx(1e8)


def _row(name, exec_idx, R, median=1.0, cin=64, err=0.1, t="attn", alt=None):
    return {"name": name, "exec_idx": exec_idx, "R": R, "absmax_median": median, "in_channels": cin,
            "rankable_channels": cin >= 8, "local_error_mean": err, "layer_type": t, "layer_type_alt": alt or t}


def test_ranking_excludes_small_inputs_and_dead_medians_and_breaks_ties_by_execution_order():
    rows = [_row("stem", 0, 50.0, cin=3), _row("a", 1, 5.0), _row("b", 2, 9.0), _row("dead", 3, 1e8, median=1e-7),
            _row("c", 4, 9.0)]
    ranked = l1.rank_layers(rows)
    assert [r["name"] for r in ranked] == ["b", "c", "a"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    by = {r["name"]: r for r in rows}
    assert by["stem"]["exclusion"] == "< 8 input channels" and by["stem"]["rank"] == ""
    assert by["dead"]["exclusion"].startswith("median absmax") and not by["dead"]["ranked"]


def test_degenerate_ranking():
    few = [_row(str(i), i, float(i)) for i in range(11)]
    assert any("< 12" in r for r in l1.degenerate_reasons(few))
    ok = [_row(str(i), i, float(i)) for i in range(12)]
    assert l1.degenerate_reasons(ok) == []
    # 7 of 12 share R to 6 significant figures -> degenerate; 6 of 12 does not
    same = [_row(str(i), i, 3.0000001 if i < 7 else float(i + 10)) for i in range(12)]
    assert any("share R" in r for r in l1.degenerate_reasons(same))
    half = [_row(str(i), i, 3.0 if i < 6 else float(i + 10)) for i in range(12)]
    assert l1.degenerate_reasons(half) == []


def test_holm_matches_a_hand_computed_example():
    adjusted, reject = l1.holm([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert reject == [True, False, False]
    assert l1.holm([0.9, 0.8])[0] == [1.0, 1.0]


def _fake_spearman(x, y):
    return 0.5, 0.01 * len(x)


def test_correlations_primary_global_types_below_ten_listed_and_no_alt_when_identical():
    ranked = [_row(f"a{i}", i, float(i), t="attn") for i in range(12)] + \
             [_row(f"m{i}", 20 + i, float(i), t="mlp") for i in range(9)]
    out = l1.correlations("deit_tiny", ranked, _fake_spearman)
    glob = [r for r in out if r["scope"] == "global"]
    assert len(glob) == 1 and glob[0]["n"] == 21 and glob[0]["status"] == "primary"
    by_type = {r["layer_type"]: r for r in out if r["scope"] == "type"}
    assert by_type["attn"]["status"] == "descriptive" and by_type["attn"]["n"] == 12 and by_type["attn"]["rho"] == 0.5
    assert by_type["mlp"]["status"] == "listed (n < 10)" and by_type["mlp"]["rho"] == ""
    assert "conv" not in by_type and {r["classification"] for r in out} == {"main"}


def test_correlations_add_the_alt_classification_when_it_differs():
    ranked = [_row(f"q{i}", i, float(i), t="attn") for i in range(14)] + \
             [_row(f"g{i}", 20 + i, float(i), t="attn", alt="conv") for i in range(14)] + \
             [_row(f"c{i}", 40 + i, float(i), t="conv") for i in range(45)]
    out = l1.correlations("efficientvit_b1", ranked, _fake_spearman)
    alt = {r["layer_type"]: r for r in out if r["classification"] == "alt"}
    main = {r["layer_type"]: r for r in out if r["classification"] == "main" and r["scope"] == "type"}
    assert (main["attn"]["n"], main["conv"]["n"]) == (28, 45)
    assert (alt["attn"]["n"], alt["conv"]["n"]) == (14, 59)


def test_holm_applies_only_to_the_three_primary_rows():
    rows = []
    for m in ("deit_tiny", "mobilevit", "efficientvit_b1"):
        rows += l1.correlations(m, [_row(f"{m}{i}", i, float(i)) for i in range(12)], _fake_spearman)
    l1.apply_holm(rows)
    primary = [r for r in rows if r["status"] == "primary"]
    assert len(primary) == 3 and all(r["p_holm"] == pytest.approx(0.36) for r in primary)
    assert all(r["p_holm"] == "" for r in rows if r["status"] != "primary")


# -- torch / scipy -----------------------------------------------------------

def test_channel_axis_is_last_for_linear_and_dim1_for_conv():
    torch = pytest.importorskip("torch")
    x = torch.zeros(2, 5, 3)
    x[1, 4, 2] = -7.0
    assert l1.LayerAccumulator("Linear").channel_absmax(x).tolist() == [0.0, 0.0, 7.0]
    xc = torch.zeros(2, 3, 4, 4)
    xc[0, 1, 2, 3] = 5.0
    assert l1.LayerAccumulator("Conv2d").channel_absmax(xc).tolist() == [0.0, 5.0, 0.0]


def test_accumulated_moments_and_range_equal_the_pooled_values():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    batches = [torch.randn(4, 6, 8, generator=g) * 3 + 1 for _ in range(3)]
    acc = l1.LayerAccumulator("Linear")
    for b in batches:
        acc.update_input(b)
    pooled = torch.cat([b.flatten() for b in batches]).double().tolist()
    for got, want in zip(acc.moments, _moments(pooled)):
        assert got == pytest.approx(want, rel=1e-6)
    assert acc.max == pytest.approx(max(pooled)) and acc.min == pytest.approx(min(pooled))


def test_per_image_errors_group_the_batch_as_the_outer_factor_of_dim0():
    torch = pytest.importorskip("torch")
    y = torch.ones(2 * 4, 3, 5)  # MobileViT-style (B * patch_area, N, C), B = 2
    y_q = y.clone()
    y_q[4:] *= 1.5  # only image 1's patches are off
    errors = l1.per_image_errors(y_q, y, batch_size=2)
    assert errors[0] == pytest.approx(0.0) and errors[1] == pytest.approx(0.5)
    with pytest.raises(RuntimeError):
        l1.per_image_errors(torch.ones(3, 2), torch.ones(3, 2), batch_size=2)


def test_spearman_wrapper():
    pytest.importorskip("scipy")
    rho, p = l1._spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 100])
    assert rho == pytest.approx(1.0) and p < 0.05


# -- grouping check and driver (no torch: run_model is replaced) --------------

def test_check_batch_and_tolerance():
    assert l1.CHECK_BATCH == 16 and l1.GROUPING_RTOL == 1e-2


def _check(first_alone, last_alone, first_in, last_in, names=("a", "b")):
    return {"alone_first": {n: (first_alone, first_alone) for n in names},
            "alone_last": {n: (last_alone, last_alone) for n in names},
            "batch": {n: (first_in, last_in) for n in names}}


def test_grouping_passes_within_one_percent_for_first_and_last_image():
    assert l1.grouping_problems(["a", "b"], _check(0.1, 0.2, 0.1005, 0.2015)) == []


def test_grouping_fails_on_the_last_image_alone():
    problems = l1.grouping_problems(["a", "b"], _check(0.1, 0.2, 0.1, 0.25))
    assert len(problems) == 2 and all("last image" in p for p in problems)


def test_grouping_fails_on_missing_layer_or_non_finite_error():
    assert any("missing" in p for p in l1.grouping_problems(["a", "c"], _check(0.1, 0.2, 0.1, 0.2)))
    assert l1.grouping_problems(["a"], _check(float("nan"), 0.2, float("nan"), 0.2, names=("a",)))


def _fake_images():
    return [{"pool_idx": i, "filename": f"{i}.jpg", "path": f"{i}.jpg"} for i in range(512)], {"x.jpg"}


def _fake_layers(model):
    return [{"model": model, "exec_idx": i, "name": f"l{i}", "kind": "Linear", "layer_type": "attn",
             "layer_type_alt": "attn", "in_channels": 64, "rankable_channels": True, "absmax_max": 10.0 + i,
             "absmax_median": 1.0, "R": 10.0 + i, "local_error_mean": 0.01 * i, "local_error_n": 512,
             "kurtosis_excess": 0.0, "act_min": -1.0, "act_max": 1.0, "act_range": 2.0} for i in range(12)]


@pytest.fixture
def driver(tmp_path, monkeypatch):
    inventory = tmp_path / "layers.csv"
    inventory.write_text("model,name\n" + "".join(f"{m},l0\n" for m in l1.MODELS))
    monkeypatch.setattr(l1, "level1_images", _fake_images)
    monkeypatch.setattr(l1, "_spearman", lambda x, y: (1.0, 0.001))
    out = tmp_path / "out"
    return ["--inventory", str(inventory), "--out-dir", str(out), "--device", "cpu"], out


def test_failed_check_in_one_model_writes_no_correlations(driver, monkeypatch):
    argv, out = driver
    monkeypatch.setattr(l1, "run_model", lambda m, inv, im, dev, bs:
                        ([], ["l3: first image alone 0.1 vs in batch 0.3 (grouping)"]) if m == "mobilevit"
                        else (_fake_layers(m), []))
    assert l1.main(argv) == 1
    assert not (out / "correlations.csv").exists()
    assert "FAILED" in (out / "checks.csv").read_text() and "no correlations written" in (out / "run.log").read_text()


def test_successful_run_writes_correlations_with_holm(driver, monkeypatch):
    argv, out = driver
    monkeypatch.setattr(l1, "run_model", lambda m, inv, im, dev, bs: (_fake_layers(m), []))
    assert l1.main(argv) == 0
    text = (out / "correlations.csv").read_text()
    assert text.count(",primary") == 3 and "0.003" in text
    assert (out / "layers_deit_tiny.csv").exists() and (out / "images.csv").exists()


def test_refuses_to_run_over_existing_correlations(driver, monkeypatch):
    argv, out = driver
    out.mkdir()
    (out / "correlations.csv").write_text("old")
    monkeypatch.setattr(l1, "run_model", lambda *a: pytest.fail("must not run"))
    assert l1.main(argv) == 1 and (out / "correlations.csv").read_text() == "old"


def test_dry_run_loads_no_model(driver, monkeypatch):
    argv, out = driver
    monkeypatch.setattr(l1, "run_model", lambda *a: pytest.fail("must not run"))
    assert l1.main(argv + ["--dry-run"]) == 0 and not out.exists()
