"""investigations/per_image_predictions.py: the artifact plan, the three
reference sources and the accuracy check (no ONNX Runtime needed)."""

from __future__ import annotations

import csv

import pytest

from quantization.vit_brevitas.investigations import per_image_predictions as pip


def test_plan_covers_10_ablation_configs_per_model_and_13_deployment_configs_times_3_repeats():
    jobs = pip.plan()
    abl = [j for j in jobs if j["group"] == "ablation"]
    dep = [j for j in jobs if j["group"] == "deployment"]

    assert len(abl) == 3 * 10 * 3 and len(dep) == 13 * 3
    assert all(str(j["artifact"]).startswith("artifacts/vit/") for j in abl)
    assert {j["name"] for j in dep if j["model"] == "mobilevit"} == {
        f"deploy_mobilevit_percentile_{g}_{s}" for g in ("per_tensor", "per_channel") for s in ("symmetric", "asymmetric")
    } | {"deploy_mobilevit_minmax_per_tensor_symmetric"}
    assert all("__outq_bnfold" in str(j["artifact"]) for j in dep)


def test_every_single_node_gelu_config_is_compared_with_its_loose_gelu_counterpart():
    names = {j["name"] for j in pip.plan()}
    gelu1 = {n for n in names if n.endswith("_gelu1")}

    assert gelu1 == {"deploy_deit_tiny_mse_per_tensor_symmetric_gelu1", "deploy_deit_tiny_mse_per_channel_symmetric_gelu1",
                     "deploy_deit_tiny_mse_per_tensor_asymmetric_gelu1", "deploy_deit_tiny_minmax_per_tensor_symmetric_gelu1"}
    assert set(pip.SAME_AS) == gelu1
    assert all(pip.SAME_AS[n] in names and pip.SAME_AS[n] == n[: -len("_gelu1")] for n in gelu1)


def test_compare_predictions_counts_changed_images(tmp_path):
    def write(path, preds):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "filename", "label", "pred1", "top5", "correct1", "correct5"])
            for i, (p1, t5) in enumerate(preds):
                w.writerow([i, f"{i}.jpg", 0, p1, t5, 0, 0])

    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write(a, [(1, "1 2 3 4 5"), (7, "7 8 9 1 2"), (3, "3 4 5 6 7")])
    write(b, [(1, "1 2 3 4 5"), (8, "8 7 9 1 2"), (3, "3 4 5 7 6")])

    assert pip.compare_predictions(a, b) == {"images": 3, "pred1_diff": 1, "top5_diff": 2}


def test_references_from_results_csv_gate_csv_and_gate_log(tmp_path):
    results = tmp_path / "vit_accuracy_offline.csv"
    with open(results, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "repeat_id", "accuracy_metric_name", "accuracy_value"])
        w.writerow(["vit_x", "2", "top1_accuracy", "70.5"])
        w.writerow(["vit_x", "2", "top5_accuracy", "90.1"])
    gate_csv = tmp_path / "g.csv"
    gate_csv.write_text("run_id,repeat_id,top1_outq,top5_outq\nvit_y,1,63.65,91.25\n")
    log = tmp_path / "g.log"
    log.write_text("[gate] vit_z rep3: quantize + export ...\n[gate] vit_z rep3: top1 73.35%  top5 91.15%\n")

    cache = {}
    get = lambda kind, src, rid, rep: pip.reference({"ref": (kind, str(src), rid), "repeat": rep}, cache)
    assert get("results", results, "vit_x", 2) == {"top1": 70.5, "top5": 90.1}
    assert get("csv", gate_csv, "vit_y", 1) == {"top1": 63.65, "top5": 91.25}
    assert get("log", log, "vit_z", 3) == {"top1": 73.35, "top5": 91.15}
    assert get("log", log, "vit_z", 1) is None
    assert get("csv", tmp_path / "absent.csv", "vit_y", 1) is None


def _per_image(path, correct1, correct5):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "filename", "label", "pred1", "top5", "correct1", "correct5"])
        for i in range(2000):
            w.writerow([i, f"{i}.jpg", 0, 0, "0 1 2 3 4", int(i < correct1), int(i < correct5)])


def test_accuracy_check_matches_mismatches_and_flags_missing_reference(tmp_path):
    out = tmp_path / "a.csv"
    _per_image(out, correct1=1467, correct5=1821)  # 73.35 % / 91.05 %
    got = pip.accuracy_of(out)

    assert got["top1"] == pytest.approx(73.35) and got["n"] == 2000
    assert pip.check(got, {"top1": 73.35, "top5": 91.05}) == "OK"
    assert pip.check(got, {"top1": 73.40, "top5": 91.05}) == "MISMATCH"
    assert pip.check(got, None) == "NO_REFERENCE"
