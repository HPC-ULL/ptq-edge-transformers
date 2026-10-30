"""scripts/combine_edge_results.py on synthetic CSVs in a temp dir."""

from __future__ import annotations

import csv
import hashlib

from benchmarking_ondevice.power_logging.results_io import RESULT_COLUMNS
from configs import llm_f16_reference, llm_manifest
from quantization.vit_brevitas import deployment
import scripts.combine_edge_results as cer

LLM = llm_manifest.generate()[0]["run_id"]
F16 = llm_f16_reference.run_id(llm_manifest.MODELS[0])
VIT = deployment.plan()[1]["run_id"]          # a deployment row
FALLBACK = deployment.plan()[-1]["run_id"]    # a fallback reference


def _row(run_id, device, n=30, energy=300.0, ppw=None, epi=None):
    r = {c: "" for c in RESULT_COLUMNS}
    r.update(result_id=f"{run_id}__{device}", run_id=run_id, device=device, num_inferences_measured=str(n),
             energy_net_j=str(energy), energy_per_inference=str(energy / n if epi is None else epi),
             perf_per_watt="1.0" if ppw is None else str(ppw))
    return r


def _write(path, rows, cols=RESULT_COLUMNS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _run(tmp_path, *extra):
    return cer.main(["--raw-dir", str(tmp_path / "raw"), "--out-dir", str(tmp_path / "out"), *extra])


def _out(tmp_path):
    with open(tmp_path / "out" / cer.OUT_NAME, newline="") as f:
        return list(csv.DictReader(f))


def _sha(tmp_path):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path / "raw").rglob("*") if p.is_file()}


def test_includes_every_results_csv_also_a_new_device(tmp_path):
    raw = tmp_path / "raw"
    _write(raw / "results_jetson_gpu_30w.csv", [_row(VIT, "jetson_gpu_30w")])
    _write(raw / "results_new_board.csv", [_row(LLM, "new_board")])
    assert _run(tmp_path) == 0
    rows = {r["device"]: r for r in _out(tmp_path)}
    assert set(rows) == {"jetson_gpu_30w", "new_board"}
    assert (rows["jetson_gpu_30w"]["backend"], rows["jetson_gpu_30w"]["power_mode_w"]) == ("gpu", "30")
    assert (rows["new_board"]["backend"], rows["new_board"]["power_mode_w"]) == ("cpu", "")
    assert rows["new_board"]["source_file"] == "results_new_board.csv"
    assert rows["new_board"]["branch"] == "llm"


def test_ignores_accuracy_csvs_subfolders_and_bad_headers(tmp_path):
    raw = tmp_path / "raw"
    _write(raw / "results_a.csv", [_row(VIT, "a")])
    _write(raw / "vit_accuracy_offline.csv", [{"run_id": VIT, "accuracy_value": "1"}], ["run_id", "accuracy_value"])
    _write(raw / "backups" / "results_a.csv", [_row(VIT, "a")])
    _write(raw / "results_bad.csv", [{"run_id": VIT, "device": "bad"}], ["run_id", "device"])
    assert _run(tmp_path) == 0
    assert [r["source_file"] for r in _out(tmp_path)] == ["results_a.csv"]
    log = (tmp_path / "out" / cer.LOG_NAME).read_text()
    assert "IGNORED results_bad.csv" in log and "IGNORED backups/" in log and "vit_accuracy_offline.csv" in log


def test_recomputes_an_llm_row_with_the_old_formula(tmp_path, capsys):
    old = 160 / 300.0                              # (n_prompt + n_gen) / energy_net_j
    _write(tmp_path / "raw" / "results_a.csv", [_row(LLM, "a", n=30, energy=300.0, ppw=old), _row(VIT, "a", n=1000, energy=50.0, ppw=20.0)])
    assert _run(tmp_path) == 0
    rows = {r["run_id"]: r for r in _out(tmp_path)}
    assert abs(float(rows[LLM]["perf_per_watt"]) - 16.0) < 1e-9       # 160 / (300 / 30)
    assert rows[VIT]["perf_per_watt"] == "20.0"                        # already right: untouched string
    assert "perf_per_watt" in capsys.readouterr().err
    assert "perf_per_watt_original" not in _out(tmp_path)[0]


def test_duplicates_are_an_error_and_write_no_csv(tmp_path):
    raw = tmp_path / "raw"
    _write(raw / "results_a.csv", [_row(VIT, "a")])
    _write(raw / "results_a2.csv", [_row(VIT, "a")])
    assert _run(tmp_path) == 1
    assert not (tmp_path / "out" / cer.OUT_NAME).exists()


def test_unknown_run_id_is_kept_with_empty_derived_and_a_warning(tmp_path, capsys):
    _write(tmp_path / "raw" / "results_a.csv", [_row("llm_made_up_model_phase9", "a")])
    assert _run(tmp_path) == 0
    r = _out(tmp_path)[0]
    assert (r["branch"], r["model"], r["format"], r["role"]) == ("", "", "", "")
    assert "unknown run_id" in capsys.readouterr().err


def test_recognises_f16_and_roles(tmp_path):
    _write(tmp_path / "raw" / "results_a.csv", [_row(F16, "a"), _row(LLM, "a"), _row(VIT, "a"), _row(FALLBACK, "a")])
    assert _run(tmp_path) == 0
    rows = {r["run_id"]: r for r in _out(tmp_path)}
    assert (rows[F16]["role"], rows[F16]["format"], rows[F16]["branch"]) == ("F16", "F16", "llm")
    assert rows[LLM]["role"] == "cuantizado" and rows[LLM]["format"] == "Q4_K_M"
    assert rows[VIT]["role"] == "despliegue" and rows[VIT]["branch"] == "vit"
    assert rows[FALLBACK]["role"] == "referencia en fallback"


def test_incomplete_rows_are_kept_and_listed(tmp_path):
    r = _row(VIT, "a")
    r["energy_net_j"] = ""
    _write(tmp_path / "raw" / "results_a.csv", [r])
    assert _run(tmp_path) == 0
    assert len(_out(tmp_path)) == 1
    assert "results_a.csv  " + VIT in (tmp_path / "out" / cer.LOG_NAME).read_text()


def test_energy_per_inference_mismatch_warns(tmp_path, capsys):
    _write(tmp_path / "raw" / "results_a.csv", [_row(VIT, "a", n=10, energy=100.0, epi=99.0)])
    assert _run(tmp_path) == 0
    assert "energy_per_inference" in capsys.readouterr().err


def test_idempotent_and_inputs_untouched(tmp_path):
    raw = tmp_path / "raw"
    _write(raw / "results_b.csv", [_row(LLM, "b", ppw=0.5), _row(F16, "b")])
    _write(raw / "results_a.csv", [_row(VIT, "a")])
    before = _sha(tmp_path)
    assert _run(tmp_path) == 0
    first = [(tmp_path / "out" / n).read_bytes() for n in (cer.OUT_NAME, cer.LOG_NAME)]
    assert _run(tmp_path) == 0
    assert first == [(tmp_path / "out" / n).read_bytes() for n in (cer.OUT_NAME, cer.LOG_NAME)]
    assert _sha(tmp_path) == before
    assert [r["device"] for r in _out(tmp_path)] == ["a", "b", "b"]


def test_dry_run_writes_nothing_and_out_dir_inside_raw_is_refused(tmp_path):
    _write(tmp_path / "raw" / "results_a.csv", [_row(VIT, "a")])
    assert _run(tmp_path, "--dry-run") == 0
    assert not (tmp_path / "out").exists()
    assert cer.main(["--raw-dir", str(tmp_path / "raw"), "--out-dir", str(tmp_path / "raw" / "combined")]) == 2


def test_empty_labels_are_filled_from_the_manifests(tmp_path):
    _write(tmp_path / "raw" / "results_a.csv", [_row(F16, "a"), _row(LLM, "a"), _row(VIT, "a"), _row(FALLBACK, "a")])
    assert _run(tmp_path) == 0
    rows = {r["run_id"]: r for r in _out(tmp_path)}
    assert [rows[F16][k] for k in ("phase", "factor_varied", "factor_level", "bitwidth")] == ["reference", "reference", "f16", "f16"]
    assert (rows[LLM]["phase"], rows[LLM]["quant_scheme"], rows[LLM]["model_family"]) == ("phase1", "k_quant", "llm")
    assert rows[VIT]["phase"] == "deployment" and rows[VIT]["factor_varied"] in ("core", "granularity", "symmetry")
    assert rows[FALLBACK]["phase"] == "phase2" and rows[FALLBACK]["calibration_algorithm"] != ""


def test_a_label_already_in_the_csv_is_kept_with_a_warning(tmp_path, capsys):
    r = _row(LLM, "a")
    r["phase"] = "otra"
    _write(tmp_path / "raw" / "results_a.csv", [r])
    assert _run(tmp_path) == 0
    assert _out(tmp_path)[0]["phase"] == "otra"
    assert "phase='otra'" in capsys.readouterr().err


def test_alternative_f16_names_are_renamed_to_the_reference_run_id(tmp_path, capsys):
    alias = llm_manifest.generate()[0]["run_id"][:-len("q4km")] + "f16"
    _write(tmp_path / "raw" / "results_a.csv", [_row(alias, "a")])
    _write(tmp_path / "raw" / "results_b.csv", [_row(F16, "b")])
    assert _run(tmp_path) == 0
    rows = {r["device"]: r for r in _out(tmp_path)}
    assert rows["a"]["run_id"] == F16 and rows["a"]["result_id"] == f"{F16}__a"
    assert (rows["a"]["role"], rows["a"]["phase"]) == ("F16", "reference")
    assert "renamed" in capsys.readouterr().err


def test_alias_and_reference_on_the_same_device_is_a_duplicate(tmp_path):
    alias = llm_manifest.generate()[0]["run_id"][:-len("q4km")] + "f16"
    _write(tmp_path / "raw" / "results_a.csv", [_row(alias, "a"), _row(F16, "a")])
    assert _run(tmp_path) == 1
