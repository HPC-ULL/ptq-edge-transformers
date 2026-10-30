"""scripts/check_csv_additions.py against files written the way append_result writes them."""

from __future__ import annotations

import csv
import shutil

from benchmarking_ondevice.power_logging import results_io
from benchmarking_ondevice.power_logging.results_io import RESULT_COLUMNS
import scripts.check_csv_additions as cca

DEVICE = "jetson_cpu_15w"


def _row(run_id, tg="10.0"):
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(result_id=f"{run_id}__{DEVICE}", run_id=run_id, device=DEVICE, throughput_decode_tps=tg)
    return row


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "results_csv_path", lambda d: tmp_path / "raw" / f"results_{d}.csv")
    for r in ("llm_a_phase1_baseline_default_q4km", "llm_b_phase1_baseline_default_q4km"):
        results_io.append_result(DEVICE, _row(r))
    current = tmp_path / "raw" / f"results_{DEVICE}.csv"
    backup = tmp_path / "backup.csv"
    shutil.copy2(current, backup)
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        csv.writer(f).writerows([["run_id", "repeat_id", "artifact_path"],
                                 ["llm_a_reference_f16", "", "x"], ["llm_b_reference_f16", "", "y"]])
    return backup, current, manifest


def test_only_new_rows_ok(tmp_path, monkeypatch):
    backup, current, manifest = _setup(tmp_path, monkeypatch)
    results_io.append_result(DEVICE, _row("llm_a_reference_f16"))
    ok, lines = cca.compare(backup, current, manifest)
    assert ok, lines
    assert any("identical" in l for l in lines)
    assert any("without a new row: llm_b_reference_f16" in l for l in lines)  # a warning, not a failure
    assert cca.main([str(backup), str(current), "--manifest", str(manifest)]) == 0


def test_unchanged_file_ok(tmp_path, monkeypatch):
    backup, current, _ = _setup(tmp_path, monkeypatch)
    assert cca.compare(backup, current)[0]


def test_overwritten_row_fails(tmp_path, monkeypatch):
    backup, current, _ = _setup(tmp_path, monkeypatch)
    # a --force re-run of an existing row: moved to the end with new values
    results_io.append_result(DEVICE, _row("llm_a_phase1_baseline_default_q4km", tg="11.0"))
    ok, lines = cca.compare(backup, current)
    assert not ok
    assert any("changed" in l for l in lines)
    assert cca.main([str(backup), str(current)]) == 1


def test_row_outside_manifest_fails(tmp_path, monkeypatch):
    backup, current, manifest = _setup(tmp_path, monkeypatch)
    results_io.append_result(DEVICE, _row("llm_c_other"))
    ok, lines = cca.compare(backup, current, manifest)
    assert not ok and any("outside the manifest" in l for l in lines)


def test_header_change_fails(tmp_path, monkeypatch):
    backup, current, _ = _setup(tmp_path, monkeypatch)
    text = current.read_text().replace("result_id,", "rid,", 1)
    current.write_text(text)
    ok, lines = cca.compare(backup, current)
    assert not ok and "header" in lines[0]
