"""scripts/recompute_llm_perf_per_watt.py: offline fix of the LLM
perf_per_watt column in the per-device results CSVs."""

from __future__ import annotations

import csv

import pytest

import scripts.recompute_llm_perf_per_watt as recompute
from benchmarking_ondevice.runner_llm.results_io import RESULT_COLUMNS


def _row(result_id, run_id, **fields):
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(result_id=result_id, run_id=run_id, **{k: str(v) for k, v in fields.items()})
    return row


def _write(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _old_formula(energy_net_j):
    return 160 / energy_net_j  # the pre-fix value: tokens/J divided by N


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "results_jetson_cpu_50w.csv"
    _write(path, [
        _row("llm_a__dev", "llm_a", num_inferences_measured=5, energy_net_j=8.0, perf_per_watt=_old_formula(8.0)),
        _row("llm_b__dev__rep1", "llm_b", num_inferences_measured=3, energy_net_j=2.0, perf_per_watt=_old_formula(2.0)),
        _row("llm_c__dev", "llm_c", num_inferences_measured=5),  # no power measurement
        _row("vit_x__dev__rep1", "vit_x", num_inferences_measured=1000, energy_net_j=50.0,
             perf_per_watt="123.456", throughput="35.6"),
    ])
    return path


def test_llm_rows_get_the_corrected_value_with_their_own_n(csv_path):
    report = recompute.process_file(csv_path, dry_run=False, stamp="T")
    rows = {r["result_id"]: r for r in _read(csv_path)}

    assert float(rows["llm_a__dev"]["perf_per_watt"]) == pytest.approx(160 * 5 / 8.0)
    assert float(rows["llm_b__dev__rep1"]["perf_per_watt"]) == pytest.approx(160 * 3 / 2.0)
    assert report["changed"] == 2 and report["no_energy"] == 1
    assert dict(report["factors"]) == {5.0: 1, 3.0: 1} and report["factor_not_n"] == []


def test_vit_rows_and_other_columns_are_untouched(csv_path):
    before = {r["result_id"]: r for r in _read(csv_path)}
    recompute.process_file(csv_path, dry_run=False, stamp="T")
    after = {r["result_id"]: r for r in _read(csv_path)}

    assert after["vit_x__dev__rep1"] == before["vit_x__dev__rep1"]
    assert after["llm_c__dev"] == before["llm_c__dev"]
    for rid in ("llm_a__dev", "llm_b__dev__rep1"):
        assert {k: v for k, v in after[rid].items() if k != "perf_per_watt"} == \
               {k: v for k, v in before[rid].items() if k != "perf_per_watt"}


def test_a_backup_of_the_original_is_written_first(csv_path):
    original = csv_path.read_text()
    report = recompute.process_file(csv_path, dry_run=False, stamp="T")

    assert report["backup"] == csv_path.with_name(csv_path.name + ".bak_T")
    assert report["backup"].read_text() == original


def test_a_results_raw_csv_gets_its_backup_in_results_backups(tmp_path):
    raw = tmp_path / "results" / "raw"
    raw.mkdir(parents=True)
    path = raw / "results_jetson_cpu_50w.csv"
    _write(path, [_row("llm_a__dev", "llm_a", num_inferences_measured=5, energy_net_j=8.0,
                       perf_per_watt=_old_formula(8.0))])
    original = path.read_text()

    report = recompute.process_file(path, dry_run=False, stamp="T")

    assert report["backup"] == tmp_path / "results" / "backups" / "results_jetson_cpu_50w.csv.bak_T"
    assert report["backup"].read_text() == original
    assert list(raw.glob("*.bak_*")) == []  # raw/ keeps only data


def test_second_run_changes_nothing_and_writes_no_backup(csv_path):
    recompute.process_file(csv_path, dry_run=False, stamp="T1")
    fixed = csv_path.read_text()
    report = recompute.process_file(csv_path, dry_run=False, stamp="T2")

    assert report["changed"] == 0 and report["unchanged"] == 2
    assert csv_path.read_text() == fixed
    assert not csv_path.with_name(csv_path.name + ".bak_T2").exists()


def test_dry_run_writes_nothing(csv_path):
    original = csv_path.read_text()
    report = recompute.process_file(csv_path, dry_run=True, stamp="T")

    assert report["changed"] == 2 and report["backup"] is None
    assert csv_path.read_text() == original
    assert list(csv_path.parent.glob("*.bak_*")) == []


def test_a_factor_other_than_n_is_flagged(tmp_path):
    path = tmp_path / "results_x.csv"
    _write(path, [_row("llm_a__dev", "llm_a", num_inferences_measured=5, energy_net_j=8.0, perf_per_watt="1.0")])

    report = recompute.process_file(path, dry_run=True, stamp="T")

    assert report["factor_not_n"] == [("llm_a__dev", 5, pytest.approx(100.0))]
    assert recompute.main([str(path), "--dry-run"]) == 2
