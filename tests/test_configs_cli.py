"""Tests for configs/cli.py -- `python -m configs <command>`, including the
Phase 3 subcommands' precondition check and clear-error wrapping.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from configs import cli


def test_vit_and_llm_commands_call_their_manifest_writers(monkeypatch):
    vit_write = Mock()
    llm_write = Mock()
    monkeypatch.setattr(cli.vit_manifest, "write", vit_write)
    monkeypatch.setattr(cli.llm_manifest, "write", llm_write)

    cli.main(["vit"])
    vit_write.assert_called_once()
    llm_write.assert_not_called()

    cli.main(["llm"])
    llm_write.assert_called_once()


def test_phase3_commands_fail_clearly_when_report_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_phase3_report_path", lambda: tmp_path / "does_not_exist.txt")
    vit_phase3_write = Mock()
    monkeypatch.setattr(cli.vit_phase3_manifest, "write", vit_phase3_write)

    with pytest.raises(SystemExit, match="python -m analysis.phase3_selection"):
        cli.main(["vit-phase3"])

    vit_phase3_write.assert_not_called()  # precondition must block BEFORE attempting generation


def test_phase3_commands_succeed_when_report_present(tmp_path, monkeypatch):
    report_path = tmp_path / "phase3_selection_report.txt"
    report_path.write_text("(fake report)")
    monkeypatch.setattr(cli, "_phase3_report_path", lambda: report_path)

    vit_phase3_write = Mock()
    llm_phase3_write = Mock()
    monkeypatch.setattr(cli.vit_phase3_manifest, "write", vit_phase3_write)
    monkeypatch.setattr(cli.llm_phase3_manifest, "write", llm_phase3_write)

    cli.main(["vit-phase3"])
    vit_phase3_write.assert_called_once()

    cli.main(["llm-phase3"])
    llm_phase3_write.assert_called_once()


def test_phase3_commands_wrap_valueerror_with_clear_guidance(tmp_path, monkeypatch):
    report_path = tmp_path / "phase3_selection_report.txt"
    report_path.write_text("(fake report)")
    monkeypatch.setattr(cli, "_phase3_report_path", lambda: report_path)
    monkeypatch.setattr(
        cli.vit_phase3_manifest, "write",
        Mock(side_effect=ValueError("factor 'granularity': no rows match its baseline comparator config {...}")),
    )

    with pytest.raises(SystemExit, match="Phase 1-2 ViT results appear incomplete"):
        cli.main(["vit-phase3"])


def test_all_runs_all_four_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_require_phase3_report", lambda: calls.append("check"))
    monkeypatch.setattr(cli.vit_manifest, "write", lambda: calls.append("vit"))
    monkeypatch.setattr(cli.llm_manifest, "write", lambda: calls.append("llm"))
    monkeypatch.setattr(cli.vit_phase3_manifest, "write", lambda: calls.append("vit-phase3"))
    monkeypatch.setattr(cli.llm_phase3_manifest, "write", lambda: calls.append("llm-phase3"))

    cli.main(["all"])

    # Phase 1-2 (order between vit/llm doesn't matter -- both unconditional
    # and independent), but every Phase 3 write must come after BOTH Phase
    # 1-2 writes, and each Phase 3 write is preceded by its own precondition
    # check (cmd_vit_phase3/cmd_llm_phase3, not a direct .write() call).
    assert calls.index("vit") < calls.index("vit-phase3")
    assert calls.index("llm") < calls.index("llm-phase3")
    assert calls.count("check") == 2  # once per Phase 3 command


def test_all_fails_clearly_when_report_missing_after_phase1_2_succeed(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_phase3_report_path", lambda: tmp_path / "missing.txt")
    vit_write = Mock()
    llm_write = Mock()
    monkeypatch.setattr(cli.vit_manifest, "write", vit_write)
    monkeypatch.setattr(cli.llm_manifest, "write", llm_write)

    with pytest.raises(SystemExit, match="python -m analysis.phase3_selection"):
        cli.main(["all"])

    # Phase 1-2 still ran to completion before the Phase 3 precondition failed.
    vit_write.assert_called_once()
    llm_write.assert_called_once()
