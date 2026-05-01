"""Tests for Obsidian export configuration."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from on_the_record import obsidian


def test_save_load_and_clear_obsidian_config(tmp_path):
    config_path = tmp_path / "config.json"
    config = obsidian.ObsidianConfig(
        vault_path=tmp_path / "Vault",
        study_folder="OTR/Study Notes",
        cli_command="obsidian open {file}",
    )

    assert obsidian.save_obsidian_config(config, config_path) == config_path
    assert obsidian.load_obsidian_config(config_path) == config
    assert obsidian.clear_obsidian_config(config_path) is True
    assert obsidian.load_obsidian_config(config_path) is None


def test_resolve_obsidian_destination_rejects_absolute_folder(tmp_path):
    config = obsidian.ObsidianConfig(
        vault_path=tmp_path / "Vault",
        study_folder=str(tmp_path / "Other"),
    )

    with pytest.raises(ValueError, match="relative"):
        obsidian.resolve_obsidian_destination(tmp_path / "study.md", config)


def test_resolve_obsidian_destination_rejects_path_traversal(tmp_path):
    config = obsidian.ObsidianConfig(
        vault_path=tmp_path / "Vault",
        study_folder="../Other",
    )

    with pytest.raises(ValueError, match="inside"):
        obsidian.resolve_obsidian_destination(tmp_path / "study.md", config)


def test_export_study_document_copies_into_vault_with_unique_name(tmp_path):
    vault = tmp_path / "Vault"
    source = tmp_path / "2026-05-01-topic.md"
    source.write_text("# Topic\n", encoding="utf-8")
    existing = vault / "OTR" / "2026-05-01-topic.md"
    existing.parent.mkdir(parents=True)
    existing.write_text("existing", encoding="utf-8")

    config = obsidian.ObsidianConfig(vault_path=vault, study_folder="OTR")

    exported = obsidian.export_study_document_to_obsidian(source, config)

    assert exported == vault / "OTR" / "2026-05-01-topic-2.md"
    assert exported.read_text(encoding="utf-8") == "# Topic\n"
    assert existing.read_text(encoding="utf-8") == "existing"


def test_export_study_document_runs_cli_hook_with_placeholders(tmp_path, monkeypatch):
    vault = tmp_path / "Vault"
    source = tmp_path / "study.md"
    source.write_text("# Study\n", encoding="utf-8")
    calls = []

    def fake_run(command, *, check, capture_output, text):
        calls.append(
            {
                "command": command,
                "check": check,
                "capture_output": capture_output,
                "text": text,
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(obsidian.subprocess, "run", fake_run)

    config = obsidian.ObsidianConfig(
        vault_path=vault,
        study_folder="OTR",
        cli_command="obsidian open {file} --vault {vault}",
    )

    exported = obsidian.export_study_document_to_obsidian(source, config)

    assert calls == [
        {
            "command": [
                "obsidian",
                "open",
                str(exported),
                "--vault",
                str(vault),
            ],
            "check": False,
            "capture_output": True,
            "text": True,
        }
    ]


def test_cli_hook_appends_file_when_command_has_no_placeholder(tmp_path, monkeypatch):
    exported = tmp_path / "study.md"
    exported.write_text("# Study\n", encoding="utf-8")
    calls = []

    def fake_run(command, *, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(obsidian.subprocess, "run", fake_run)

    config = obsidian.ObsidianConfig(
        vault_path=tmp_path / "Vault",
        cli_command="obsidian open",
    )

    obsidian.run_obsidian_cli_hook(exported, config)

    assert calls == [["obsidian", "open", str(exported)]]
