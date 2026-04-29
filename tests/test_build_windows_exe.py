"""Tests for the Windows PyInstaller build helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_build_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_windows_exe.py"
    spec = importlib.util.spec_from_file_location("build_windows_exe", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_bundles_project_dotenv_when_present(tmp_path, monkeypatch):
    build_script = _load_build_script()
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(build_script.platform, "system", lambda: "Windows")
    monkeypatch.setattr(build_script, "ENV_FILE", env_file)
    monkeypatch.setattr(build_script, "_prepare_speaker_model", lambda: None)

    def fake_run(command, *, cwd, check):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["check"] = check

    monkeypatch.setattr(build_script.subprocess, "run", fake_run)

    assert build_script.main() == 0

    assert captured["check"] is True
    assert "--add-data" in captured["command"]
    assert f"{env_file};." in captured["command"]
