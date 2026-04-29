"""Tests for configuration helpers."""

from __future__ import annotations

import pytest

from on_the_record import config


def test_api_key_setup_hint_uses_windows_commands():
    hint = config._api_key_setup_hint("nt")

    assert '$env:OPENAI_API_KEY = "sk-..."' in hint
    assert "set OPENAI_API_KEY=sk-..." in hint


def test_load_api_key_prints_windows_guidance(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config,
        "_api_key_setup_hint",
        lambda os_name=None: (
            'PowerShell:  $env:OPENAI_API_KEY = "sk-..."\n'
            "cmd.exe:     set OPENAI_API_KEY=sk-..."
        ),
    )

    with pytest.raises(SystemExit):
        config.load_api_key()

    err = capsys.readouterr().err
    assert '$env:OPENAI_API_KEY = "sk-..."' in err
    assert "set OPENAI_API_KEY=sk-..." in err


def test_load_api_key_uses_dotenv_without_overriding_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=from-file\n"
        "EXISTING=from-file\n"
        "QUOTED=\"hello world\"\n"
        "INLINE=value # comment\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EXISTING", "from-env")
    monkeypatch.chdir(tmp_path)

    assert config.load_api_key() == "from-file"
    assert config.os.environ["EXISTING"] == "from-env"
    assert config.os.environ["QUOTED"] == "hello world"
    assert config.os.environ["INLINE"] == "value"


def test_dotenv_paths_include_pyinstaller_bundle(monkeypatch, tmp_path):
    bundle_dir = tmp_path / "bundle"
    exe_dir = tmp_path / "dist"
    exe = exe_dir / "on-the-record.exe"
    bundle_dir.mkdir()
    exe_dir.mkdir()
    exe.write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config.sys, "frozen", True, raising=False)
    monkeypatch.setattr(config.sys, "executable", str(exe))
    monkeypatch.setattr(config.sys, "_MEIPASS", str(bundle_dir), raising=False)

    assert config._dotenv_paths() == [
        tmp_path / ".env",
        exe_dir / ".env",
        bundle_dir / ".env",
    ]
