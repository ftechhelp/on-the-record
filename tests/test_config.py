"""Tests for configuration helpers."""

from __future__ import annotations

import pytest

from on_the_record import config


def test_api_key_setup_hint_uses_windows_commands():
    hint = config._api_key_setup_hint("nt")

    assert '$env:OPENAI_API_KEY = "sk-..."' in hint
    assert "set OPENAI_API_KEY=sk-..." in hint


def test_load_api_key_prints_windows_guidance(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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