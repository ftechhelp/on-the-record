"""Tests for CLI startup behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from on_the_record import cli


def test_start_does_not_load_audio_before_api_key(monkeypatch):
    args = SimpleNamespace(
        output="",
        format="txt",
        chunk_size=15,
        device=None,
        model="gpt-4o-transcribe",
        diarize=True,
    )
    audio_loaded = False

    def fake_load_api_key():
        raise SystemExit(1)

    def fake_load_audio_module():
        nonlocal audio_loaded
        audio_loaded = True
        raise AssertionError("audio module should not load before API key validation")

    monkeypatch.setattr(cli, "load_api_key", fake_load_api_key)
    monkeypatch.setattr(cli, "_load_audio_module", fake_load_audio_module)

    with pytest.raises(SystemExit):
        cli._cmd_start(args)

    assert audio_loaded is False