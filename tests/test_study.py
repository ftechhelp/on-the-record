"""Tests for Gemini study document generation."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from on_the_record import study


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_default_study_output_path_uses_markdown_suffix():
    assert study.default_study_output_path(Path("notes") / "transcript.txt") == Path(
        "notes"
    ) / "transcript_study.md"


def test_load_gemini_api_key_returns_none_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    assert study.load_gemini_api_key() is None


def test_generate_study_document_calls_gemini_and_returns_markdown(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "```markdown\n# Study Guide\n\n- Remember this.\n```"}
                            ]
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(study.urllib.request, "urlopen", fake_urlopen)

    document = study.generate_study_document(
        "[00:00:00] Speaker: Learn the core idea.",
        api_key="gemini-key",
        model="gemini-test",
        timeout=3,
    )

    assert document == "# Study Guide\n\n- Remember this."
    assert "gemini-test:generateContent" in captured["url"]
    assert "key=gemini-key" in captured["url"]
    assert captured["timeout"] == 3
    prompt = captured["body"]["contents"][0]["parts"][0]["text"]
    assert "Turn the transcript below" in prompt
    assert "Learn the core idea" in prompt


def test_generate_study_document_rejects_empty_transcript():
    with pytest.raises(ValueError):
        study.generate_study_document("   ", api_key="gemini-key")


def test_generate_study_document_wraps_http_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(b"bad key"),
        )

    monkeypatch.setattr(study.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        study.generate_study_document("transcript", api_key="bad-key")


def test_write_study_document_reads_transcript_and_writes_markdown(tmp_path, monkeypatch):
    transcript = tmp_path / "transcript.txt"
    output = tmp_path / "study.md"
    transcript.write_text("raw transcript", encoding="utf-8")

    monkeypatch.setattr(
        study,
        "generate_study_document",
        lambda transcript_text, *, api_key, model: f"# Notes\n\n{transcript_text}",
    )

    written = study.write_study_document(
        transcript,
        output,
        api_key="gemini-key",
        model="gemini-test",
    )

    assert written == output
    assert output.read_text(encoding="utf-8") == "# Notes\n\nraw transcript\n"

