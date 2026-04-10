"""Tests for on_the_record.transcribe — response parsing."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from on_the_record.transcribe import (
    TranscriptSegment,
    _parse_diarized,
    _parse_verbose,
    transcribe_chunk,
)


# ---------------------------------------------------------------------------
# Helpers — fake API response objects
# ---------------------------------------------------------------------------


@dataclass
class FakeDiarizedSegment:
    speaker: str
    text: str
    start: float
    end: float


@dataclass
class FakeVerboseSegment:
    text: str
    start: float
    end: float


class FakeDiarizedResponse:
    def __init__(self, segments):
        self.segments = segments


class FakeVerboseResponse:
    def __init__(self, segments):
        self.segments = segments


# ---------------------------------------------------------------------------
# _parse_diarized
# ---------------------------------------------------------------------------


class TestParseDiarized:
    def test_basic(self):
        resp = FakeDiarizedResponse(
            segments=[
                FakeDiarizedSegment(
                    speaker="Speaker 1", text="Hello there.", start=0.0, end=1.5
                ),
                FakeDiarizedSegment(
                    speaker="Speaker 2", text="Hi!", start=1.5, end=2.0
                ),
            ]
        )
        result = _parse_diarized(resp, offset=0.0)
        assert len(result) == 2
        assert result[0].speaker == "Speaker 1"
        assert result[0].text == "Hello there."
        assert result[1].speaker == "Speaker 2"

    def test_with_offset(self):
        resp = FakeDiarizedResponse(
            segments=[
                FakeDiarizedSegment(speaker="S1", text="Word", start=1.0, end=2.0),
            ]
        )
        result = _parse_diarized(resp, offset=30.0)
        assert result[0].start == 31.0
        assert result[0].end == 32.0

    def test_skips_empty_text(self):
        resp = FakeDiarizedResponse(
            segments=[
                FakeDiarizedSegment(speaker="S1", text="", start=0.0, end=1.0),
                FakeDiarizedSegment(speaker="S1", text="   ", start=1.0, end=2.0),
                FakeDiarizedSegment(speaker="S1", text="Real text", start=2.0, end=3.0),
            ]
        )
        result = _parse_diarized(resp, offset=0.0)
        assert len(result) == 1
        assert result[0].text == "Real text"

    def test_missing_speaker_defaults(self):
        resp = FakeDiarizedResponse(
            segments=[
                FakeDiarizedSegment(speaker=None, text="Test", start=0.0, end=1.0),
            ]
        )
        result = _parse_diarized(resp, offset=0.0)
        assert result[0].speaker == "Unknown"

    def test_empty_segments(self):
        resp = FakeDiarizedResponse(segments=[])
        result = _parse_diarized(resp, offset=0.0)
        assert result == []

    def test_no_segments_attribute(self):
        resp = MagicMock(spec=[])  # no .segments attribute
        result = _parse_diarized(resp, offset=0.0)
        assert result == []


# ---------------------------------------------------------------------------
# _parse_verbose
# ---------------------------------------------------------------------------


class TestParseVerbose:
    def test_basic(self):
        resp = FakeVerboseResponse(
            segments=[
                FakeVerboseSegment(text="Hello.", start=0.0, end=1.0),
            ]
        )
        result = _parse_verbose(resp, offset=0.0)
        assert len(result) == 1
        assert result[0].speaker == "Speaker"
        assert result[0].text == "Hello."

    def test_with_offset(self):
        resp = FakeVerboseResponse(
            segments=[
                FakeVerboseSegment(text="Word", start=5.0, end=6.0),
            ]
        )
        result = _parse_verbose(resp, offset=15.0)
        assert result[0].start == 20.0
        assert result[0].end == 21.0


# ---------------------------------------------------------------------------
# transcribe_chunk — mocked API call
# ---------------------------------------------------------------------------


class TestTranscribeChunk:
    @patch("on_the_record.transcribe.OpenAI")
    def test_diarize_model(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        fake_resp = FakeDiarizedResponse(
            segments=[
                FakeDiarizedSegment(speaker="Alice", text="Hello", start=0.0, end=1.0),
            ]
        )
        mock_client.audio.transcriptions.create.return_value = fake_resp

        result = transcribe_chunk(
            b"fake-wav",
            api_key="sk-test",
            model="gpt-4o-transcribe-diarize",
            chunk_offset=0.0,
        )

        assert len(result) == 1
        assert result[0].speaker == "Alice"
        assert result[0].text == "Hello"

        # Verify the API was called with diarized_json format
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["response_format"] == "diarized_json"
        assert call_kwargs["chunking_strategy"] == "auto"

    @patch("on_the_record.transcribe.OpenAI")
    def test_non_diarize_model(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        fake_resp = FakeVerboseResponse(
            segments=[
                FakeVerboseSegment(text="Hi", start=0.0, end=0.5),
            ]
        )
        mock_client.audio.transcriptions.create.return_value = fake_resp

        result = transcribe_chunk(
            b"fake-wav",
            api_key="sk-test",
            model="gpt-4o-transcribe",
            chunk_offset=0.0,
        )

        assert len(result) == 1
        assert result[0].speaker == "Speaker"

        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["response_format"] == "verbose_json"
        assert "chunking_strategy" not in call_kwargs
