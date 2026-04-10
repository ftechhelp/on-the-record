"""Tests for on_the_record.writer — all output formats."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from on_the_record.transcribe import TranscriptSegment
from on_the_record.writer import (
    TxtWriter,
    MdWriter,
    JsonWriter,
    get_writer,
    SUPPORTED_FORMATS,
    _format_timestamp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(speaker="Alice", text="Hello there.", start=0.0, end=1.5),
        TranscriptSegment(speaker="Bob", text="Hi Alice!", start=1.5, end=2.8),
    ]


# ---------------------------------------------------------------------------
# _format_timestamp
# ---------------------------------------------------------------------------


class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0.0) == "00:00:00"

    def test_seconds_only(self):
        assert _format_timestamp(45.7) == "00:00:45"

    def test_minutes(self):
        assert _format_timestamp(125.0) == "00:02:05"

    def test_hours(self):
        assert _format_timestamp(3661.0) == "01:01:01"


# ---------------------------------------------------------------------------
# TxtWriter
# ---------------------------------------------------------------------------


class TestTxtWriter:
    def test_write_segments(self, tmp_path: Path):
        out = tmp_path / "test.txt"
        writer = TxtWriter(out)
        writer.write_segments(_make_segments())

        content = out.read_text()
        assert "[00:00:00] Alice: Hello there." in content
        assert "[00:00:01] Bob: Hi Alice!" in content

    def test_append_mode(self, tmp_path: Path):
        out = tmp_path / "test.txt"
        writer = TxtWriter(out)
        writer.write_segments(_make_segments()[:1])
        writer.write_segments(_make_segments()[1:])

        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "sub" / "dir" / "test.txt"
        writer = TxtWriter(out)
        writer.write_segments(_make_segments()[:1])
        assert out.exists()


# ---------------------------------------------------------------------------
# MdWriter
# ---------------------------------------------------------------------------


class TestMdWriter:
    def test_write_segments(self, tmp_path: Path):
        out = tmp_path / "test.md"
        writer = MdWriter(out)
        writer.write_segments(_make_segments())

        content = out.read_text()
        assert "# Transcript" in content
        assert "**Alice**" in content
        assert "**Bob**" in content
        assert "**[00:00:00]**" in content

    def test_header_written_once(self, tmp_path: Path):
        out = tmp_path / "test.md"
        writer = MdWriter(out)
        writer.write_segments(_make_segments()[:1])
        writer.write_segments(_make_segments()[1:])

        content = out.read_text()
        assert content.count("# Transcript") == 1


# ---------------------------------------------------------------------------
# JsonWriter
# ---------------------------------------------------------------------------


class TestJsonWriter:
    def test_write_segments(self, tmp_path: Path):
        out = tmp_path / "test.json"
        writer = JsonWriter(out)
        writer.write_segments(_make_segments())

        data = json.loads(out.read_text())
        assert len(data) == 2
        assert data[0]["speaker"] == "Alice"
        assert data[0]["text"] == "Hello there."
        assert data[0]["start"] == 0.0
        assert data[0]["end"] == 1.5
        assert data[0]["start_formatted"] == "00:00:00"

    def test_append_mode(self, tmp_path: Path):
        out = tmp_path / "test.json"
        writer = JsonWriter(out)
        writer.write_segments(_make_segments()[:1])
        writer.write_segments(_make_segments()[1:])

        data = json.loads(out.read_text())
        assert len(data) == 2

    def test_handles_corrupt_existing(self, tmp_path: Path):
        out = tmp_path / "test.json"
        out.write_text("not valid json")

        writer = JsonWriter(out)
        writer.write_segments(_make_segments()[:1])

        data = json.loads(out.read_text())
        assert len(data) == 1


# ---------------------------------------------------------------------------
# get_writer factory
# ---------------------------------------------------------------------------


class TestGetWriter:
    def test_all_formats(self, tmp_path: Path):
        for fmt in SUPPORTED_FORMATS:
            writer = get_writer(fmt, tmp_path / f"test.{fmt}")
            assert writer is not None

    def test_invalid_format(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unsupported output format"):
            get_writer("csv", tmp_path / "test.csv")

    def test_context_manager(self, tmp_path: Path):
        out = tmp_path / "test.txt"
        with get_writer("txt", out) as w:
            w.write_segments(_make_segments())
        assert out.exists()
