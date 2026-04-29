"""Output writers for transcript segments.

Supports plain text (.txt), Markdown (.md), and JSON (.json) formats.
All writers append to an existing file so they work with the streaming
chunk-by-chunk workflow.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from on_the_record.transcribe import TranscriptSegment

logger = logging.getLogger("on_the_record.writer")


def _format_timestamp(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS``."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TranscriptWriter(ABC):
    """Base class for transcript output writers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def write_segments(self, segments: list[TranscriptSegment]) -> None:
        """Append *segments* to the output file."""

    @abstractmethod
    def finalize(self) -> None:
        """Perform any cleanup when recording stops (e.g. close JSON array)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finalize()


# ---------------------------------------------------------------------------
# Plain text writer
# ---------------------------------------------------------------------------


class TxtWriter(TranscriptWriter):
    """Writes segments as ``[HH:MM:SS] Speaker: text``."""

    def write_segments(self, segments: list[TranscriptSegment]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for seg in segments:
                ts = _format_timestamp(seg.start)
                f.write(f"[{ts}] {seg.speaker}: {seg.text}\n")
        logger.debug("Wrote %d segment(s) to %s", len(segments), self.path)

    def finalize(self) -> None:
        pass  # nothing to do for plain text


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------


class MdWriter(TranscriptWriter):
    """Writes segments as Markdown with timestamps as headings."""

    _header_written: bool = False

    def write_segments(self, segments: list[TranscriptSegment]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            if not self._header_written:
                f.write("# Transcript\n\n")
                self._header_written = True

            for seg in segments:
                ts = _format_timestamp(seg.start)
                f.write(f"**[{ts}]** **{seg.speaker}**: {seg.text}\n\n")
        logger.debug("Wrote %d segment(s) to %s", len(segments), self.path)

    def finalize(self) -> None:
        pass


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------


class JsonWriter(TranscriptWriter):
    """Writes segments as a JSON array.

    On each :meth:`write_segments` call the existing array is loaded from
    disk (if present), the new segments are appended, and the whole array
    is rewritten.  This is simple and correct for the expected file sizes
    (a few MB of text at most).
    """

    def write_segments(self, segments: list[TranscriptSegment]) -> None:
        existing: list[dict] = []
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Could not parse existing JSON at %s; overwriting.", self.path
                )

        for seg in segments:
            existing.append(
                {
                    "speaker": seg.speaker,
                    "text": seg.text,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "start_formatted": _format_timestamp(seg.start),
                    "end_formatted": _format_timestamp(seg.end),
                }
            )

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
            f.write("\n")

        logger.debug(
            "Wrote %d segment(s) to %s (total %d)",
            len(segments),
            self.path,
            len(existing),
        )

    def finalize(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_WRITERS: dict[str, type[TranscriptWriter]] = {
    "txt": TxtWriter,
    "md": MdWriter,
    "json": JsonWriter,
}

SUPPORTED_FORMATS = tuple(_WRITERS.keys())


def get_writer(fmt: str, path: str | Path) -> TranscriptWriter:
    """Return a writer instance for the given format.

    Raises ``ValueError`` if *fmt* is not one of the supported formats.
    """
    cls = _WRITERS.get(fmt)
    if cls is None:
        raise ValueError(
            f"Unsupported output format '{fmt}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    return cls(path)


def rewrite_segments(fmt: str, path: str | Path, segments: list[TranscriptSegment]) -> None:
    """Replace *path* with *segments* using the selected transcript format."""
    output_path = Path(path)
    if output_path.exists():
        output_path.unlink()

    with get_writer(fmt, output_path) as writer:
        writer.write_segments(segments)
