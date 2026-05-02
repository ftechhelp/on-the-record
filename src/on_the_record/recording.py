"""Reusable recording and transcription workflow.

This module contains the CLI-independent recording loop used by both the
terminal command and the future macOS app engine.
"""

from __future__ import annotations

import importlib
import logging
import queue
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass
from typing import Any

from on_the_record.config import Config
from on_the_record.transcribe import TranscriptSegment, transcribe_chunk
from on_the_record.writer import get_writer

logger = logging.getLogger("on_the_record.recording")

CAPTURE_POLL_INTERVAL = 0.1
_CAPTURE_COMPLETE = object()

RecordingEventCallback = Callable[[str, dict[str, Any]], None]
WriterFactory = Callable[[str, str], Any]
TranscribeFunction = Callable[..., list[TranscriptSegment]]


@dataclass(frozen=True)
class RecordingResult:
    """Summary returned after a recording session completes."""

    elapsed_seconds: float
    total_segments: int
    output_path: str


def load_audio_module():
    """Import audio backends only when recording actually needs them."""
    return importlib.import_module("on_the_record.audio")


def queued_chunks(
    recorder,
    *,
    poll_interval: float = CAPTURE_POLL_INTERVAL,
) -> Generator[Any, None, None]:
    """Yield recorded chunks while capture continues on a background thread."""

    chunk_queue: queue.Queue[object] = queue.Queue()
    capture_error: Exception | None = None

    def _capture() -> None:
        nonlocal capture_error

        try:
            for chunk in recorder.record():
                chunk_queue.put(chunk)
        except Exception as exc:
            capture_error = exc
        finally:
            chunk_queue.put(_CAPTURE_COMPLETE)

    capture_thread = threading.Thread(
        target=_capture,
        name="audio-capture",
        daemon=True,
    )
    capture_thread.start()

    try:
        while True:
            try:
                item = chunk_queue.get(timeout=poll_interval)
            except queue.Empty:
                if capture_error is not None and not capture_thread.is_alive():
                    raise capture_error
                continue

            if item is _CAPTURE_COMPLETE:
                if capture_error is not None:
                    raise capture_error
                break
            yield item
    finally:
        recorder.stop()
        capture_thread.join()


class RecordingSession:
    """Run one streaming audio recording and transcription session."""

    def __init__(
        self,
        config: Config,
        *,
        audio_module=None,
        writer_factory: WriterFactory = get_writer,
        transcribe: TranscribeFunction = transcribe_chunk,
        event_callback: RecordingEventCallback | None = None,
        poll_interval: float = CAPTURE_POLL_INTERVAL,
    ) -> None:
        self.config = config
        self.audio_module = audio_module or load_audio_module()
        self.writer_factory = writer_factory
        self.transcribe = transcribe
        self.event_callback = event_callback
        self.poll_interval = poll_interval
        self.recorder = None

    def request_stop(self) -> None:
        """Ask the active recorder to stop after the current chunk."""
        if self.recorder is not None:
            self.recorder.stop()
        self._emit("stop_requested")

    def run(self) -> RecordingResult:
        """Run the session synchronously until capture stops."""
        self.recorder = self.audio_module.AudioRecorder(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            chunk_seconds=self.config.chunk_seconds,
            device_name=self.config.device_name,
            include_system_audio=self.config.include_system_audio,
            include_microphone=self.config.include_microphone,
            silence_threshold=self.config.silence_threshold,
        )
        writer = self.writer_factory(
            self.config.output_format,
            self.config.output_path,
        )
        total_segments = 0
        start_time = time.monotonic()

        self._emit(
            "recording_started",
            output_path=self.config.output_path,
            output_format=self.config.output_format,
            model=self.config.model,
            chunk_seconds=self.config.chunk_seconds,
            include_system_audio=self.config.include_system_audio,
            include_microphone=self.config.include_microphone,
            device_name=self.config.device_name,
        )

        try:
            with writer:
                for chunk in queued_chunks(
                    self.recorder,
                    poll_interval=self.poll_interval,
                ):
                    self._emit(
                        "transcription_started",
                        chunk_index=chunk.chunk_index,
                        offset=chunk.start_time_offset,
                    )
                    wav_bytes = chunk.to_wav()

                    try:
                        segments = self.transcribe(
                            wav_bytes,
                            api_key=self.config.api_key,
                            model=self.config.model,
                            chunk_offset=chunk.start_time_offset,
                        )
                    except Exception as exc:
                        logger.error(
                            "Transcription failed for chunk %d: %s",
                            chunk.chunk_index,
                            exc,
                        )
                        self._emit(
                            "transcription_failed",
                            chunk_index=chunk.chunk_index,
                            error=str(exc),
                        )
                        continue

                    if segments:
                        writer.write_segments(segments)
                        total_segments += len(segments)
                        self._emit(
                            "segments_written",
                            chunk_index=chunk.chunk_index,
                            segment_count=len(segments),
                            segments=[asdict(segment) for segment in segments],
                        )
                    else:
                        self._emit(
                            "no_speech_detected",
                            chunk_index=chunk.chunk_index,
                        )
        except Exception as exc:
            self._emit("fatal_error", error=str(exc))
            raise

        elapsed = time.monotonic() - start_time
        result = RecordingResult(
            elapsed_seconds=elapsed,
            total_segments=total_segments,
            output_path=self.config.output_path,
        )
        self._emit(
            "recording_stopped",
            elapsed_seconds=elapsed,
            total_segments=total_segments,
            output_path=self.config.output_path,
        )
        return result

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self.event_callback is None:
            return
        self.event_callback(event_type, payload)
