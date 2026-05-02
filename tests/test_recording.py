"""Tests for the reusable recording session controller."""

from __future__ import annotations

from types import SimpleNamespace

from on_the_record.config import Config
from on_the_record.recording import RecordingSession
from on_the_record.transcribe import TranscriptSegment


class _FakeChunk:
    def __init__(self, chunk_index: int):
        self.chunk_index = chunk_index
        self.start_time_offset = float(chunk_index * 15)

    def to_wav(self) -> bytes:
        return f"chunk-{self.chunk_index}".encode("ascii")


class _FakeRecorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stopped = False

    def stop(self):
        self.stopped = True

    def record(self):
        yield _FakeChunk(0)
        yield _FakeChunk(1)


class _RecordingWriter:
    def __init__(self):
        self.written: list[list[str]] = []
        self.finalized = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finalized = True
        return None

    def write_segments(self, segments):
        self.written.append([segment.text for segment in segments])


def test_recording_session_writes_segments_and_emits_events():
    writer = _RecordingWriter()
    events = []

    def fake_transcribe(wav_bytes, *, api_key, model, chunk_offset):
        return [
            TranscriptSegment(
                speaker="Speaker",
                text=wav_bytes.decode("ascii"),
                start=chunk_offset,
                end=chunk_offset + 1,
            )
        ]

    session = RecordingSession(
        Config(api_key="test-key", output_path="transcript.txt"),
        audio_module=SimpleNamespace(AudioRecorder=_FakeRecorder),
        writer_factory=lambda fmt, path: writer,
        transcribe=fake_transcribe,
        event_callback=lambda event_type, payload: events.append((event_type, payload)),
    )

    result = session.run()

    assert result.total_segments == 2
    assert result.output_path == "transcript.txt"
    assert writer.written == [["chunk-0"], ["chunk-1"]]
    assert writer.finalized is True
    assert [event_type for event_type, _ in events] == [
        "recording_started",
        "transcription_started",
        "segments_written",
        "transcription_started",
        "segments_written",
        "recording_stopped",
    ]


def test_recording_session_continues_after_transcription_error():
    writer = _RecordingWriter()
    events = []

    def fake_transcribe(wav_bytes, *, api_key, model, chunk_offset):
        if wav_bytes == b"chunk-0":
            raise RuntimeError("temporary failure")
        return [
            TranscriptSegment(
                speaker="Speaker",
                text="recovered",
                start=chunk_offset,
                end=chunk_offset + 1,
            )
        ]

    session = RecordingSession(
        Config(api_key="test-key", output_path="transcript.txt"),
        audio_module=SimpleNamespace(AudioRecorder=_FakeRecorder),
        writer_factory=lambda fmt, path: writer,
        transcribe=fake_transcribe,
        event_callback=lambda event_type, payload: events.append((event_type, payload)),
    )

    result = session.run()

    assert result.total_segments == 1
    assert writer.written == [["recovered"]]
    assert any(event_type == "transcription_failed" for event_type, _ in events)
