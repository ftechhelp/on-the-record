"""Tests for CLI startup behavior."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from on_the_record import cli
from on_the_record.transcribe import TranscriptSegment


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


class _FakeChunk:
    def __init__(self, chunk_index: int):
        self.chunk_index = chunk_index
        self.start_time_offset = float(chunk_index * 15)

    def to_wav(self) -> bytes:
        return f"chunk-{self.chunk_index}".encode("ascii")


class _RecordingWriter:
    def __init__(self):
        self.written: list[list[str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def write_segments(self, segments):
        self.written.append([segment.text for segment in segments])


def _make_args():
    return SimpleNamespace(
        output="transcript.txt",
        format="txt",
        chunk_size=15,
        device=None,
        model="gpt-4o-transcribe",
        diarize=False,
    )


def test_start_keeps_capturing_while_transcribing(monkeypatch):
    args = _make_args()
    first_transcription_started = threading.Event()
    second_chunk_captured = threading.Event()
    release_first_transcription = threading.Event()
    writer = _RecordingWriter()

    class FakeRecorder:
        def __init__(self, **kwargs):
            self.stopped = False

        def stop(self):
            self.stopped = True

        def record(self):
            yield _FakeChunk(0)
            assert first_transcription_started.wait(timeout=1)
            second_chunk_captured.set()
            yield _FakeChunk(1)

    def fake_transcribe(wav_bytes, *, api_key, model, chunk_offset):
        chunk_name = wav_bytes.decode("ascii")
        if chunk_name == "chunk-0":
            first_transcription_started.set()
            assert second_chunk_captured.wait(timeout=1)
            release_first_transcription.set()
        else:
            assert release_first_transcription.is_set()

        return [
            TranscriptSegment(
                speaker="Speaker",
                text=chunk_name,
                start=chunk_offset,
                end=chunk_offset + 1,
            )
        ]

    fake_audio_module = SimpleNamespace(
        AudioRecorder=FakeRecorder,
        _IS_MACOS=False,
        _sck_available=False,
    )

    monkeypatch.setattr(cli, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(cli, "_load_audio_module", lambda: fake_audio_module)
    monkeypatch.setattr(cli, "get_writer", lambda fmt, path: writer)
    monkeypatch.setattr(cli, "transcribe_chunk", fake_transcribe)
    monkeypatch.setattr(cli.signal, "signal", lambda *args, **kwargs: None)

    cli._cmd_start(args)

    assert second_chunk_captured.is_set()
    assert writer.written == [["chunk-0"], ["chunk-1"]]


def test_start_exits_when_capture_thread_fails(monkeypatch):
    args = _make_args()
    writer = _RecordingWriter()

    class FakeRecorder:
        def __init__(self, **kwargs):
            self.stopped = False

        def stop(self):
            self.stopped = True

        def record(self):
            yield _FakeChunk(0)
            raise RuntimeError("capture failed")

    def fake_transcribe(wav_bytes, *, api_key, model, chunk_offset):
        return [
            TranscriptSegment(
                speaker="Speaker",
                text=wav_bytes.decode("ascii"),
                start=chunk_offset,
                end=chunk_offset + 1,
            )
        ]

    fake_audio_module = SimpleNamespace(
        AudioRecorder=FakeRecorder,
        _IS_MACOS=False,
        _sck_available=False,
    )

    monkeypatch.setattr(cli, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(cli, "_load_audio_module", lambda: fake_audio_module)
    monkeypatch.setattr(cli, "get_writer", lambda fmt, path: writer)
    monkeypatch.setattr(cli, "transcribe_chunk", fake_transcribe)
    monkeypatch.setattr(cli.signal, "signal", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit):
        cli._cmd_start(args)

    assert writer.written == [["chunk-0"]]


def test_start_drains_buffered_chunks_after_stop(monkeypatch):
    args = _make_args()
    first_transcription_started = threading.Event()
    second_chunk_captured = threading.Event()
    writer = _RecordingWriter()
    handlers = {}

    class FakeRecorder:
        def __init__(self, **kwargs):
            self.stop_event = threading.Event()

        def stop(self):
            self.stop_event.set()

        def record(self):
            yield _FakeChunk(0)
            assert first_transcription_started.wait(timeout=1)
            second_chunk_captured.set()
            yield _FakeChunk(1)
            assert self.stop_event.wait(timeout=1)

    def fake_transcribe(wav_bytes, *, api_key, model, chunk_offset):
        chunk_name = wav_bytes.decode("ascii")
        if chunk_name == "chunk-0":
            first_transcription_started.set()
            assert second_chunk_captured.wait(timeout=1)
            handlers[cli.signal.SIGINT](cli.signal.SIGINT, None)

        return [
            TranscriptSegment(
                speaker="Speaker",
                text=chunk_name,
                start=chunk_offset,
                end=chunk_offset + 1,
            )
        ]

    fake_audio_module = SimpleNamespace(
        AudioRecorder=FakeRecorder,
        _IS_MACOS=False,
        _sck_available=False,
    )

    monkeypatch.setattr(cli, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(cli, "_load_audio_module", lambda: fake_audio_module)
    monkeypatch.setattr(cli, "get_writer", lambda fmt, path: writer)
    monkeypatch.setattr(cli, "transcribe_chunk", fake_transcribe)
    monkeypatch.setattr(
        cli.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )

    cli._cmd_start(args)

    assert writer.written == [["chunk-0"], ["chunk-1"]]


def test_start_polls_while_waiting_for_first_chunk(monkeypatch):
    args = _make_args()
    writer = _RecordingWriter()
    release_first_chunk = threading.Event()
    recorder_created = threading.Event()

    class FakeRecorder:
        def __init__(self, **kwargs):
            recorder_created.set()

        def stop(self):
            return None

        def record(self):
            assert release_first_chunk.wait(timeout=1)
            yield _FakeChunk(0)

    def fake_transcribe(wav_bytes, *, api_key, model, chunk_offset):
        return [
            TranscriptSegment(
                speaker="Speaker",
                text=wav_bytes.decode("ascii"),
                start=chunk_offset,
                end=chunk_offset + 1,
            )
        ]

    fake_audio_module = SimpleNamespace(
        AudioRecorder=FakeRecorder,
        _IS_MACOS=False,
        _sck_available=False,
    )

    monkeypatch.setattr(cli, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(cli, "_load_audio_module", lambda: fake_audio_module)
    monkeypatch.setattr(cli, "get_writer", lambda fmt, path: writer)
    monkeypatch.setattr(cli, "transcribe_chunk", fake_transcribe)
    monkeypatch.setattr(cli.signal, "signal", lambda *args, **kwargs: None)

    thread = threading.Thread(target=cli._cmd_start, args=(args,))
    thread.start()

    assert recorder_created.wait(timeout=1)
    time.sleep(cli._CAPTURE_POLL_INTERVAL * 2)
    assert thread.is_alive() is True
    release_first_chunk.set()
    thread.join(timeout=2)

    assert thread.is_alive() is False
    assert writer.written == [["chunk-0"]]