"""Tests for CLI startup behavior."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
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
        self.speakers: list[list[str]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def write_segments(self, segments):
        self.written.append([segment.text for segment in segments])
        self.speakers.append([segment.speaker for segment in segments])


def _make_args():
    return SimpleNamespace(
        output="transcript.txt",
        format="txt",
        chunk_size=15,
        device=None,
        model="gpt-4o-transcribe",
        diarize=False,
        study_doc=True,
        study_output=None,
        gemini_model="gemini-2.5-flash",
        recognize_speakers=False,
        enroll_speakers=False,
        speaker_threshold=0.78,
        speaker_profiles=None,
        speaker_save_samples=False,
    )


class _AudioFakeChunk(_FakeChunk):
    def __init__(self, chunk_index: int):
        super().__init__(chunk_index)
        self.audio = np.ones(15 * 16_000, dtype=np.float32)
        self.sample_rate = 16_000
        self.channels = 1


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


def test_start_generates_study_document_after_recording(monkeypatch):
    args = _make_args()
    writer = _RecordingWriter()
    generated = {}

    class FakeRecorder:
        def __init__(self, **kwargs):
            return None

        def stop(self):
            return None

        def record(self):
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
    monkeypatch.setattr(cli, "load_gemini_api_key", lambda: "gemini-key")
    monkeypatch.setattr(cli.signal, "signal", lambda *args, **kwargs: None)

    def fake_write_study_document(transcript_path, output_path, *, api_key, model):
        generated["transcript_path"] = transcript_path
        generated["output_path"] = str(output_path)
        generated["api_key"] = api_key
        generated["model"] = model
        return output_path

    monkeypatch.setattr(cli, "write_study_document", fake_write_study_document)

    cli._cmd_start(args)

    assert generated == {
        "transcript_path": "transcript.txt",
        "output_path": "transcript_study.md",
        "api_key": "gemini-key",
        "model": "gemini-2.5-flash",
    }


def test_start_skips_study_document_without_gemini_key(monkeypatch):
    args = _make_args()
    writer = _RecordingWriter()

    class FakeRecorder:
        def __init__(self, **kwargs):
            return None

        def stop(self):
            return None

        def record(self):
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
    monkeypatch.setattr(cli, "load_gemini_api_key", lambda: None)
    monkeypatch.setattr(cli.signal, "signal", lambda *args, **kwargs: None)

    def fail_write_study_document(*args, **kwargs):
        raise AssertionError("study document should be skipped without Gemini key")

    monkeypatch.setattr(cli, "write_study_document", fail_write_study_document)

    cli._cmd_start(args)

    assert writer.written == [["chunk-0"]]


def test_start_honors_no_study_doc(monkeypatch):
    args = _make_args()
    args.study_doc = False
    writer = _RecordingWriter()

    class FakeRecorder:
        def __init__(self, **kwargs):
            return None

        def stop(self):
            return None

        def record(self):
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
    monkeypatch.setattr(cli, "load_gemini_api_key", lambda: "gemini-key")
    monkeypatch.setattr(cli.signal, "signal", lambda *args, **kwargs: None)

    def fail_write_study_document(*args, **kwargs):
        raise AssertionError("study document should be disabled")

    monkeypatch.setattr(cli, "write_study_document", fail_write_study_document)

    cli._cmd_start(args)

    assert writer.written == [["chunk-0"]]


def test_start_parser_enables_speaker_features_by_default():
    parser = cli._build_parser()

    args = parser.parse_args(["start"])

    assert args.recognize_speakers is None
    assert args.enroll_speakers is None


def test_start_parser_can_disable_speaker_features():
    parser = cli._build_parser()

    args = parser.parse_args(["start", "--no-recognize-speakers", "--no-enroll-speakers"])

    assert args.recognize_speakers is False
    assert args.enroll_speakers is False


def test_default_speaker_session_skips_missing_optional_dependencies(monkeypatch):
    class FakeUnavailable(RuntimeError):
        pass

    fake_speaker_module = SimpleNamespace(
        SpeakerRecognitionUnavailable=FakeUnavailable,
        SpeakerProfileStore=lambda path: SimpleNamespace(load=lambda: None),
        create_speaker_backend=lambda: (_ for _ in ()).throw(FakeUnavailable("missing")),
        DEFAULT_SPEAKER_THRESHOLD=0.78,
    )

    monkeypatch.setattr(cli, "_load_speaker_recognition_module", lambda: fake_speaker_module)

    session = cli._make_speaker_session(
        SimpleNamespace(
            recognize_speakers=None,
            enroll_speakers=None,
            speaker_profiles=None,
            speaker_threshold=0.78,
            speaker_save_samples=False,
        )
    )

    assert session is None


def test_explicit_speaker_session_errors_when_dependencies_missing(monkeypatch):
    class FakeUnavailable(RuntimeError):
        pass

    fake_speaker_module = SimpleNamespace(
        SpeakerRecognitionUnavailable=FakeUnavailable,
        SpeakerProfileStore=lambda path: SimpleNamespace(load=lambda: None),
        create_speaker_backend=lambda: (_ for _ in ()).throw(FakeUnavailable("missing")),
        DEFAULT_SPEAKER_THRESHOLD=0.78,
    )

    monkeypatch.setattr(cli, "_load_speaker_recognition_module", lambda: fake_speaker_module)

    with pytest.raises(SystemExit):
        cli._make_speaker_session(
            SimpleNamespace(
                recognize_speakers=True,
                enroll_speakers=None,
                speaker_profiles=None,
                speaker_threshold=0.78,
                speaker_save_samples=False,
            )
        )


def test_start_applies_known_speaker_names_while_streaming(monkeypatch):
    args = _make_args()
    args.recognize_speakers = True
    writer = _RecordingWriter()

    class FakeRecorder:
        def __init__(self, **kwargs):
            return None

        def stop(self):
            return None

        def record(self):
            yield _AudioFakeChunk(0)

    class FakeSpeakerSession:
        def resolve_segment(self, **kwargs):
            assert kwargs["speaker_label"] == "Speaker 1"
            assert kwargs["audio"].size == 16_000
            return "Alice"

    def fake_transcribe(wav_bytes, *, api_key, model, chunk_offset):
        return [
            TranscriptSegment(
                speaker="Speaker 1",
                text="hello",
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
    monkeypatch.setattr(cli, "_make_speaker_session", lambda args: FakeSpeakerSession())
    monkeypatch.setattr(cli, "get_writer", lambda fmt, path: writer)
    monkeypatch.setattr(cli, "transcribe_chunk", fake_transcribe)
    monkeypatch.setattr(cli.signal, "signal", lambda *args, **kwargs: None)

    cli._cmd_start(args)

    assert writer.speakers == [["Alice"]]


def test_start_enrolls_unknown_speakers_and_rewrites_before_study(monkeypatch):
    args = _make_args()
    args.enroll_speakers = True
    writer = _RecordingWriter()
    order = []

    class FakeRecorder:
        def __init__(self, **kwargs):
            return None

        def stop(self):
            return None

        def record(self):
            yield _AudioFakeChunk(0)

    class FakeSpeakerSession:
        def resolve_segment(self, **kwargs):
            return "Speaker 1"

        def prompt_for_unknowns(self, *, output=None):
            order.append("prompt")
            return {"Speaker 1": "Bob"}

    def fake_transcribe(wav_bytes, *, api_key, model, chunk_offset):
        return [
            TranscriptSegment(
                speaker="Speaker 1",
                text="hello",
                start=chunk_offset,
                end=chunk_offset + 1,
            )
        ]

    def fake_rewrite_segments(fmt, path, segments):
        order.append("rewrite")
        assert segments[0].speaker == "Bob"

    def fake_write_study_document(transcript_path, output_path, *, api_key, model):
        order.append("study")
        return output_path

    fake_audio_module = SimpleNamespace(
        AudioRecorder=FakeRecorder,
        _IS_MACOS=False,
        _sck_available=False,
    )

    monkeypatch.setattr(cli, "load_api_key", lambda: "test-key")
    monkeypatch.setattr(cli, "_load_audio_module", lambda: fake_audio_module)
    monkeypatch.setattr(cli, "_make_speaker_session", lambda args: FakeSpeakerSession())
    monkeypatch.setattr(cli, "get_writer", lambda fmt, path: writer)
    monkeypatch.setattr(cli, "rewrite_segments", fake_rewrite_segments)
    monkeypatch.setattr(cli, "transcribe_chunk", fake_transcribe)
    monkeypatch.setattr(cli, "load_gemini_api_key", lambda: "gemini-key")
    monkeypatch.setattr(cli, "write_study_document", fake_write_study_document)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.signal, "signal", lambda *args, **kwargs: None)

    cli._cmd_start(args)

    assert writer.speakers == [["Speaker 1"]]
    assert order == ["prompt", "rewrite", "study"]


def test_speakers_test_backend_reports_embedding_dimensions(monkeypatch, capsys):
    class FakeBackend:
        def embed(self, audio, sample_rate):
            assert audio.shape == (16_000,)
            assert sample_rate == 16_000
            return [0.1, 0.2, 0.3]

    class FakeUnavailable(RuntimeError):
        pass

    fake_speaker_module = SimpleNamespace(
        SpeakerRecognitionUnavailable=FakeUnavailable,
        create_speaker_backend=lambda: FakeBackend(),
    )

    monkeypatch.setattr(cli, "_load_speaker_recognition_module", lambda: fake_speaker_module)

    cli._cmd_speakers_test_backend(SimpleNamespace())

    assert capsys.readouterr().out == "Speaker backend OK: embedding dimensions=3\n"


def test_speakers_test_backend_errors_when_unavailable(monkeypatch):
    class FakeUnavailable(RuntimeError):
        pass

    fake_speaker_module = SimpleNamespace(
        SpeakerRecognitionUnavailable=FakeUnavailable,
        create_speaker_backend=lambda: (_ for _ in ()).throw(FakeUnavailable("missing")),
    )

    monkeypatch.setattr(cli, "_load_speaker_recognition_module", lambda: fake_speaker_module)

    with pytest.raises(SystemExit):
        cli._cmd_speakers_test_backend(SimpleNamespace())
