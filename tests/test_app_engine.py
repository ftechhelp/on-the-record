"""Tests for the JSON-lines app engine."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

from on_the_record.app_engine import JsonLineEngine, build_config, build_study_options
from on_the_record.recording import RecordingResult


@dataclass
class _FakeDevice:
    name: str
    id: str
    is_loopback: bool


def _events(output: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_engine_handles_ping():
    output = io.StringIO()
    engine = JsonLineEngine(input_stream=io.StringIO(), output_stream=output)

    assert engine.handle_line('{"id": "1", "command": "ping"}') is True

    assert _events(output) == [{"event": "pong", "request_id": "1"}]


def test_engine_lists_devices():
    output = io.StringIO()
    fake_audio = type(
        "FakeAudioModule",
        (),
        {
            "list_devices": staticmethod(
                lambda: [_FakeDevice("System Audio", "screencapturekit", True)]
            )
        },
    )
    engine = JsonLineEngine(
        input_stream=io.StringIO(),
        output_stream=output,
        audio_module_loader=lambda: fake_audio,
    )

    engine.handle_line('{"id": "devices", "command": "list_devices"}')

    assert _events(output) == [
        {
            "event": "devices",
            "request_id": "devices",
            "devices": [
                {
                    "name": "System Audio",
                    "id": "screencapturekit",
                    "is_loopback": True,
                }
            ],
        }
    ]


def test_engine_starts_recording_session():
    output = io.StringIO()
    created = {}

    class FakeSession:
        def __init__(self, config, *, event_callback):
            created["config"] = config
            self.event_callback = event_callback

        def run(self):
            self.event_callback("recording_started", {"output_path": "out.txt"})
            return RecordingResult(
                elapsed_seconds=1.0,
                total_segments=2,
                output_path="out.txt",
            )

        def request_stop(self):
            return None

    engine = JsonLineEngine(
        input_stream=io.StringIO(),
        output_stream=output,
        session_factory=FakeSession,
    )

    engine.handle_line(
        json.dumps(
            {
                "id": "start",
                "command": "start_recording",
                "payload": {"api_key": "test-key", "output_path": "out.txt"},
            }
        )
    )
    assert engine._session_thread is not None
    engine._session_thread.join(timeout=1)

    events = _events(output)
    assert created["config"].api_key == "test-key"
    assert events[0] == {"event": "start_accepted", "request_id": "start"}
    assert events[1] == {"event": "recording_started", "output_path": "out.txt"}
    assert events[2] == {
        "event": "recording_finished",
        "request_id": "start",
        "result": {
            "elapsed_seconds": 1.0,
            "total_segments": 2,
            "output_path": "out.txt",
        },
    }


def test_engine_generates_study_document_when_enabled():
    output = io.StringIO()
    calls = []

    class FakeSession:
        def __init__(self, config, *, event_callback):
            self.event_callback = event_callback

        def run(self):
            return RecordingResult(
                elapsed_seconds=1.0,
                total_segments=2,
                output_path="out.txt",
            )

        def request_stop(self):
            return None

    def fake_study_writer(transcript_path, output_path, *, api_key, model):
        calls.append((transcript_path, output_path, api_key, model))
        return Path("study.md")

    engine = JsonLineEngine(
        input_stream=io.StringIO(),
        output_stream=output,
        session_factory=FakeSession,
        study_document_writer=fake_study_writer,
    )

    engine.handle_line(
        json.dumps(
            {
                "id": "start",
                "command": "start_recording",
                "payload": {
                    "api_key": "test-key",
                    "output_path": "out.txt",
                    "study_doc_enabled": True,
                    "gemini_api_key": "gemini-key",
                    "gemini_model": "gemini-test",
                },
            }
        )
    )
    assert engine._session_thread is not None
    engine._session_thread.join(timeout=1)

    events = _events(output)
    assert calls == [("out.txt", None, "gemini-key", "gemini-test")]
    assert events[1] == {
        "event": "study_doc_started",
        "request_id": "start",
        "transcript_path": "out.txt",
        "model": "gemini-test",
    }
    assert events[2] == {
        "event": "study_doc_written",
        "request_id": "start",
        "output_path": "study.md",
    }


def test_engine_skips_enabled_study_document_without_gemini_key():
    output = io.StringIO()

    class FakeSession:
        def __init__(self, config, *, event_callback):
            return None

        def run(self):
            return RecordingResult(
                elapsed_seconds=1.0,
                total_segments=1,
                output_path="out.txt",
            )

        def request_stop(self):
            return None

    engine = JsonLineEngine(
        input_stream=io.StringIO(),
        output_stream=output,
        session_factory=FakeSession,
    )

    engine.handle_line(
        json.dumps(
            {
                "id": "start",
                "command": "start_recording",
                "payload": {
                    "api_key": "test-key",
                    "output_path": "out.txt",
                    "study_doc_enabled": True,
                    "gemini_api_key": " ",
                },
            }
        )
    )
    assert engine._session_thread is not None
    engine._session_thread.join(timeout=1)

    events = _events(output)
    assert events[1] == {
        "event": "study_doc_skipped",
        "request_id": "start",
        "reason": "missing_gemini_api_key",
    }


def test_build_config_uses_app_payload_values():
    config = build_config(
        {
            "api_key": "test-key",
            "output_path": "meeting",
            "format": "md",
            "chunk_seconds": 30,
            "include_system_audio": False,
            "include_microphone": True,
            "diarize": True,
        }
    )

    assert config.output_path == "meeting.md"
    assert config.output_format == "md"
    assert config.chunk_seconds == 30
    assert config.include_system_audio is False
    assert config.include_microphone is True
    assert config.model == "gpt-4o-transcribe-diarize"


def test_build_study_options_uses_app_payload_values():
    options = build_study_options(
        {
            "study_doc_enabled": True,
            "gemini_api_key": "gemini-key",
            "gemini_model": "gemini-test",
            "study_output_path": "study.md",
        }
    )

    assert options == {
        "enabled": True,
        "api_key": "gemini-key",
        "model": "gemini-test",
        "output_path": "study.md",
    }
