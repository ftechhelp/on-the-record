"""JSON-lines engine for a native macOS shell.

The future Swift app can launch this module as a bundled Python process,
send commands on stdin, and receive structured events on stdout.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from on_the_record.config import (
    Config,
    DEFAULT_CHANNELS,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_FORMAT,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SILENCE_THRESHOLD,
    load_dotenv,
)
from on_the_record.recording import RecordingSession


class JsonLineEngine:
    """Handle JSON command messages and emit JSON event messages."""

    def __init__(
        self,
        *,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
        session_factory: Callable[..., RecordingSession] = RecordingSession,
        audio_module_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.session_factory = session_factory
        self.audio_module_loader = audio_module_loader or self._load_audio_module
        self._session: RecordingSession | None = None
        self._session_thread: threading.Thread | None = None
        self._write_lock = threading.Lock()

    def run(self) -> None:
        """Run the command loop until stdin closes or a shutdown command arrives."""
        self.emit("ready")
        for line in self.input_stream:
            if not self.handle_line(line):
                break

    def handle_line(self, line: str) -> bool:
        """Handle one command line. Return False when the loop should stop."""
        line = line.strip()
        if not line:
            return True

        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            self.emit("error", error=f"Invalid JSON: {exc}")
            return True

        command = message.get("command")
        payload = message.get("payload") or {}
        request_id = message.get("id")

        try:
            if command == "ping":
                self.emit("pong", request_id=request_id)
            elif command == "list_devices":
                self._handle_list_devices(request_id=request_id)
            elif command == "start_recording":
                self._handle_start_recording(payload, request_id=request_id)
            elif command == "stop_recording":
                self._handle_stop_recording(request_id=request_id)
            elif command == "shutdown":
                self._handle_stop_recording(request_id=request_id)
                self.emit("shutdown", request_id=request_id)
                return False
            else:
                self.emit(
                    "error",
                    request_id=request_id,
                    error=f"Unknown command: {command}",
                )
        except Exception as exc:
            self.emit("error", request_id=request_id, error=str(exc))

        return True

    def emit(self, event: str, **payload: Any) -> None:
        message = {"event": event, **_json_safe(payload)}
        with self._write_lock:
            self.output_stream.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.output_stream.flush()

    def _handle_list_devices(self, *, request_id: Any = None) -> None:
        audio_module = self.audio_module_loader()
        devices = [asdict(device) for device in audio_module.list_devices()]
        self.emit("devices", request_id=request_id, devices=devices)

    def _handle_start_recording(
        self,
        payload: dict[str, Any],
        *,
        request_id: Any = None,
    ) -> None:
        if self._session_thread is not None and self._session_thread.is_alive():
            self.emit(
                "error",
                request_id=request_id,
                error="Recording is already running.",
            )
            return

        config = build_config(payload)
        session = self.session_factory(
            config,
            event_callback=self._forward_recording_event,
        )
        self._session = session
        self._session_thread = threading.Thread(
            target=self._run_session,
            args=(session, request_id),
            name="app-recording-session",
            daemon=True,
        )
        self.emit("start_accepted", request_id=request_id)
        self._session_thread.start()

    def _handle_stop_recording(self, *, request_id: Any = None) -> None:
        if self._session is None:
            self.emit("stop_ignored", request_id=request_id, reason="not_recording")
            return
        self._session.request_stop()
        self.emit("stop_requested", request_id=request_id)

    def _run_session(self, session: RecordingSession, request_id: Any) -> None:
        try:
            result = session.run()
        except Exception as exc:
            self.emit("recording_error", request_id=request_id, error=str(exc))
        else:
            self.emit(
                "recording_finished",
                request_id=request_id,
                result=asdict(result),
            )
        finally:
            if self._session is session:
                self._session = None

    def _forward_recording_event(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.emit(event_type, **payload)

    @staticmethod
    def _load_audio_module():
        from on_the_record import audio

        return audio


def build_config(payload: dict[str, Any]) -> Config:
    """Build a :class:`Config` from a JSON command payload."""
    load_dotenv()
    api_key = str(
        payload.get("api_key") or os.environ.get("OPENAI_API_KEY") or ""
    ).strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to start recording.")

    output_format = str(payload.get("format") or DEFAULT_FORMAT)
    output_path = str(payload.get("output_path") or _default_output_path(output_format))
    path = Path(output_path)
    if path.suffix.lstrip(".") != output_format:
        path = path.with_suffix(f".{output_format}")

    model = str(payload.get("model") or DEFAULT_MODEL)
    if bool(payload.get("diarize", True)) and "diarize" not in model:
        model = "gpt-4o-transcribe-diarize"

    return Config(
        api_key=api_key,
        sample_rate=int(payload.get("sample_rate") or DEFAULT_SAMPLE_RATE),
        channels=int(payload.get("channels") or DEFAULT_CHANNELS),
        chunk_seconds=int(payload.get("chunk_seconds") or DEFAULT_CHUNK_SECONDS),
        device_name=payload.get("device_name"),
        include_system_audio=bool(payload.get("include_system_audio", True)),
        include_microphone=bool(payload.get("include_microphone", True)),
        silence_threshold=float(
            payload.get("silence_threshold") or DEFAULT_SILENCE_THRESHOLD
        ),
        output_path=str(path),
        output_format=output_format,
        model=model,
    )


def _default_output_path(output_format: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"./transcript_{timestamp}.{output_format}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    JsonLineEngine().run()


if __name__ == "__main__":
    main()
