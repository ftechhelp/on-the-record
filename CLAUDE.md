# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**on-the-record** is a cross-platform CLI tool that captures system audio (loopback), transcribes it with optional OpenAI speaker diarization, and writes results in txt/md/json formats. It supports macOS 13+ natively via ScreenCaptureKit, older macOS via BlackHole, Linux via PulseAudio/PipeWire, and Windows via WASAPI.

## Commands

```bash
# Install dependencies (use uv)
uv sync
uv sync --group dev   # include dev dependencies (pytest, pytest-mock)

# Run the CLI
uv run on-the-record start
uv run on-the-record list-devices
uv run on-the-record test-audio

# Run tests
uv run pytest
uv run pytest tests/test_audio.py    # single file
uv run pytest tests/test_audio.py::test_name   # single test
uv run pytest -v                     # verbose

# Package (run on the target OS — PyInstaller does not cross-compile)
uv sync --group build
uv run python scripts/build_windows_exe.py    # headless CLI -> dist/on-the-record.exe
uv run python scripts/build_windows_tray.py   # tray app    -> dist/On The Record.exe
powershell -File scripts\build_windows_installer.ps1   # Inno Setup (auto-finds ISCC.exe)
scripts/build_macos_app.sh                     # macOS menu bar -> dist/On The Record.app
```

No lint or type-check commands are configured in pyproject.toml.

## Entry points (pyproject `[project.scripts]`)

| Command | Module | Purpose |
|---------|--------|---------|
| `on-the-record` | `cli:main` | User-facing CLI (`start`, `list-devices`, `test-audio`, `config obsidian`) |
| `on-the-record-engine` | `app_engine:main` | JSON-lines bridge for the macOS Swift app (stdin commands → stdout events) |
| `on-the-record-tray` | `tray:main` | Windows system tray app |

## Architecture

```
cli.py              CLI entry point (start, list-devices, test-audio subcommands)
config.py           Config dataclass with defaults (sample_rate=16kHz, chunk_seconds=15, silence_threshold=0.003)
audio.py            AudioRecorder — yields AudioChunk objects; silence detection; WAV encoding
macos_audio.py      macOS 13+ ScreenCaptureKit backend (PyObjC); raw CMSampleBuffer extraction
transcribe.py       OpenAI API integration; parses diarized/verbose responses; exponential backoff retry
writer.py           Abstract TranscriptWriter; TxtWriter, MdWriter, JsonWriter implementations
recording.py        RecordingSession — CLI-independent recording/transcription loop; emits events
app_engine.py       JSON-lines engine (build_config/build_study_options) for the native shells
tray.py             Windows system tray app (pystray + tkinter); runs RecordingSession in-process
```

**Native shells**: macOS uses a Swift menu bar app (`macos/OnTheRecordMenuBar/`) that launches `app_engine.py` as a subprocess over JSON-lines. Windows uses `tray.py` (entry point `on-the-record-tray`), which reuses `build_config`/`build_study_options` and runs `RecordingSession` in-process — no subprocess/IPC. API keys: macOS Keychain vs Windows Credential Manager (`keyring`). Both build to `dist/On The Record.{app,exe}` via `scripts/build_macos_app.sh` / `scripts/build_windows_tray.py`; the Windows installer is `windows/installer/on-the-record.iss` (Inno Setup).

**Data flow**: system audio → 15s chunks at 16 kHz → silence check → WAV bytes → OpenAI API → diarized segments → appended to output file.

**Platform abstraction**: `AudioRecorder` uses `macos_audio.py` on macOS 13+ and falls back to the `soundcard` library everywhere else. Both backends yield the same `AudioChunk` type.

**Writer pattern**: `get_writer(format)` returns the appropriate `TranscriptWriter` subclass. Writers use `__enter__`/`__exit__` for format-specific finalization (e.g., JsonWriter closes the JSON array on exit).

**Streaming**: `AudioRecorder.record()` is a generator — the main loop in `cli.py` iterates chunks for real-time transcription without buffering the full recording.

## Key Constants (config.py)

| Setting | Default | Notes |
|---------|---------|-------|
| `sample_rate` | 16,000 Hz | Required by OpenAI Whisper |
| `chunk_seconds` | 15 | Configurable via `--chunk-size` |
| `silence_threshold` | 0.003 RMS | Skips near-silent chunks to reduce API cost |
| `model` | `gpt-4o-transcribe` | Switches to `gpt-4o-transcribe-diarize` when `--diarize` |

## Environment

- `OPENAI_API_KEY` must be set for transcription (not needed for `list-devices` or `test-audio`)
- `GEMINI_API_KEY` enables post-recording Gemini study-document generation (optional)
- Keys can also come from a `.env` file (`config.load_dotenv`). On Windows builds, a root `.env` present at build time is **bundled into the exe** and takes precedence at runtime — rebuild after changing embedded values, and treat the binary as containing those secrets.
- macOS requires Screen Recording permission for ScreenCaptureKit

## Conventions

- **Lazy startup**: keep CLI/engine startup lazy enough that an API-key validation failure does not initialize Windows COM or audio state. `recording.load_audio_module()` imports the audio backend only when recording actually begins.
- **Platform fallbacks**: do not remove platform backends because they are hard to test locally. Windows uses WASAPI loopback via `soundcard` (watch COM init and PyInstaller hidden imports); macOS 13+ uses ScreenCaptureKit, macOS 12 and older use the BlackHole path, Linux uses PulseAudio/PipeWire monitor sources — all behind the shared `AudioChunk` abstraction.
- **After code changes**, rebuild the affected packaged targets (Windows CLI exe, Windows tray exe, macOS app) and report whether each succeeded or why it could not run (e.g. wrong OS, `iscc` missing). Update `README.md` when a change affects user-facing behavior, setup, commands, config, or build steps.
- Do not commit changes or create branches unless explicitly asked.
