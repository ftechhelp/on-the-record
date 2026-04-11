# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**on-the-record** is a cross-platform CLI tool that captures system audio (loopback), transcribes it with optional speaker diarization using OpenAI's API, and writes results in txt/md/json formats. It supports macOS 13+ natively via ScreenCaptureKit, older macOS via BlackHole, Linux via PulseAudio/PipeWire, and Windows via WASAPI.

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
uv run pytest -v                     # verbose
```

No lint or type-check commands are configured in pyproject.toml.

## Architecture

```
cli.py              CLI entry point (start, list-devices, test-audio subcommands)
config.py           Config dataclass with defaults (sample_rate=16kHz, chunk_seconds=15, silence_threshold=0.003)
audio.py            AudioRecorder — yields AudioChunk objects; silence detection; WAV encoding
macos_audio.py      macOS 13+ ScreenCaptureKit backend (PyObjC); raw CMSampleBuffer extraction
transcribe.py       OpenAI API integration; parses diarized/verbose responses; exponential backoff retry
writer.py           Abstract TranscriptWriter; TxtWriter, MdWriter, JsonWriter implementations
```

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
- macOS requires Screen Recording permission for ScreenCaptureKit
