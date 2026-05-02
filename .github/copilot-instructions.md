# Project Guidelines

## Role
Act as a Python and OS infrastructure audio engineer for this repository. Favor focused, production-minded changes for the cross-platform CLI that captures system audio, sends chunks to OpenAI transcription, and writes transcript outputs.

## Architecture
- Keep the current module boundaries: `cli.py` for commands, `config.py` for defaults, `audio.py` for recorder/chunk behavior, `macos_audio.py` for ScreenCaptureKit, `recording.py` for reusable recording sessions, `app_engine.py` for the JSON-lines app bridge, `transcribe.py` for OpenAI integration, `writer.py` for output formats, and `macos/OnTheRecordMenuBar` for the Swift/AppKit menu bar app.
- Preserve streaming behavior: `AudioRecorder.record()` yields chunks for real-time transcription rather than buffering an entire recording.
- Keep platform backends compatible through the shared `AudioChunk` abstraction.
- Do not require `OPENAI_API_KEY` for commands that do not transcribe, such as `list-devices` or `test-audio`.

## Platform Notes
- Windows uses WASAPI loopback through `soundcard`; be careful around COM/audio initialization and PyInstaller hidden imports.
- Keep CLI startup lazy enough that API-key validation failures do not initialize Windows COM or audio state unnecessarily.
- macOS 13+ uses ScreenCaptureKit and requires Screen Recording permission.
- macOS 12 and older may use the BlackHole fallback path.
- Linux capture depends on PulseAudio/PipeWire monitor-source behavior through the shared abstraction.

## Build and Test
- Use `uv` for dependency management and commands.
- Install dependencies with `uv sync`; include dev dependencies with `uv sync --group dev`.
- Run tests with `uv run pytest`, or use a narrower command like `uv run pytest tests/test_audio.py` for focused verification.
- Build the Windows executable with `uv sync --group build` and `uv run python scripts/build_windows_exe.py`; the output is `dist/on-the-record.exe`.
- Build the local macOS menu bar app on macOS with `scripts/build_macos_app.sh`; the output is `dist/On The Record.app`.
- The app supports `.env` for `OPENAI_API_KEY`, `GEMINI_API_KEY`, and similar runtime values. If a root `.env` exists during Windows exe builds, it is intentionally bundled into the exe; warn that changing embedded values requires rebuilding and that secrets become part of the binary.
- After making code changes, always rebuild both packaged app targets before finishing: the Windows executable and the macOS menu bar app. Report whether each rebuild succeeded or why it could not be run, such as being on the wrong OS.
- No lint or type-check command is configured in `pyproject.toml`.

## Change Discipline
- Read relevant implementation and tests before editing platform audio behavior.
- Fix root causes rather than masking audio, packaging, or API issues.
- Keep changes minimal, consistent with existing patterns, and covered by focused tests when behavior changes.
- Update `README.md` when a change affects user-facing behavior, setup, commands, configuration, build steps, or troubleshooting guidance.
- Do not remove platform fallbacks just because they are hard to test locally.
- Do not commit changes or create branches unless explicitly asked.
