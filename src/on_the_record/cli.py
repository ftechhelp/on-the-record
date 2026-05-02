"""Command-line interface for on-the-record.

Subcommands
-----------
start          Begin recording and transcribing system audio.
list-devices   Show available audio devices.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import signal
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from on_the_record import __version__
from on_the_record.config import (
    Config,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_FORMAT,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SILENCE_THRESHOLD,
    load_api_key,
)
from on_the_record.obsidian import (
    ObsidianConfig,
    clear_obsidian_config,
    config_file_path,
    export_study_document_to_obsidian,
    load_obsidian_config,
    save_obsidian_config,
)
from on_the_record.recording import CAPTURE_POLL_INTERVAL, RecordingSession
from on_the_record.study import (
    DEFAULT_GEMINI_MODEL,
    load_gemini_api_key,
    write_named_study_document,
)
from on_the_record.transcribe import transcribe_chunk
from on_the_record.writer import SUPPORTED_FORMATS, get_writer

logger = logging.getLogger("on_the_record")

_CAPTURE_POLL_INTERVAL = CAPTURE_POLL_INTERVAL


def _resolve_audio_sources(args: argparse.Namespace) -> tuple[bool, bool]:
    include_system_audio = not getattr(args, "microphone_only", False)
    include_microphone = not getattr(args, "no_microphone", False)

    if not include_system_audio and not include_microphone:
        print(
            "Error: --microphone-only and --no-microphone cannot be used together.",
            file=sys.stderr,
        )
        sys.exit(1)

    return include_system_audio, include_microphone


def _format_audio_sources(
    *,
    include_system_audio: bool,
    include_microphone: bool,
    system_name: str | None = None,
    microphone_name: str | None = None,
) -> str:
    sources = []
    if include_system_audio:
        sources.append(system_name or "system audio")
    if include_microphone:
        sources.append(microphone_name or "microphone")
    return " + ".join(sources)


def _maybe_generate_study_document(
    transcript_path: str,
    *,
    enabled: bool,
    output_path: str | None,
    model: str,
    total_segments: int,
    obsidian_enabled: bool | None = None,
    obsidian_vault: str | None = None,
    obsidian_folder: str | None = None,
    obsidian_cli_command: str | None = None,
) -> None:
    """Generate a Markdown study document after recording, if configured."""
    if not enabled:
        logger.info("Study document generation disabled.")
        return
    if total_segments == 0:
        logger.info("Skipping study document generation because no transcript segments were written.")
        return

    gemini_api_key = load_gemini_api_key()
    if gemini_api_key is None:
        logger.warning(
            "Skipping study document generation because GEMINI_API_KEY is not set."
        )
        return

    study_path = Path(output_path) if output_path else None
    if study_path:
        logger.info("Generating Gemini study document: %s", study_path)
    else:
        logger.info("Generating Gemini study document with a Gemini title.")

    try:
        written_path = write_named_study_document(
            transcript_path,
            study_path,
            api_key=gemini_api_key,
            model=model,
        )
    except Exception as exc:
        logger.error("Study document generation failed: %s", exc)
        return

    logger.info("Study document written to %s", written_path)
    _maybe_export_study_document_to_obsidian(
        written_path,
        enabled=obsidian_enabled,
        vault_path=obsidian_vault,
        study_folder=obsidian_folder,
        cli_command=obsidian_cli_command,
    )


def _maybe_export_study_document_to_obsidian(
    study_path: str | Path,
    *,
    enabled: bool | None,
    vault_path: str | None,
    study_folder: str | None,
    cli_command: str | None,
) -> None:
    if enabled is False:
        logger.info("Obsidian export disabled.")
        return

    config = _resolve_obsidian_config(
        vault_path=vault_path,
        study_folder=study_folder,
        cli_command=cli_command,
    )

    if config is None:
        if enabled is True:
            logger.warning(
                "Skipping Obsidian export because no Obsidian vault is configured."
            )
        return

    try:
        exported_path = export_study_document_to_obsidian(study_path, config)
    except Exception as exc:
        logger.error("Obsidian export failed: %s", exc)
        return

    logger.info("Study document exported to Obsidian: %s", exported_path)


def _resolve_obsidian_config(
    *,
    vault_path: str | None,
    study_folder: str | None,
    cli_command: str | None,
) -> ObsidianConfig | None:
    saved_config = load_obsidian_config()
    if vault_path is None and study_folder is None and cli_command is None:
        return saved_config

    resolved_vault = Path(vault_path).expanduser() if vault_path else None
    if resolved_vault is None:
        if saved_config is None:
            return None
        resolved_vault = saved_config.vault_path

    return ObsidianConfig(
        vault_path=resolved_vault,
        study_folder=(
            study_folder
            if study_folder is not None
            else saved_config.study_folder if saved_config else ""
        ),
        cli_command=(
            cli_command
            if cli_command is not None
            else saved_config.cli_command if saved_config else None
        ),
        run_cli_after_export=(
            saved_config.run_cli_after_export if saved_config else True
        ),
    )


def _load_audio_module():
    """Import audio backends only when a command actually needs them."""
    return importlib.import_module("on_the_record.audio")


def _log_recording_event(event_type: str, payload: dict) -> None:
    if event_type == "transcription_started":
        logger.info(
            "Transcribing chunk %d (offset %.1f s) …",
            payload["chunk_index"],
            payload["offset"],
        )
    elif event_type == "transcription_failed":
        logger.error(
            "Transcription failed for chunk %d: %s",
            payload["chunk_index"],
            payload["error"],
        )
    elif event_type == "segments_written":
        for segment in payload["segments"]:
            logger.info("  %s: %s", segment["speaker"], segment["text"][:80])
    elif event_type == "no_speech_detected":
        logger.info("  (no speech detected)")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    """Configure verbose logging to stderr."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("on_the_record")
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)


# ---------------------------------------------------------------------------
# ``start`` command
# ---------------------------------------------------------------------------


def _cmd_start(args: argparse.Namespace) -> None:
    """Main recording + transcription loop."""
    include_system_audio, include_microphone = _resolve_audio_sources(args)
    api_key = load_api_key()
    audio_module = _load_audio_module()

    # Resolve output path
    output_path = args.output
    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./transcript_{ts}.{args.format}"

    # Ensure the extension matches the format
    path = Path(output_path)
    if path.suffix.lstrip(".") != args.format:
        path = path.with_suffix(f".{args.format}")

    model = args.model
    # If using the diarize model, force the model name
    if args.diarize and "diarize" not in model:
        model = "gpt-4o-transcribe-diarize"

    config = Config(
        api_key=api_key,
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=1,
        chunk_seconds=args.chunk_size,
        device_name=args.device,
        include_system_audio=include_system_audio,
        include_microphone=include_microphone,
        silence_threshold=DEFAULT_SILENCE_THRESHOLD,
        output_path=str(path),
        output_format=args.format,
        model=model,
    )

    logger.info("on-the-record v%s", __version__)
    logger.info("Model:       %s", config.model)
    logger.info("Output:      %s", config.output_path)
    logger.info("Format:      %s", config.output_format)
    logger.info("Chunk size:  %d s", config.chunk_seconds)
    logger.info(
        "Sources:     %s",
        _format_audio_sources(
            include_system_audio=config.include_system_audio,
            include_microphone=config.include_microphone,
        ),
    )
    if config.device_name:
        logger.info("Device:      %s", config.device_name)
    if audio_module._IS_MACOS and config.include_system_audio:
        if audio_module._sck_available:
            logger.info(
                "Using ScreenCaptureKit for native system audio capture.\n"
                "           You may be prompted to grant Screen Recording permission."
            )
        else:
            logger.info(
                "Tip: Make sure your system output is set to a Multi-Output Device\n"
                "           that includes both your speakers and BlackHole.\n"
                "           Otherwise you'll only capture microphone input, not speaker audio."
            )
    logger.info("Press Ctrl+C to stop recording.\n")

    session = RecordingSession(
        config,
        audio_module=audio_module,
        writer_factory=get_writer,
        transcribe=transcribe_chunk,
        event_callback=_log_recording_event,
        poll_interval=_CAPTURE_POLL_INTERVAL,
    )

    # Graceful shutdown on Ctrl+C / SIGTERM
    def _shutdown(signum, frame):
        logger.info("Received signal %d — stopping …", signum)
        session.request_stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        result = session.run()
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        sys.exit(1)

    logger.info(
        "Done. Recorded %.0f s, wrote %d segment(s) to %s",
        result.elapsed_seconds,
        result.total_segments,
        result.output_path,
    )
    _maybe_generate_study_document(
        config.output_path,
        enabled=getattr(args, "study_doc", True),
        output_path=getattr(args, "study_output", None),
        model=getattr(args, "gemini_model", DEFAULT_GEMINI_MODEL),
        total_segments=result.total_segments,
        obsidian_enabled=getattr(args, "obsidian", None),
        obsidian_vault=getattr(args, "obsidian_vault", None),
        obsidian_folder=getattr(args, "obsidian_folder", None),
        obsidian_cli_command=getattr(args, "obsidian_cli_command", None),
    )


# ---------------------------------------------------------------------------
# ``config obsidian`` command
# ---------------------------------------------------------------------------


def _cmd_config_obsidian(args: argparse.Namespace) -> None:
    """Show, save, or clear persistent Obsidian export settings."""
    if args.clear:
        changed = clear_obsidian_config()
        if changed:
            print(f"Cleared Obsidian config from {config_file_path()}")
        else:
            print("No Obsidian config was set.")
        return

    if args.show:
        _print_obsidian_config(load_obsidian_config())
        return

    saved_config = load_obsidian_config()
    has_updates = any(
        value is not None
        for value in (args.vault, args.folder, args.cli_command)
    ) or args.clear_cli_command
    if not has_updates:
        _print_obsidian_config(saved_config)
        return

    vault_path = Path(args.vault).expanduser() if args.vault else None
    if vault_path is None:
        if saved_config is None:
            print("Error: --vault is required for the first Obsidian config.", file=sys.stderr)
            sys.exit(1)
        vault_path = saved_config.vault_path

    if args.clear_cli_command:
        cli_command = None
    elif args.cli_command is not None:
        cli_command = args.cli_command.strip() or None
    elif saved_config is not None:
        cli_command = saved_config.cli_command
    else:
        cli_command = None

    config = ObsidianConfig(
        vault_path=vault_path,
        study_folder=(
            args.folder
            if args.folder is not None
            else saved_config.study_folder if saved_config else ""
        ),
        cli_command=cli_command,
        run_cli_after_export=True,
    )
    path = save_obsidian_config(config)
    print(f"Saved Obsidian config to {path}")
    _print_obsidian_config(config)


def _print_obsidian_config(config: ObsidianConfig | None) -> None:
    if config is None:
        print(f"No Obsidian config set. Config path: {config_file_path()}")
        return

    print("Obsidian config:")
    print(f"  Vault:        {config.vault_path}")
    print(f"  Study folder: {config.study_folder or '.'}")
    print(f"  CLI command:  {config.cli_command or '(none)'}")
    print(f"  Config path:  {config_file_path()}")


# ---------------------------------------------------------------------------
# ``list-devices`` command
# ---------------------------------------------------------------------------


def _cmd_list_devices(args: argparse.Namespace) -> None:
    """Print available audio devices."""
    audio_module = _load_audio_module()
    devices = audio_module.list_devices()
    if not devices:
        print("No audio devices found.", file=sys.stderr)
        sys.exit(1)

    print(f"{'Type':<14} {'Name'}")
    print(f"{'----':<14} {'----'}")
    for dev in devices:
        if dev.id == "screencapturekit":
            kind = "system"
        elif dev.is_loopback:
            kind = "loopback" if not audio_module._IS_MACOS else "virtual"
        else:
            kind = "input"
        print(f"{kind:<14} {dev.name}")


# ---------------------------------------------------------------------------
# ``test-audio`` command
# ---------------------------------------------------------------------------


def _cmd_test_audio(args: argparse.Namespace) -> None:
    """Record a few seconds from a device and report audio levels.

    This helps diagnose whether audio is actually flowing through the
    selected device — no API key required.
    """
    import numpy as np

    audio_module = _load_audio_module()
    device_name = args.device
    duration = args.seconds
    sample_rate = DEFAULT_SAMPLE_RATE
    include_system_audio, include_microphone = _resolve_audio_sources(args)

    # Use ScreenCaptureKit if available and no explicit non-SCK device chosen.
    use_sck = include_system_audio and audio_module._sck_available and (
        device_name is None or device_name == "screencapturekit"
    )

    if use_sck:
        from on_the_record.macos_audio import SystemAudioRecorder

        microphone = (
            audio_module._get_microphone_device()
            if include_microphone
            else None
        )
        source_description = _format_audio_sources(
            include_system_audio=True,
            include_microphone=include_microphone,
            microphone_name=microphone.name if microphone is not None else None,
        )
        print(f"Recording {duration}s from {source_description} …\n")
        sck = SystemAudioRecorder(sample_rate=sample_rate, channels=1)
        with ExitStack() as stack:
            sck = stack.enter_context(sck)
            microphone_recorder = None
            if microphone is not None:
                microphone_recorder = stack.enter_context(
                    microphone.recorder(samplerate=sample_rate, channels=1)
                )
            readers = {"system": lambda: sck.read_chunk(sample_rate * duration)}
            if microphone_recorder is not None:
                readers["microphone"] = lambda: microphone_recorder.record(
                    numframes=sample_rate * duration
                )
            recordings = audio_module._record_sources_concurrently(readers)
        audio = audio_module._mix_audio_sources(*recordings.values())
    else:
        mic = (
            audio_module._get_capture_device(device_name)
            if include_system_audio
            else None
        )
        microphone = (
            audio_module._get_microphone_device(exclude_device=mic)
            if include_microphone
            else None
        )
        source_description = _format_audio_sources(
            include_system_audio=include_system_audio,
            include_microphone=include_microphone,
            system_name=mic.name if mic is not None else None,
            microphone_name=microphone.name if microphone is not None else None,
        )
        print(f"Recording {duration}s from {source_description} …\n")

        with ExitStack() as stack:
            system_recorder = None
            microphone_recorder = None
            if mic is not None:
                system_recorder = stack.enter_context(
                    mic.recorder(samplerate=sample_rate, channels=1)
                )
            if microphone is not None:
                microphone_recorder = stack.enter_context(
                    microphone.recorder(samplerate=sample_rate, channels=1)
                )
            readers = {}
            if system_recorder is not None:
                readers["system"] = lambda: system_recorder.record(
                    numframes=sample_rate * duration
                )
            if microphone_recorder is not None:
                readers["microphone"] = lambda: microphone_recorder.record(
                    numframes=sample_rate * duration
                )
            recordings = audio_module._record_sources_concurrently(readers)

        audio = audio_module._mix_audio_sources(*recordings.values())

    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))

    print(f"  Samples:   {audio.size}")
    print(f"  Peak:      {peak:.6f}")
    print(f"  RMS:       {rms:.6f}")
    print(f"  Threshold: {DEFAULT_SILENCE_THRESHOLD:.6f}")
    print()

    if peak < 0.000001:
        print(
            "RESULT: No audio detected at all — the device is receiving pure silence."
        )
        print()
        if audio_module._IS_MACOS and not use_sck:
            print("This usually means one of:")
            print("  1. Your system output is NOT set to the Multi-Output Device")
            print("     Fix: SwitchAudioSource -s 'Multi-Output Device'")
            print("  2. The Multi-Output Device doesn't include BlackHole")
            print("     Fix: Open Audio MIDI Setup and check the device config")
            print("  3. Nothing is actually playing on your system right now")
        elif audio_module._IS_MACOS and use_sck:
            print("This usually means nothing is playing on your system right now.")
            print("Try playing some audio and run the test again.")
    elif rms < DEFAULT_SILENCE_THRESHOLD:
        print(
            f"RESULT: Audio detected but very quiet (RMS {rms:.6f} < threshold {DEFAULT_SILENCE_THRESHOLD})."
        )
        print("The silence detection would skip this. Try playing louder audio,")
        print("or lower the threshold with a future --silence-threshold flag.")
    else:
        print(
            f"RESULT: Audio is flowing! RMS {rms:.6f} is above threshold {DEFAULT_SILENCE_THRESHOLD}."
        )
        print("You're good to go — run 'on-the-record start' to begin transcribing.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="on-the-record",
        description="Capture system audio and transcribe it with speaker diarization.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command")

    # -- start ---------------------------------------------------------------
    start = sub.add_parser("start", help="Start recording and transcribing.")
    start.add_argument(
        "-o",
        "--output",
        default="",
        help="Output file path. Defaults to ./transcript_<timestamp>.<format>",
    )
    start.add_argument(
        "-f",
        "--format",
        choices=SUPPORTED_FORMATS,
        default=DEFAULT_FORMAT,
        help=f"Output format (default: {DEFAULT_FORMAT}).",
    )
    start.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Audio chunk size in seconds (default: {DEFAULT_CHUNK_SECONDS}).",
    )
    start.add_argument(
        "-d",
        "--device",
        default=None,
        help="Audio device name (substring match). Use 'list-devices' to see options.",
    )
    start.add_argument(
        "--no-microphone",
        action="store_true",
        help="Capture system audio only; do not record your microphone.",
    )
    start.add_argument(
        "--microphone-only",
        "--no-system-audio",
        action="store_true",
        help="Capture microphone only; do not record system/output audio.",
    )
    start.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL}).",
    )
    start.add_argument(
        "--diarize",
        action="store_true",
        default=True,
        help="Enable speaker diarization (default: enabled).",
    )
    start.add_argument(
        "--no-diarize",
        action="store_false",
        dest="diarize",
        help="Disable speaker diarization.",
    )
    start.add_argument(
        "--study-doc",
        action="store_true",
        default=True,
        help="Generate a Gemini Markdown study document after recording (default: enabled when GEMINI_API_KEY is set).",
    )
    start.add_argument(
        "--no-study-doc",
        action="store_false",
        dest="study_doc",
        help="Disable Gemini study document generation.",
    )
    start.add_argument(
        "--study-output",
        default=None,
        help="Study document output path. Defaults to a Gemini-titled Markdown file.",
    )
    start.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model for study document generation (default: {DEFAULT_GEMINI_MODEL}).",
    )
    start.add_argument(
        "--obsidian",
        action="store_true",
        default=None,
        help="Export the Gemini study document to the configured Obsidian vault.",
    )
    start.add_argument(
        "--no-obsidian",
        action="store_false",
        dest="obsidian",
        help="Disable Obsidian export for this run.",
    )
    start.add_argument(
        "--obsidian-vault",
        default=None,
        help="Obsidian vault path for this run, overriding saved config.",
    )
    start.add_argument(
        "--obsidian-folder",
        default=None,
        help="Vault-relative folder for this run's study document.",
    )
    start.add_argument(
        "--obsidian-cli-command",
        default=None,
        help="External command to run after Obsidian export. {file} and {vault} are supported.",
    )
    start.set_defaults(func=_cmd_start)

    # -- config --------------------------------------------------------------
    config_parser = sub.add_parser("config", help="Manage persistent settings.")
    config_sub = config_parser.add_subparsers(dest="config_command")

    obsidian = config_sub.add_parser(
        "obsidian",
        help="Configure Obsidian study-document export.",
    )
    obsidian.add_argument(
        "--vault",
        default=None,
        help="Path to your Obsidian vault.",
    )
    obsidian.add_argument(
        "--folder",
        default=None,
        help="Vault-relative folder for OTR study documents.",
    )
    obsidian.add_argument(
        "--cli-command",
        default=None,
        help="Optional command to run after export. {file} and {vault} are supported.",
    )
    obsidian.add_argument(
        "--clear-cli-command",
        action="store_true",
        help="Remove the configured Obsidian CLI hook.",
    )
    obsidian.add_argument(
        "--show",
        action="store_true",
        help="Show the current Obsidian config.",
    )
    obsidian.add_argument(
        "--clear",
        action="store_true",
        help="Clear the Obsidian config.",
    )
    obsidian.set_defaults(func=_cmd_config_obsidian)

    # -- list-devices --------------------------------------------------------
    ld = sub.add_parser("list-devices", help="List available audio devices.")
    ld.set_defaults(func=_cmd_list_devices)

    # -- test-audio ----------------------------------------------------------
    ta = sub.add_parser(
        "test-audio",
        help="Record a few seconds and report audio levels (no API key needed).",
    )
    ta.add_argument(
        "-d",
        "--device",
        default=None,
        help="Audio device name (substring match).",
    )
    ta.add_argument(
        "-s",
        "--seconds",
        type=int,
        default=5,
        help="Seconds to record (default: 5).",
    )
    ta.add_argument(
        "--no-microphone",
        action="store_true",
        help="Test system audio only; do not record your microphone.",
    )
    ta.add_argument(
        "--microphone-only",
        "--no-system-audio",
        action="store_true",
        help="Test microphone only; do not record system/output audio.",
    )
    ta.set_defaults(func=_cmd_test_audio)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command or not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
