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
import queue
import signal
import sys
import threading
import time
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
from on_the_record.transcribe import transcribe_chunk
from on_the_record.writer import SUPPORTED_FORMATS, get_writer

logger = logging.getLogger("on_the_record")

_CAPTURE_COMPLETE = object()
_CAPTURE_POLL_INTERVAL = 0.1


def _queued_chunks(recorder):
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
                item = chunk_queue.get(timeout=_CAPTURE_POLL_INTERVAL)
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


def _load_audio_module():
    """Import audio backends only when a command actually needs them."""
    return importlib.import_module("on_the_record.audio")


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
    if config.device_name:
        logger.info("Device:      %s", config.device_name)
    if audio_module._IS_MACOS:
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

    recorder = audio_module.AudioRecorder(
        sample_rate=config.sample_rate,
        channels=config.channels,
        chunk_seconds=config.chunk_seconds,
        device_name=config.device_name,
        silence_threshold=config.silence_threshold,
    )

    # Graceful shutdown on Ctrl+C / SIGTERM
    def _shutdown(signum, frame):
        logger.info("Received signal %d — stopping …", signum)
        recorder.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    writer = get_writer(config.output_format, config.output_path)
    total_segments = 0
    start_time = time.monotonic()

    try:
        with writer:
            for chunk in _queued_chunks(recorder):
                logger.info(
                    "Transcribing chunk %d (offset %.1f s) …",
                    chunk.chunk_index,
                    chunk.start_time_offset,
                )
                wav_bytes = chunk.to_wav()

                try:
                    segments = transcribe_chunk(
                        wav_bytes,
                        api_key=config.api_key,
                        model=config.model,
                        chunk_offset=chunk.start_time_offset,
                    )
                except Exception as exc:
                    logger.error(
                        "Transcription failed for chunk %d: %s", chunk.chunk_index, exc
                    )
                    continue

                if segments:
                    writer.write_segments(segments)
                    total_segments += len(segments)
                    for seg in segments:
                        logger.info("  %s: %s", seg.speaker, seg.text[:80])
                else:
                    logger.info("  (no speech detected)")
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        sys.exit(1)

    elapsed = time.monotonic() - start_time
    logger.info(
        "Done. Recorded %.0f s, wrote %d segment(s) to %s",
        elapsed,
        total_segments,
        config.output_path,
    )


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

    # Use ScreenCaptureKit if available and no explicit non-SCK device chosen.
    use_sck = audio_module._sck_available and (
        device_name is None or device_name == "screencapturekit"
    )

    if use_sck:
        from on_the_record.macos_audio import SystemAudioRecorder

        microphone = audio_module._get_microphone_device()
        print(
            f"Recording {duration}s via ScreenCaptureKit plus microphone '{microphone.name}' …\n"
        )
        sck = SystemAudioRecorder(sample_rate=sample_rate, channels=1)
        with sck, microphone.recorder(
            samplerate=sample_rate, channels=1
        ) as microphone_recorder:
            recordings = audio_module._record_sources_concurrently(
                {
                    "system": lambda: sck.read_chunk(sample_rate * duration),
                    "microphone": lambda: microphone_recorder.record(
                        numframes=sample_rate * duration
                    ),
                }
            )
        audio = audio_module._mix_audio_sources(
            recordings["system"],
            recordings["microphone"],
        )
    else:
        mic = audio_module._get_capture_device(device_name)
        microphone = audio_module._get_microphone_device(exclude_device=mic)
        print(
            f"Recording {duration}s from '{mic.name}' plus microphone '{microphone.name}' …\n"
        )

        with mic.recorder(
            samplerate=sample_rate, channels=1
        ) as system_recorder, microphone.recorder(
            samplerate=sample_rate, channels=1
        ) as microphone_recorder:
            recordings = audio_module._record_sources_concurrently(
                {
                    "system": lambda: system_recorder.record(
                        numframes=sample_rate * duration
                    ),
                    "microphone": lambda: microphone_recorder.record(
                        numframes=sample_rate * duration
                    ),
                }
            )

        audio = audio_module._mix_audio_sources(
            recordings["system"],
            recordings["microphone"],
        )

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
    start.set_defaults(func=_cmd_start)

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
    ta.set_defaults(func=_cmd_test_audio)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
