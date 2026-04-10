"""Command-line interface for on-the-record.

Subcommands
-----------
start          Begin recording and transcribing system audio.
list-devices   Show available audio devices.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from on_the_record import __version__
from on_the_record.audio import AudioRecorder, list_devices
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
    logger.info("Press Ctrl+C to stop recording.\n")

    recorder = AudioRecorder(
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
            for chunk in recorder.record():
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
    devices = list_devices()
    if not devices:
        print("No audio devices found.", file=sys.stderr)
        sys.exit(1)

    print(f"{'Type':<12} {'Name'}")
    print(f"{'----':<12} {'----'}")
    for dev in devices:
        kind = "loopback" if dev.is_loopback else "input"
        print(f"{kind:<12} {dev.name}")


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
