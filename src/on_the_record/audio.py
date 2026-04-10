"""Audio capture from system loopback devices.

Handles cross-platform loopback audio capture using the ``soundcard`` library,
silence detection, chunking, and WAV encoding of audio buffers.
"""

from __future__ import annotations

import io
import logging
import platform
import struct
import sys
import threading
import wave
from dataclasses import dataclass, field
from typing import Generator

import numpy as np

try:
    import soundcard as sc
except Exception as exc:  # pragma: no cover – hardware-dependent
    sc = None  # type: ignore[assignment]
    _sc_import_error = exc
else:
    _sc_import_error = None

logger = logging.getLogger("on_the_record.audio")


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


@dataclass
class AudioDevice:
    """Simplified representation of an audio device."""

    name: str
    id: str
    is_loopback: bool


def list_devices() -> list[AudioDevice]:
    """Return all available audio capture devices, including loopbacks.

    Raises ``RuntimeError`` if soundcard could not be imported (usually means
    no audio subsystem is available).
    """
    _ensure_soundcard()
    devices: list[AudioDevice] = []

    # Regular microphones
    for mic in sc.all_microphones(include_loopback=False):
        devices.append(AudioDevice(name=mic.name, id=mic.id, is_loopback=False))

    # Loopback / monitor sources
    for mic in sc.all_microphones(include_loopback=True):
        # soundcard returns *all* mics when include_loopback=True on some
        # platforms, so filter to only the ones that are actually loopbacks.
        if mic.isloopback:
            devices.append(AudioDevice(name=mic.name, id=mic.id, is_loopback=True))

    return devices


def _get_loopback(device_name: str | None = None):
    """Return a soundcard microphone object for the requested loopback device.

    If *device_name* is ``None``, the default speaker's loopback is used.
    """
    _ensure_soundcard()

    if device_name is not None:
        # Try to find a loopback whose name contains the requested string.
        for mic in sc.all_microphones(include_loopback=True):
            if mic.isloopback and device_name.lower() in mic.name.lower():
                return mic
        # Fallback: try exact id match.
        for mic in sc.all_microphones(include_loopback=True):
            if mic.isloopback and device_name == mic.id:
                return mic
        print(
            f"Error: no loopback device matching '{device_name}' found.\n"
            "Run 'on-the-record list-devices' to see available devices.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Default: loopback for the default speaker.
    default_speaker = sc.default_speaker()
    if default_speaker is None:
        _no_loopback_error()

    loopback = sc.get_microphone(default_speaker.name, include_loopback=True)
    if loopback is None:
        _no_loopback_error()
    return loopback


def _ensure_soundcard() -> None:
    if sc is None:
        raise RuntimeError(
            f"Could not import soundcard: {_sc_import_error}\n"
            "Make sure you have a working audio subsystem installed."
        ) from _sc_import_error


def _no_loopback_error() -> None:
    os_name = platform.system()
    hint = ""
    if os_name == "Darwin":
        hint = (
            "\nOn macOS you need a virtual audio loopback driver.\n"
            "Install BlackHole: https://existential.audio/blackhole/\n"
            "Then create a Multi-Output Device in Audio MIDI Setup that\n"
            "includes both your speakers and BlackHole."
        )
    elif os_name == "Linux":
        hint = (
            "\nOn Linux, make sure PulseAudio or PipeWire is running.\n"
            "Monitor sources should be available automatically."
        )
    print(
        f"Error: could not find a loopback audio device.{hint}",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------


def is_silent(audio: np.ndarray, threshold: float) -> bool:
    """Return ``True`` if the RMS energy of *audio* is below *threshold*.

    *audio* should be a 1-D float array with values in [-1, 1].
    """
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return rms < threshold


# ---------------------------------------------------------------------------
# WAV encoding
# ---------------------------------------------------------------------------


def encode_wav(audio: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """Encode a float32 numpy array as 16-bit PCM WAV bytes.

    *audio* values are expected in the range [-1.0, 1.0].
    """
    # Ensure mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # float -> int16
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Audio capture / chunking
# ---------------------------------------------------------------------------


@dataclass
class AudioChunk:
    """A chunk of captured audio."""

    audio: np.ndarray
    sample_rate: int
    channels: int
    chunk_index: int
    start_time_offset: float  # seconds since recording started

    def to_wav(self) -> bytes:
        """Encode this chunk as WAV bytes."""
        return encode_wav(self.audio, self.sample_rate, self.channels)


@dataclass
class AudioRecorder:
    """Records audio from a loopback device and yields chunks.

    Usage::

        recorder = AudioRecorder(sample_rate=16000, chunk_seconds=15)
        for chunk in recorder.record():
            # chunk is an AudioChunk
            wav_bytes = chunk.to_wav()
    """

    sample_rate: int = 16_000
    channels: int = 1
    chunk_seconds: int = 15
    device_name: str | None = None
    silence_threshold: float = 0.003

    # Internal state
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )

    def stop(self) -> None:
        """Signal the recorder to stop after the current chunk."""
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def record(self) -> Generator[AudioChunk, None, None]:
        """Yield ``AudioChunk`` objects until :meth:`stop` is called.

        Each chunk contains ``chunk_seconds`` of audio. Silent chunks are
        detected but still yielded (with a log message) so the caller can
        decide what to do.
        """
        loopback = _get_loopback(self.device_name)
        num_frames = self.sample_rate * self.chunk_seconds
        chunk_index = 0

        logger.info(
            "Recording from '%s' at %d Hz, %d-second chunks",
            loopback.name,
            self.sample_rate,
            self.chunk_seconds,
        )

        with loopback.recorder(
            samplerate=self.sample_rate, channels=self.channels
        ) as rec:
            while not self._stop_event.is_set():
                try:
                    data = rec.record(numframes=num_frames)
                except Exception:
                    if self._stop_event.is_set():
                        break
                    raise

                # soundcard returns float64 arrays; flatten to 1-D
                audio = data.flatten().astype(np.float32)

                silent = is_silent(audio, self.silence_threshold)
                if silent:
                    logger.debug("Chunk %d is silent — skipping", chunk_index)
                    chunk_index += 1
                    continue

                offset = chunk_index * self.chunk_seconds
                logger.info("Chunk %d captured (%.1f s offset)", chunk_index, offset)

                yield AudioChunk(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    chunk_index=chunk_index,
                    start_time_offset=offset,
                )
                chunk_index += 1

        logger.info("Recording stopped.")
