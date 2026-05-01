"""Audio capture from system loopback devices.

Handles cross-platform audio capture using:

- **ScreenCaptureKit** (macOS 13+) — native system audio capture, no virtual
  device required.  Falls through to soundcard if unavailable.
- **soundcard** — cross-platform loopback via PulseAudio/PipeWire (Linux),
  WASAPI (Windows), or virtual audio devices like BlackHole (macOS < 13).
"""

from __future__ import annotations

import io
import logging
import platform
import struct
import sys
import threading
import time
import warnings
from contextlib import ExitStack
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generator

import numpy as np


def _patch_soundcard_windows_numpy_compat(mediafoundation_module=None) -> None:
    """Patch soundcard's Windows recorder to work with NumPy 2.

    soundcard's WASAPI backend still uses ``numpy.fromstring`` on a raw audio
    buffer. NumPy 2 removed binary-mode ``fromstring``, so we copy the buffer
    via ``frombuffer`` before the capture buffer is released.
    """
    if platform.system() != "Windows":
        return

    if mediafoundation_module is None:
        try:
            import soundcard.mediafoundation as mediafoundation_module
        except Exception:
            return

    recorder_type = getattr(mediafoundation_module, "_Recorder", None)
    if recorder_type is None:
        return

    original_record_chunk = getattr(recorder_type, "_record_chunk", None)
    if original_record_chunk is None or getattr(
        original_record_chunk, "_on_the_record_numpy2_compat", False
    ):
        return

    def _record_chunk(self):
        while self._capture_available_frames() == 0:
            if self._idle_start_time is None:
                self._idle_start_time = time.perf_counter_ns()

            default_block_length, minimum_block_length = self.deviceperiod
            time.sleep(minimum_block_length / 4)
            elapsed_time_ns = time.perf_counter_ns() - self._idle_start_time
            if elapsed_time_ns / 1_000_000_000 > default_block_length * 4:
                num_frames = int(self.samplerate * elapsed_time_ns / 1_000_000_000)
                num_channels = len(set(self.channelmap))
                self._idle_start_time += elapsed_time_ns
                return np.zeros([num_frames * num_channels], dtype=np.float32)

        self._idle_start_time = None
        data_ptr, nframes, flags = self._capture_buffer()
        if data_ptr == mediafoundation_module._ffi.NULL:
            raise RuntimeError("Could not create capture buffer")

        buffer = mediafoundation_module._ffi.buffer(
            data_ptr, nframes * 4 * len(set(self.channelmap))
        )
        chunk = np.frombuffer(buffer, dtype=np.float32).copy()

        if flags & mediafoundation_module._ole32.AUDCLNT_BUFFERFLAGS_SILENT:
            chunk[:] = 0
        if self._is_first_frame:
            flags &= ~mediafoundation_module._ole32.AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY
            self._is_first_frame = False
        if flags & mediafoundation_module._ole32.AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY:
            mediafoundation_module.warnings.warn(
                "data discontinuity in recording",
                mediafoundation_module.SoundcardRuntimeWarning,
            )
        if nframes > 0:
            self._capture_release(nframes)
            return chunk
        return np.zeros([0], dtype=np.float32)

    _record_chunk._on_the_record_numpy2_compat = True
    recorder_type._record_chunk = _record_chunk

# Suppress soundcard's macOS loopback warning — we handle this ourselves.
warnings.filterwarnings("ignore", message=".*macOS does not support loopback.*")

try:
    import soundcard as sc
except Exception as exc:  # pragma: no cover – hardware-dependent
    sc = None  # type: ignore[assignment]
    _sc_import_error = exc
else:
    _sc_import_error = None
    _patch_soundcard_windows_numpy_compat()

# ScreenCaptureKit backend (macOS 13+)
_sck_available = False
try:
    from on_the_record.macos_audio import (
        is_available as _sck_is_available,
        import_error as _sck_import_error,
    )

    _sck_available = _sck_is_available()
    if _sck_available:
        from on_the_record.macos_audio import SystemAudioRecorder
except Exception as _sck_exc:  # pragma: no cover
    _sck_available = False

logger = logging.getLogger("on_the_record.audio")


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


# Well-known virtual audio device names used for loopback on macOS.
_VIRTUAL_DEVICE_KEYWORDS: tuple[str, ...] = (
    "blackhole",
    "soundflower",
    "loopback",
    "existential",
)

_IS_MACOS = platform.system() == "Darwin"


@dataclass
class AudioDevice:
    """Simplified representation of an audio device."""

    name: str
    id: str
    is_loopback: bool


def _is_virtual_device(name: str) -> bool:
    """Heuristic: return *True* if *name* looks like a virtual audio device."""
    lower = name.lower()
    return any(kw in lower for kw in _VIRTUAL_DEVICE_KEYWORDS)


def list_devices() -> list[AudioDevice]:
    """Return all available audio capture devices, including loopbacks.

    On macOS 13+ with ScreenCaptureKit available, a synthetic
    ``System Audio (ScreenCaptureKit)`` device is listed first — this
    captures all system audio natively without a virtual device.

    On macOS without ScreenCaptureKit, ``soundcard`` is used instead.
    Virtual audio devices (e.g. BlackHole) appear as regular inputs and
    are tagged as ``virtual`` so the user knows to select them.

    Raises ``RuntimeError`` if soundcard could not be imported (usually means
    no audio subsystem is available).
    """
    devices: list[AudioDevice] = []

    # On macOS 13+, ScreenCaptureKit provides native system audio capture.
    if _sck_available:
        devices.append(
            AudioDevice(
                name="System Audio (ScreenCaptureKit)",
                id="screencapturekit",
                is_loopback=True,
            )
        )

    _ensure_soundcard()
    seen_ids: set[str] = set()

    # On all platforms, gather regular microphones first.
    for mic in sc.all_microphones(include_loopback=False):
        if mic.id in seen_ids:
            continue
        seen_ids.add(mic.id)
        devices.append(
            AudioDevice(
                name=mic.name,
                id=mic.id,
                # On macOS, flag virtual devices as loopback-capable.
                is_loopback=_IS_MACOS and _is_virtual_device(mic.name),
            )
        )

    # Loopback / monitor sources (Linux & Windows).
    for mic in sc.all_microphones(include_loopback=True):
        if mic.id in seen_ids:
            continue
        seen_ids.add(mic.id)
        if mic.isloopback:
            devices.append(AudioDevice(name=mic.name, id=mic.id, is_loopback=True))

    return devices


def _get_capture_device(device_name: str | None = None):
    """Return a soundcard microphone object for audio capture.

    On Linux/Windows this prefers a true loopback device.  On macOS, where
    loopback is not natively supported, it falls back to matching a regular
    input device (e.g. BlackHole) by name.

    If *device_name* is ``None``, the default speaker's loopback is used
    (Linux/Windows) or a known virtual device is auto-detected (macOS).
    """
    _ensure_soundcard()

    if device_name is not None:
        return _find_device_by_name(device_name)

    # --- No explicit device: auto-detect ---

    if not _IS_MACOS:
        # Linux / Windows — try default speaker loopback.
        default_speaker = sc.default_speaker()
        if default_speaker is not None:
            loopback = sc.get_microphone(default_speaker.name, include_loopback=True)
            if loopback is not None:
                return loopback
        _no_loopback_error()

    # macOS — auto-detect a virtual audio device.
    for mic in sc.all_microphones(include_loopback=False):
        if _is_virtual_device(mic.name):
            logger.info("Auto-detected virtual device: '%s'", mic.name)
            return mic

    _no_loopback_error()


def _get_microphone_device(exclude_device=None):
    """Return the default physical microphone for local voice capture."""
    _ensure_soundcard()

    default_microphone = sc.default_microphone()
    if _is_usable_microphone(default_microphone, exclude_device):
        return default_microphone

    for microphone in sc.all_microphones(include_loopback=False):
        if _is_usable_microphone(microphone, exclude_device):
            return microphone

    raise RuntimeError(
        "Could not find a usable microphone to record your voice. "
        "Check your system input device and run 'on-the-record list-devices'."
    )


def _is_usable_microphone(microphone, exclude_device=None) -> bool:
    if microphone is None:
        return False
    if _same_audio_device(microphone, exclude_device):
        return False
    if getattr(microphone, "isloopback", False):
        return False
    if _IS_MACOS and _is_virtual_device(getattr(microphone, "name", "")):
        return False
    return True


def _same_audio_device(left, right) -> bool:
    if left is None or right is None:
        return False
    left_id = getattr(left, "id", None)
    right_id = getattr(right, "id", None)
    if left_id is not None and right_id is not None and left_id == right_id:
        return True
    return getattr(left, "name", None) == getattr(right, "name", None)


def _find_device_by_name(device_name: str):
    """Find a capture device whose name contains *device_name*.

    Searches loopback devices first (Linux/Windows), then falls back to
    regular inputs (required on macOS).
    """
    lower = device_name.lower()

    # 1. True loopback devices.
    for mic in sc.all_microphones(include_loopback=True):
        if mic.isloopback and lower in mic.name.lower():
            return mic

    # 2. Regular input devices (covers macOS virtual devices).
    for mic in sc.all_microphones(include_loopback=False):
        if lower in mic.name.lower():
            return mic

    # 3. Exact id match as last resort.
    for mic in sc.all_microphones(include_loopback=True):
        if device_name == mic.id:
            return mic

    print(
        f"Error: no audio device matching '{device_name}' found.\n"
        "Run 'on-the-record list-devices' to see available devices.",
        file=sys.stderr,
    )
    sys.exit(1)


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
        if _sck_available:
            hint = (
                "\nScreenCaptureKit is available but auto-detection failed.\n"
                "Try running without --device to use system audio capture."
            )
        else:
            hint = (
                "\nOn macOS you need a virtual audio loopback driver.\n"
                "Install BlackHole: https://existential.audio/blackhole/\n"
                "Then create a Multi-Output Device in Audio MIDI Setup that\n"
                "includes both your speakers and BlackHole.\n"
                "\nAlternatively, upgrade to macOS 13+ for native system audio\n"
                "capture via ScreenCaptureKit (no virtual device needed)."
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


def _as_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert recorded audio to a 1-D float32 mono array."""
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32).flatten()


def _mix_audio_sources(*sources: np.ndarray) -> np.ndarray:
    """Mix mono audio sources, padding shorter inputs with silence."""
    mono_sources = [_as_mono_float32(source) for source in sources if source.size > 0]
    if not mono_sources:
        return np.array([], dtype=np.float32)

    target_size = max(source.size for source in mono_sources)
    aligned_sources: list[np.ndarray] = []
    for source in mono_sources:
        if source.size < target_size:
            source = np.pad(source, (0, target_size - source.size))
        aligned_sources.append(source)

    mixed = np.sum(aligned_sources, axis=0)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def _record_sources_concurrently(
    readers: dict[str, Callable[[], np.ndarray]],
) -> dict[str, np.ndarray]:
    """Read multiple blocking audio sources at the same time."""
    results: dict[str, np.ndarray] = {}
    errors: dict[str, Exception] = {}

    def _read_source(source_name: str, reader: Callable[[], np.ndarray]) -> None:
        try:
            results[source_name] = reader()
        except Exception as exc:
            errors[source_name] = exc

    threads = [
        threading.Thread(
            target=_read_source,
            args=(source_name, reader),
            name=f"audio-{source_name}",
        )
        for source_name, reader in readers.items()
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        source_name, error = next(iter(errors.items()))
        raise RuntimeError(f"Audio capture failed for {source_name}: {error}") from error

    return results


def _source_description(
    *,
    system_name: str | None,
    microphone_name: str | None,
) -> str:
    source_names = []
    if system_name is not None:
        source_names.append(f"system audio '{system_name}'")
    if microphone_name is not None:
        source_names.append(f"microphone '{microphone_name}'")
    return " plus ".join(source_names)


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
    include_system_audio: bool = True
    include_microphone: bool = True
    silence_threshold: float = 0.003

    # Internal state
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.include_system_audio and not self.include_microphone:
            raise ValueError("At least one audio source must be enabled.")

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

        On macOS 13+, prefers ScreenCaptureKit for native system audio
        capture.  Falls back to soundcard if ScreenCaptureKit is
        unavailable or the user explicitly chose a soundcard device.
        """
        use_sck = self.include_system_audio and _sck_available and (
            self.device_name is None or self.device_name == "screencapturekit"
        )

        if use_sck:
            yield from self._record_screencapturekit()
        else:
            yield from self._record_soundcard()

    def _record_screencapturekit(self) -> Generator[AudioChunk, None, None]:
        """Record using ScreenCaptureKit (macOS 13+)."""
        num_samples = self.sample_rate * self.chunk_seconds
        chunk_index = 0

        microphone = _get_microphone_device() if self.include_microphone else None

        logger.info(
            "Recording %s at %d Hz, %d-second chunks",
            _source_description(
                system_name="ScreenCaptureKit",
                microphone_name=microphone.name if microphone is not None else None,
            ),
            self.sample_rate,
            self.chunk_seconds,
        )

        sck_recorder = SystemAudioRecorder(
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

        with ExitStack() as stack:
            sck_recorder = stack.enter_context(sck_recorder)
            microphone_recorder = None
            if microphone is not None:
                microphone_recorder = stack.enter_context(
                    microphone.recorder(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                    )
                )

            while not self._stop_event.is_set():
                readers: dict[str, Callable[[], np.ndarray]] = {
                    "system": lambda: sck_recorder.read_chunk(num_samples),
                }
                if microphone_recorder is not None:
                    readers["microphone"] = lambda: microphone_recorder.record(
                        numframes=num_samples
                    )

                try:
                    recordings = _record_sources_concurrently(readers)
                except Exception:
                    if self._stop_event.is_set():
                        break
                    raise

                audio = _mix_audio_sources(*recordings.values())

                rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                peak = float(np.max(np.abs(audio)))
                silent = rms < self.silence_threshold

                logger.info(
                    "Chunk %d — RMS: %.6f  Peak: %.6f  Threshold: %.6f  %s",
                    chunk_index,
                    rms,
                    peak,
                    self.silence_threshold,
                    "SILENT (skipping)" if silent else "OK",
                )

                if silent:
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

    def _record_soundcard(self) -> Generator[AudioChunk, None, None]:
        """Record using soundcard (cross-platform fallback)."""
        loopback = (
            _get_capture_device(self.device_name)
            if self.include_system_audio
            else None
        )
        microphone = (
            _get_microphone_device(exclude_device=loopback)
            if self.include_microphone
            else None
        )
        num_frames = self.sample_rate * self.chunk_seconds
        chunk_index = 0

        logger.info(
            "Recording %s at %d Hz, %d-second chunks",
            _source_description(
                system_name=loopback.name if loopback is not None else None,
                microphone_name=microphone.name if microphone is not None else None,
            ),
            self.sample_rate,
            self.chunk_seconds,
        )

        with ExitStack() as stack:
            system_recorder = None
            microphone_recorder = None
            if loopback is not None:
                system_recorder = stack.enter_context(
                    loopback.recorder(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                    )
                )
            if microphone is not None:
                microphone_recorder = stack.enter_context(
                    microphone.recorder(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                    )
                )

            while not self._stop_event.is_set():
                readers: dict[str, Callable[[], np.ndarray]] = {}
                if system_recorder is not None:
                    readers["system"] = lambda: system_recorder.record(
                        numframes=num_frames
                    )
                if microphone_recorder is not None:
                    readers["microphone"] = lambda: microphone_recorder.record(
                        numframes=num_frames
                    )

                try:
                    recordings = _record_sources_concurrently(readers)
                except Exception:
                    if self._stop_event.is_set():
                        break
                    raise

                audio = _mix_audio_sources(*recordings.values())

                rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                peak = float(np.max(np.abs(audio)))
                silent = rms < self.silence_threshold

                logger.info(
                    "Chunk %d — RMS: %.6f  Peak: %.6f  Threshold: %.6f  %s",
                    chunk_index,
                    rms,
                    peak,
                    self.silence_threshold,
                    "SILENT (skipping)" if silent else "OK",
                )

                if silent:
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
