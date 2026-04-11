"""macOS system audio capture using ScreenCaptureKit.

Available on macOS 13 (Ventura) and later.  ScreenCaptureKit captures
system audio natively — no virtual audio device (BlackHole) required.
The user still hears audio through their speakers normally.

The only permission required is **Screen Recording** (even though we only
capture audio, Apple bundles this under the same permission).

Implementation notes
--------------------
- On macOS 13, ScreenCaptureKit requires a minimal video stream alongside
  audio (2×2 px, 1 FPS).  On macOS 14+, ``capturesAudioOnly`` is available.
- PyObjC protocol methods are implemented by naming the Python method to
  match the Objective-C selector with colons replaced by underscores.
- ``CMBlockBufferGetDataPointer`` is NOT exposed by PyObjC; we use
  ``CMBlockBufferCopyDataBytes`` instead.
- The ``dispatch`` Python package was removed from ``pyobjc-core`` in newer
  releases; we create GCD queues directly via ctypes / libSystem instead.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from typing import Generator

import numpy as np

logger = logging.getLogger("on_the_record.macos_audio")

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_AVAILABLE = False
_IMPORT_ERROR: str | None = None


def _parse_macos_version() -> tuple[int, int]:
    """Return (major, minor) macOS version."""
    ver = platform.mac_ver()[0]
    parts = ver.split(".")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


def _create_serial_dispatch_queue(label: bytes):
    """Create a GCD serial dispatch queue without the ``dispatch`` package.

    Uses ctypes to call ``dispatch_queue_create`` directly from libSystem,
    then wraps the result in a PyObjC object so it can be passed to
    ScreenCaptureKit methods that expect a ``dispatch_queue_t``.
    """
    import ctypes
    import ctypes.util
    import objc

    lib = ctypes.CDLL(ctypes.util.find_library("System") or "libSystem.B.dylib")
    lib.dispatch_queue_create.restype = ctypes.c_void_p
    lib.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    ptr = lib.dispatch_queue_create(label, None)  # None → serial queue
    if not ptr:
        raise RuntimeError("dispatch_queue_create returned NULL")
    return objc.objc_object(c_void_p=ptr)


if platform.system() == "Darwin":
    try:
        major, minor = _parse_macos_version()
        if major < 13:
            _IMPORT_ERROR = f"macOS 13+ required, got {major}.{minor}"
        else:
            import objc
            import ScreenCaptureKit  # noqa: F401 — PyObjC framework
            import CoreMedia  # noqa: F401
            import Quartz  # noqa: F401
            from Foundation import NSObject, NSRunLoop, NSDate

            _AVAILABLE = True
    except ImportError as exc:
        _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = "Not macOS"


def is_available() -> bool:
    """Return True if ScreenCaptureKit can be used on this system."""
    return _AVAILABLE


def import_error() -> str | None:
    """Return the reason ScreenCaptureKit is unavailable, or None."""
    return _IMPORT_ERROR


# ---------------------------------------------------------------------------
# Audio sample extraction from CMSampleBuffer
# ---------------------------------------------------------------------------


def _extract_audio_samples(sample_buffer) -> np.ndarray | None:
    """Convert a CMSampleBuffer containing audio into a float32 numpy array.

    Returns None if the buffer cannot be read.
    """
    import CoreMedia

    block_buf = CoreMedia.CMSampleBufferGetDataBuffer(sample_buffer)
    if block_buf is None:
        return None

    length = CoreMedia.CMBlockBufferGetDataLength(block_buf)
    if length == 0:
        return None

    # Copy raw bytes out of the CMBlockBuffer.
    status, data = CoreMedia.CMBlockBufferCopyDataBytes(block_buf, 0, length, None)
    if status != 0:
        logger.warning("CMBlockBufferCopyDataBytes failed with status %d", status)
        return None

    # The audio format from ScreenCaptureKit is 32-bit float, interleaved.
    # Convert the raw bytes to numpy float32.
    try:
        samples = np.frombuffer(data, dtype=np.float32)
    except ValueError:
        logger.warning("Could not interpret %d bytes as float32 audio", length)
        return None

    return samples


# ---------------------------------------------------------------------------
# SCStreamOutput delegate and SystemAudioRecorder — only defined when available
# ---------------------------------------------------------------------------

if _AVAILABLE:

    class _AudioStreamOutput(NSObject):
        """Objective-C delegate that receives audio buffers from SCStream.

        Buffers are accumulated into an internal list which can be drained
        by the Python consumer thread.
        """

        def init(self):  # noqa: D401 — Obj-C naming convention
            self = objc.super(_AudioStreamOutput, self).init()
            if self is None:
                return None
            self._lock = threading.Lock()
            self._buffers: list[np.ndarray] = []
            self._error: Exception | None = None
            return self

        # SCStreamOutput protocol — audio callback
        def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
            """Called by ScreenCaptureKit for each audio buffer."""
            # output_type 1 = audio (SCStreamOutputTypeAudio)
            if output_type != 1:
                return

            samples = _extract_audio_samples(sample_buffer)
            if samples is not None and samples.size > 0:
                with self._lock:
                    self._buffers.append(samples)

        # SCStreamDelegate — error callback
        def stream_didStopWithError_(self, stream, error):
            logger.error("SCStream stopped with error: %s", error)
            with self._lock:
                self._error = RuntimeError(f"SCStream error: {error}")

        def drain(self) -> list[np.ndarray]:
            """Return and clear accumulated audio buffers."""
            with self._lock:
                bufs = self._buffers
                self._buffers = []
                return bufs

        def get_error(self) -> Exception | None:
            with self._lock:
                return self._error

    class SystemAudioRecorder:
        """Captures system audio using ScreenCaptureKit.

        This is the macOS-native replacement for soundcard-based loopback
        capture.  It yields raw float32 numpy arrays of audio samples.

        Usage::

            recorder = SystemAudioRecorder(sample_rate=16000)
            recorder.start()
            # ... periodically call recorder.read_chunk() ...
            recorder.stop()
        """

        def __init__(self, sample_rate: int = 16_000, channels: int = 1):
            self.sample_rate = sample_rate
            self.channels = channels
            self._stream = None
            self._output: _AudioStreamOutput | None = None
            self._queue = None
            self._started = False

        def start(self) -> None:
            """Request Screen Recording permission and start the audio stream."""
            import ScreenCaptureKit
            import Quartz

            if self._started:
                return

            # --- 1. Get shareable content (triggers permission prompt) ----------
            content = self._get_shareable_content()
            if content is None:
                raise RuntimeError(
                    "Could not get shareable content.  Grant Screen Recording "
                    "permission in System Settings > Privacy & Security."
                )

            # --- 2. Build an SCContentFilter for the full display ---------------
            displays = content.displays()
            if not displays:
                raise RuntimeError("No displays found.")
            display = displays[0]

            # Exclude no applications — capture everything.
            content_filter = ScreenCaptureKit.SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(
                display, [], []
            )

            # --- 3. Stream configuration ----------------------------------------
            config = ScreenCaptureKit.SCStreamConfiguration.alloc().init()

            # Audio settings
            config.setCapturesAudio_(True)
            config.setSampleRate_(self.sample_rate)
            config.setChannelCount_(self.channels)

            # Check if capturesAudioOnly is available (macOS 14+)
            major, _ = _parse_macos_version()
            if major >= 14 and hasattr(config, "setCapturesAudioOnly_"):
                config.setCapturesAudioOnly_(True)
                logger.info("Using audio-only capture (macOS 14+)")
            else:
                # macOS 13 — need minimal video stream
                config.setWidth_(2)
                config.setHeight_(2)
                config.setMinimumFrameInterval_(
                    Quartz.CMTimeMake(1, 1)  # 1 FPS
                )
                config.setShowsCursor_(False)
                logger.info("Using minimal video + audio capture (macOS 13)")

            # --- 4. Create and start the stream ----------------------------------
            self._output = _AudioStreamOutput.alloc().init()
            self._queue = _create_serial_dispatch_queue(b"on_the_record.audio")

            self._stream = (
                ScreenCaptureKit.SCStream.alloc().initWithFilter_configuration_delegate_(
                    content_filter, config, self._output
                )
            )

            # Add the output handler for audio
            success = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self._output,
                1,  # SCStreamOutputTypeAudio
                self._queue,
                None,
            )
            if not success:
                raise RuntimeError("Failed to add stream output for audio.")

            # Start the stream (async with completion handler)
            start_event = threading.Event()
            start_error: list[Exception] = []

            def _on_start(error):
                if error is not None:
                    start_error.append(RuntimeError(f"SCStream start failed: {error}"))
                start_event.set()

            self._stream.startCaptureWithCompletionHandler_(_on_start)

            # Pump the run loop briefly to allow the completion handler to fire.
            deadline = time.monotonic() + 10.0
            while not start_event.is_set() and time.monotonic() < deadline:
                NSRunLoop.currentRunLoop().runUntilDate_(
                    NSDate.dateWithTimeIntervalSinceNow_(0.1)
                )

            if not start_event.is_set():
                raise RuntimeError("Timed out waiting for SCStream to start.")
            if start_error:
                raise start_error[0]

            self._started = True
            logger.info(
                "ScreenCaptureKit stream started (%d Hz, %d ch)",
                self.sample_rate,
                self.channels,
            )

        def read_chunk(self, num_samples: int) -> np.ndarray:
            """Block until *num_samples* audio samples have been captured.

            Returns a 1-D float32 numpy array.  Raises if the stream has
            encountered an error.
            """
            if not self._started or self._output is None:
                raise RuntimeError("Stream not started.")

            collected: list[np.ndarray] = []
            total = 0

            while total < num_samples:
                err = self._output.get_error()
                if err is not None:
                    raise err

                buffers = self._output.drain()
                for buf in buffers:
                    # ScreenCaptureKit may deliver interleaved multi-channel data.
                    # If we requested mono (channels=1), it should already be mono,
                    # but reshape & average just in case.
                    if self.channels == 1 and buf.ndim == 1:
                        # Already mono
                        pass
                    elif buf.size % self.channels == 0:
                        buf = buf.reshape(-1, self.channels).mean(axis=1).astype(np.float32)

                    collected.append(buf)
                    total += buf.size

                if total < num_samples:
                    # Sleep briefly to let more audio arrive.
                    time.sleep(0.05)

            # Concatenate and trim to exact size.
            audio = np.concatenate(collected)
            return audio[:num_samples]

        def stop(self) -> None:
            """Stop the audio stream."""
            if not self._started or self._stream is None:
                return

            stop_event = threading.Event()

            def _on_stop(error):
                if error is not None:
                    logger.warning("SCStream stop error: %s", error)
                stop_event.set()

            self._stream.stopCaptureWithCompletionHandler_(_on_stop)

            deadline = time.monotonic() + 5.0
            while not stop_event.is_set() and time.monotonic() < deadline:
                NSRunLoop.currentRunLoop().runUntilDate_(
                    NSDate.dateWithTimeIntervalSinceNow_(0.1)
                )

            self._started = False
            self._stream = None
            logger.info("ScreenCaptureKit stream stopped.")

        def _get_shareable_content(self):
            """Synchronously fetch SCShareableContent."""
            import ScreenCaptureKit

            result: list = [None]
            event = threading.Event()

            def _handler(content, error):
                if error is not None:
                    logger.error("getShareableContent error: %s", error)
                else:
                    result[0] = content
                event.set()

            ScreenCaptureKit.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                True,  # exclude desktop windows
                True,  # on-screen windows only
                _handler,
            )

            deadline = time.monotonic() + 10.0
            while not event.is_set() and time.monotonic() < deadline:
                NSRunLoop.currentRunLoop().runUntilDate_(
                    NSDate.dateWithTimeIntervalSinceNow_(0.1)
                )

            return result[0]

        def __enter__(self):
            self.start()
            return self

        def __exit__(self, *args):
            self.stop()
