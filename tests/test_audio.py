"""Tests for on_the_record.audio — silence detection and WAV encoding."""

from __future__ import annotations

import struct
import time
import types
import wave
import io
import warnings

import numpy as np
import pytest

import on_the_record.audio as audio_module
from on_the_record.audio import encode_wav, is_silent


# ---------------------------------------------------------------------------
# is_silent
# ---------------------------------------------------------------------------


class TestIsSilent:
    def test_empty_array(self):
        assert is_silent(np.array([], dtype=np.float32), threshold=0.003) is True

    def test_digital_silence(self):
        silence = np.zeros(16000, dtype=np.float32)
        assert is_silent(silence, threshold=0.003) is True

    def test_near_silence(self):
        # Very low amplitude noise — below threshold
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.001, 16000).astype(np.float32)
        assert is_silent(noise, threshold=0.003) is True

    def test_audible_signal(self):
        # Sine wave at a reasonable amplitude
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        signal = 0.5 * np.sin(2 * np.pi * 440 * t)
        assert is_silent(signal, threshold=0.003) is False

    def test_threshold_boundary(self):
        # Constant signal right at the threshold should not be silent
        constant = np.full(16000, 0.004, dtype=np.float32)
        assert is_silent(constant, threshold=0.003) is False

        # Just below threshold should be silent
        constant_low = np.full(16000, 0.002, dtype=np.float32)
        assert is_silent(constant_low, threshold=0.003) is True


# ---------------------------------------------------------------------------
# encode_wav
# ---------------------------------------------------------------------------


class TestEncodeWav:
    def test_produces_valid_wav(self):
        # 1 second of silence at 16kHz
        audio = np.zeros(16000, dtype=np.float32)
        wav_bytes = encode_wav(audio, sample_rate=16000, channels=1)

        # Should be parseable as a WAV file
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000

    def test_stereo_to_mono(self):
        # Stereo input (2-D array) should be averaged to mono
        audio = np.zeros((16000, 2), dtype=np.float32)
        audio[:, 0] = 0.5
        audio[:, 1] = -0.5
        wav_bytes = encode_wav(audio, sample_rate=16000, channels=1)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getnframes() == 16000

    def test_clipping(self):
        # Values outside [-1, 1] should be clipped, not overflow
        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        wav_bytes = encode_wav(audio, sample_rate=16000, channels=1)

        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(3)
            samples = struct.unpack("<3h", frames)
            assert samples[0] == 32767  # clipped to max
            assert samples[1] == -32767  # clipped to min (32767 not 32768)
            assert abs(samples[2] - 16383) <= 1  # 0.5 * 32767 ≈ 16383

    def test_wav_starts_with_riff(self):
        audio = np.zeros(100, dtype=np.float32)
        wav_bytes = encode_wav(audio, sample_rate=16000)
        assert wav_bytes[:4] == b"RIFF"


class TestSoundcardWindowsCompat:
    def test_patch_replaces_fromstring_path_with_frombuffer_copy(self, monkeypatch):
        payload = bytearray(np.array([0.25, -0.5], dtype=np.float32).tobytes())

        class FakeFFI:
            NULL = object()

            @staticmethod
            def buffer(data_ptr, size):
                assert size == len(payload)
                return memoryview(payload)

        class FakeRecorder:
            def _record_chunk(self):
                raise AssertionError("patch should replace this method")

            def __init__(self):
                self._idle_start_time = None
                self._is_first_frame = True
                self.channelmap = [0]
                self.deviceperiod = (0.05, 0.01)
                self.samplerate = 16000
                self.released = []

            def _capture_available_frames(self):
                return 1

            def _capture_buffer(self):
                return object(), 2, 0

            def _capture_release(self, nframes):
                self.released.append(nframes)

        fake_mediafoundation = types.SimpleNamespace(
            _Recorder=FakeRecorder,
            _ffi=FakeFFI,
            _ole32=types.SimpleNamespace(
                AUDCLNT_BUFFERFLAGS_SILENT=0x2,
                AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY=0x1,
            ),
            SoundcardRuntimeWarning=RuntimeWarning,
            warnings=warnings,
            time=time,
        )

        monkeypatch.setattr(audio_module.platform, "system", lambda: "Windows")

        audio_module._patch_soundcard_windows_numpy_compat(fake_mediafoundation)

        recorder = FakeRecorder()
        chunk = recorder._record_chunk()

        assert np.allclose(chunk, np.array([0.25, -0.5], dtype=np.float32))
        assert chunk.base is None
        assert recorder.released == [2]
