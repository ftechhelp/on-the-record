"""Tests for on_the_record.audio — silence detection and WAV encoding."""

from __future__ import annotations

import struct
import sys
import time
import types
import wave
import io
import warnings

import numpy as np
import pytest

import on_the_record.audio as audio_module
import on_the_record.macos_audio as macos_audio_module
from on_the_record.audio import AudioRecorder, encode_wav, is_silent


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


class TestScreenCaptureKitCompat:
    def test_make_cm_time_uses_coremedia(self, monkeypatch):
        fake_coremedia = types.SimpleNamespace(
            CMTimeMake=lambda value, timescale: ("cm-time", value, timescale)
        )

        monkeypatch.setitem(sys.modules, "CoreMedia", fake_coremedia)

        assert macos_audio_module._make_cm_time(1, 1) == ("cm-time", 1, 1)


class TestAudioMixing:
    def test_mix_audio_sources_pads_and_clips(self):
        system_audio = np.array([0.6, 0.0, -0.2], dtype=np.float32)
        microphone_audio = np.array([0.5, 0.4], dtype=np.float32)

        mixed = audio_module._mix_audio_sources(system_audio, microphone_audio)

        assert mixed.dtype == np.float32
        assert np.allclose(mixed, np.array([1.0, 0.4, -0.2], dtype=np.float32))

    def test_get_microphone_device_skips_loopback_and_virtual_devices(self, monkeypatch):
        loopback = types.SimpleNamespace(
            name="Speakers Loopback",
            id="loopback",
            isloopback=True,
        )
        virtual = types.SimpleNamespace(
            name="BlackHole 2ch",
            id="blackhole",
            isloopback=False,
        )
        physical = types.SimpleNamespace(
            name="Built-in Microphone",
            id="mic",
            isloopback=False,
        )
        fake_soundcard = types.SimpleNamespace(
            default_microphone=lambda: virtual,
            all_microphones=lambda include_loopback=False: [loopback, virtual, physical],
        )

        monkeypatch.setattr(audio_module, "sc", fake_soundcard)
        monkeypatch.setattr(audio_module, "_IS_MACOS", True)

        assert audio_module._get_microphone_device(exclude_device=loopback) is physical


class _FakeRecorderContext:
    def __init__(self, audio: np.ndarray):
        self.audio = audio

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def record(self, numframes: int):
        return self.audio[:numframes]


class _FakeDevice:
    def __init__(self, name: str, audio: np.ndarray):
        self.name = name
        self.id = name.lower().replace(" ", "-")
        self.isloopback = False
        self.audio = audio

    def recorder(self, *, samplerate: int, channels: int):
        return _FakeRecorderContext(self.audio)


class TestAudioRecorderMicrophoneMix:
    def test_requires_at_least_one_audio_source(self):
        with pytest.raises(ValueError, match="At least one audio source"):
            AudioRecorder(include_system_audio=False, include_microphone=False)

    def test_soundcard_recording_mixes_loopback_and_microphone(self, monkeypatch):
        loopback = _FakeDevice(
            "Speakers Loopback",
            np.array([[0.2], [0.2], [0.0], [0.0]], dtype=np.float32),
        )
        microphone = _FakeDevice(
            "Built-in Microphone",
            np.array([[0.0], [0.4], [0.4], [0.0]], dtype=np.float32),
        )

        monkeypatch.setattr(audio_module, "_get_capture_device", lambda device: loopback)
        monkeypatch.setattr(
            audio_module,
            "_get_microphone_device",
            lambda exclude_device=None: microphone,
        )

        recorder = AudioRecorder(sample_rate=4, chunk_seconds=1, silence_threshold=0.0)
        chunks = recorder._record_soundcard()

        try:
            chunk = next(chunks)
        finally:
            chunks.close()

        assert np.allclose(
            chunk.audio,
            np.array([0.2, 0.6, 0.4, 0.0], dtype=np.float32),
        )
        assert chunk.chunk_index == 0
        assert chunk.start_time_offset == 0

    def test_soundcard_recording_can_skip_microphone(self, monkeypatch):
        loopback = _FakeDevice(
            "Speakers Loopback",
            np.array([[0.2], [0.3], [0.0], [0.0]], dtype=np.float32),
        )

        monkeypatch.setattr(audio_module, "_get_capture_device", lambda device: loopback)

        def fail_get_microphone_device(exclude_device=None):
            raise AssertionError("microphone should not be opened")

        monkeypatch.setattr(
            audio_module,
            "_get_microphone_device",
            fail_get_microphone_device,
        )

        recorder = AudioRecorder(
            sample_rate=4,
            chunk_seconds=1,
            include_microphone=False,
            silence_threshold=0.0,
        )
        chunks = recorder._record_soundcard()

        try:
            chunk = next(chunks)
        finally:
            chunks.close()

        assert np.allclose(
            chunk.audio,
            np.array([0.2, 0.3, 0.0, 0.0], dtype=np.float32),
        )

    def test_soundcard_recording_can_skip_system_audio(self, monkeypatch):
        microphone = _FakeDevice(
            "Built-in Microphone",
            np.array([[0.0], [0.4], [0.5], [0.0]], dtype=np.float32),
        )

        def fail_get_capture_device(device_name=None):
            raise AssertionError("system audio should not be opened")

        monkeypatch.setattr(audio_module, "_get_capture_device", fail_get_capture_device)
        monkeypatch.setattr(
            audio_module,
            "_get_microphone_device",
            lambda exclude_device=None: microphone,
        )

        recorder = AudioRecorder(
            sample_rate=4,
            chunk_seconds=1,
            include_system_audio=False,
            silence_threshold=0.0,
        )
        chunks = recorder._record_soundcard()

        try:
            chunk = next(chunks)
        finally:
            chunks.close()

        assert np.allclose(
            chunk.audio,
            np.array([0.0, 0.4, 0.5, 0.0], dtype=np.float32),
        )

    def test_screencapturekit_recording_mixes_system_and_microphone(self, monkeypatch):
        microphone = _FakeDevice(
            "Built-in Microphone",
            np.array([[0.0], [0.5], [0.0], [0.0]], dtype=np.float32),
        )

        class FakeSystemAudioRecorder:
            def __init__(self, *, sample_rate: int, channels: int):
                self.audio = np.array([0.25, 0.0, 0.25, 0.0], dtype=np.float32)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def read_chunk(self, num_samples: int):
                return self.audio[:num_samples]

        monkeypatch.setattr(
            audio_module,
            "SystemAudioRecorder",
            FakeSystemAudioRecorder,
            raising=False,
        )
        monkeypatch.setattr(
            audio_module,
            "_get_microphone_device",
            lambda exclude_device=None: microphone,
        )

        recorder = AudioRecorder(sample_rate=4, chunk_seconds=1, silence_threshold=0.0)
        chunks = recorder._record_screencapturekit()

        try:
            chunk = next(chunks)
        finally:
            chunks.close()

        assert np.allclose(
            chunk.audio,
            np.array([0.25, 0.5, 0.25, 0.0], dtype=np.float32),
        )
        assert chunk.chunk_index == 0
        assert chunk.start_time_offset == 0

    def test_screencapturekit_recording_can_skip_microphone(self, monkeypatch):
        class FakeSystemAudioRecorder:
            def __init__(self, *, sample_rate: int, channels: int):
                self.audio = np.array([0.25, 0.0, 0.35, 0.0], dtype=np.float32)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def read_chunk(self, num_samples: int):
                return self.audio[:num_samples]

        monkeypatch.setattr(
            audio_module,
            "SystemAudioRecorder",
            FakeSystemAudioRecorder,
            raising=False,
        )

        def fail_get_microphone_device(exclude_device=None):
            raise AssertionError("microphone should not be opened")

        monkeypatch.setattr(
            audio_module,
            "_get_microphone_device",
            fail_get_microphone_device,
        )

        recorder = AudioRecorder(
            sample_rate=4,
            chunk_seconds=1,
            include_microphone=False,
            silence_threshold=0.0,
        )
        chunks = recorder._record_screencapturekit()

        try:
            chunk = next(chunks)
        finally:
            chunks.close()

        assert np.allclose(
            chunk.audio,
            np.array([0.25, 0.0, 0.35, 0.0], dtype=np.float32),
        )
