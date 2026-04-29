"""Local speaker recognition and enrollment support."""

from __future__ import annotations

import io
import json
import os
import sys
import wave
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, TextIO

import numpy as np


DEFAULT_SPEAKER_THRESHOLD = 0.78
DEFAULT_CLUSTER_THRESHOLD = 0.72
DEFAULT_MIN_SEGMENT_SECONDS = 1.0
DEFAULT_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
BUNDLED_SPEAKER_MODEL_DIR = "speechbrain-model"


class SpeakerRecognitionUnavailable(RuntimeError):
    """Raised when optional speaker recognition dependencies are missing."""


class SpeakerEmbeddingBackend(Protocol):
    """Computes a speaker embedding from mono audio."""

    def embed(self, audio: np.ndarray, sample_rate: int) -> list[float]:
        """Return a speaker embedding for *audio*."""


@dataclass
class SpeakerProfile:
    """A locally stored speaker identity."""

    id: str
    display_name: str
    embedding: list[float]
    sample_count: int = 1
    sample_paths: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())


@dataclass
class SpeakerMatch:
    """Best matching saved speaker profile."""

    profile: SpeakerProfile
    score: float


@dataclass
class UnknownSpeakerCluster:
    """A cluster of unmatched segments within one recording session."""

    id: str
    temporary_name: str
    embedding: list[float]
    segment_indices: list[int] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)
    sample_paths: list[str] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_speaker_profiles_dir() -> Path:
    """Return the platform-specific directory for speaker profiles."""
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
        return root / "on-the-record" / "speakers"

    if sys_platform() == "darwin":
        return Path.home() / "Library" / "Application Support" / "on-the-record" / "speakers"

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return root / "on-the-record" / "speakers"


def sys_platform() -> str:
    """Return ``sys.platform`` through a wrapper for simple tests."""
    import sys

    return sys.platform


def cosine_similarity(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    """Return cosine similarity for two embedding vectors."""
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Return the normalized arithmetic mean of embeddings."""
    if not embeddings:
        raise ValueError("at least one embedding is required")

    centroid = np.mean(np.asarray(embeddings, dtype=np.float32), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0.0:
        centroid = centroid / norm
    return centroid.astype(float).tolist()


def encode_wav_sample(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float audio as 16-bit PCM WAV bytes."""
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=1)
    pcm = np.clip(mono, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(sample_rate)
        file.writeframes(pcm.tobytes())
    return buffer.getvalue()


class SpeakerProfileStore:
    """Loads and saves local speaker profiles."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else default_speaker_profiles_dir()
        self.path = self.directory / "profiles.json"
        self.profiles: list[SpeakerProfile] = []

    def load(self) -> None:
        """Load profiles from disk if they exist."""
        if not self.path.exists():
            self.profiles = []
            return

        with open(self.path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        self.profiles = [SpeakerProfile(**item) for item in payload.get("profiles", [])]

    def save(self) -> None:
        """Persist profiles to disk."""
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "profiles": [profile.__dict__ for profile in self.profiles]}
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")

    def list_profiles(self) -> list[SpeakerProfile]:
        """Return profiles sorted by display name."""
        return sorted(self.profiles, key=lambda profile: profile.display_name.casefold())

    def find_by_id(self, profile_id: str) -> SpeakerProfile | None:
        """Find a profile by id."""
        return next((profile for profile in self.profiles if profile.id == profile_id), None)

    def find_by_name(self, display_name: str) -> SpeakerProfile | None:
        """Find a profile by display name, case-insensitively."""
        normalized = display_name.casefold()
        return next(
            (profile for profile in self.profiles if profile.display_name.casefold() == normalized),
            None,
        )

    def rename(self, profile_id: str, display_name: str) -> bool:
        """Rename a profile. Returns whether a profile changed."""
        profile = self.find_by_id(profile_id)
        if profile is None:
            return False
        profile.display_name = display_name
        profile.updated_at = _utc_now()
        self.save()
        return True

    def remove(self, profile_id: str) -> bool:
        """Remove a profile. Returns whether a profile was removed."""
        before = len(self.profiles)
        self.profiles = [profile for profile in self.profiles if profile.id != profile_id]
        if len(self.profiles) == before:
            return False
        self.save()
        return True

    def best_match(
        self,
        embedding: list[float],
        *,
        threshold: float = DEFAULT_SPEAKER_THRESHOLD,
    ) -> SpeakerMatch | None:
        """Return the best profile match when it clears *threshold*."""
        best: SpeakerMatch | None = None
        for profile in self.profiles:
            score = cosine_similarity(embedding, profile.embedding)
            if best is None or score > best.score:
                best = SpeakerMatch(profile=profile, score=score)

        if best is None or best.score < threshold:
            return None
        return best

    def enroll(
        self,
        display_name: str,
        embeddings: list[list[float]],
        *,
        sample_paths: list[str] | None = None,
    ) -> SpeakerProfile:
        """Create or update a profile with new embeddings."""
        if not display_name.strip():
            raise ValueError("speaker name cannot be empty")
        if not embeddings:
            raise ValueError("at least one embedding is required")

        now = _utc_now()
        profile = self.find_by_name(display_name.strip())
        new_count = len(embeddings)
        new_centroid = average_embeddings(embeddings)

        if profile is None:
            profile = SpeakerProfile(
                id=str(uuid.uuid4()),
                display_name=display_name.strip(),
                embedding=new_centroid,
                sample_count=new_count,
                sample_paths=list(sample_paths or []),
                created_at=now,
                updated_at=now,
            )
            self.profiles.append(profile)
        else:
            weighted = [profile.embedding] * max(profile.sample_count, 1) + embeddings
            profile.embedding = average_embeddings(weighted)
            profile.sample_count += new_count
            profile.sample_paths.extend(sample_paths or [])
            profile.updated_at = now

        self.save()
        return profile


class SpeechBrainEmbeddingBackend:
    """SpeechBrain ECAPA speaker embedding backend."""

    def __init__(self, model: str = DEFAULT_SPEAKER_MODEL, savedir: str | Path | None = None) -> None:
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
        except ImportError as exc:
            raise SpeakerRecognitionUnavailable(
                "Speaker recognition dependencies are not installed. "
                "Install them with `uv sync --extra speaker`. On Windows, use Python 3.11 or 3.12."
            ) from exc

        bundled_model_dir = _bundled_speaker_model_dir()
        model_source = str(bundled_model_dir) if bundled_model_dir else model
        model_savedir = bundled_model_dir or savedir or Path.home() / ".cache" / "on-the-record" / "speechbrain"
        local_strategy = LocalStrategy.NO_LINK if bundled_model_dir else LocalStrategy.COPY_SKIP_CACHE
        overrides = {"pretrained_path": str(model_savedir)} if bundled_model_dir else None

        self._torch = torch
        classifier_kwargs = {
            "source": model_source,
            "savedir": str(model_savedir),
            "local_strategy": local_strategy,
        }
        if overrides is not None:
            classifier_kwargs["overrides"] = overrides

        self._classifier = EncoderClassifier.from_hparams(
            **classifier_kwargs,
        )

    def embed(self, audio: np.ndarray, sample_rate: int) -> list[float]:
        """Compute a normalized speaker embedding for mono audio."""
        del sample_rate
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim > 1:
            mono = np.mean(mono, axis=1)
        signal = self._torch.from_numpy(mono).unsqueeze(0)
        with self._torch.no_grad():
            embedding = self._classifier.encode_batch(signal).squeeze().detach().cpu().numpy()
        return average_embeddings([np.asarray(embedding, dtype=np.float32).tolist()])


class SpeakerRecognitionSession:
    """Recognizes known speakers and clusters unknown speakers in one run."""

    def __init__(
        self,
        store: SpeakerProfileStore,
        backend: SpeakerEmbeddingBackend,
        *,
        threshold: float = DEFAULT_SPEAKER_THRESHOLD,
        cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
        min_segment_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS,
        save_samples: bool = False,
        sample_directory: str | Path | None = None,
    ) -> None:
        self.store = store
        self.backend = backend
        self.threshold = threshold
        self.cluster_threshold = cluster_threshold
        self.min_segment_seconds = min_segment_seconds
        self.save_samples = save_samples
        self.sample_directory = Path(sample_directory) if sample_directory else store.directory / "samples"
        self.clusters: list[UnknownSpeakerCluster] = []
        self._cluster_embeddings: dict[str, list[list[float]]] = {}

    def resolve_segment(
        self,
        *,
        segment_index: int,
        speaker_label: str,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
    ) -> str:
        """Return the resolved speaker name or a temporary cluster label."""
        if audio.size < int(self.min_segment_seconds * sample_rate):
            return speaker_label

        embedding = self.backend.embed(audio, sample_rate)
        match = self.store.best_match(embedding, threshold=self.threshold)
        if match is not None:
            return match.profile.display_name

        cluster = self._find_or_create_cluster(embedding)
        cluster.segment_indices.append(segment_index)
        if text.strip() and len(cluster.snippets) < 3:
            cluster.snippets.append(text.strip())
        if self.save_samples and not cluster.sample_paths:
            cluster.sample_paths.append(str(self._save_cluster_sample(cluster, audio, sample_rate)))
        return cluster.temporary_name

    def prompt_for_unknowns(
        self,
        *,
        input_func: Callable[[str], str] = input,
        output: TextIO | None = None,
    ) -> dict[str, str]:
        """Prompt for unknown cluster names and enroll them.

        Returns a mapping from temporary cluster labels to final display names.
        """
        if not self.clusters:
            return {}

        stream = output
        mapping: dict[str, str] = {}
        for cluster in self.clusters:
            if stream is not None:
                print(f"\nUnknown speaker: {cluster.temporary_name}", file=stream)
                for snippet in cluster.snippets:
                    print(f"  {snippet[:120]}", file=stream)

            display_name = input_func(
                f"Name for {cluster.temporary_name} (blank to keep label): "
            ).strip()
            if not display_name:
                continue

            embeddings = self._cluster_embeddings.get(cluster.id, [cluster.embedding])
            self.store.enroll(display_name, embeddings, sample_paths=cluster.sample_paths)
            mapping[cluster.temporary_name] = display_name

        return mapping

    def _find_or_create_cluster(self, embedding: list[float]) -> UnknownSpeakerCluster:
        best_cluster: UnknownSpeakerCluster | None = None
        best_score = 0.0
        for cluster in self.clusters:
            score = cosine_similarity(embedding, cluster.embedding)
            if score > best_score:
                best_cluster = cluster
                best_score = score

        if best_cluster is not None and best_score >= self.cluster_threshold:
            embeddings = self._cluster_embeddings[best_cluster.id]
            embeddings.append(embedding)
            best_cluster.embedding = average_embeddings(embeddings)
            return best_cluster

        cluster_number = len(self.clusters) + 1
        cluster = UnknownSpeakerCluster(
            id=str(uuid.uuid4()),
            temporary_name=f"Unknown Speaker {cluster_number}",
            embedding=embedding,
        )
        self.clusters.append(cluster)
        self._cluster_embeddings[cluster.id] = [embedding]
        return cluster

    def _save_cluster_sample(
        self,
        cluster: UnknownSpeakerCluster,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Path:
        self.sample_directory.mkdir(parents=True, exist_ok=True)
        path = self.sample_directory / f"{cluster.id}.wav"
        path.write_bytes(encode_wav_sample(audio, sample_rate))
        return path


def create_speaker_backend() -> SpeakerEmbeddingBackend:
    """Create the default optional speaker embedding backend."""
    return SpeechBrainEmbeddingBackend()


def _bundled_speaker_model_dir() -> Path | None:
    """Return the PyInstaller-bundled SpeechBrain model directory, if present."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if not bundle_root:
        return None

    model_dir = Path(bundle_root) / BUNDLED_SPEAKER_MODEL_DIR
    return model_dir if (model_dir / "hyperparams.yaml").exists() else None
