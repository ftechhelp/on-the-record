"""Tests for local speaker recognition support."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from on_the_record.speaker_recognition import (
    SpeakerProfileStore,
    SpeakerRecognitionSession,
    average_embeddings,
    cosine_similarity,
)


class FakeBackend:
    def embed(self, audio: np.ndarray, sample_rate: int) -> list[float]:
        del sample_rate
        if audio[0] > 0:
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_cosine_similarity():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_average_embeddings_normalizes_centroid():
    result = average_embeddings([[2.0, 0.0], [2.0, 0.0]])

    assert result == [1.0, 0.0]


def test_profile_store_enrolls_and_matches(tmp_path: Path):
    store = SpeakerProfileStore(tmp_path)
    store.load()

    profile = store.enroll("Alice", [[1.0, 0.0]])
    store.load()
    match = store.best_match([0.95, 0.05], threshold=0.8)

    assert profile.display_name == "Alice"
    assert match is not None
    assert match.profile.display_name == "Alice"
    assert json.loads((tmp_path / "profiles.json").read_text())["profiles"][0]["display_name"] == "Alice"


def test_profile_store_updates_existing_profile(tmp_path: Path):
    store = SpeakerProfileStore(tmp_path)
    store.load()

    first = store.enroll("Alice", [[1.0, 0.0]])
    second = store.enroll("alice", [[1.0, 0.0]])

    assert first.id == second.id
    assert second.sample_count == 2
    assert len(store.profiles) == 1


def test_session_recognizes_known_speaker(tmp_path: Path):
    store = SpeakerProfileStore(tmp_path)
    store.load()
    store.enroll("Alice", [[1.0, 0.0]])
    session = SpeakerRecognitionSession(store, FakeBackend(), threshold=0.8)

    speaker = session.resolve_segment(
        segment_index=0,
        speaker_label="Speaker 1",
        text="hello",
        audio=np.ones(16_000, dtype=np.float32),
        sample_rate=16_000,
    )

    assert speaker == "Alice"


def test_session_preserves_unknown_label_and_prompts_for_name(tmp_path: Path):
    store = SpeakerProfileStore(tmp_path)
    store.load()
    session = SpeakerRecognitionSession(store, FakeBackend(), threshold=0.99)

    speaker = session.resolve_segment(
        segment_index=0,
        speaker_label="A",
        text="please remember me",
        audio=np.ones(16_000, dtype=np.float32),
        sample_rate=16_000,
    )
    mapping = session.prompt_for_unknowns(input_func=lambda prompt: "Bob")

    assert speaker == "A"
    assert mapping == {"A": "Bob"}
    assert store.find_by_name("Bob") is not None


def test_session_prompts_once_per_unknown_label(tmp_path: Path):
    store = SpeakerProfileStore(tmp_path)
    store.load()
    session = SpeakerRecognitionSession(store, FakeBackend(), threshold=0.99)
    prompts = []

    for index, text in enumerate(["first", "second"]):
        speaker = session.resolve_segment(
            segment_index=index,
            speaker_label="A",
            text=text,
            audio=np.ones(16_000, dtype=np.float32),
            sample_rate=16_000,
        )
        assert speaker == "A"

    mapping = session.prompt_for_unknowns(input_func=lambda prompt: prompts.append(prompt) or "Bob")

    assert prompts == ["Name for A (blank to keep label): "]
    assert mapping == {"A": "Bob"}
    assert store.find_by_name("Bob").sample_count == 2


def test_session_prompt_handles_eof_after_partial_mapping(tmp_path: Path):
    store = SpeakerProfileStore(tmp_path)
    store.load()
    session = SpeakerRecognitionSession(store, FakeBackend(), threshold=0.99)
    responses = iter(["Alice"])

    session.resolve_segment(
        segment_index=0,
        speaker_label="A",
        text="first",
        audio=np.ones(16_000, dtype=np.float32),
        sample_rate=16_000,
    )
    session.resolve_segment(
        segment_index=1,
        speaker_label="B",
        text="second",
        audio=np.zeros(16_000, dtype=np.float32),
        sample_rate=16_000,
    )

    def input_func(prompt: str) -> str:
        try:
            return next(responses)
        except StopIteration as exc:
            raise EOFError from exc

    mapping = session.prompt_for_unknowns(input_func=input_func)

    assert mapping == {"A": "Alice"}


def test_session_saves_samples_only_when_enabled(tmp_path: Path):
    store = SpeakerProfileStore(tmp_path)
    store.load()
    session = SpeakerRecognitionSession(
        store,
        FakeBackend(),
        threshold=0.99,
        save_samples=True,
        sample_directory=tmp_path / "samples",
    )

    session.resolve_segment(
        segment_index=0,
        speaker_label="Speaker 1",
        text="sample me",
        audio=np.ones(16_000, dtype=np.float32),
        sample_rate=16_000,
    )
    session.prompt_for_unknowns(input_func=lambda prompt: "Carol")
    profile = store.find_by_name("Carol")

    assert profile is not None
    assert len(profile.sample_paths) == 1
    assert Path(profile.sample_paths[0]).exists()

