"""OpenAI transcription with speaker diarization.

Sends WAV audio chunks to the ``gpt-4o-transcribe-diarize`` endpoint and
returns structured transcript segments.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

logger = logging.getLogger("on_the_record.transcribe")

# Retriable error types and backoff settings
_RETRIABLE = (APIConnectionError, RateLimitError)
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0  # seconds

# Pricing per 1M tokens (input, output) by model prefix.
# Verified against OpenAI pricing page as of 2025-05.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-transcribe-diarize": (2.50, 10.00),
    "gpt-4o-transcribe": (2.50, 10.00),
    "gpt-4o-mini-transcribe": (1.25, 5.00),
}


@dataclass
class TranscriptSegment:
    """A single diarized transcript segment."""

    speaker: str
    text: str
    start: float  # seconds
    end: float  # seconds


@dataclass
class UsageInfo:
    """Token usage and estimated cost for one transcription call."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        """Estimated cost in USD based on known model pricing."""
        price_per_1m = _MODEL_PRICING.get(self.model)
        if price_per_1m is None:
            # Fall back to gpt-4o-transcribe pricing for unknown models
            price_per_1m = _MODEL_PRICING["gpt-4o-transcribe"]
        input_cost = self.input_tokens / 1_000_000 * price_per_1m[0]
        output_cost = self.output_tokens / 1_000_000 * price_per_1m[1]
        return input_cost + output_cost

    def __add__(self, other: "UsageInfo") -> "UsageInfo":
        return UsageInfo(
            model=self.model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


def transcribe_chunk(
    wav_bytes: bytes,
    *,
    api_key: str,
    model: str = "gpt-4o-transcribe-diarize",
    chunk_offset: float = 0.0,
) -> tuple[list[TranscriptSegment], UsageInfo]:
    """Transcribe a WAV audio chunk via the OpenAI API.

    Parameters
    ----------
    wav_bytes:
        Raw WAV file bytes (16-bit PCM, 16 kHz).
    api_key:
        OpenAI API key.
    model:
        Model to use.  Defaults to ``gpt-4o-transcribe-diarize`` for
        speaker diarization.  Can also be ``gpt-4o-transcribe`` or
        ``gpt-4o-mini-transcribe`` (no diarization).
    chunk_offset:
        Time offset (seconds) of this chunk from the start of the
        recording.  Added to each segment's start/end so timestamps
        are relative to the overall session.

    Returns
    -------
    tuple[list[TranscriptSegment], UsageInfo]
        Parsed transcript segments with speaker labels/timestamps, and
        token usage with estimated cost for this API call.
    """
    client = OpenAI(api_key=api_key)

    use_diarize = "diarize" in model
    response_format = "diarized_json" if use_diarize else "verbose_json"

    # Build the request kwargs
    kwargs: dict = dict(
        model=model,
        file=("chunk.wav", wav_bytes, "audio/wav"),
        response_format=response_format,
    )

    # chunking_strategy is required for diarized model when audio > 30 s
    # We always set it to "auto" for safety.
    if use_diarize:
        kwargs["chunking_strategy"] = "auto"

    segments: list[TranscriptSegment] = []
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            logger.debug(
                "Sending chunk to %s (attempt %d/%d, %.1f s offset)",
                model,
                attempt + 1,
                _MAX_RETRIES,
                chunk_offset,
            )
            result = client.audio.transcriptions.create(**kwargs)
            last_exc = None
            break
        except _RETRIABLE as exc:
            last_exc = exc
            wait = _INITIAL_BACKOFF * (2**attempt)
            logger.warning(
                "Retriable API error (%s), retrying in %.1f s …",
                exc,
                wait,
            )
            time.sleep(wait)
        except APIError as exc:
            # Non-retriable API error — raise immediately.
            logger.error("OpenAI API error: %s", exc)
            raise

    if last_exc is not None:
        logger.error("All %d API attempts failed.", _MAX_RETRIES)
        raise last_exc  # type: ignore[misc]

    # Extract token usage from the response if the API provides it.
    raw_usage = getattr(result, "usage", None)
    usage = UsageInfo(
        model=model,
        input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
    )

    # Parse the response into TranscriptSegment objects.
    if use_diarize:
        segments = _parse_diarized(result, chunk_offset)
    else:
        segments = _parse_verbose(result, chunk_offset)

    logger.info(
        "Transcribed %d segment(s) from chunk at offset %.1f s (tokens: %d in / %d out, ~$%.4f)",
        len(segments),
        chunk_offset,
        usage.input_tokens,
        usage.output_tokens,
        usage.cost_usd,
    )
    return segments, usage


def _parse_diarized(result, offset: float) -> list[TranscriptSegment]:
    """Parse a ``diarized_json`` response."""
    segments: list[TranscriptSegment] = []

    # The response object has a `.segments` attribute when using
    # response_format="diarized_json".
    raw_segments = getattr(result, "segments", None) or []

    for seg in raw_segments:
        speaker = getattr(seg, "speaker", None) or "Unknown"
        text = getattr(seg, "text", "") or ""
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", 0.0))

        text = text.strip()
        if not text:
            continue

        segments.append(
            TranscriptSegment(
                speaker=speaker,
                text=text,
                start=start + offset,
                end=end + offset,
            )
        )
    return segments


def _parse_verbose(result, offset: float) -> list[TranscriptSegment]:
    """Parse a ``verbose_json`` response (no diarization)."""
    segments: list[TranscriptSegment] = []

    raw_segments = getattr(result, "segments", None) or []

    for seg in raw_segments:
        text = getattr(seg, "text", "") or ""
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", 0.0))

        text = text.strip()
        if not text:
            continue

        segments.append(
            TranscriptSegment(
                speaker="Speaker",
                text=text,
                start=start + offset,
                end=end + offset,
            )
        )
    return segments
