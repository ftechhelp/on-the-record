"""Configuration and constants for on-the-record."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from environment and CLI flags."""

    # OpenAI
    api_key: str

    # Audio capture
    sample_rate: int = 16_000
    channels: int = 1
    chunk_seconds: int = 15
    device_name: str | None = None

    # Silence detection — RMS threshold below which a chunk is considered silent.
    # A 16-bit PCM signal has a max amplitude of 32767; 0.003 is well below any
    # meaningful speech but above digital silence / noise floor.
    silence_threshold: float = 0.003

    # Output
    output_path: str = ""
    output_format: str = "txt"  # txt | md | json

    # Model
    model: str = "gpt-4o-transcribe"


def _api_key_setup_hint(os_name: str | None = None) -> str:
    """Return platform-appropriate environment variable instructions."""
    if (os_name or os.name) == "nt":
        return (
            "PowerShell:  $env:OPENAI_API_KEY = \"sk-...\"\n"
            "cmd.exe:     set OPENAI_API_KEY=sk-..."
        )
    return "export OPENAI_API_KEY='sk-...'"


def load_api_key() -> str:
    """Load the OpenAI API key from the environment.

    Returns the key string, or prints an error and exits if not set.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print(
            "Error: OPENAI_API_KEY environment variable is not set.\n"
            "Get an API key at https://platform.openai.com/api-keys\n"
            "Then set it with one of:\n"
            f"{_api_key_setup_hint()}",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


# Default sample rate used across the application.
DEFAULT_SAMPLE_RATE: int = 16_000
DEFAULT_CHANNELS: int = 1
DEFAULT_CHUNK_SECONDS: int = 15
DEFAULT_SILENCE_THRESHOLD: float = 0.003
DEFAULT_FORMAT: str = "txt"
DEFAULT_MODEL: str = "gpt-4o-transcribe"
