"""Configuration and constants for on-the-record."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


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
    include_system_audio: bool = True
    include_microphone: bool = True

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


def load_dotenv() -> None:
    """Load environment values from .env files without overriding real env vars."""
    for path in _dotenv_paths():
        if path.is_file():
            _load_dotenv_file(path)


def _dotenv_paths() -> list[Path]:
    """Return candidate .env paths in precedence order."""
    candidates = [Path.cwd() / ".env"]

    executable = getattr(sys, "executable", "")
    if getattr(sys, "frozen", False) and executable:
        candidates.append(Path(executable).resolve().parent / ".env")

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / ".env")

    seen: set[Path] = set()
    paths: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            paths.append(resolved)
    return paths


def _load_dotenv_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from *path*."""
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = _parse_dotenv_line(line)
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_dotenv_line(line: str) -> tuple[str | None, str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None, ""
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None, ""

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None, ""

    return key, _parse_dotenv_value(value.strip())


def _parse_dotenv_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        inner = value[1:-1]
        if value[0] == '"':
            return bytes(inner, "utf-8").decode("unicode_escape")
        return inner

    return value.split(" #", 1)[0].strip()


def load_api_key() -> str:
    """Load the OpenAI API key from the environment.

    Returns the key string, or prints an error and exits if not set.
    """
    load_dotenv()
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
