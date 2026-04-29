"""Build a Windows executable for on-the-record with PyInstaller."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
ENTRYPOINT = SRC_DIR / "on_the_record" / "__main__.py"
EXE_PATH = PROJECT_ROOT / "dist" / "on-the-record.exe"
SPEAKER_MODEL_DIR = PROJECT_ROOT / "build" / "speechbrain-model"
ENV_FILE = PROJECT_ROOT / ".env"


def _prepare_speaker_model() -> None:
    """Download the SpeechBrain model files that are bundled into the exe."""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
    except ImportError as exc:
        raise RuntimeError(
            "Speaker-recognition build dependencies are missing. "
            "Run `uv sync --extra speaker --group build` before building the executable."
        ) from exc

    print(f"Preparing bundled SpeechBrain speaker model at {SPEAKER_MODEL_DIR}")
    SPEAKER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(SPEAKER_MODEL_DIR),
        local_strategy=LocalStrategy.COPY_SKIP_CACHE,
    )


def main() -> int:
    if platform.system() != "Windows":
        print(
            "This build script must be run on Windows to produce a Windows executable.",
            file=sys.stderr,
        )
        return 1

    try:
        _prepare_speaker_model()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    env_data_args: list[str] = []
    if ENV_FILE.is_file():
        print("Bundling project .env into the executable.")
        env_data_args = ["--add-data", f"{ENV_FILE};."]

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "on-the-record",
        "--paths",
        str(SRC_DIR),
        "--hidden-import",
        "on_the_record.audio",
        "--hidden-import",
        "on_the_record.speaker_recognition",
        "--hidden-import",
        "speechbrain.inference.speaker",
        "--hidden-import",
        "speechbrain.utils.fetching",
        "--hidden-import",
        "torch",
        "--hidden-import",
        "torchaudio",
        "--add-data",
        f"{SPEAKER_MODEL_DIR};speechbrain-model",
        *env_data_args,
        "--collect-all",
        "speechbrain",
        "--collect-all",
        "torch",
        "--collect-all",
        "torchaudio",
        "--collect-all",
        "huggingface_hub",
        "--collect-all",
        "hyperpyyaml",
        "--collect-all",
        "soundfile",
        "--collect-all",
        "sentencepiece",
        "--collect-submodules",
        "soundcard",
        "--collect-data",
        "soundcard",
        "--collect-binaries",
        "soundcard",
        str(ENTRYPOINT),
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"Built executable at {EXE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())