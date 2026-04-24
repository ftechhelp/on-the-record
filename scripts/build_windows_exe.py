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


def main() -> int:
    if platform.system() != "Windows":
        print(
            "This build script must be run on Windows to produce a Windows executable.",
            file=sys.stderr,
        )
        return 1

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