"""Build the Windows system tray app for on-the-record with PyInstaller.

Produces a windowed (no-console) single-file ``On The Record.exe`` that runs
the system tray app. This is separate from ``build_windows_exe.py``, which
still builds the headless ``on-the-record.exe`` CLI.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
ENTRYPOINT = SRC_DIR / "on_the_record" / "tray_main.py"
ICON_PATH = PROJECT_ROOT / "windows" / "assets" / "on-the-record.ico"
EXE_PATH = PROJECT_ROOT / "dist" / "On The Record.exe"
ENV_FILE = PROJECT_ROOT / ".env"


def _ensure_icon() -> None:
    if ICON_PATH.is_file():
        return
    print("Generating app icon …")
    from scripts.generate_icon import main as generate_icon  # type: ignore

    generate_icon()


def main() -> int:
    if platform.system() != "Windows":
        print(
            "This build script must be run on Windows to produce a Windows executable.",
            file=sys.stderr,
        )
        return 1

    # Allow `from scripts.generate_icon import ...` regardless of CWD.
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(SRC_DIR))
    _ensure_icon()

    env_data_args: list[str] = []
    if ENV_FILE.is_file():
        print("Bundling project .env into the executable.")
        env_data_args = ["--add-data", f"{ENV_FILE};."]

    icon_args = ["--icon", str(ICON_PATH)] if ICON_PATH.is_file() else []

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "On The Record",
        "--paths",
        str(SRC_DIR),
        *icon_args,
        "--hidden-import",
        "on_the_record.audio",
        # keyring loads its Windows Credential Manager backend dynamically.
        "--hidden-import",
        "keyring.backends.Windows",
        "--collect-submodules",
        "keyring.backends",
        "--collect-submodules",
        "pystray",
        *env_data_args,
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
