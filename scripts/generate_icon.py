"""Generate the Windows app icon (.ico) for the tray exe and installer.

Reuses :func:`on_the_record.tray.create_icon_image` so the installed exe icon
matches the idle tray icon. Writes a multi-resolution ``.ico`` to
``windows/assets/on-the-record.ico``.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ICON_PATH = PROJECT_ROOT / "windows" / "assets" / "on-the-record.ico"
ICON_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)]


def main() -> int:
    from on_the_record.tray import create_icon_image

    ICON_PATH.parent.mkdir(parents=True, exist_ok=True)
    image = create_icon_image(recording=False).resize((256, 256))
    image.save(ICON_PATH, format="ICO", sizes=ICON_SIZES)
    print(f"Wrote icon to {ICON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
