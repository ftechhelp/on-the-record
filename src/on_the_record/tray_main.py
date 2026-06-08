"""PyInstaller entry point for the Windows system tray app.

PyInstaller needs a concrete script file as its entry point; this keeps that
entry trivial and delegates to :func:`on_the_record.tray.main`.
"""

from on_the_record.tray import main

if __name__ == "__main__":
    main()
