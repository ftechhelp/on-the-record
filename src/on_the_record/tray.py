"""Windows system tray app for on-the-record.

Provides a tray icon with start/stop/settings controls, mirroring the macOS
menu bar app (``macos/OnTheRecordMenuBar``). The Python recording engine runs
*in-process* on a background thread — reusing :func:`build_config` and
:class:`RecordingSession` — so there is no separate engine process or IPC.

Threading model
---------------
``tkinter`` must run on the main thread, so :func:`main` owns a hidden Tk root
and calls ``root.mainloop()``. The pystray icon runs detached on a background
thread via ``icon.run_detached()``. Menu callbacks fire on the pystray thread
and marshal any GUI work onto the Tk thread with ``root.after(0, ...)``. The
recording itself runs on its own daemon thread.

Secrets (OpenAI / Gemini API keys) live in Windows Credential Manager via the
``keyring`` library; non-secret options live in ``settings.json`` under
``%APPDATA%/On The Record``.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import keyring
import pystray
from PIL import Image, ImageDraw

from on_the_record.app_engine import build_config, build_study_options
from on_the_record.config import load_dotenv
from on_the_record.obsidian import (
    ObsidianConfig,
    clear_obsidian_config,
    export_study_document_to_obsidian,
    load_obsidian_config,
    save_obsidian_config,
)
from on_the_record.recording import RecordingSession
from on_the_record.study import DEFAULT_GEMINI_MODEL, write_named_study_document

APP_NAME = "On The Record"
KEYRING_SERVICE = "on-the-record"
OPENAI_ACCOUNT = "OPENAI_API_KEY"
GEMINI_ACCOUNT = "GEMINI_API_KEY"

FORMATS = ("txt", "md", "json")
SOURCE_MODES = ("both", "system", "microphone")
MIN_CHUNK_SECONDS = 5


# --------------------------------------------------------------------------- #
# Settings + secret storage
# --------------------------------------------------------------------------- #


@dataclass
class AppSettings:
    """Non-secret recording options persisted to ``settings.json``."""

    output_directory: str = ""
    format: str = "txt"
    source_mode: str = "both"  # both | system | microphone
    chunk_seconds: int = 15
    diarize: bool = True
    study_doc_enabled: bool = False
    gemini_model: str = DEFAULT_GEMINI_MODEL

    @property
    def includes_system_audio(self) -> bool:
        return self.source_mode in ("both", "system")

    @property
    def includes_microphone(self) -> bool:
        return self.source_mode in ("both", "microphone")


def _config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / APP_NAME


def _settings_path() -> Path:
    return _config_dir() / "settings.json"


def _log(message: str) -> None:
    """Append a diagnostic line to ``%APPDATA%/On The Record/tray.log``.

    The windowed exe has no console, so this is how errors become visible.
    """
    try:
        path = _config_dir() / "tray.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
    except OSError:
        pass


def _default_output_directory() -> Path:
    return Path.home() / "Documents" / APP_NAME


def load_settings() -> AppSettings:
    """Load settings from disk, falling back to defaults for missing keys."""
    settings = AppSettings(output_directory=str(_default_output_directory()))
    path = _settings_path()
    if not path.is_file():
        return settings
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return settings
    known = {field for field in asdict(settings)}
    for key, value in data.items():
        if key in known:
            setattr(settings, key, value)
    if not settings.output_directory:
        settings.output_directory = str(_default_output_directory())
    settings.chunk_seconds = max(MIN_CHUNK_SECONDS, int(settings.chunk_seconds))
    return settings


def save_settings(settings: AppSettings) -> None:
    """Persist non-secret settings to ``settings.json``."""
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


def load_secret(account: str) -> str:
    """Read an API key from Windows Credential Manager (empty string if unset)."""
    try:
        return keyring.get_password(KEYRING_SERVICE, account) or ""
    except keyring.errors.KeyringError:
        return ""


def effective_secret(account: str) -> str:
    """Return the key actually in effect for *account*.

    Credential Manager wins, then a value from the bundled/project ``.env`` or
    environment — the same precedence as ``build_config`` /
    ``build_study_options``. This lets the Settings window reflect keys baked
    into a bundled ``.env`` instead of showing a misleading blank field.
    (Account names match the env var names, e.g. ``OPENAI_API_KEY``.)
    """
    stored = load_secret(account)
    if stored:
        return stored
    load_dotenv()
    return os.environ.get(account, "").strip()


def save_secret(account: str, value: str) -> None:
    """Store (or clear) an API key in Windows Credential Manager."""
    value = (value or "").strip()
    try:
        if value:
            keyring.set_password(KEYRING_SERVICE, account, value)
        else:
            try:
                keyring.delete_password(KEYRING_SERVICE, account)
            except keyring.errors.PasswordDeleteError:
                pass
    except keyring.errors.KeyringError:
        pass


def _make_output_path(settings: AppSettings) -> str:
    directory = settings.output_directory or str(_default_output_directory())
    Path(directory).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(directory) / f"transcript_{timestamp}.{settings.format}")


def _build_payload(
    settings: AppSettings,
    openai_key: str,
    gemini_key: str,
    output_path: str,
) -> dict[str, Any]:
    """Assemble the start-recording payload consumed by :func:`build_config`."""
    return {
        "api_key": openai_key,
        "output_path": output_path,
        "format": settings.format,
        "chunk_seconds": settings.chunk_seconds,
        "include_system_audio": settings.includes_system_audio,
        "include_microphone": settings.includes_microphone,
        "diarize": settings.diarize,
        "study_doc_enabled": settings.study_doc_enabled,
        "gemini_api_key": gemini_key,
        "gemini_model": settings.gemini_model,
    }


# --------------------------------------------------------------------------- #
# Recording controller (in-process)
# --------------------------------------------------------------------------- #

class RecordingController:
    """Drive a :class:`RecordingSession` on a background thread.

    Mirrors ``JsonLineEngine._handle_start_recording`` / ``_run_session`` from
    :mod:`on_the_record.app_engine`, but invokes the engine in-process and
    forwards events through a plain callback instead of JSON-lines.
    """

    def __init__(self, on_event) -> None:
        self._on_event = on_event
        self._session: RecordingSession | None = None
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self.last_output_path: str | None = None

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stop_requested(self) -> bool:
        """True once a stop has been asked for but recording is still winding down."""
        return self._stop_requested and self.is_recording

    def start(self, payload: dict[str, Any], study_options: dict[str, Any]) -> None:
        if self.is_recording:
            raise RuntimeError("Recording is already running.")
        config = build_config(payload)  # raises ValueError if the key is missing
        session = RecordingSession(config, event_callback=self._on_event)
        self._session = session
        self._stop_requested = False
        self.last_output_path = config.output_path
        self._thread = threading.Thread(
            target=self._run,
            args=(session, study_options),
            name="tray-recording",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_requested = True
        if self._session is not None:
            self._session.request_stop()

    def _run(self, session: RecordingSession, study_options: dict[str, Any]) -> None:
        try:
            result = session.run()
        except Exception as exc:  # surfaced to the user as a notification
            self._on_event("recording_error", {"error": str(exc)})
            return
        finally:
            if self._session is session:
                self._session = None
        self._maybe_generate_study_document(result, study_options)
        self._on_event(
            "recording_finished",
            {
                "output_path": result.output_path,
                "total_segments": result.total_segments,
                "elapsed_seconds": result.elapsed_seconds,
            },
        )

    def _maybe_generate_study_document(self, result, options: dict[str, Any]) -> None:
        # Same guards as JsonLineEngine._maybe_generate_study_document.
        if not options["enabled"] or result.total_segments == 0 or not options["api_key"]:
            return
        self.generate_study_document(result.output_path, options)

    def generate_study_document(
        self, transcript_path: str, options: dict[str, Any]
    ) -> None:
        """Generate a study document for *transcript_path*, then export to Obsidian.

        Shared by the post-recording flow and the manual "Generate Study
        Document…" menu action, so a transcript can be turned into a note even
        when the automatic step did not run. Safe to call from any thread — it
        only emits events.
        """
        if not options["api_key"]:
            self._on_event(
                "study_doc_failed", {"error": "Gemini API key is not set."}
            )
            return
        self._on_event("study_doc_started", {"transcript_path": transcript_path})
        try:
            written = write_named_study_document(
                transcript_path,
                options["output_path"],
                api_key=options["api_key"],
                model=options["model"],
            )
        except Exception as exc:
            self._on_event("study_doc_failed", {"error": str(exc)})
            return
        self._on_event("study_doc_written", {"output_path": str(written)})
        self._maybe_export_to_obsidian(written)

    def _maybe_export_to_obsidian(self, study_path) -> None:
        """Copy the study document into the configured Obsidian vault, if any.

        Mirrors ``cli._maybe_export_study_document_to_obsidian``: a saved vault
        config means export; no config means skip.
        """
        try:
            config = load_obsidian_config()
        except Exception as exc:  # corrupt config.json
            self._on_event("obsidian_export_failed", {"error": f"config error: {exc}"})
            return
        if config is None:
            return
        try:
            exported = export_study_document_to_obsidian(study_path, config)
        except Exception as exc:
            self._on_event("obsidian_export_failed", {"error": str(exc)})
            return
        self._on_event("obsidian_exported", {"output_path": str(exported)})


# --------------------------------------------------------------------------- #
# Tray icon imagery
# --------------------------------------------------------------------------- #


def create_icon_image(recording: bool = False) -> Image.Image:
    """Draw the tray icon — a blue disc when idle, red with a dot when recording."""
    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    disc = (24, 119, 242) if not recording else (220, 38, 38)
    draw.ellipse((4, 4, size - 4, size - 4), fill=disc)
    if recording:
        # White centre dot to read as a "REC" indicator.
        draw.ellipse((size // 2 - 10, size // 2 - 10, size // 2 + 10, size // 2 + 10), fill=(255, 255, 255))
    else:
        # Simple waveform tick marks.
        for offset, height in ((-14, 10), (-4, 20), (6, 14), (16, 8)):
            x = size // 2 + offset
            draw.rectangle((x, size // 2 - height, x + 4, size // 2 + height), fill=(255, 255, 255))
    return image


# --------------------------------------------------------------------------- #
# Tray application
# --------------------------------------------------------------------------- #


class TrayApp:
    """Owns the tray icon, the hidden Tk root, and the recording controller."""

    def __init__(self) -> None:
        import tkinter as tk

        self.settings = load_settings()
        self.controller = RecordingController(self._on_recording_event)
        self._settings_window: Any = None

        # Hidden Tk root lives on the main thread. tkinter is not thread-safe, so
        # menu callbacks (which fire on the pystray thread) cannot touch Tk
        # directly — not even via root.after(). They enqueue work here, and the
        # main thread drains the queue from a periodic poll it schedules itself.
        self._task_queue: queue.Queue = queue.Queue()
        self._root = tk.Tk()
        self._root.withdraw()

        self.icon = pystray.Icon(
            "on-the-record",
            icon=create_icon_image(recording=False),
            title=APP_NAME,
            menu=self._build_menu(),
        )

    # -- menu ---------------------------------------------------------------- #

    def _build_menu(self) -> "pystray.Menu":
        Item = pystray.MenuItem
        return pystray.Menu(
            Item(
                "Start Recording",
                self._on_start,
                enabled=lambda _item: not self.controller.is_recording,
                default=True,
            ),
            Item(
                lambda _item: "Stopping…" if self.controller.stop_requested else "Stop Recording",
                self._on_stop,
                enabled=lambda _item: self.controller.is_recording and not self.controller.stop_requested,
            ),
            pystray.Menu.SEPARATOR,
            Item("Settings…", self._on_settings),
            Item("List Devices", self._on_list_devices),
            Item("Open Last Transcript", self._on_open_last, enabled=lambda _item: bool(self.controller.last_output_path)),
            Item("Generate Study Document…", self._on_generate_study),
            pystray.Menu.SEPARATOR,
            Item("Quit", self._on_quit),
        )

    # -- helpers ------------------------------------------------------------- #

    def _ui(self, func) -> None:
        """Schedule *func* to run on the Tk (main) thread.

        Safe to call from any thread — it only touches the thread-safe queue;
        :meth:`_pump_queue` (running on the main thread) invokes *func*.
        """
        self._task_queue.put(func)

    def _pump_queue(self) -> None:
        """Drain queued GUI work on the main thread, then re-arm the poll.

        A failing task must not kill the pump (that would freeze every menu
        action), so each task is isolated and its error surfaced to the user.
        """
        try:
            while True:
                task = self._task_queue.get_nowait()
                try:
                    task()
                except Exception as exc:  # one bad task must not stop the pump
                    _log(f"GUI task error: {exc!r}")
                    self._notify(f"Error: {exc}")
        except queue.Empty:
            pass
        self._root.after(100, self._pump_queue)

    def _notify(self, message: str, title: str = APP_NAME) -> None:
        try:
            self.icon.notify(message, title)
        except Exception:
            pass

    def _show_error(self, message: str) -> None:
        """Surface an error in a modal dialog instead of a transient balloon.

        Balloon notifications show one at a time, so a failure message can be
        clobbered by the "Recording stopped" balloon that follows it before the
        user notices. A dialog stays up until dismissed. Safe to call from any
        thread — it logs immediately and marshals the dialog onto the Tk thread.
        """
        _log(f"error shown to user: {message}")

        def show() -> None:
            from tkinter import messagebox

            messagebox.showerror(APP_NAME, message)

        self._ui(show)

    def _refresh(self, recording: bool | None = None) -> None:
        if recording is not None:
            self.icon.icon = create_icon_image(recording=recording)
            self.icon.title = f"{APP_NAME} — Recording" if recording else APP_NAME
        self.icon.update_menu()

    # -- recording event callback (runs on the recording thread) ------------- #

    def _on_recording_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "recording_started":
            self._refresh(recording=True)
            self._notify("Recording started.")
        elif event_type == "recording_finished":
            self._refresh(recording=False)
            segments = payload.get("total_segments", 0)
            self._notify(f"Recording stopped — {segments} segment(s) written.")
        elif event_type == "recording_error":
            self._refresh(recording=False)
            self._show_error(f"Recording failed:\n\n{payload.get('error', 'unknown')}")
        elif event_type == "study_doc_started":
            self._notify("Creating study document with Gemini…")
        elif event_type == "study_doc_written":
            self._notify("Study document written.")
        elif event_type == "study_doc_failed":
            self._show_error(
                f"Study document generation failed:\n\n{payload.get('error', 'unknown')}"
            )
        elif event_type == "obsidian_exported":
            self._notify("Study document exported to Obsidian.")
        elif event_type == "obsidian_export_failed":
            self._show_error(
                f"Obsidian export failed:\n\n{payload.get('error', 'unknown')}"
            )

    # -- menu callbacks (run on the pystray thread) -------------------------- #

    def _on_start(self, _icon=None, _item=None) -> None:
        if self.controller.is_recording:
            return
        openai_key = load_secret(OPENAI_ACCOUNT)
        gemini_key = load_secret(GEMINI_ACCOUNT)
        output_path = _make_output_path(self.settings)
        payload = _build_payload(self.settings, openai_key, gemini_key, output_path)
        try:
            self.controller.start(payload, build_study_options(payload))
        except ValueError:
            self._notify("Set your OpenAI API key in Settings before recording.")
            self._ui(self._open_settings_window)
        except Exception as exc:
            self._notify(f"Could not start recording: {exc}")
        else:
            self._refresh()

    def _on_stop(self, _icon=None, _item=None) -> None:
        if not self.controller.is_recording or self.controller.stop_requested:
            return
        self.controller.stop()
        # Capture stops within ~1s, but the final chunk still has to be
        # transcribed — tell the user the command registered so it doesn't feel
        # stuck. "recording_finished" resets the title when it actually ends.
        self.icon.title = f"{APP_NAME} — Stopping…"
        self._notify("Stopping — finishing the current chunk…")
        self._refresh()

    def _on_settings(self, _icon=None, _item=None) -> None:
        self._ui(self._open_settings_window)

    def _on_list_devices(self, _icon=None, _item=None) -> None:
        self._ui(self._show_devices)

    def _on_open_last(self, _icon=None, _item=None) -> None:
        path = self.controller.last_output_path
        if path and Path(path).is_file():
            try:
                os.startfile(path)  # noqa: S606 — Windows-only, opens in default app
            except OSError as exc:
                self._notify(f"Could not open transcript: {exc}")
        else:
            self._notify("No transcript available yet.")

    def _on_generate_study(self, _icon=None, _item=None) -> None:
        self._ui(self._prompt_and_generate_study)

    def _on_quit(self, _icon=None, _item=None) -> None:
        self.controller.stop()
        self.icon.stop()
        self._ui(self._root.destroy)

    # -- Tk windows (run on the main thread) --------------------------------- #

    def _show_devices(self) -> None:
        from tkinter import messagebox

        try:
            from on_the_record import audio

            devices = audio.list_devices()
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not list devices:\n{exc}")
            return
        lines = [
            f"{'[loopback] ' if d.is_loopback else ''}{d.name}" for d in devices
        ] or ["No audio devices found."]
        messagebox.showinfo(f"{APP_NAME} — Audio Devices", "\n".join(lines))

    def _prompt_and_generate_study(self) -> None:
        """Pick a transcript (Tk thread), then generate its study document.

        The Gemini call is network-bound, so it runs on a daemon thread to keep
        the Tk loop responsive; progress and results surface as notifications via
        the controller's event callback. Obsidian export follows automatically
        when a vault is configured.
        """
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Select a transcript",
            initialdir=self.settings.output_directory or str(Path.home()),
            filetypes=[
                ("Transcripts", "*.txt *.md *.json"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        gemini_key = effective_secret(GEMINI_ACCOUNT)
        if not gemini_key:
            self._notify("Set your Gemini API key in Settings to generate study documents.")
            self._open_settings_window()
            return

        options = {
            "enabled": True,
            "api_key": gemini_key,
            "model": self.settings.gemini_model,
            "output_path": None,
        }
        # generate_study_document emits "study_doc_started", which notifies.
        threading.Thread(
            target=self.controller.generate_study_document,
            args=(path, options),
            name="tray-study-doc",
            daemon=True,
        ).start()

    def _open_settings_window(self) -> None:
        if self._settings_window is not None and self._settings_window.exists():
            self._settings_window.focus()
            return
        try:
            self._settings_window = SettingsWindow(
                self._root,
                self.settings,
                on_saved=self._on_settings_saved,
            )
        except Exception as exc:
            _log(f"Failed to open Settings window: {exc!r}")
            self._notify(f"Could not open Settings: {exc}")
            raise

    def _on_settings_saved(self, settings: AppSettings) -> None:
        self.settings = settings
        self._refresh()

    # -- lifecycle ----------------------------------------------------------- #

    def run(self) -> None:
        # pystray's icon.run() blocks pumping its own Win32 message loop, so run
        # it on a dedicated thread and let Tk's mainloop own the main thread.
        # (icon.run_detached() instead blocks the main thread on Windows, which
        # starves Tk's mainloop and the _pump_queue poll — Settings never opens.)
        icon_thread = threading.Thread(target=self.icon.run, name="tray-icon", daemon=True)
        icon_thread.start()
        self._root.after(100, self._pump_queue)  # start polling for GUI work
        self._root.mainloop()


# --------------------------------------------------------------------------- #
# Settings window
# --------------------------------------------------------------------------- #


class SettingsWindow:
    """A tkinter settings dialog mirroring the macOS SettingsWindowController."""

    def __init__(self, parent, settings: AppSettings, *, on_saved) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._on_saved = on_saved
        self._win = tk.Toplevel(parent)
        self._win.title(f"{APP_NAME} — Settings")
        self._win.resizable(False, False)

        frame = ttk.Frame(self._win, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        row = 0

        # Variables (seeded from current settings + Credential Manager).
        self.openai_var = tk.StringVar(value=effective_secret(OPENAI_ACCOUNT))
        self.gemini_var = tk.StringVar(value=effective_secret(GEMINI_ACCOUNT))
        self.output_var = tk.StringVar(value=settings.output_directory)
        self.format_var = tk.StringVar(value=settings.format)
        self.source_var = tk.StringVar(value=settings.source_mode)
        self.chunk_var = tk.IntVar(value=settings.chunk_seconds)
        self.diarize_var = tk.BooleanVar(value=settings.diarize)
        self.study_var = tk.BooleanVar(value=settings.study_doc_enabled)
        self.gemini_model_var = tk.StringVar(value=settings.gemini_model)

        # Obsidian export config is stored separately (shared with the CLI).
        try:
            obsidian = load_obsidian_config()
        except Exception:
            obsidian = None
        self.obsidian_vault_var = tk.StringVar(
            value=str(obsidian.vault_path) if obsidian else ""
        )
        self.obsidian_folder_var = tk.StringVar(
            value=obsidian.study_folder if obsidian else ""
        )

        def add_label(text: str) -> None:
            nonlocal row
            ttk.Label(frame, text=text).grid(row=row, column=0, sticky="w", pady=(6, 0))

        add_label("OpenAI API key")
        ttk.Entry(frame, textvariable=self.openai_var, show="•", width=44).grid(
            row=row, column=1, columnspan=2, sticky="we", pady=(6, 0)
        )
        row += 1

        add_label("Gemini API key")
        ttk.Entry(frame, textvariable=self.gemini_var, show="•", width=44).grid(
            row=row, column=1, columnspan=2, sticky="we", pady=(6, 0)
        )
        row += 1

        add_label("Output folder")
        ttk.Entry(frame, textvariable=self.output_var, width=36).grid(
            row=row, column=1, sticky="we", pady=(6, 0)
        )
        ttk.Button(frame, text="Browse…", command=self._browse).grid(
            row=row, column=2, sticky="we", padx=(6, 0), pady=(6, 0)
        )
        row += 1

        add_label("Format")
        ttk.Combobox(
            frame, textvariable=self.format_var, values=list(FORMATS), state="readonly", width=12
        ).grid(row=row, column=1, sticky="w", pady=(6, 0))
        row += 1

        add_label("Audio source")
        ttk.Combobox(
            frame, textvariable=self.source_var, values=list(SOURCE_MODES), state="readonly", width=12
        ).grid(row=row, column=1, sticky="w", pady=(6, 0))
        row += 1

        add_label("Chunk seconds")
        ttk.Spinbox(frame, from_=MIN_CHUNK_SECONDS, to=120, textvariable=self.chunk_var, width=10).grid(
            row=row, column=1, sticky="w", pady=(6, 0)
        )
        row += 1

        ttk.Checkbutton(frame, text="Speaker diarization", variable=self.diarize_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        row += 1

        ttk.Checkbutton(
            frame, text="Generate Gemini study document", variable=self.study_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))
        row += 1

        add_label("Gemini model")
        ttk.Entry(frame, textvariable=self.gemini_model_var, width=28).grid(
            row=row, column=1, columnspan=2, sticky="we", pady=(6, 0)
        )
        row += 1

        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="we", pady=(12, 2)
        )
        row += 1
        ttk.Label(frame, text="Obsidian export (study docs) — leave vault empty to disable").grid(
            row=row, column=0, columnspan=3, sticky="w"
        )
        row += 1

        add_label("Vault folder")
        ttk.Entry(frame, textvariable=self.obsidian_vault_var, width=36).grid(
            row=row, column=1, sticky="we", pady=(6, 0)
        )
        ttk.Button(frame, text="Browse…", command=self._browse_vault).grid(
            row=row, column=2, sticky="we", padx=(6, 0), pady=(6, 0)
        )
        row += 1

        add_label("Study subfolder")
        ttk.Entry(frame, textvariable=self.obsidian_folder_var, width=36).grid(
            row=row, column=1, columnspan=2, sticky="we", pady=(6, 0)
        )
        row += 1

        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=3, sticky="e", pady=(16, 0))
        ttk.Button(buttons, text="Cancel", command=self._win.destroy).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(buttons, text="Save", command=self._save).grid(row=0, column=1)

        frame.columnconfigure(1, weight=1)
        # The root is withdrawn, so make this a normal top-level (taskbar button,
        # normal stacking) and force it to the foreground — otherwise it can open
        # hidden behind the active window.
        self._win.update_idletasks()
        self._win.deiconify()
        self._win.lift()
        self._win.attributes("-topmost", True)
        self._win.after(400, lambda: self._win.attributes("-topmost", False))
        self._win.focus_force()

    def _browse(self) -> None:
        from tkinter import filedialog

        chosen = filedialog.askdirectory(initialdir=self.output_var.get() or str(Path.home()))
        if chosen:
            self.output_var.set(chosen)

    def _browse_vault(self) -> None:
        from tkinter import filedialog

        chosen = filedialog.askdirectory(
            initialdir=self.obsidian_vault_var.get() or str(Path.home())
        )
        if chosen:
            self.obsidian_vault_var.set(chosen)

    def _save(self) -> None:
        from tkinter import messagebox

        try:
            chunk = max(MIN_CHUNK_SECONDS, int(self.chunk_var.get()))
        except (TypeError, ValueError):
            chunk = MIN_CHUNK_SECONDS

        settings = AppSettings(
            output_directory=self.output_var.get().strip() or str(_default_output_directory()),
            format=self.format_var.get() or "txt",
            source_mode=self.source_var.get() or "both",
            chunk_seconds=chunk,
            diarize=bool(self.diarize_var.get()),
            study_doc_enabled=bool(self.study_var.get()),
            gemini_model=self.gemini_model_var.get().strip() or DEFAULT_GEMINI_MODEL,
        )
        try:
            save_settings(settings)
            save_secret(OPENAI_ACCOUNT, self.openai_var.get())
            save_secret(GEMINI_ACCOUNT, self.gemini_var.get())
            self._save_obsidian()
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not save settings:\n{exc}")
            return
        self._on_saved(settings)
        self._win.destroy()

    def _save_obsidian(self) -> None:
        """Persist Obsidian export config; an empty vault disables export.

        Reuses the CLI's config store and preserves any cli_command /
        run_cli_after_export already configured there.
        """
        vault = self.obsidian_vault_var.get().strip()
        folder = self.obsidian_folder_var.get().strip()
        if not vault:
            clear_obsidian_config()
            return
        try:
            existing = load_obsidian_config()
        except Exception:
            existing = None
        save_obsidian_config(
            ObsidianConfig(
                vault_path=Path(vault).expanduser(),
                study_folder=folder,
                cli_command=existing.cli_command if existing else None,
                run_cli_after_export=existing.run_cli_after_export if existing else True,
            )
        )

    def exists(self) -> bool:
        try:
            return bool(self._win.winfo_exists())
        except Exception:
            return False

    def focus(self) -> None:
        try:
            self._win.lift()
            self._win.focus_force()
        except Exception:
            pass


def main() -> None:
    """Entry point for the ``on-the-record-tray`` script and the bundled exe."""
    try:
        TrayApp().run()
    except Exception as exc:
        _log(f"Fatal: {exc!r}")
        raise


if __name__ == "__main__":
    main()
