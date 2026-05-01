"""Obsidian vault export support for generated study documents."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CONFIG_DIR_NAME = "on-the-record"
CONFIG_FILE_NAME = "config.json"


@dataclass(frozen=True)
class ObsidianConfig:
    """Persistent Obsidian export settings."""

    vault_path: Path
    study_folder: str = ""
    cli_command: str | None = None
    run_cli_after_export: bool = True

    def to_json(self) -> dict[str, object]:
        return {
            "vault_path": str(self.vault_path),
            "study_folder": self.study_folder,
            "cli_command": self.cli_command,
            "run_cli_after_export": self.run_cli_after_export,
        }


def config_file_path() -> Path:
    """Return the user-level OTR configuration file path."""
    return _config_dir() / CONFIG_FILE_NAME


def load_obsidian_config(path: str | Path | None = None) -> ObsidianConfig | None:
    """Load the persisted Obsidian config, if one exists."""
    config_path = Path(path) if path is not None else config_file_path()
    if not config_path.is_file():
        return None

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid config JSON in {config_path}: {exc}") from exc

    obsidian_data = data.get("obsidian")
    if not obsidian_data:
        return None
    if not isinstance(obsidian_data, dict):
        raise ValueError("Invalid Obsidian config: expected an object.")

    return _obsidian_config_from_json(obsidian_data)


def save_obsidian_config(
    config: ObsidianConfig,
    path: str | Path | None = None,
) -> Path:
    """Persist Obsidian export settings and return the config path."""
    config_path = Path(path) if path is not None else config_file_path()
    data: dict[str, object] = {}
    if config_path.is_file():
        data = json.loads(config_path.read_text(encoding="utf-8"))

    data["obsidian"] = config.to_json()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return config_path


def clear_obsidian_config(path: str | Path | None = None) -> bool:
    """Remove persisted Obsidian settings. Return True when something changed."""
    config_path = Path(path) if path is not None else config_file_path()
    if not config_path.is_file():
        return False

    data = json.loads(config_path.read_text(encoding="utf-8"))
    if "obsidian" not in data:
        return False

    del data["obsidian"]
    if data:
        config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    else:
        config_path.unlink()
    return True


def resolve_obsidian_destination(
    source_path: str | Path,
    config: ObsidianConfig,
) -> Path:
    """Return the vault destination path for a study document."""
    source_path = Path(source_path)
    vault_path = config.vault_path.expanduser().resolve(strict=False)
    folder = Path(config.study_folder).expanduser()
    if folder.is_absolute():
        raise ValueError("Obsidian study folder must be relative to the vault.")

    destination_dir = (vault_path / folder).resolve(strict=False)
    if not destination_dir.is_relative_to(vault_path):
        raise ValueError("Obsidian study folder must stay inside the vault.")

    return destination_dir / source_path.name


def export_study_document_to_obsidian(
    source_path: str | Path,
    config: ObsidianConfig,
    *,
    run_cli_hook: bool = True,
) -> Path:
    """Copy a generated study document into the configured Obsidian vault."""
    source_path = Path(source_path)
    destination_path = _unique_path(resolve_obsidian_destination(source_path, config))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)

    if run_cli_hook and config.cli_command and config.run_cli_after_export:
        run_obsidian_cli_hook(destination_path, config)

    return destination_path


def run_obsidian_cli_hook(
    exported_path: str | Path,
    config: ObsidianConfig,
) -> subprocess.CompletedProcess[str]:
    """Run the configured Obsidian CLI hook for an exported file."""
    exported_path = Path(exported_path)
    command = _build_cli_command(config.cli_command or "", exported_path, config)
    if not command:
        raise ValueError("Obsidian CLI command is empty.")

    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def _config_dir() -> Path:
    if os.name == "nt":
        root = os.environ.get("APPDATA")
        if root:
            return Path(root) / CONFIG_DIR_NAME
        return Path.home() / "AppData" / "Roaming" / CONFIG_DIR_NAME

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / CONFIG_DIR_NAME

    root = os.environ.get("XDG_CONFIG_HOME")
    if root:
        return Path(root) / CONFIG_DIR_NAME
    return Path.home() / ".config" / CONFIG_DIR_NAME


def _obsidian_config_from_json(data: dict[str, object]) -> ObsidianConfig:
    vault_path = data.get("vault_path")
    if not isinstance(vault_path, str) or not vault_path.strip():
        raise ValueError("Obsidian vault_path is required.")

    study_folder = data.get("study_folder", "")
    if not isinstance(study_folder, str):
        raise ValueError("Obsidian study_folder must be a string.")

    cli_command = data.get("cli_command")
    if cli_command is not None and not isinstance(cli_command, str):
        raise ValueError("Obsidian cli_command must be a string.")

    run_cli_after_export = data.get("run_cli_after_export", True)
    if not isinstance(run_cli_after_export, bool):
        raise ValueError("Obsidian run_cli_after_export must be true or false.")

    return ObsidianConfig(
        vault_path=Path(vault_path).expanduser(),
        study_folder=study_folder,
        cli_command=cli_command or None,
        run_cli_after_export=run_cli_after_export,
    )


def _build_cli_command(
    command: str,
    exported_path: Path,
    config: ObsidianConfig,
) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        return []

    replacements = {
        "{file}": str(exported_path),
        "{vault}": str(config.vault_path.expanduser()),
    }
    has_placeholder = any(
        placeholder in part for part in parts for placeholder in replacements
    )
    resolved = [
        _replace_placeholders(part, replacements)
        for part in parts
    ]
    if not has_placeholder:
        resolved.append(str(exported_path))
    return resolved


def _replace_placeholders(part: str, replacements: dict[str, str]) -> str:
    for placeholder, value in replacements.items():
        part = part.replace(placeholder, value)
    return part


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not choose a unique path for {path}")
