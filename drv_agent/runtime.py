from __future__ import annotations

import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


class AdapterUnavailableError(RuntimeError):
    pass


def resolve_path(workspace_root: str | Path, candidate: str | None) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate)
    if path.is_absolute():
        return path
    return Path(workspace_root).joinpath(path).resolve()


def ensure_sys_path(path: str | Path | None) -> None:
    if path is None:
        return
    path_str = str(Path(path).resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def import_or_raise(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise AdapterUnavailableError(f"Missing dependency '{module_name}'. {install_hint}") from exc


def parse_json_like(text: str) -> dict[str, Any]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def run_command_template(
    template: str,
    substitutions: dict[str, Any],
    *,
    cwd: str | Path | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    formatted = template.format(**substitutions)
    command = shlex.split(formatted)
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
