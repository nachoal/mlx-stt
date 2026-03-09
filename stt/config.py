from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _python_has_module(python_executable: str, module: str) -> bool:
    try:
        proc = subprocess.run(
            [python_executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return proc.returncode == 0


def resolve_shared_python() -> str | None:
    explicit = os.environ.get("STT_SHARED_PYTHON")
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return str(path)
        return None

    for candidate in ("python3", "python"):
        if _python_has_module(candidate, "mlx_audio"):
            return candidate
    return None


def env_sample(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser()
    if path.exists():
        return path
    return None
