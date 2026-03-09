from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any


def stt_home() -> Path:
    explicit = os.environ.get("STT_HOME")
    if explicit:
        return Path(explicit).expanduser()
    if os.uname().sysname == "Darwin":
        return Path("~/Library/Application Support/mlx-stt").expanduser()
    return Path("~/.local/share/mlx-stt").expanduser()


def config_path() -> Path:
    return stt_home() / "config.json"


def default_runtime_dir() -> Path:
    return stt_home() / "runtime"


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def save_config(data: dict[str, Any]) -> Path:
    home = stt_home()
    home.mkdir(parents=True, exist_ok=True)
    path = config_path()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


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

    config = load_config()
    configured = config.get("runtime_python")
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return str(path)

    for candidate in ("python3", "python"):
        if _python_has_module(candidate, "mlx_audio"):
            return candidate
    return None


def resolve_parakeet_binary() -> str | None:
    explicit = os.environ.get("STT_PARAKEET_BINARY")
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return str(path)
        return None

    config = load_config()
    configured = config.get("parakeet_binary")
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return str(path)

    from .utils import which

    return which("parakeet-mlx")


def env_sample(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser()
    if path.exists():
        return path
    return None
