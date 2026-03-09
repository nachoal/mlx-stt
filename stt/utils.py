from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


def run_command(
    cmd: list[str],
    *,
    check: bool = True,
    live: bool = False,
) -> subprocess.CompletedProcess[str]:
    if live:
        proc = subprocess.run(cmd, text=True)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        stderr = (proc.stderr or "").strip() or (proc.stdout or "").strip()
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{stderr}")
    return proc


def which(name: str) -> str | None:
    return shutil.which(name)


def file_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".flac", ".aac", ".webm"}:
        return "audio"
    if ext in {".mp4", ".mkv", ".mov", ".avi", ".m4v"}:
        return "video"
    return "other"


def audio_duration(path: Path) -> float | None:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def convert_video_to_wav(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
    )
    return output_path


def json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
