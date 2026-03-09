from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from .config import resolve_shared_python
from .constants import PARAKEET_BINARY, PARAKEET_MODEL, QWEN_MODELS
from .utils import audio_duration, convert_video_to_wav, file_kind, run_command, which


@dataclass
class TranscriptionResult:
    backend: str
    model: str
    text: str
    success: bool
    total_time: float | None
    audio_duration: float | None
    rtf: float | None
    stderr: str | None = None
    command: list[str] | None = None
    output_paths: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_rtf(total_time: float | None, duration: float | None) -> float | None:
    if total_time is None or not duration:
        return None
    return total_time / duration


def _prepare_input(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if file_kind(path) != "video":
        return path, None
    tmp = tempfile.TemporaryDirectory(prefix="stt-video-")
    wav_path = Path(tmp.name) / f"{path.stem}.wav"
    convert_video_to_wav(path, wav_path)
    return wav_path, tmp


def _run_shared_python(code: str) -> dict[str, Any]:
    shared_python = resolve_shared_python()
    if not shared_python:
        raise RuntimeError("No Python with mlx_audio found. Set STT_SHARED_PYTHON.")
    proc = run_command([shared_python, "-c", code], check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"No JSON payload found:\n{proc.stdout}")


def transcribe_qwen(path: Path, *, model_key: str, language: str = "auto") -> TranscriptionResult:
    prepared_path, tmp = _prepare_input(path)
    model_id = QWEN_MODELS[model_key]
    language_arg = None if language == "auto" else language.title()
    code = f"""
import json
import time
from mlx_audio.stt import load

path = {str(prepared_path)!r}
language = {language_arg!r}
started = time.time()
model = load({model_id!r})
if language is None:
    result = model.generate(path)
else:
    result = model.generate(path, language=language)
elapsed = time.time() - started
print(json.dumps({{
    "text": result.text,
    "elapsed": elapsed,
    "audio_duration": getattr(result, "audio_duration", None),
}}))
"""
    try:
        payload = _run_shared_python(code)
        duration = payload.get("audio_duration") or audio_duration(prepared_path)
        return TranscriptionResult(
            backend=model_key,
            model=model_id,
            text=payload.get("text", ""),
            success=True,
            total_time=payload.get("elapsed"),
            audio_duration=duration,
            rtf=_compute_rtf(payload.get("elapsed"), duration),
        )
    except Exception as exc:
        return TranscriptionResult(
            backend=model_key,
            model=model_id,
            text="",
            success=False,
            total_time=None,
            audio_duration=audio_duration(prepared_path),
            rtf=None,
            stderr=str(exc),
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


def transcribe_mlx_parakeet(path: Path) -> TranscriptionResult:
    prepared_path, tmp = _prepare_input(path)
    code = f"""
import json
import time
from mlx_audio.stt import load

path = {str(prepared_path)!r}
started = time.time()
model = load({PARAKEET_MODEL!r})
result = model.generate(path, chunk_duration=120.0, overlap_duration=15.0)
elapsed = time.time() - started
print(json.dumps({{
    "text": getattr(result, "text", ""),
    "elapsed": elapsed,
    "audio_duration": getattr(result, "audio_duration", None),
}}))
"""
    try:
        payload = _run_shared_python(code)
        duration = payload.get("audio_duration") or audio_duration(prepared_path)
        return TranscriptionResult(
            backend="mlx-parakeet",
            model=PARAKEET_MODEL,
            text=payload.get("text", ""),
            success=True,
            total_time=payload.get("elapsed"),
            audio_duration=duration,
            rtf=_compute_rtf(payload.get("elapsed"), duration),
        )
    except Exception as exc:
        return TranscriptionResult(
            backend="mlx-parakeet",
            model=PARAKEET_MODEL,
            text="",
            success=False,
            total_time=None,
            audio_duration=audio_duration(prepared_path),
            rtf=None,
            stderr=str(exc),
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


def transcribe_parakeet_cli(
    path: Path,
    *,
    output_format: str = "txt",
    output_dir: Path | None = None,
    output_name: str = "transcript",
) -> TranscriptionResult:
    prepared_path, tmp_input = _prepare_input(path)
    managed_dir = output_dir is None
    tmp_output = tempfile.TemporaryDirectory(prefix="stt-asr-") if managed_dir else None
    target_dir = output_dir or Path(tmp_output.name)
    target_dir.mkdir(parents=True, exist_ok=True)

    if which(PARAKEET_BINARY) is None:
        return TranscriptionResult(
            backend="parakeet-mlx",
            model=PARAKEET_MODEL,
            text="",
            success=False,
            total_time=None,
            audio_duration=audio_duration(prepared_path),
            rtf=None,
            stderr="parakeet-mlx not found in PATH",
        )

    command = [
        PARAKEET_BINARY,
        str(prepared_path),
        "--model",
        PARAKEET_MODEL,
        "--output-dir",
        str(target_dir),
        "--output-format",
        output_format,
        "--output-template",
        output_name,
        "--decoding",
        "greedy",
        "--chunk-duration",
        "120",
        "--overlap-duration",
        "15",
    ]

    started = time.time()
    proc = run_command(command, check=False)
    elapsed = time.time() - started

    paths = {
        "txt": target_dir / f"{output_name}.txt",
        "json": target_dir / f"{output_name}.json",
        "srt": target_dir / f"{output_name}.srt",
        "vtt": target_dir / f"{output_name}.vtt",
    }
    text = ""
    if paths["txt"].exists():
        text = paths["txt"].read_text().strip()
    elif paths["json"].exists():
        try:
            data = json.loads(paths["json"].read_text())
            text = str(data.get("text") or data.get("transcript") or "").strip()
        except json.JSONDecodeError:
            text = paths["json"].read_text().strip()
    else:
        for key in ("srt", "vtt"):
            if paths[key].exists():
                lines = []
                for raw_line in paths[key].read_text().splitlines():
                    line = raw_line.strip()
                    if not line or line.isdigit() or "-->" in line:
                        continue
                    lines.append(line)
                text = " ".join(lines).strip()
                break

    duration = audio_duration(prepared_path)
    result = TranscriptionResult(
        backend="parakeet-mlx",
        model=PARAKEET_MODEL,
        text=text,
        success=proc.returncode == 0,
        total_time=elapsed,
        audio_duration=duration,
        rtf=_compute_rtf(elapsed, duration),
        stderr=proc.stderr.strip() or None,
        command=command,
        output_paths={key: str(value) for key, value in paths.items() if value.exists()} or None,
    )

    if managed_dir:
        result.output_paths = None
    if tmp_input is not None:
        tmp_input.cleanup()
    if managed_dir and tmp_output is not None:
        tmp_output.cleanup()
    return result
