from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from .config import resolve_parakeet_binary, resolve_shared_python
from .constants import DIARIZATION_MODEL, DIARIZATION_REVISION, PARAKEET_MODEL, QWEN_MODELS
from .utils import audio_duration, convert_media_to_wav, is_pcm16k_mono_wav, needs_wav_normalization, run_command

WORKER = Path(__file__).resolve().parent / "runtime_worker.py"


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
    segments: list[dict[str, Any]] | None = None
    speakers: int | None = None
    timings: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_rtf(total_time: float | None, duration: float | None) -> float | None:
    if total_time is None or not duration:
        return None
    return total_time / duration


def _prepare_input(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if not needs_wav_normalization(path):
        return path, None
    tmp = tempfile.TemporaryDirectory(prefix="stt-audio-")
    wav_path = Path(tmp.name) / f"{path.stem}.wav"
    convert_media_to_wav(path, wav_path)
    return wav_path, tmp


def _last_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"No JSON payload found:\n{stdout}")


def _run_worker(request: dict[str, Any]) -> dict[str, Any]:
    shared_python = resolve_shared_python()
    if not shared_python:
        raise RuntimeError("No Python with mlx_audio found. Run `stt setup` or set STT_SHARED_PYTHON.")
    proc = run_command([shared_python, str(WORKER), json.dumps(request)], check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return _last_json_line(proc.stdout)


def _prepare_pcm16k(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if is_pcm16k_mono_wav(path):
        return path, None
    tmp = tempfile.TemporaryDirectory(prefix="stt-audio-")
    wav_path = Path(tmp.name) / f"{path.stem}.wav"
    convert_media_to_wav(path, wav_path)
    return wav_path, tmp


def transcribe_in_runtime(
    path: Path,
    *,
    backend: str,
    model: str,
    language: str | None = None,
    segment: bool = True,
    diarize: bool = False,
) -> TranscriptionResult:
    """Transcribes in the runtime worker, per speaker-pure speech segment (diarizer-planned) or as one file."""
    prepared_path = path
    tmp: tempfile.TemporaryDirectory[str] | None = None
    try:
        prepared_path, tmp = _prepare_pcm16k(path)
        payload = _run_worker(
            {
                "audio": str(prepared_path),
                "backend": backend,
                "asr_model": model,
                "language": language,
                "segment": segment or diarize,
                "diarize": diarize,
                "diarization_model": DIARIZATION_MODEL,
                "diarization_revision": DIARIZATION_REVISION,
            }
        )
        duration = payload.get("audio_duration") or audio_duration(prepared_path)
        return TranscriptionResult(
            backend=backend,
            model=model,
            text=payload.get("text", ""),
            success=True,
            total_time=payload.get("elapsed"),
            audio_duration=duration,
            rtf=_compute_rtf(payload.get("elapsed"), duration),
            segments=payload.get("segments"),
            speakers=payload.get("speakers"),
            timings=payload.get("timings"),
        )
    except Exception as exc:
        return TranscriptionResult(
            backend=backend,
            model=model,
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


def transcribe_qwen(
    path: Path,
    *,
    model_key: str,
    language: str = "auto",
    diarize: bool = False,
    segment: bool = False,
) -> TranscriptionResult:
    # Whole-file decoding measured better for Qwen3 on Spanish; speaker labels and subtitle cues need
    # per-turn segments.
    language_arg = None if language == "auto" else language.title()
    return transcribe_in_runtime(
        path, backend=model_key, model=QWEN_MODELS[model_key], language=language_arg, segment=segment, diarize=diarize
    )


def transcribe_mlx_parakeet(path: Path, *, language: str = "auto", diarize: bool = False) -> TranscriptionResult:
    # Always per speech turn: long windows make Parakeet v3 drop whole sentences. The worker narrows
    # the windows further for non-English hints, where Parakeet otherwise drifts into English.
    return transcribe_in_runtime(
        path, backend="mlx-parakeet", model=PARAKEET_MODEL, language=language, segment=True, diarize=diarize
    )


def transcribe_parakeet_cli(
    path: Path,
    *,
    output_format: str = "txt",
    output_dir: Path | None = None,
    output_name: str = "transcript",
) -> TranscriptionResult:
    prepared_path = path
    tmp_input: tempfile.TemporaryDirectory[str] | None = None
    managed_dir = output_dir is None
    tmp_output = tempfile.TemporaryDirectory(prefix="stt-asr-") if managed_dir else None
    target_dir = output_dir or Path(tmp_output.name)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        prepared_path, tmp_input = _prepare_input(path)
        parakeet_binary = resolve_parakeet_binary()
        if parakeet_binary is None:
            return TranscriptionResult(
                backend="parakeet-mlx",
                model=PARAKEET_MODEL,
                text="",
                success=False,
                total_time=None,
                audio_duration=audio_duration(prepared_path),
                rtf=None,
                stderr="parakeet-mlx not found. Run `stt setup` or set STT_PARAKEET_BINARY.",
            )

        command = [
            parakeet_binary,
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
        return result
    except Exception as exc:
        return TranscriptionResult(
            backend="parakeet-mlx",
            model=PARAKEET_MODEL,
            text="",
            success=False,
            total_time=None,
            audio_duration=audio_duration(prepared_path),
            rtf=None,
            stderr=str(exc),
        )
    finally:
        if tmp_input is not None:
            tmp_input.cleanup()
        if managed_dir and tmp_output is not None:
            tmp_output.cleanup()
