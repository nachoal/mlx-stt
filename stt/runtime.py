from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
import sys

from .config import default_runtime_dir, save_config, stt_home
from .constants import PARAKEET_MODEL, QWEN_MODELS
from .utils import run_command, which


@dataclass
class SetupResult:
    home_dir: str
    runtime_dir: str
    runtime_python: str
    parakeet_binary: str
    config_path: str
    ffmpeg: str | None
    downloaded_models: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _runtime_python(runtime_dir: Path) -> Path:
    return runtime_dir / "bin" / "python"


def _runtime_parakeet(runtime_dir: Path) -> Path:
    return runtime_dir / "bin" / "parakeet-mlx"


def _log(message: str) -> None:
    print(f"[stt setup] {message}", file=sys.stderr, flush=True)


def _warm_model(runtime_python: Path, model_id: str) -> None:
    code = (
        "from mlx_audio.stt import load\n"
        f"load({model_id!r})\n"
        "print('ok')\n"
    )
    run_command([str(runtime_python), "-c", code])


def bootstrap_runtime(
    *,
    runtime_dir: Path | None = None,
    download_models: str = "core",
    install_ffmpeg: bool = False,
    live: bool = False,
) -> SetupResult:
    if which("uv") is None:
        raise RuntimeError("uv is required for setup. Install uv first.")

    ffmpeg_path = which("ffmpeg")
    if install_ffmpeg and ffmpeg_path is None:
        brew = which("brew")
        if brew is None:
            raise RuntimeError("ffmpeg is missing and Homebrew is not available.")
        run_command([brew, "install", "ffmpeg"], live=live)
        ffmpeg_path = which("ffmpeg")

    target_dir = (runtime_dir or default_runtime_dir()).expanduser()
    _log(f"creating runtime at {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["uv", "venv", str(target_dir)], live=live)

    runtime_python = _runtime_python(target_dir)
    if not runtime_python.exists():
        raise RuntimeError(f"Runtime python not created: {runtime_python}")

    _log("installing mlx-audio runtime packages")
    run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(runtime_python),
            "mlx-audio==0.3.1",
            "transformers==5.0.0rc3",
        ],
        live=live,
    )
    _log("installing parakeet-mlx cli into runtime")
    run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(runtime_python),
            "parakeet-mlx==0.5.0",
        ],
        live=live,
    )

    downloaded_models: list[str] = []
    if download_models not in {"none", "core", "all"}:
        raise RuntimeError("download_models must be one of: none, core, all")

    if download_models in {"core", "all"}:
        for model_id in [PARAKEET_MODEL, QWEN_MODELS["qwen3-asr-0.6b"]]:
            _log(f"warming model cache for {model_id}")
            _warm_model(runtime_python, model_id)
            downloaded_models.append(model_id)
    if download_models == "all":
        model_id = QWEN_MODELS["qwen3-asr-1.7b"]
        _log(f"warming model cache for {model_id}")
        _warm_model(runtime_python, model_id)
        downloaded_models.append(model_id)

    config_data = {
        "runtime_dir": str(target_dir),
        "runtime_python": str(runtime_python),
        "parakeet_binary": str(_runtime_parakeet(target_dir)),
        "ffmpeg": ffmpeg_path,
        "downloaded_models": downloaded_models,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path = save_config(config_data)

    return SetupResult(
        home_dir=str(stt_home()),
        runtime_dir=str(target_dir),
        runtime_python=str(runtime_python),
        parakeet_binary=str(_runtime_parakeet(target_dir)),
        config_path=str(path),
        ffmpeg=ffmpeg_path,
        downloaded_models=downloaded_models,
    )
