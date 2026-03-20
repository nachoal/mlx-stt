from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys

from .benchmark import benchmark_file, benchmark_repo_samples
from .config import config_path, default_runtime_dir, load_config, resolve_parakeet_binary, resolve_shared_python, stt_home
from .recommend import recommend_backend
from .runtime import bootstrap_runtime
from .transcribe import transcribe_mlx_parakeet, transcribe_parakeet_cli, transcribe_qwen
from .utils import json_print, run_command, which


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local transcription CLI with backend recommendations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recommend_parser = subparsers.add_parser("recommend", help="Recommend the best local STT backend")
    recommend_parser.add_argument("file", help="Audio or video file")
    recommend_parser.add_argument("--language", default="auto", help="Expected language")
    recommend_parser.add_argument("--speed", action="store_true", help="Bias toward speed")
    recommend_parser.add_argument("--accuracy", action="store_true", help="Bias toward accuracy")
    recommend_parser.add_argument("--output-format", default="txt", choices=["txt", "json", "srt", "vtt", "all"])
    recommend_parser.add_argument("--json", action="store_true", help="Print JSON output")

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe using the recommended or explicit backend")
    transcribe_parser.add_argument("file", help="Audio or video file")
    transcribe_parser.add_argument("--language", default="auto", help="Expected language")
    transcribe_parser.add_argument("--speed", action="store_true", help="Bias toward speed")
    transcribe_parser.add_argument("--accuracy", action="store_true", help="Bias toward accuracy")
    transcribe_parser.add_argument("--output-format", default="txt", choices=["txt", "json", "srt", "vtt", "all"])
    transcribe_parser.add_argument("--output-dir", help="Directory to write transcript artifacts")
    transcribe_parser.add_argument("--output-name", default="transcript", help="Output filename stem")
    transcribe_parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "mlx-parakeet", "parakeet-mlx", "qwen3-asr-0.6b", "qwen3-asr-1.7b"],
        help="Force a backend",
    )
    transcribe_parser.add_argument("--json", action="store_true", help="Print JSON output")

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark the local STT backends")
    benchmark_parser.add_argument("file", nargs="?", help="Audio or video file")
    benchmark_parser.add_argument("--reference-text", help="Reference transcript for WER")
    benchmark_parser.add_argument("--language", default="auto", help="Expected language")
    benchmark_parser.add_argument("--suite", choices=["repo-samples"], help="Run the built-in benchmark suite")
    benchmark_parser.add_argument("--json", action="store_true", help="Print JSON output")

    doctor_parser = subparsers.add_parser("doctor", help="Inspect local runtime state")
    doctor_parser.add_argument("--json", action="store_true", help="Print JSON output")

    setup_parser = subparsers.add_parser("setup", help="Create an isolated runtime and pre-download models")
    setup_parser.add_argument("--runtime-dir", help="Override the runtime directory")
    setup_parser.add_argument(
        "--download-models",
        default="core",
        choices=["none", "core", "all"],
        help="Which models to warm into the local cache",
    )
    setup_parser.add_argument("--install-ffmpeg", action="store_true", help="Install ffmpeg with Homebrew if missing")
    setup_parser.add_argument("--json", action="store_true", help="Print JSON output")

    return parser.parse_args(argv)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _external_package_version(python_executable: str, name: str) -> str | None:
    if which(python_executable) is None:
        return None
    code = (
        "from importlib.metadata import version, PackageNotFoundError\n"
        f"name={name!r}\n"
        "try:\n"
        "    print(version(name))\n"
        "except PackageNotFoundError:\n"
        "    print('NOT_INSTALLED')\n"
    )
    proc = run_command([python_executable, "-c", code], check=False)
    value = (proc.stdout.strip() or proc.stderr.strip() or "").strip()
    if value == "NOT_INSTALLED":
        return None
    return value or None


def command_recommend(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    recommendation = recommend_backend(
        path=path,
        language=args.language,
        speed_priority=args.speed,
        accuracy_priority=args.accuracy,
        output_format=args.output_format,
    )
    payload = recommendation.to_dict()
    payload["file"] = str(path)
    if args.json:
        json_print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_transcribe(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    if args.backend == "auto":
        recommendation = recommend_backend(
            path=path,
            language=args.language,
            speed_priority=args.speed,
            accuracy_priority=args.accuracy,
            output_format=args.output_format,
        )
        backend = recommendation.backend
    else:
        recommendation = None
        backend = args.backend

    if backend == "qwen3-asr-0.6b":
        result = transcribe_qwen(path, model_key="qwen3-asr-0.6b", language=args.language)
    elif backend == "qwen3-asr-1.7b":
        result = transcribe_qwen(path, model_key="qwen3-asr-1.7b", language=args.language)
    elif backend == "mlx-parakeet":
        result = transcribe_mlx_parakeet(path, language=args.language)
    else:
        result = transcribe_parakeet_cli(
            path,
            output_format=args.output_format,
            output_dir=output_dir,
            output_name=args.output_name,
        )

    if output_dir is not None and backend != "parakeet-mlx":
        output_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        if args.output_format in {"txt", "all"}:
            txt_path = output_dir / f"{args.output_name}.txt"
            txt_path.write_text(result.text)
            written["txt"] = str(txt_path)
        if args.output_format in {"json", "all"}:
            json_path = output_dir / f"{args.output_name}.json"
            json_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
            written["json"] = str(json_path)
        result.output_paths = written or None

    payload = {"result": result.to_dict(), "file": str(path)}
    if recommendation is not None:
        payload["recommendation"] = recommendation.to_dict()
    if args.json:
        json_print(payload)
    else:
        print(result.text)
    return 0 if result.success else 1


def command_benchmark(args: argparse.Namespace) -> int:
    if args.suite == "repo-samples":
        rows = benchmark_repo_samples()
        if not rows:
            raise SystemExit(
                "No repo sample fixtures configured. Set STT_SAMPLE_ENGLISH and/or STT_SAMPLE_SPANISH."
            )
    else:
        if not args.file:
            raise SystemExit("benchmark requires FILE or --suite repo-samples")
        path = Path(args.file).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        rows = benchmark_file(path, reference_text=args.reference_text, language_hint=args.language)

    if args.json:
        json_print({"rows": rows})
    else:
        for row in rows:
            label = row.get("case", Path(row["audio_path"]).stem)
            print(
                f"{label:<18} {row['variant']:<16} success={row['success']} "
                f"wer={row.get('wer')} text={row['text'][:100]}"
            )
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    shared_python = resolve_shared_python()
    configured = load_config()
    payload = {
        "python": sys.executable,
        "stt_home": str(stt_home()),
        "config_path": str(config_path()),
        "config": configured,
        "shared_python": shared_python,
        "shared_python_exists": bool(shared_python),
        "ffmpeg": which("ffmpeg"),
        "ffprobe": which("ffprobe"),
        "parakeet_mlx_binary": resolve_parakeet_binary(),
        "versions": {
            "stt": _package_version("stt"),
            "parakeet-mlx": _external_package_version(shared_python or "python3", "parakeet-mlx") if (shared_python or which("python3")) else None,
            "mlx-audio": _external_package_version(shared_python, "mlx-audio") if shared_python else None,
            "transformers": _external_package_version(shared_python, "transformers") if shared_python else None,
        },
        "recommendations": {
            "english_fast_short": "mlx-parakeet",
            "english_subtitles_or_long": "parakeet-mlx",
            "multilingual_default": "qwen3-asr-0.6b",
            "multilingual_accuracy": "qwen3-asr-1.7b",
        },
    }
    if args.json:
        json_print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def command_setup(args: argparse.Namespace) -> int:
    runtime_dir = Path(args.runtime_dir).expanduser().resolve() if args.runtime_dir else default_runtime_dir()
    result = bootstrap_runtime(
        runtime_dir=runtime_dir,
        download_models=args.download_models,
        install_ffmpeg=args.install_ffmpeg,
        live=not args.json,
    )
    payload = result.to_dict()
    if args.json:
        json_print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "recommend":
        return command_recommend(args)
    if args.command == "transcribe":
        return command_transcribe(args)
    if args.command == "benchmark":
        return command_benchmark(args)
    if args.command == "doctor":
        return command_doctor(args)
    if args.command == "setup":
        return command_setup(args)
    raise SystemExit(f"Unknown command: {args.command}")
