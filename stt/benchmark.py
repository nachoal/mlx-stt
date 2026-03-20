from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

from .config import env_sample
from .constants import REPO_SAMPLE_SUITE
from .transcribe import transcribe_mlx_parakeet, transcribe_parakeet_cli, transcribe_qwen


def _normalize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return cleaned.split()


def _levenshtein(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if left_item == right_item else 1),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str | None, hypothesis: str) -> float | None:
    if not reference:
        return None
    ref_words = _normalize(reference)
    hyp_words = _normalize(hypothesis)
    if not ref_words:
        return None
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def benchmark_file(path: Path, *, reference_text: str | None, language_hint: str) -> list[dict[str, Any]]:
    runs = [
        ("qwen3-asr-0.6b", transcribe_qwen(path, model_key="qwen3-asr-0.6b", language=language_hint)),
        ("qwen3-asr-1.7b", transcribe_qwen(path, model_key="qwen3-asr-1.7b", language=language_hint)),
        ("mlx-parakeet", transcribe_mlx_parakeet(path, language=language_hint)),
        ("parakeet-mlx", transcribe_parakeet_cli(path)),
    ]
    rows: list[dict[str, Any]] = []
    for variant, result in runs:
        row = asdict(result)
        row["variant"] = variant
        row["reference_text"] = reference_text
        row["language_hint"] = language_hint
        row["wer"] = word_error_rate(reference_text, result.text)
        row["audio_path"] = str(path)
        rows.append(row)
    return rows


def benchmark_repo_samples() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_map = {
        "english": {
            "env_path": "STT_SAMPLE_ENGLISH",
            "reference_env": "STT_SAMPLE_ENGLISH_TEXT",
            "fallback_reference": "Hello. This is a test.",
            "language_hint": "english",
        },
        "spanish": {
            "env_path": "STT_SAMPLE_SPANISH",
            "reference_env": "STT_SAMPLE_SPANISH_TEXT",
            "fallback_reference": (
                "Marina de las Aduanas en los puertos, y en el caso de las aduanas "
                "terrestres, el Ejército, la Secretaría de la Defensa Nacional. "
                "Un billón doscientos cincuenta mil millones de pesos, y además aquí "
                "quiero agradecer también."
            ),
            "language_hint": "spanish",
        },
    }
    for name in REPO_SAMPLE_SUITE:
        config = sample_map[name]
        path = env_sample(config["env_path"])
        if path is None:
            continue
        reference_text = os.environ.get(config["reference_env"], config["fallback_reference"])
        case_rows = benchmark_file(
            path,
            reference_text=reference_text,
            language_hint=config["language_hint"],
        )
        for row in case_rows:
            row["case"] = name
        rows.extend(case_rows)
    return rows
