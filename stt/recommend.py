from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .constants import LONG_AUDIO_THRESHOLD_SECONDS, PARAKEET_MODEL, QWEN_MODELS, SHORT_CLIP_THRESHOLD_SECONDS
from .utils import audio_duration, file_kind


@dataclass
class Recommendation:
    backend: str
    model: str
    rationale: list[str]
    language: str
    output_format: str
    duration_seconds: float | None
    speed_priority: bool
    accuracy_priority: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_language(language: str | None) -> str:
    if not language:
        return "auto"
    return language.strip().lower() or "auto"


def recommend_backend(
    *,
    path: Path,
    language: str = "auto",
    speed_priority: bool = False,
    accuracy_priority: bool = False,
    output_format: str = "txt",
) -> Recommendation:
    kind = file_kind(path)
    if kind not in {"audio", "video"}:
        raise ValueError(f"Expected audio/video input, got {kind}: {path}")

    duration = audio_duration(path)
    normalized_language = normalize_language(language)
    rationale: list[str] = []

    # Parakeet v3 drifts into English on real non-English speech (real Spanish talks: 35% WER vs 7.5%
    # for Qwen3-ASR 0.6B), so only English skips the language-based rules below.
    english = normalized_language == "english"

    if english and output_format in {"srt", "vtt", "all"}:
        rationale.append("English subtitle cues come from segmented MLX Parakeet sentence timestamps.")
        return Recommendation(
            backend="mlx-parakeet",
            model=PARAKEET_MODEL,
            rationale=rationale,
            language=normalized_language,
            output_format=output_format,
            duration_seconds=duration,
            speed_priority=speed_priority,
            accuracy_priority=accuracy_priority,
        )

    if english and duration is not None and duration > LONG_AUDIO_THRESHOLD_SECONDS:
        rationale.append("Speech-turn segmentation keeps very long English audio stable on MLX Parakeet.")
        return Recommendation(
            backend="mlx-parakeet",
            model=PARAKEET_MODEL,
            rationale=rationale,
            language=normalized_language,
            output_format=output_format,
            duration_seconds=duration,
            speed_priority=speed_priority,
            accuracy_priority=accuracy_priority,
        )

    if english:
        if speed_priority and duration is not None and duration <= SHORT_CLIP_THRESHOLD_SECONDS:
            rationale.append("English short clip + speed favors direct MLX Parakeet.")
            return Recommendation(
                backend="mlx-parakeet",
                model=PARAKEET_MODEL,
                rationale=rationale,
                language=normalized_language,
                output_format=output_format,
                duration_seconds=duration,
                speed_priority=speed_priority,
                accuracy_priority=accuracy_priority,
            )
        if accuracy_priority:
            rationale.append("English + accuracy favors Qwen3-ASR 1.7B.")
            return Recommendation(
                backend="qwen3-asr-1.7b",
                model=QWEN_MODELS["qwen3-asr-1.7b"],
                rationale=rationale,
                language=normalized_language,
                output_format=output_format,
                duration_seconds=duration,
                speed_priority=speed_priority,
                accuracy_priority=accuracy_priority,
            )
        rationale.append("Default English recommendation is segmented MLX Parakeet.")
        return Recommendation(
            backend="mlx-parakeet",
            model=PARAKEET_MODEL,
            rationale=rationale,
            language=normalized_language,
            output_format=output_format,
            duration_seconds=duration,
            speed_priority=speed_priority,
            accuracy_priority=accuracy_priority,
        )

    if normalized_language not in {"auto", "english"}:
        if accuracy_priority:
            rationale.append("Known non-English audio favors Qwen3-ASR 1.7B.")
            return Recommendation(
                backend="qwen3-asr-1.7b",
                model=QWEN_MODELS["qwen3-asr-1.7b"],
                rationale=rationale,
                language=normalized_language,
                output_format=output_format,
                duration_seconds=duration,
                speed_priority=speed_priority,
                accuracy_priority=accuracy_priority,
            )
        rationale.append("Known non-English audio favors Qwen3-ASR 0.6B.")
        return Recommendation(
            backend="qwen3-asr-0.6b",
            model=QWEN_MODELS["qwen3-asr-0.6b"],
            rationale=rationale,
            language=normalized_language,
            output_format=output_format,
            duration_seconds=duration,
            speed_priority=speed_priority,
            accuracy_priority=accuracy_priority,
        )

    if accuracy_priority:
        rationale.append("Unknown language + accuracy favors Qwen3-ASR 1.7B.")
        return Recommendation(
            backend="qwen3-asr-1.7b",
            model=QWEN_MODELS["qwen3-asr-1.7b"],
            rationale=rationale,
            language=normalized_language,
            output_format=output_format,
            duration_seconds=duration,
            speed_priority=speed_priority,
            accuracy_priority=accuracy_priority,
        )

    rationale.append("Unknown language favors Qwen3-ASR 0.6B for local multilingual STT.")
    return Recommendation(
        backend="qwen3-asr-0.6b",
        model=QWEN_MODELS["qwen3-asr-0.6b"],
        rationale=rationale,
        language=normalized_language,
        output_format=output_format,
        duration_seconds=duration,
        speed_priority=speed_priority,
        accuracy_priority=accuracy_priority,
    )
