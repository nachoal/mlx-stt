from __future__ import annotations

from pathlib import Path

QWEN_MODELS = {
    "qwen3-asr-0.6b": "mlx-community/Qwen3-ASR-0.6B-8bit",
    "qwen3-asr-1.7b": "mlx-community/Qwen3-ASR-1.7B-8bit",
}

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
PARAKEET_BINARY = "parakeet-mlx"
LONG_AUDIO_THRESHOLD_SECONDS = 3600.0
SHORT_CLIP_THRESHOLD_SECONDS = 180.0

REPO_SAMPLE_SUITE = [
    "english",
    "spanish",
]
