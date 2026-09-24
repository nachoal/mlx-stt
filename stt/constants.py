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

DIARIZATION_MODEL = "nvidia/Nemotron-3-Diarization"
DIARIZATION_REVISION = "a435e9867d79e789e90053f9b6d6834053af564a"

RUNTIME_PACKAGES = [
    "mlx==0.32.2",
    "mlx-audio==0.5.5",
    "transformers==5.17.0",
]
PARAKEET_CLI_PACKAGE = "parakeet-mlx==0.5.2"

REPO_SAMPLE_SUITE = [
    "english",
    "spanish",
]
