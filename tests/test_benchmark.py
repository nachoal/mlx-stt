from pathlib import Path

from stt.benchmark import benchmark_file, benchmark_repo_samples, word_error_rate
from stt.transcribe import TranscriptionResult


def test_word_error_rate_exact_match():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_word_error_rate_detects_difference():
    assert word_error_rate("hello world", "hello there") > 0.0


def test_benchmark_file_shapes_rows(monkeypatch):
    fake = TranscriptionResult(
        backend="fake",
        model="fake-model",
        text="hello world",
        success=True,
        total_time=1.0,
        audio_duration=2.0,
        rtf=0.5,
    )

    monkeypatch.setattr("stt.benchmark.transcribe_qwen", lambda path, model_key, language: fake)
    monkeypatch.setattr(
        "stt.benchmark.transcribe_mlx_parakeet",
        lambda path, language="auto": fake,
    )
    monkeypatch.setattr("stt.benchmark.transcribe_parakeet_cli", lambda path: fake)

    rows = benchmark_file(Path("clip.wav"), reference_text="hello world", language_hint="english")
    assert len(rows) == 4
    assert {row["variant"] for row in rows} == {"qwen3-asr-0.6b", "qwen3-asr-1.7b", "mlx-parakeet", "parakeet-mlx"}


def test_benchmark_repo_samples_uses_env(monkeypatch, tmp_path):
    english = tmp_path / "english.wav"
    spanish = tmp_path / "spanish.wav"
    english.write_text("x")
    spanish.write_text("x")

    monkeypatch.setenv("STT_SAMPLE_ENGLISH", str(english))
    monkeypatch.setenv("STT_SAMPLE_SPANISH", str(spanish))
    monkeypatch.setattr("stt.benchmark.benchmark_file", lambda path, reference_text, language_hint: [{"audio_path": str(path), "reference_text": reference_text, "language_hint": language_hint}])

    rows = benchmark_repo_samples()
    assert len(rows) == 2
    assert rows[0]["audio_path"] == str(english)
    assert rows[1]["audio_path"] == str(spanish)
