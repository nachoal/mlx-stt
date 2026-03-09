from pathlib import Path

from stt.recommend import recommend_backend


def test_recommend_english_short_speed_prefers_mlx_parakeet(monkeypatch):
    monkeypatch.setattr("stt.recommend.file_kind", lambda path: "audio")
    monkeypatch.setattr("stt.recommend.audio_duration", lambda path: 30.0)
    result = recommend_backend(
        path=Path("clip.wav"),
        language="english",
        speed_priority=True,
        accuracy_priority=False,
        output_format="txt",
    )
    assert result.backend == "mlx-parakeet"


def test_recommend_subtitles_prefers_parakeet_cli(monkeypatch):
    monkeypatch.setattr("stt.recommend.file_kind", lambda path: "audio")
    monkeypatch.setattr("stt.recommend.audio_duration", lambda path: 30.0)
    result = recommend_backend(
        path=Path("clip.wav"),
        language="english",
        output_format="srt",
    )
    assert result.backend == "parakeet-mlx"


def test_recommend_spanish_prefers_qwen(monkeypatch):
    monkeypatch.setattr("stt.recommend.file_kind", lambda path: "audio")
    monkeypatch.setattr("stt.recommend.audio_duration", lambda path: 20.0)
    result = recommend_backend(
        path=Path("clip.wav"),
        language="spanish",
    )
    assert result.backend == "qwen3-asr-0.6b"


def test_recommend_unknown_language_accuracy_prefers_qwen_17(monkeypatch):
    monkeypatch.setattr("stt.recommend.file_kind", lambda path: "audio")
    monkeypatch.setattr("stt.recommend.audio_duration", lambda path: 20.0)
    result = recommend_backend(
        path=Path("clip.wav"),
        language="auto",
        accuracy_priority=True,
    )
    assert result.backend == "qwen3-asr-1.7b"


def test_recommend_long_audio_prefers_parakeet_cli(monkeypatch):
    monkeypatch.setattr("stt.recommend.file_kind", lambda path: "audio")
    monkeypatch.setattr("stt.recommend.audio_duration", lambda path: 7200.0)
    result = recommend_backend(
        path=Path("clip.wav"),
        language="english",
        speed_priority=True,
    )
    assert result.backend == "parakeet-mlx"
