import json
from pathlib import Path

import pytest

from stt import cli
from stt.transcribe import TranscriptionResult


def _result(**overrides):
    base = dict(
        backend="mlx-parakeet",
        model="m",
        text="hello there general kenobi",
        success=True,
        total_time=1.0,
        audio_duration=4.0,
        rtf=0.25,
        segments=[
            {"start": 0.0, "end": 1.5, "text": "hello there", "speaker": "speaker_0"},
            {"start": 2.0, "end": 3.5, "text": "general kenobi", "speaker": "speaker_1"},
        ],
        speakers=2,
    )
    base.update(overrides)
    return TranscriptionResult(**base)


def test_transcribe_parser_accepts_diarize():
    args = cli.parse_args(["transcribe", "clip.wav", "--diarize", "--json"])
    assert args.diarize is True


def test_diarize_rejects_external_parakeet_cli(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_text("x")
    with pytest.raises(SystemExit, match="--diarize needs an in-runtime backend"):
        cli.main(["transcribe", str(clip), "--backend", "parakeet-mlx", "--diarize"])


def test_diarize_passes_through_and_prints_speaker_lines(monkeypatch, tmp_path, capsys):
    clip = tmp_path / "clip.wav"
    clip.write_text("x")
    calls = {}

    def fake_parakeet(path, language="auto", diarize=False):
        calls["diarize"] = diarize
        return _result()

    monkeypatch.setattr(cli, "transcribe_mlx_parakeet", fake_parakeet)
    assert cli.main(["transcribe", str(clip), "--backend", "mlx-parakeet", "--diarize"]) == 0
    assert calls["diarize"] is True
    assert capsys.readouterr().out.strip() == "speaker_0: hello there\nspeaker_1: general kenobi"


def test_write_outputs_all_formats(tmp_path):
    written = cli._write_outputs(_result(), tmp_path, "t", "all", speakers=True)
    assert set(written) == {"txt", "json", "srt", "vtt"}
    assert Path(written["txt"]).read_text() == "speaker_0: hello there\nspeaker_1: general kenobi"
    assert json.loads(Path(written["json"]).read_text())["speakers"] == 2
    assert "speaker_1: general kenobi" in Path(written["srt"]).read_text()
    assert Path(written["vtt"]).read_text().startswith("WEBVTT")


def test_plain_output_without_diarize_ignores_speakers(tmp_path):
    written = cli._write_outputs(_result(), tmp_path, "t", "txt", speakers=False)
    assert Path(written["txt"]).read_text() == "hello there general kenobi"


def test_qwen_subtitles_request_per_turn_segments(monkeypatch, tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_text("x")
    calls = {}

    def fake_qwen(path, model_key, language="auto", diarize=False, segment=False):
        calls.update(model_key=model_key, segment=segment, diarize=diarize)
        return _result(backend=model_key)

    monkeypatch.setattr(cli, "transcribe_qwen", fake_qwen)
    out = tmp_path / "out"
    assert cli.main(["transcribe", str(clip), "--backend", "qwen3-asr-0.6b", "--output-format", "srt",
                     "--output-dir", str(out), "--json"]) == 0
    assert calls == {"model_key": "qwen3-asr-0.6b", "segment": True, "diarize": False}
    assert (out / "transcript.srt").read_text().startswith("1\n00:00:00,000 --> 00:00:01,500\nhello there")
