import json
from pathlib import Path
from types import SimpleNamespace

from stt.transcribe import _prepare_input
from stt.utils import file_kind, needs_wav_normalization


def test_file_kind_recognizes_opus():
    assert file_kind(Path("voice.opus")) == "audio"


def test_needs_wav_normalization_detects_opus(monkeypatch, tmp_path):
    voice_note = tmp_path / "voice.ogg"
    voice_note.write_text("x")

    monkeypatch.setattr(
        "stt.utils.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"streams": [{"codec_type": "audio", "codec_name": "opus"}]}),
        ),
    )

    assert needs_wav_normalization(voice_note) is True


def test_needs_wav_normalization_skips_pcm_wav(monkeypatch, tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_text("x")

    monkeypatch.setattr(
        "stt.utils.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le"}]}),
        ),
    )

    assert needs_wav_normalization(clip) is False


def test_prepare_input_normalizes_to_temp_wav(monkeypatch, tmp_path):
    voice_note = tmp_path / "voice.ogg"
    voice_note.write_text("x")
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setattr("stt.transcribe.needs_wav_normalization", lambda path: True)

    def fake_convert(input_path: Path, output_path: Path) -> Path:
        calls.append((input_path, output_path))
        output_path.write_text("wav")
        return output_path

    monkeypatch.setattr("stt.transcribe.convert_media_to_wav", fake_convert)

    prepared_path, tmp = _prepare_input(voice_note)

    assert tmp is not None
    assert prepared_path.suffix == ".wav"
    assert prepared_path.exists()
    assert calls == [(voice_note, prepared_path)]

    tmp.cleanup()


def test_prepare_input_leaves_supported_pcm_input(monkeypatch, tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_text("x")

    monkeypatch.setattr("stt.transcribe.needs_wav_normalization", lambda path: False)

    prepared_path, tmp = _prepare_input(clip)

    assert prepared_path == clip
    assert tmp is None
