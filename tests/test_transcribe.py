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


def test_transcribe_segmented_sends_worker_request_and_parses_payload(monkeypatch, tmp_path):
    from stt import transcribe

    clip = tmp_path / "clip.wav"
    clip.write_text("x")
    seen: dict = {}
    payload = {
        "text": "hola adiós",
        "segments": [
            {"start": 0.1, "end": 1.0, "text": "hola", "speaker": "speaker_0"},
            {"start": 1.5, "end": 2.0, "text": "adiós", "speaker": "speaker_1"},
        ],
        "speakers": 2,
        "audio_duration": 2.0,
        "elapsed": 0.5,
        "timings": {"diarization_seconds": 0.01},
    }

    def fake_run(cmd, check=True, live=False):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="progress noise\n" + json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(transcribe, "is_pcm16k_mono_wav", lambda path: True)
    monkeypatch.setattr(transcribe, "resolve_shared_python", lambda: "/runtime/bin/python")
    monkeypatch.setattr(transcribe, "run_command", fake_run)

    result = transcribe.transcribe_mlx_parakeet(clip, language="spanish", diarize=True)

    python, worker, request_json = seen["cmd"]
    request = json.loads(request_json)
    assert python == "/runtime/bin/python"
    assert worker.endswith("runtime_worker.py")
    assert request["backend"] == "mlx-parakeet"
    assert request["audio"] == str(clip)
    assert request["diarize"] is True
    assert request["segment"] is True
    assert request["language"] == "spanish"
    assert request["diarization_model"] == "nvidia/Nemotron-3-Diarization"
    assert result.success
    assert result.speakers == 2
    assert result.segments[1]["speaker"] == "speaker_1"
    assert result.rtf == 0.25


def test_transcribe_qwen_passes_titled_language_and_reports_worker_failure(monkeypatch, tmp_path):
    from stt import transcribe

    clip = tmp_path / "clip.wav"
    clip.write_text("x")
    requests: list[dict] = []

    def fake_run(cmd, check=True, live=False):
        requests.append(json.loads(cmd[2]))
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(transcribe, "is_pcm16k_mono_wav", lambda path: True)
    monkeypatch.setattr(transcribe, "resolve_shared_python", lambda: "/runtime/bin/python")
    monkeypatch.setattr(transcribe, "run_command", fake_run)
    monkeypatch.setattr(transcribe, "audio_duration", lambda path: 2.0)

    result = transcribe.transcribe_qwen(clip, model_key="qwen3-asr-0.6b", language="spanish")

    assert requests[0]["language"] == "Spanish"
    assert requests[0]["diarize"] is False
    assert requests[0]["segment"] is False
    assert result.success is False
    assert result.stderr == "boom"


def test_transcribe_qwen_segments_when_diarizing(monkeypatch, tmp_path):
    from stt import transcribe

    clip = tmp_path / "clip.wav"
    clip.write_text("x")
    requests: list[dict] = []

    def fake_run(cmd, check=True, live=False):
        requests.append(json.loads(cmd[2]))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"text": "", "segments": [], "speakers": 0}), stderr="")

    monkeypatch.setattr(transcribe, "is_pcm16k_mono_wav", lambda path: True)
    monkeypatch.setattr(transcribe, "resolve_shared_python", lambda: "/runtime/bin/python")
    monkeypatch.setattr(transcribe, "run_command", fake_run)
    monkeypatch.setattr(transcribe, "audio_duration", lambda path: 1.0)

    transcribe.transcribe_qwen(clip, model_key="qwen3-asr-1.7b", diarize=True)

    assert requests[0]["segment"] is True
    assert requests[0]["asr_model"] == "mlx-community/Qwen3-ASR-1.7B-8bit"
