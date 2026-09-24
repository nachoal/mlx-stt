"""Runs inside the stt runtime (mlx, numpy, mlx-audio), never in the CLI environment.

Diarizes a 16 kHz mono PCM WAV with Nemotron-3-Diarization, plans speaker-pure speech segments, and
transcribes each segment with the requested backend (or the whole file when `segment` is false).
Prints one JSON object on the last stdout line.

    <runtime-python> runtime_worker.py '{"audio": "...", "backend": "mlx-parakeet", "asr_model": "...", ...}'
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stt.nemotron_diar_mlx import HOP, SAMPLE_RATE, Nemotron3Diarizer  # noqa: E402
from stt.segments import Segment, parakeet_decode_options, plan_segments, renumber_speakers  # noqa: E402

FRAME_SECONDS = HOP / SAMPLE_RATE


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError("expected 16 kHz mono 16-bit PCM WAV")
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return data.astype(np.float32) / 32768.0


def speaker_runs(probs: np.ndarray, threshold: float = 0.5) -> list[tuple[float, float, int]]:
    """Maximal runs of 10 ms frames with the same dominant active speaker."""
    if len(probs) == 0:
        return []
    active = probs.max(axis=1) > threshold
    dominant = np.where(active, probs.argmax(axis=1), -1)
    edges = np.flatnonzero(np.diff(np.concatenate([[-2], dominant, [-2]])))
    return [
        (start * FRAME_SECONDS, end * FRAME_SECONDS, int(dominant[start]))
        for start, end in zip(edges[:-1], edges[1:])
        if dominant[start] >= 0
    ]


def transcribe_clip(model, backend: str, clip, offset: float, speaker: int, language: str | None) -> list[dict]:
    import mlx.core as mx

    audio = mx.array(clip)
    if backend == "mlx-parakeet":
        result = model.generate(audio, **parakeet_decode_options(language, len(clip) / SAMPLE_RATE))
        return [
            {"start": round(offset + float(s.start), 3), "end": round(offset + float(s.end), 3),
             "text": s.text.strip(), "speaker": speaker}
            for s in getattr(result, "sentences", [])
            if s.text.strip()
        ]
    result = model.generate(audio) if language is None else model.generate(audio, language=language)
    text = (getattr(result, "text", "") or "").strip()
    return [{"start": round(offset, 3), "end": round(offset + len(clip) / SAMPLE_RATE, 3), "text": text, "speaker": speaker}] if text else []


def main() -> None:
    request = json.loads(sys.argv[1])
    started = time.time()
    audio = read_wav(request["audio"])
    duration = len(audio) / SAMPLE_RATE

    t0 = time.time()
    if request.get("segment", True):
        diarizer = Nemotron3Diarizer.from_pretrained(request["diarization_model"], revision=request["diarization_revision"])
        plan = plan_segments(speaker_runs(diarizer.diarize(audio)), duration)
    else:
        plan = [Segment(0.0, duration, 0)] if duration > 0 else []
    diarization_seconds = time.time() - t0

    from mlx_audio.stt import load

    t0 = time.time()
    model = load(request["asr_model"])
    load_seconds = time.time() - t0

    t0 = time.time()
    segments: list[dict] = []
    for segment in plan:
        clip = audio[int(segment.start * SAMPLE_RATE) : int(segment.end * SAMPLE_RATE)]
        if len(clip) >= SAMPLE_RATE // 10:
            segments.extend(transcribe_clip(model, request["backend"], clip, segment.start, segment.speaker, request.get("language")))
    asr_seconds = time.time() - t0

    speakers = renumber_speakers(segments) if request.get("diarize") else None
    if not request.get("diarize"):
        for item in segments:
            item.pop("speaker", None)

    print(json.dumps({
        "text": " ".join(item["text"] for item in segments).strip(),
        "segments": segments,
        "speakers": speakers,
        "audio_duration": duration,
        "elapsed": time.time() - started,
        "timings": {
            "diarization_seconds": round(diarization_seconds, 3),
            "asr_load_seconds": round(load_seconds, 3),
            "asr_seconds": round(asr_seconds, 3),
            "planned_segments": len(plan),
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
