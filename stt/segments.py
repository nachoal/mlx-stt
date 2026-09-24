"""Speech-turn segmentation and transcript formats.

Standard library only: the CLI imports it for output formats, and the runtime worker imports it to
plan the segments the ASR backends transcribe. Parakeet v3 drops whole sentences when it decodes
long windows that contain pauses; transcribing one speaker turn at a time avoids that and keeps each
language switch in its own window.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

# mlx-audio never gives Parakeet v3 a language context, so on real non-English speech it drifts into
# English unless each decode window stays short (real Spanish talks: 35% WER per speech turn, 7.9% with
# these windows). The windows hurt English and clean read speech, so they apply only to non-English hints.
NON_ENGLISH_PARAKEET_WINDOW = {"chunk_duration": 4.75, "overlap_duration": 1.0}
LONG_PARAKEET_WINDOW = {"chunk_duration": 30.0, "overlap_duration": 5.0}
LONG_CLIP_SECONDS = 60.0
ENGLISH_HINTS = {"", "auto", "english", "en"}

MAX_GAP_SECONDS = 1.0
MAX_SEGMENT_SECONDS = 20.0
MIN_RUN_SECONDS = 0.2
EDGE_PAD_SECONDS = 1.0
MAX_SILENCE_SECONDS = 2.0


@dataclass
class Segment:
    start: float
    end: float
    speaker: int


def plan_segments(
    runs: list[tuple[float, float, int]],
    total_seconds: float,
    *,
    max_gap: float = MAX_GAP_SECONDS,
    max_length: float = MAX_SEGMENT_SECONDS,
    min_run: float = MIN_RUN_SECONDS,
    edge_pad: float = EDGE_PAD_SECONDS,
    max_silence: float = MAX_SILENCE_SECONDS,
) -> list[Segment]:
    """Turns time-ordered speech runs `(start, end, speaker)` into speaker-pure segments that tile the audio.

    Same-speaker runs merge across pauses up to `max_gap`, up to `max_length` per segment. Runs shorter
    than `min_run` join the running segment. Cuts fall in the middle of the silence between segments, so no
    audio is dropped, except that silences longer than `max_silence` keep only `edge_pad` next to speech.
    """
    if total_seconds <= 0:
        return []
    if not runs:
        return [Segment(0.0, total_seconds, 0)]

    cores: list[list[float | int]] = []
    for start, end, speaker in runs:
        fits = bool(cores) and start - cores[-1][1] <= max_gap and end - cores[-1][0] <= max_length
        if fits and end - start < min_run:
            cores[-1][1] = end
        elif fits and cores[-1][2] == speaker:
            cores[-1][1] = end
        else:
            cores.append([start, end, speaker])

    split: list[tuple[float, float, int]] = []
    for start, end, speaker in cores:
        pieces = max(1, math.ceil((end - start) / max_length))
        step = (end - start) / pieces
        split.extend((start + i * step, start + (i + 1) * step, int(speaker)) for i in range(pieces))

    def share(gap: float, between: bool) -> float:
        if gap <= max_silence:
            return gap / 2 if between else gap
        return edge_pad

    planned: list[Segment] = []
    for i, (start, end, speaker) in enumerate(split):
        before = start - (split[i - 1][1] if i else 0.0)
        after = (split[i + 1][0] if i + 1 < len(split) else total_seconds) - end
        lo = start - share(max(before, 0.0), between=i > 0)
        hi = end + share(max(after, 0.0), between=i + 1 < len(split))
        planned.append(Segment(max(0.0, lo), min(total_seconds, hi), speaker))
    return planned


def parakeet_decode_options(language: str | None, clip_seconds: float) -> dict[str, float]:
    """Decode windows for one Parakeet segment: short for non-English hints, chunked for overlong clips."""
    if (language or "").strip().lower() not in ENGLISH_HINTS:
        return dict(NON_ENGLISH_PARAKEET_WINDOW)
    if clip_seconds > LONG_CLIP_SECONDS:
        return dict(LONG_PARAKEET_WINDOW)
    return {}


def renumber_speakers(segments: list[dict]) -> int:
    """Relabels speakers `speaker_0, speaker_1, ...` in order of first appearance; returns the speaker count."""
    order: dict[int, str] = {}
    for segment in segments:
        raw = segment.get("speaker")
        if raw is None:
            continue
        if raw not in order:
            order[raw] = f"speaker_{len(order)}"
        segment["speaker"] = order[raw]
    return len(order)


def merge_turns(segments: list[dict]) -> list[dict]:
    """Joins consecutive segments of one speaker into turns."""
    turns: list[dict] = []
    for segment in segments:
        if turns and turns[-1].get("speaker") == segment.get("speaker"):
            turns[-1]["end"] = segment["end"]
            turns[-1]["text"] = f"{turns[-1]['text']} {segment['text']}".strip()
        else:
            turns.append(dict(segment))
    return turns


def to_text(segments: list[dict], *, speakers: bool) -> str:
    if not speakers:
        return " ".join(segment["text"] for segment in segments if segment["text"]).strip()
    return "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in merge_turns(segments) if turn["text"])


def _timestamp(seconds: float, separator: str) -> str:
    millis = max(0, round(seconds * 1000))
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _cue_text(segment: dict, speakers: bool) -> str:
    if speakers and segment.get("speaker"):
        return f"{segment['speaker']}: {segment['text']}"
    return segment["text"]


def to_srt(segments: list[dict], *, speakers: bool) -> str:
    blocks = []
    for index, segment in enumerate((s for s in segments if s["text"]), start=1):
        blocks.append(
            f"{index}\n{_timestamp(segment['start'], ',')} --> {_timestamp(segment['end'], ',')}\n"
            f"{_cue_text(segment, speakers)}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: list[dict], *, speakers: bool) -> str:
    blocks = ["WEBVTT\n"]
    for segment in segments:
        if segment["text"]:
            blocks.append(
                f"{_timestamp(segment['start'], '.')} --> {_timestamp(segment['end'], '.')}\n"
                f"{_cue_text(segment, speakers)}\n"
            )
    return "\n".join(blocks)
