import pytest

from stt.segments import Segment, merge_turns, plan_segments, renumber_speakers, to_srt, to_text, to_vtt


def test_plan_segments_without_speech_covers_the_whole_file():
    assert plan_segments([], 12.0) == [Segment(0.0, 12.0, 0)]
    assert plan_segments([], 0.0) == []


def test_plan_segments_merges_one_speaker_across_short_pauses():
    runs = [(1.0, 3.0, 0), (3.5, 6.0, 0), (6.4, 9.0, 0)]
    assert plan_segments(runs, 10.0) == [Segment(0.0, 10.0, 0)]


def test_plan_segments_splits_at_a_speaker_change_mid_silence():
    runs = [(0.5, 4.0, 0), (4.6, 8.0, 1)]
    first, second = plan_segments(runs, 9.0)
    assert (first.speaker, second.speaker) == (0, 1)
    assert first.end == pytest.approx(4.3)
    assert second.start == pytest.approx(4.3)


def test_plan_segments_tiles_short_silences_and_trims_long_ones():
    runs = [(1.0, 3.0, 0), (4.0, 6.0, 1), (20.0, 22.0, 0)]
    segments = plan_segments(runs, 30.0)
    assert segments[0].start == 0.0
    assert segments[0].end == pytest.approx(segments[1].start)
    assert segments[1].end == pytest.approx(7.0)
    assert segments[2].start == pytest.approx(19.0)
    assert segments[2].end == pytest.approx(23.0)


def test_plan_segments_caps_segment_length():
    runs = [(float(t), t + 0.9, 0) for t in range(0, 60)]
    segments = plan_segments(runs, 60.0)
    assert len(segments) >= 3
    assert all(s.end - s.start <= 20.0 + 1.0 for s in segments)


def test_plan_segments_hard_splits_one_overlong_run():
    segments = plan_segments([(0.0, 50.0, 2)], 50.0)
    assert len(segments) == 3
    assert {s.speaker for s in segments} == {2}
    assert segments[-1].end == 50.0


def test_plan_segments_absorbs_short_dominance_flips():
    runs = [(0.0, 3.0, 0), (3.0, 3.1, 1), (3.1, 6.0, 0)]
    assert [s.speaker for s in plan_segments(runs, 6.0)] == [0]


def test_renumber_speakers_uses_first_appearance_order():
    segments = [{"speaker": 3, "text": "a"}, {"speaker": 1, "text": "b"}, {"speaker": 3, "text": "c"}]
    assert renumber_speakers(segments) == 2
    assert [s["speaker"] for s in segments] == ["speaker_0", "speaker_1", "speaker_0"]


def test_merge_turns_joins_consecutive_segments_of_one_speaker():
    turns = merge_turns([
        {"start": 0.0, "end": 1.0, "text": "hola", "speaker": "speaker_0"},
        {"start": 1.0, "end": 2.0, "text": "qué tal", "speaker": "speaker_0"},
        {"start": 2.5, "end": 3.0, "text": "bien", "speaker": "speaker_1"},
    ])
    assert [(t["speaker"], t["text"], t["end"]) for t in turns] == [
        ("speaker_0", "hola qué tal", 2.0),
        ("speaker_1", "bien", 3.0),
    ]


def test_text_and_cue_formats():
    segments = [
        {"start": 0.5, "end": 2.25, "text": "Hello there.", "speaker": "speaker_0"},
        {"start": 3.0, "end": 3661.5, "text": "General Kenobi.", "speaker": "speaker_1"},
    ]
    assert to_text(segments, speakers=False) == "Hello there. General Kenobi."
    assert to_text(segments, speakers=True) == "speaker_0: Hello there.\nspeaker_1: General Kenobi."
    srt = to_srt(segments, speakers=True)
    assert "1\n00:00:00,500 --> 00:00:02,250\nspeaker_0: Hello there.\n" in srt
    assert "01:01:01,500" in srt
    vtt = to_vtt(segments, speakers=False)
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:03.000 --> 01:01:01.500\nGeneral Kenobi.\n" in vtt


def test_parakeet_windows_stay_short_for_non_english_hints():
    from stt.segments import parakeet_decode_options

    assert parakeet_decode_options("spanish", 12.0) == {"chunk_duration": 4.75, "overlap_duration": 1.0}
    assert parakeet_decode_options("fr", 3.0) == {"chunk_duration": 4.75, "overlap_duration": 1.0}


def test_parakeet_windows_leave_english_and_auto_segments_whole():
    from stt.segments import parakeet_decode_options

    assert parakeet_decode_options("english", 18.0) == {}
    assert parakeet_decode_options("auto", 18.0) == {}
    assert parakeet_decode_options(None, 18.0) == {}
    assert parakeet_decode_options("english", 75.0) == {"chunk_duration": 30.0, "overlap_duration": 5.0}
