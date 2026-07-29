"""Tests for the VideoContext model and its consistency guarantees."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from vision_use.models.context import (
    ActivitySpan,
    AnalyzerRun,
    ContextSection,
    Keyframe,
    Shot,
    TimeSpan,
    Transcript,
    TranscriptSegment,
    VideoContext,
    Word,
)
from vision_use.models.ids import (
    is_valid,
    keyframe_id,
    segment_id,
    shot_id,
    speaker_id,
    word_id,
)
from vision_use.models.media import (
    FrameRate,
    MediaProbe,
    SourceInfo,
    VideoStreamInfo,
)

DIGEST = "0" * 64


def _probe(duration_s: float = 10.0) -> MediaProbe:
    return MediaProbe(
        source=SourceInfo(
            path=Path("clip.mp4"),
            sha256=DIGEST,
            size_bytes=1,
            container="mp4",
            duration_s=duration_s,
        ),
        video=VideoStreamInfo(
            index=0,
            codec="h264",
            width=1920,
            height=1080,
            frame_rate=FrameRate(num=25, den=1),
        ),
    )


def _shot(index: int, start: float, end: float, **kwargs: object) -> Shot:
    return Shot(
        id=shot_id(index),
        index=index,
        span=TimeSpan(start_s=start, end_s=end),
        **kwargs,
    )


def _word(index: int, start: float, end: float, text: str) -> Word:
    return Word(id=word_id(index), span=TimeSpan(start_s=start, end_s=end), text=text)


# --- ids -------------------------------------------------------------------


def test_ids_are_zero_padded_and_sort_chronologically() -> None:
    """Lexicographic order must match chronological order, or diffs read wrong."""
    ids = [shot_id(i) for i in (0, 2, 10, 100)]

    assert ids == sorted(ids)
    assert ids[0] == "sh_0000"


def test_word_ids_are_wider_than_shot_ids() -> None:
    """An hour of speech is ~10k words; four digits would overflow into resorting."""
    assert len(word_id(0)) > len(shot_id(0))


def test_ids_widen_past_their_padding() -> None:
    assert shot_id(123456) == "sh_123456"


@pytest.mark.parametrize(
    "value", [shot_id(1), keyframe_id(1), word_id(1), segment_id(1), speaker_id(1)]
)
def test_generated_ids_validate(value: str) -> None:
    assert is_valid(value)


@pytest.mark.parametrize("value", ["", "sh", "sh_", "xx_0001", "sh_00a1", "SH_0001"])
def test_malformed_ids_rejected(value: str) -> None:
    assert not is_valid(value)


def test_negative_index_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        shot_id(-1)


# --- TimeSpan --------------------------------------------------------------


def test_span_is_half_open() -> None:
    """Half-open intervals let adjacent shots tile without overlapping."""
    span = TimeSpan(start_s=1.0, end_s=2.0)

    assert span.contains(1.0)
    assert not span.contains(2.0)


def test_adjacent_spans_do_not_overlap() -> None:
    left = TimeSpan(start_s=0.0, end_s=1.0)
    right = TimeSpan(start_s=1.0, end_s=2.0)

    assert not left.overlaps(right)


def test_overlapping_spans_detected() -> None:
    assert TimeSpan(start_s=0.0, end_s=1.5).overlaps(TimeSpan(start_s=1.0, end_s=2.0))


def test_reversed_span_rejected() -> None:
    with pytest.raises(ValidationError, match="before it starts"):
        TimeSpan(start_s=2.0, end_s=1.0)


def test_negative_span_rejected() -> None:
    with pytest.raises(ValidationError):
        TimeSpan(start_s=-1.0, end_s=1.0)


def test_zero_length_span_allowed() -> None:
    """A detector may legitimately report an instantaneous event."""
    assert TimeSpan(start_s=1.0, end_s=1.0).duration_s == 0.0


# --- context consistency ---------------------------------------------------


def test_minimal_context_is_probe_only() -> None:
    context = VideoContext(probe=_probe())

    assert context.has(ContextSection.PROBE)
    assert not context.has(ContextSection.SHOTS)
    assert context.duration_s == 10.0


def test_missing_reports_unpopulated_sections() -> None:
    context = VideoContext(probe=_probe())

    assert context.missing([ContextSection.SHOTS, ContextSection.TRANSCRIPT]) == [
        ContextSection.SHOTS,
        ContextSection.TRANSCRIPT,
    ]


def test_has_is_driven_by_provenance_not_emptiness() -> None:
    """A video with genuinely no speech differs from one never transcribed."""
    context = VideoContext(
        probe=_probe(),
        transcript=Transcript(),
        analyzers={
            "asr": AnalyzerRun(
                name="asr",
                version="1",
                params_hash=DIGEST,
                produced=(ContextSection.TRANSCRIPT,),
            )
        },
    )

    assert context.has(ContextSection.TRANSCRIPT)
    assert context.transcript is not None
    assert context.transcript.words == ()


def test_unordered_shots_rejected() -> None:
    with pytest.raises(ValidationError, match="chronological"):
        VideoContext(probe=_probe(), shots=(_shot(0, 5.0, 6.0), _shot(1, 1.0, 2.0)))


def test_duplicate_shot_ids_rejected() -> None:
    duplicate = _shot(0, 2.0, 3.0)

    with pytest.raises(ValidationError, match="duplicate id"):
        VideoContext(probe=_probe(), shots=(_shot(0, 0.0, 1.0), duplicate))


def test_overlapping_shots_rejected() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        VideoContext(probe=_probe(), shots=(_shot(0, 0.0, 2.0), _shot(1, 1.0, 3.0)))


def test_dangling_keyframe_reference_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown keyframe"):
        VideoContext(probe=_probe(), shots=(_shot(0, 0.0, 1.0, keyframe_id="kf_0009"),))


def test_dangling_shot_reference_on_keyframe_rejected() -> None:
    keyframe = Keyframe(
        id=keyframe_id(0), shot_id="sh_0009", t_s=0.5, cache_key=DIGEST, width=16, height=9
    )

    with pytest.raises(ValidationError, match="unknown shot"):
        VideoContext(probe=_probe(), keyframes=(keyframe,))


def test_valid_cross_references_accepted() -> None:
    keyframe = Keyframe(
        id=keyframe_id(0),
        shot_id=shot_id(0),
        t_s=0.5,
        cache_key=DIGEST,
        width=16,
        height=9,
    )
    context = VideoContext(
        probe=_probe(),
        shots=(_shot(0, 0.0, 1.0, keyframe_id=keyframe_id(0)),),
        keyframes=(keyframe,),
    )

    assert context.keyframe_by_id(keyframe_id(0)) is keyframe


def test_segment_referencing_unknown_word_rejected() -> None:
    segment = TranscriptSegment(
        id=segment_id(0),
        span=TimeSpan(start_s=0.0, end_s=1.0),
        text="hello",
        word_ids=("w_000099",),
    )

    with pytest.raises(ValidationError, match="unknown words"):
        Transcript(words=(_word(0, 0.0, 0.5, "hello"),), segments=(segment,))


def test_unordered_words_rejected() -> None:
    with pytest.raises(ValidationError, match="chronological"):
        Transcript(words=(_word(0, 5.0, 6.0, "late"), _word(1, 1.0, 2.0, "early")))


# --- lookups ---------------------------------------------------------------


def _talking_context() -> VideoContext:
    words = (
        _word(0, 0.0, 0.4, "our"),
        _word(1, 0.4, 0.9, "pricing"),
        _word(2, 0.9, 1.4, "is"),
        _word(3, 1.4, 2.0, "simple"),
        _word(4, 3.0, 3.5, "pricing"),
    )
    return VideoContext(
        probe=_probe(),
        shots=(_shot(0, 0.0, 2.0), _shot(1, 2.0, 4.0)),
        transcript=Transcript(language="en", words=words),
        audio_activity=(
            ActivitySpan(span=TimeSpan(start_s=0.0, end_s=2.0), rms_db=-18.0, is_silence=False),
            ActivitySpan(span=TimeSpan(start_s=2.0, end_s=3.0), rms_db=-60.0, is_silence=True),
        ),
        analyzers={
            "asr": AnalyzerRun(
                name="asr",
                version="1",
                params_hash=DIGEST,
                produced=(ContextSection.TRANSCRIPT,),
            )
        },
    )


def test_shot_at_finds_covering_shot() -> None:
    context = _talking_context()

    assert context.shot_at(0.5) is not None
    assert context.shot_at(0.5).id == shot_id(0)  # type: ignore[union-attr]
    assert context.shot_at(2.0).id == shot_id(1)  # type: ignore[union-attr]
    assert context.shot_at(99.0) is None


def test_shot_by_id_missing_returns_none() -> None:
    assert _talking_context().shot_by_id("sh_9999") is None


def test_words_between_includes_straddling_words() -> None:
    """A word crossing the boundary is part of what is said; dropping it truncates."""
    words = _talking_context().words_between(0.5, 1.0)

    assert [word.text for word in words] == ["pricing", "is"]


def test_text_between_joins_words() -> None:
    assert _talking_context().text_between(0.0, 2.0) == "our pricing is simple"


def test_words_between_without_transcript_is_empty() -> None:
    assert VideoContext(probe=_probe()).words_between(0.0, 5.0) == []


def test_find_text_returns_every_occurrence_with_timings() -> None:
    matches = list(_talking_context().find_text("pricing"))

    assert len(matches) == 2
    assert matches[0][0].span.start_s == 0.4
    assert matches[1][0].span.start_s == 3.0


def test_find_text_matches_multi_word_phrases() -> None:
    (match,) = list(_talking_context().find_text("pricing is simple"))

    assert match[0].span.start_s == 0.4
    assert match[-1].span.end_s == 2.0


def test_find_text_ignores_case_and_edge_punctuation() -> None:
    context = VideoContext(
        probe=_probe(),
        transcript=Transcript(words=(_word(0, 0.0, 0.5, "Pricing,"),)),
    )

    assert len(list(context.find_text("pricing"))) == 1


def test_find_text_without_transcript_yields_nothing() -> None:
    assert list(VideoContext(probe=_probe()).find_text("anything")) == []


def test_find_text_with_blank_needle_yields_nothing() -> None:
    assert list(_talking_context().find_text("   ")) == []


# --- determinism -----------------------------------------------------------


def test_context_round_trips_through_json() -> None:
    original = _talking_context()

    assert VideoContext.model_validate_json(original.model_dump_json()) == original


def test_serialization_is_byte_stable() -> None:
    """M1's exit criterion depends on this: no timestamps, no set ordering."""
    assert _talking_context().model_dump_json() == _talking_context().model_dump_json()


def test_analyzer_run_carries_no_timestamp() -> None:
    """A wall-clock field would break byte-stability and nothing consumes it."""
    fields = set(AnalyzerRun.model_fields)

    assert not {field for field in fields if "time" in field or field.endswith("at")}
