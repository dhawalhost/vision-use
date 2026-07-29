"""Tests for the media description models."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError
from vision_use.models.media import (
    AudioStreamInfo,
    FrameRate,
    MediaProbe,
    SourceInfo,
    VideoStreamInfo,
)


def _video(**overrides: object) -> VideoStreamInfo:
    base: dict[str, object] = {
        "index": 0,
        "codec": "h264",
        "width": 1920,
        "height": 1080,
        "frame_rate": FrameRate(num=25, den=1),
    }
    return VideoStreamInfo(**{**base, **overrides})


def test_ntsc_rate_is_exact() -> None:
    """The whole reason frame rate is rational: 30000/1001 has no float form."""
    rate = FrameRate(num=30000, den=1001)

    assert rate.fraction == Fraction(30000, 1001)
    assert rate.frame_duration_s == Fraction(1001, 30000)
    assert str(rate) == "30000/1001"


def test_integer_rate_renders_without_denominator() -> None:
    assert str(FrameRate(num=25, den=1)) == "25"


def test_frame_duration_accumulates_without_drift() -> None:
    """Summing exact frame durations must land on an exact second boundary."""
    rate = FrameRate(num=30000, den=1001)
    total = sum((rate.frame_duration_s for _ in range(30000)), Fraction(0))

    assert total == Fraction(1001)


def test_from_fraction_round_trips() -> None:
    assert FrameRate.from_fraction(Fraction(24000, 1001)).fraction == Fraction(24000, 1001)


def test_zero_denominator_rejected() -> None:
    with pytest.raises(ValidationError):
        FrameRate(num=25, den=0)


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [(0, (1920, 1080)), (90, (1080, 1920)), (180, (1920, 1080)), (270, (1080, 1920))],
)
def test_rotation_swaps_display_dimensions(rotation: int, expected: tuple[int, int]) -> None:
    stream = _video(rotation=rotation)

    assert (stream.display_width, stream.display_height) == expected


def test_portrait_detection_uses_rotation() -> None:
    """Phone footage is stored landscape with a rotation flag; naive checks fail."""
    assert _video(rotation=90).is_portrait
    assert not _video(rotation=0).is_portrait


def test_invalid_rotation_rejected() -> None:
    with pytest.raises(ValidationError):
        _video(rotation=45)


def test_source_requires_full_sha256() -> None:
    with pytest.raises(ValidationError):
        SourceInfo(
            path=Path("a.mp4"),
            sha256="abc",
            size_bytes=1,
            container="mp4",
            duration_s=1.0,
        )


def _source() -> SourceInfo:
    return SourceInfo(
        path=Path("a.mp4"),
        sha256="0" * 64,
        size_bytes=1,
        container="mp4",
        duration_s=1.0,
    )


def test_probe_rejects_colliding_stream_indices() -> None:
    audio = AudioStreamInfo(index=0, codec="aac", sample_rate=48000, channels=2)

    with pytest.raises(ValidationError):
        MediaProbe(source=_source(), video=_video(index=0), audio=audio)


def test_has_audio_reflects_presence() -> None:
    silent = MediaProbe(source=_source(), video=_video())
    assert not silent.has_audio

    audio = AudioStreamInfo(index=1, codec="aac", sample_rate=48000, channels=2)
    assert MediaProbe(source=_source(), video=_video(), audio=audio).has_audio


def test_probe_round_trips_through_json() -> None:
    """Every document VisionUse emits must survive a JSON round trip unchanged."""
    original = MediaProbe(
        source=_source(),
        video=_video(frame_rate=FrameRate(num=30000, den=1001), rotation=90),
        audio=AudioStreamInfo(index=1, codec="aac", sample_rate=48000, channels=2),
    )

    assert MediaProbe.model_validate_json(original.model_dump_json()) == original
