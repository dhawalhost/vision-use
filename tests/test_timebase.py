"""Tests for time quantisation and frame mapping."""

from __future__ import annotations

import pytest
from vision_use.models.media import FrameRate
from vision_use.timebase import (
    TIME_PRECISION,
    duration_frames,
    frame_index_at,
    frame_start_s,
    quantize_s,
    snap_to_frame,
)

PAL = FrameRate(num=25, den=1)
NTSC = FrameRate(num=30000, den=1001)


def test_quantize_rounds_to_precision() -> None:
    assert quantize_s(1.23456789) == 1.234568


def test_quantize_is_idempotent() -> None:
    """Re-quantising must be a no-op, or repeated merges would drift."""
    once = quantize_s(0.1 + 0.2)

    assert quantize_s(once) == once


def test_quantize_kills_float_representation_noise() -> None:
    """The whole point: 0.1+0.2 must not serialise as 0.30000000000000004."""
    assert quantize_s(0.1 + 0.2) == 0.3


def test_precision_is_microseconds() -> None:
    assert TIME_PRECISION == 6


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, 0), (0.039, 0), (0.04, 1), (1.0, 25), (2.5, 62)],
)
def test_frame_index_at_pal(seconds: float, expected: int) -> None:
    assert frame_index_at(seconds, PAL) == expected


def test_frame_index_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match="negative"):
        frame_index_at(-0.1, PAL)


def test_frame_start_round_trips() -> None:
    for index in (0, 1, 999, 100_000):
        assert frame_index_at(frame_start_s(index, PAL), PAL) == index


def test_frame_start_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="negative"):
        frame_start_s(-1, PAL)


def test_ntsc_round_trip_survives_an_hour() -> None:
    """Float rate arithmetic drifts a whole frame over this span; rational does not."""
    index = 107_892  # one hour at 30000/1001

    assert frame_index_at(frame_start_s(index, NTSC), NTSC) == index


def test_snap_lands_on_a_frame_boundary() -> None:
    snapped = snap_to_frame(1.2345, PAL)

    assert snapped == frame_start_s(frame_index_at(1.2345, PAL), PAL)
    assert snapped <= 1.2345


def test_snap_is_idempotent() -> None:
    once = snap_to_frame(3.7777, NTSC)

    assert snap_to_frame(once, NTSC) == once


@pytest.mark.parametrize(("seconds", "expected"), [(0.0, 0), (1.0, 25), (1.999, 49)])
def test_duration_frames(seconds: float, expected: int) -> None:
    assert duration_frames(seconds, PAL) == expected


def test_duration_frames_rejects_negative() -> None:
    with pytest.raises(ValueError, match="negative"):
        duration_frames(-1.0, PAL)
