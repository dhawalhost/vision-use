"""Integration tests for media probing.

These need real media, so they generate clips with ffmpeg and skip when it is
unavailable.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from vision_use.errors import ProbeError, UnsupportedMediaError
from vision_use.video.probe import _normalise_rotation, probe

from tests.conftest import MakeClip, requires_ffmpeg

pytestmark = pytest.mark.integration


@requires_ffmpeg
def test_probe_reads_dimensions_and_rate(make_clip: MakeClip) -> None:
    result = probe(make_clip(width=320, height=240, fps="25"))

    assert (result.video.width, result.video.height) == (320, 240)
    assert result.video.frame_rate.fraction == Fraction(25, 1)
    assert result.video.codec == "h264"


@requires_ffmpeg
def test_probe_reads_ntsc_rate_exactly(make_clip: MakeClip) -> None:
    """A float round trip would turn 30000/1001 into 29.97 and lose exactness."""
    result = probe(make_clip(name="ntsc", fps="30000/1001"))

    assert result.video.frame_rate.fraction == Fraction(30000, 1001)


@requires_ffmpeg
def test_probe_detects_audio(make_clip: MakeClip) -> None:
    result = probe(make_clip(with_audio=True))

    assert result.has_audio
    assert result.audio is not None
    assert result.audio.channels >= 1
    assert result.audio.index != result.video.index


@requires_ffmpeg
def test_probe_handles_silent_media(make_clip: MakeClip) -> None:
    result = probe(make_clip(name="silent", with_audio=False))

    assert not result.has_audio
    assert result.audio is None


@requires_ffmpeg
def test_probe_reports_duration(make_clip: MakeClip) -> None:
    result = probe(make_clip(name="twosec", seconds=2.0))

    assert result.source.duration_s == pytest.approx(2.0, abs=0.15)


@requires_ffmpeg
def test_probe_is_deterministic(make_clip: MakeClip) -> None:
    """Probing is a cache input, so identical input must give identical output."""
    clip = make_clip(name="determinism")

    assert probe(clip) == probe(clip)


@requires_ffmpeg
def test_skip_hash_zeroes_digest(make_clip: MakeClip) -> None:
    result = probe(make_clip(), compute_hash=False)

    assert result.source.sha256 == "0" * 64


@requires_ffmpeg
def test_frame_count_is_plausible(make_clip: MakeClip) -> None:
    result = probe(make_clip(name="counted", seconds=2.0, fps="25"))

    assert result.video.frame_count is not None
    assert 45 <= result.video.frame_count <= 55


def test_missing_file_raises_with_remedy(tmp_path: Path) -> None:
    with pytest.raises(ProbeError) as excinfo:
        probe(tmp_path / "absent.mp4")

    assert excinfo.value.remedy is not None


def test_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ProbeError):
        probe(tmp_path)


def test_non_media_raises(tmp_path: Path) -> None:
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a video", encoding="utf-8")

    with pytest.raises((ProbeError, UnsupportedMediaError)):
        probe(junk)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, 0), (90, 90), (-90, 270), (270, 270), (360, 0), (-270, 90), (45, 0), (89.6, 90)],
)
def test_rotation_normalisation(raw: float, expected: int) -> None:
    assert _normalise_rotation(raw) == expected
