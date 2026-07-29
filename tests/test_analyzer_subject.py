"""Tests for the subject analyzer.

What this measures is *where the frame changes*, which is a stand-in for where
the subject is, not a detection of one. The tests are written to that claim:
they assert it finds motion and that it says so honestly when there is none,
not that it finds people.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from vision_use.cache.store import CacheStore
from vision_use.context.analyzer import AnalyzerRequest
from vision_use.context.analyzers.subject import SubjectAnalyzer
from vision_use.context.runner import analyze
from vision_use.models.context import ContextSection, Shot, TimeSpan, VideoContext
from vision_use.video.probe import probe

from tests.conftest import requires_ffmpeg


def _clip(target: Path, filters: list[str], seconds: float = 3.0) -> Path:
    """Render a fixture. Generated, never committed, per CONTRIBUTING.md."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            *filters,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        capture_output=True,
        check=True,
    )
    return target


def _moving_box(target: Path, x: int) -> Path:
    """A still background with one box moving at horizontal position ``x``."""
    return _clip(
        target,
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:size=640x360:rate=25:duration=3",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:size=80x120:rate=25:duration=3",
            "-filter_complex",
            f"[0][1]overlay=x={x}:y=120+20*sin(2*PI*t)",
        ],
    )


def _still(target: Path) -> Path:
    return _clip(target, ["-f", "lavfi", "-i", "color=c=gray:size=640x360:rate=25:duration=3"])


def _subject_of(source: Path, cache: Path) -> tuple[float, float]:
    """Return the located centre and confidence for a single-shot clip."""
    context = analyze(source, [ContextSection.SUBJECT], cache=CacheStore(cache))
    span = context.subject[0]
    return span.centre_x, span.confidence or 0.0


# --- locating a subject ----------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_a_subject_on_the_right_is_found_on_the_right(tmp_path: Path) -> None:
    """The whole point: a crop must not assume the subject was centred."""
    source = _moving_box(tmp_path / "right.mp4", x=460)

    centre_x, confidence = _subject_of(source, tmp_path / "cache")

    # True centre is (460 + 40) / 640 = 0.781.
    assert centre_x == pytest.approx(0.78, abs=0.06)
    assert confidence > 0.5


@pytest.mark.integration
@requires_ffmpeg
def test_a_subject_on_the_left_is_found_on_the_left(tmp_path: Path) -> None:
    source = _moving_box(tmp_path / "left.mp4", x=60)

    centre_x, confidence = _subject_of(source, tmp_path / "cache")

    # True centre is (60 + 40) / 640 = 0.156.
    assert centre_x == pytest.approx(0.16, abs=0.06)
    assert confidence > 0.5


@pytest.mark.integration
@requires_ffmpeg
def test_the_two_are_actually_distinguished(tmp_path: Path) -> None:
    """A measurement that returns the same answer either way measures nothing."""
    left, _ = _subject_of(_moving_box(tmp_path / "l.mp4", x=60), tmp_path / "cache")
    right, _ = _subject_of(_moving_box(tmp_path / "r.mp4", x=460), tmp_path / "cache")

    assert right - left > 0.4


# --- saying so when there is nothing ---------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_a_still_frame_reports_no_confidence(tmp_path: Path) -> None:
    """Nothing moved, so there is no evidence -- not weak evidence."""
    centre_x, confidence = _subject_of(_still(tmp_path / "still.mp4"), tmp_path / "cache")

    assert confidence == 0.0
    assert centre_x == 0.5


@pytest.mark.integration
@requires_ffmpeg
def test_motion_everywhere_is_low_confidence(tmp_path: Path) -> None:
    """`testsrc` changes across the whole frame; that is not a subject."""
    source = _clip(
        tmp_path / "busy.mp4",
        ["-f", "lavfi", "-i", "testsrc=size=640x360:rate=25:duration=3"],
    )

    _, confidence = _subject_of(source, tmp_path / "cache")

    assert confidence < 0.5


def test_no_shots_means_no_subject_and_a_warning(tmp_path: Path) -> None:
    """Averaging across a whole file with no shot structure would be a guess."""
    analyzer = SubjectAnalyzer()
    source = _still(tmp_path / "still.mp4") if _has_ffmpeg() else tmp_path / "absent.mp4"
    if not source.exists():
        pytest.skip("ffmpeg is not installed")

    context = VideoContext(probe=probe(source))
    result = analyzer.run(
        AnalyzerRequest(source=source, context=context, cache=CacheStore(tmp_path / "cache"))
    )

    assert result.subject == ()
    assert result.warnings


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


# --- contract --------------------------------------------------------------


def test_the_analyzer_declares_what_it_needs() -> None:
    """Shots, because averaging a subject across a cut places the crop nowhere."""
    analyzer = SubjectAnalyzer()

    assert ContextSection.SHOTS in analyzer.requires
    assert ContextSection.SUBJECT in analyzer.produces


def test_params_cover_what_changes_the_result() -> None:
    """An incidental value causes needless cache misses; a missing one is worse."""
    assert set(SubjectAnalyzer().params()) == {
        "sample_rate_hz",
        "max_samples",
        "flat_concentration",
    }


@pytest.mark.integration
@requires_ffmpeg
def test_a_subject_span_covers_its_shot(tmp_path: Path) -> None:
    source = _moving_box(tmp_path / "span.mp4", x=460)

    context = analyze(source, [ContextSection.SUBJECT], cache=CacheStore(tmp_path / "cache"))

    shot: Shot = context.shots[0]
    span: TimeSpan = context.subject[0].span
    assert context.subject[0].shot_id == shot.id
    assert (span.start_s, span.end_s) == (shot.span.start_s, shot.span.end_s)


@pytest.mark.integration
@requires_ffmpeg
def test_the_result_is_deterministic(tmp_path: Path) -> None:
    """Two analyses of the same file must agree, or nothing can be cached."""
    source = _moving_box(tmp_path / "det.mp4", x=460)

    first = analyze(source, [ContextSection.SUBJECT], cache=CacheStore(tmp_path / "a"))
    second = analyze(source, [ContextSection.SUBJECT], cache=CacheStore(tmp_path / "b"))

    assert first.subject == second.subject
