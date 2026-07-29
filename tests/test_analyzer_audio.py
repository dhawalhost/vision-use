"""Tests for audio activity detection."""

from __future__ import annotations

import subprocess
from itertools import pairwise
from pathlib import Path

import pytest
from vision_use.cache.store import CacheStore
from vision_use.context.analyzer import AnalyzerRequest
from vision_use.context.analyzers.audio import (
    SILENCE_FLOOR_DB,
    AudioActivityAnalyzer,
    _to_db,
    _Window,
)
from vision_use.context.registry import default_registry, registry
from vision_use.context.runner import analyze, context_digest
from vision_use.models.context import ContextSection, VideoContext
from vision_use.video.probe import probe

from tests.conftest import FFMPEG, MakeClip, requires_ffmpeg

S = ContextSection


@pytest.fixture(scope="session")
def gap_clip(media_dir: Path) -> Path:
    """A 3s clip: tone, one second of digital silence, tone.

    Built rather than committed so the silence is exact and the boundaries are
    known, which is what makes the assertions below meaningful.
    """
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")

    path = media_dir / "tone-gap.mp4"
    if path.exists():
        return path

    completed = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=25:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-filter_complex",
            "[1:a]volume=enable='between(t,1,2)':volume=0[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    return path


# --- construction ----------------------------------------------------------


def test_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="positive"):
        AudioActivityAnalyzer(window_s=0)


def test_rejects_negative_min_span() -> None:
    with pytest.raises(ValueError, match="negative"):
        AudioActivityAnalyzer(min_span_s=-1)


def test_params_cover_everything_that_changes_output() -> None:
    params = AudioActivityAnalyzer(window_s=0.05, silence_db=-30.0, min_span_s=0.5).params()

    assert params == {
        "window_s": 0.05,
        "silence_db": -30.0,
        "min_span_s": 0.5,
        "rate": 16_000,
    }


def test_declares_only_audio_activity() -> None:
    analyzer = AudioActivityAnalyzer()

    assert analyzer.produces == {S.AUDIO_ACTIVITY}
    assert analyzer.requires == frozenset()


# --- level conversion ------------------------------------------------------


def test_full_scale_is_zero_db() -> None:
    assert _to_db(1.0) == 0.0


def test_half_scale_is_about_minus_six_db() -> None:
    assert _to_db(0.5) == pytest.approx(-6.02, abs=0.01)


@pytest.mark.parametrize("rms", [0.0, -0.0])
def test_digital_silence_is_floored_not_infinite(rms: float) -> None:
    """log10(0) is undefined and an infinity would not survive a JSON round trip."""
    assert _to_db(rms) == SILENCE_FLOOR_DB


def test_very_quiet_is_clamped_to_the_floor() -> None:
    assert _to_db(1e-12) == SILENCE_FLOOR_DB


def test_levels_are_rounded_for_cross_platform_determinism() -> None:
    """Float summation is not bit-identical across platforms; rounding fixes that."""
    value = _to_db(0.333333333333)

    assert value == round(value, 2)


# --- span construction (no media needed) -----------------------------------


def _spans(
    windows: list[tuple[float, float, float]],
    duration: float,
    **kwargs: float,
) -> list[tuple[float, float, bool]]:
    analyzer = AudioActivityAnalyzer(**kwargs)
    built = analyzer._build_spans(
        [_Window(start_s=a, end_s=b, rms_db=db) for a, b, db in windows], duration
    )
    return [(s.span.start_s, s.span.end_s, s.is_silence) for s in built]


def _run(
    start: float, count: int, db: float, window: float = 0.02
) -> list[tuple[float, float, float]]:
    return [
        (round(start + i * window, 6), round(start + (i + 1) * window, 6), db)
        for i in range(count)
    ]


def test_no_windows_yields_no_spans() -> None:
    assert _spans([], 5.0) == []


def test_uniform_audio_is_one_span() -> None:
    spans = _spans(_run(0.0, 50, -20.0), 1.0)

    assert spans == [(0.0, 1.0, False)]


def test_silence_between_speech_is_detected() -> None:
    windows = _run(0.0, 50, -20.0) + _run(1.0, 50, -80.0) + _run(2.0, 50, -20.0)

    spans = _spans(windows, 3.0)

    assert [s[2] for s in spans] == [False, True, False]
    assert spans[1][0] == pytest.approx(1.0)
    assert spans[1][1] == pytest.approx(2.0)


def test_spans_tile_the_source_without_gaps() -> None:
    windows = _run(0.0, 50, -20.0) + _run(1.0, 50, -80.0)

    spans = _spans(windows, 2.0)

    assert spans[0][0] == 0.0
    assert spans[-1][1] == 2.0
    for left, right in pairwise(spans):
        assert left[1] == right[0]


def test_brief_flicker_is_absorbed() -> None:
    """A single quiet window between speech is a plosive, not a silence."""
    windows = _run(0.0, 50, -20.0) + _run(1.0, 2, -80.0) + _run(1.04, 50, -20.0)

    spans = _spans(windows, 3.0, min_span_s=0.2)

    assert spans == [(0.0, 3.0, False)]


def test_no_two_adjacent_spans_share_a_kind() -> None:
    """Regression: absorbing a runt used to leave two touching same-kind spans."""
    windows = _run(0.0, 50, -20.0) + _run(1.0, 2, -80.0) + _run(1.04, 50, -20.0)

    spans = _spans(windows, 3.0, min_span_s=0.2)

    kinds = [s[2] for s in spans]
    assert all(a != b for a, b in pairwise(kinds))


def test_short_leading_run_merges_forwards() -> None:
    """A leading runt has no predecessor to absorb it, so it merges into its successor."""
    windows = _run(0.0, 2, -80.0) + _run(0.04, 50, -20.0)

    spans = _spans(windows, 2.0, min_span_s=0.2)

    assert spans == [(0.0, 2.0, False)]


def test_threshold_is_inclusive() -> None:
    """A window exactly at the threshold counts as silence, per the documented rule."""
    spans = _spans(_run(0.0, 50, -40.0), 1.0, silence_db=-40.0)

    assert spans[0][2] is True


def test_threshold_shifts_classification() -> None:
    quiet = _run(0.0, 50, -35.0)

    assert _spans(quiet, 1.0, silence_db=-40.0)[0][2] is False
    assert _spans(quiet, 1.0, silence_db=-30.0)[0][2] is True


def test_final_span_extends_to_full_duration() -> None:
    """Windows stop at the last decoded sample; the span must still cover the source."""
    spans = _spans(_run(0.0, 10, -20.0), 5.0)

    assert spans[-1][1] == 5.0


# --- against real media ----------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_detects_the_silent_gap(gap_clip: Path, tmp_path: Path) -> None:
    context = analyze(
        gap_clip,
        [S.AUDIO_ACTIVITY],
        cache=CacheStore(tmp_path / "cache"),
        registry=default_registry([AudioActivityAnalyzer()]),
    )

    silences = [s for s in context.audio_activity if s.is_silence]
    assert len(silences) == 1
    assert silences[0].span.start_s == pytest.approx(1.0, abs=0.15)
    assert silences[0].span.end_s == pytest.approx(2.0, abs=0.15)


@pytest.mark.integration
@requires_ffmpeg
def test_silence_is_much_quieter_than_speech(gap_clip: Path, tmp_path: Path) -> None:
    context = analyze(
        gap_clip,
        [S.AUDIO_ACTIVITY],
        cache=CacheStore(tmp_path / "cache"),
        registry=default_registry([AudioActivityAnalyzer()]),
    )

    silent = next(s for s in context.audio_activity if s.is_silence)
    active = next(s for s in context.audio_activity if not s.is_silence)

    assert silent.rms_db < active.rms_db - 30


@pytest.mark.integration
@requires_ffmpeg
def test_activity_spans_cover_the_source(gap_clip: Path, tmp_path: Path) -> None:
    context = analyze(
        gap_clip,
        [S.AUDIO_ACTIVITY],
        cache=CacheStore(tmp_path / "cache"),
        registry=default_registry([AudioActivityAnalyzer()]),
    )

    spans = context.audio_activity
    assert spans[0].span.start_s == 0.0
    assert spans[-1].span.end_s == pytest.approx(context.duration_s, abs=0.05)


@pytest.mark.integration
@requires_ffmpeg
def test_silent_video_yields_empty_activity_and_a_warning(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    """Video with no audio track is legitimate input, not an error."""
    clip = make_clip(name="noaudio", with_audio=False)
    request = AnalyzerRequest(
        source=clip,
        context=VideoContext(probe=probe(clip)),
        cache=CacheStore(tmp_path / "cache"),
    )

    result = AudioActivityAnalyzer().run(request)

    assert result.audio_activity == ()
    assert result.warnings
    assert "no audio" in result.warnings[0]


@pytest.mark.integration
@requires_ffmpeg
def test_analysed_but_silent_still_records_provenance(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    """'No audio' must stay distinguishable from 'never analysed'."""
    context = analyze(
        make_clip(name="noaudio2", with_audio=False),
        [S.AUDIO_ACTIVITY],
        cache=CacheStore(tmp_path / "cache"),
        registry=default_registry([AudioActivityAnalyzer()]),
    )

    assert context.has(S.AUDIO_ACTIVITY)
    assert context.audio_activity == ()


@pytest.mark.integration
@requires_ffmpeg
def test_analysis_is_byte_identical_across_runs(gap_clip: Path, tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    audio_registry = default_registry([AudioActivityAnalyzer()])

    first = analyze(gap_clip, [S.AUDIO_ACTIVITY], cache=store, registry=audio_registry)
    second = analyze(gap_clip, [S.AUDIO_ACTIVITY], cache=store, registry=audio_registry)

    assert first.model_dump_json() == second.model_dump_json()
    assert context_digest(first) == context_digest(second)


@pytest.mark.integration
@requires_ffmpeg
def test_cold_and_warm_cache_agree(gap_clip: Path, tmp_path: Path) -> None:
    audio_registry = default_registry([AudioActivityAnalyzer()])

    cold = analyze(
        gap_clip, [S.AUDIO_ACTIVITY], cache=CacheStore(tmp_path / "a"), registry=audio_registry
    )
    warm_store = CacheStore(tmp_path / "b")
    analyze(gap_clip, [S.AUDIO_ACTIVITY], cache=warm_store, registry=audio_registry)
    warm = analyze(gap_clip, [S.AUDIO_ACTIVITY], cache=warm_store, registry=audio_registry)

    assert context_digest(cold) == context_digest(warm)


@pytest.mark.integration
@requires_ffmpeg
def test_both_sections_compose_in_one_context(gap_clip: Path, tmp_path: Path) -> None:
    """Two independent analyzers must populate one context without interfering."""
    context = analyze(
        gap_clip, [S.SHOTS, S.AUDIO_ACTIVITY], cache=CacheStore(tmp_path / "cache")
    )

    assert context.has(S.SHOTS)
    assert context.has(S.AUDIO_ACTIVITY)
    assert context.shots
    assert context.audio_activity
    assert len(context.analyzers) == 2


def test_registered_as_a_builtin() -> None:
    assert "audio.activity" in registry
    assert registry.producer_of(S.AUDIO_ACTIVITY) is not None
