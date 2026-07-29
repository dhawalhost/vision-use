"""Tests for silence removal.

The measurement that matters is that speech survives. Cutting a word in half is
the failure a viewer notices, and it is what this skill's own validation exists
to catch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner
from vision_use.cache.store import CacheStore
from vision_use.cli.main import app
from vision_use.models.context import (
    ActivitySpan,
    AnalyzerRun,
    ContextSection,
    TimeSpan,
    VideoContext,
)
from vision_use.models.ids import clip_id, source_id, track_id
from vision_use.models.media import FrameRate, MediaProbe, SourceInfo, VideoStreamInfo
from vision_use.models.plan import Clip, EditPlan, Source, Timeline, Track
from vision_use.planning import save_plan, straight_through
from vision_use.skills import SkillOutcome, SkillRequest, apply_patch, run_skill
from vision_use.skills.silence import MIN_SPEECH_KEPT, RemoveSilence, _keeps, _Span
from vision_use.timebase import frame_index_at, frame_start_s
from vision_use.video.probe import probe

from tests.conftest import FFMPEG, requires_ffmpeg

runner = CliRunner()
PAL = FrameRate(num=25, den=1)
DIGEST = "f" * 64
PRESENT = Path(__file__)


def _context(*spans: tuple[float, float, bool], duration_s: float = 10.0) -> VideoContext:
    """Build a context whose audio activity is exactly ``spans``."""
    return VideoContext(
        probe=MediaProbe(
            source=SourceInfo(
                path=PRESENT,
                sha256=DIGEST,
                size_bytes=1,
                container="mp4",
                duration_s=duration_s,
            ),
            video=VideoStreamInfo(index=0, codec="h264", width=640, height=480, frame_rate=PAL),
        ),
        audio_activity=tuple(
            ActivitySpan(
                span=TimeSpan(start_s=start, end_s=end),
                rms_db=-60.0 if silent else -20.0,
                is_silence=silent,
            )
            for start, end, silent in spans
        ),
        analyzers={
            "audio.activity": AnalyzerRun(
                name="audio.activity",
                version="1",
                params_hash=DIGEST,
                produced=(ContextSection.AUDIO_ACTIVITY,),
            )
        },
    )


def _plan(duration_s: float = 10.0) -> EditPlan:
    source = Source(id=source_id(0), path=PRESENT, sha256=DIGEST, duration_s=duration_s)
    return EditPlan(
        sources={source.id: source},
        timeline=Timeline(
            frame_rate=PAL,
            width=640,
            height=480,
            tracks=(
                Track(
                    id=track_id(0),
                    kind="video",
                    clips=(
                        Clip(
                            id=clip_id(0),
                            source_id=source.id,
                            source_in_s=0.0,
                            source_out_s=duration_s,
                            timeline_start_s=0.0,
                        ),
                    ),
                ),
            ),
        ),
    )


# --- construction ----------------------------------------------------------


def test_rejects_empty_thresholds() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RemoveSilence(thresholds=())


def test_rejects_negative_padding() -> None:
    with pytest.raises(ValueError, match="negative"):
        RemoveSilence(pad_s=-1.0)


def test_requires_audio_activity() -> None:
    """The runtime backfills it; the skill must not analyse audio itself."""
    assert RemoveSilence().meta.requires == {ContextSection.AUDIO_ACTIVITY}


# --- span arithmetic -------------------------------------------------------


def _clip(in_s: float, out_s: float) -> Clip:
    return Clip(
        id=clip_id(0),
        source_id=source_id(0),
        source_in_s=in_s,
        source_out_s=out_s,
        timeline_start_s=0.0,
    )


def test_no_silence_keeps_the_whole_clip() -> None:
    assert _keeps(_clip(0.0, 10.0), [], 0.1) == (_Span(0.0, 10.0),)


def test_a_middle_silence_splits_the_clip() -> None:
    keeps = _keeps(_clip(0.0, 10.0), [_Span(4.0, 6.0)], 0.0)

    assert keeps == (_Span(0.0, 4.0), _Span(6.0, 10.0))


def test_padding_keeps_air_either_side_of_the_cut() -> None:
    """Cutting the instant the waveform drops clips the tail off the last word."""
    keeps = _keeps(_clip(0.0, 10.0), [_Span(4.0, 6.0)], 0.5)

    assert keeps == (_Span(0.0, 4.5), _Span(5.5, 10.0))


def test_padding_never_inverts_a_short_silence() -> None:
    """Padding wider than the silence itself must not produce a negative span."""
    keeps = _keeps(_clip(0.0, 10.0), [_Span(4.0, 4.2)], 0.5)

    assert keeps == (_Span(0.0, 10.0),)


def test_leading_silence_is_trimmed() -> None:
    keeps = _keeps(_clip(0.0, 10.0), [_Span(0.0, 3.0)], 0.0)

    assert keeps == (_Span(3.0, 10.0),)


def test_trailing_silence_is_trimmed() -> None:
    keeps = _keeps(_clip(0.0, 10.0), [_Span(7.0, 10.0)], 0.0)

    assert keeps == (_Span(0.0, 7.0),)


def test_runt_keeps_are_dropped() -> None:
    """A span of a few frames between cuts is a flash, not a shot."""
    keeps = _keeps(_clip(0.0, 10.0), [_Span(0.0, 4.0), _Span(4.05, 8.0)], 0.0)

    assert all(span.duration_s >= 0.2 for span in keeps)


def test_multiple_silences_produce_multiple_keeps() -> None:
    keeps = _keeps(_clip(0.0, 12.0), [_Span(3.0, 4.0), _Span(7.0, 8.0)], 0.0)

    assert len(keeps) == 3


# --- lifecycle -------------------------------------------------------------


def _run(context: VideoContext, plan: EditPlan) -> SkillOutcome:
    return run_skill(
        RemoveSilence(),
        PRESENT,
        plan,
        SkillRequest(prompt="cut the dead air"),
        context=context,
    )


def test_a_long_silence_is_cut() -> None:
    context = _context((0.0, 4.0, False), (4.0, 7.0, True), (7.0, 10.0, False))

    outcome = _run(context, _plan())

    assert outcome.accepted is not None
    assert outcome.plan.duration_s < 10.0


def test_a_clip_with_no_silence_is_left_alone() -> None:
    """Nothing to cut is a legitimate outcome, not a failure."""
    context = _context((0.0, 10.0, False))

    outcome = _run(context, _plan())

    assert outcome.accepted is None
    assert outcome.attempts == ()


def test_short_pauses_are_not_cut() -> None:
    """A pause between sentences is speech, not dead air."""
    context = _context((0.0, 4.0, False), (4.0, 4.2, True), (4.2, 10.0, False))

    outcome = _run(context, _plan())

    assert outcome.accepted is None


def test_the_cautious_threshold_is_tried_first() -> None:
    context = _context((0.0, 4.0, False), (4.0, 7.0, True), (7.0, 10.0, False))

    outcome = _run(context, _plan())

    assert outcome.accepted is not None
    assert outcome.accepted.strategy.name == "over-0.8s"


def test_the_cut_becomes_several_clips_laid_end_to_end() -> None:
    context = _context(
        (0.0, 3.0, False),
        (3.0, 5.0, True),
        (5.0, 7.0, False),
        (7.0, 9.0, True),
        (9.0, 10.0, False),
    )

    outcome = _run(context, _plan())

    assert outcome.accepted is not None
    clips = outcome.plan.timeline.clips()
    assert len(clips) == 3
    cursor = 0.0
    for clip in clips:
        assert clip.timeline_start_s == pytest.approx(cursor, abs=0.05)
        cursor += clip.duration_s


def test_validation_reports_how_much_speech_survived() -> None:
    context = _context((0.0, 4.0, False), (4.0, 7.0, True), (7.0, 10.0, False))

    outcome = _run(context, _plan())

    assert outcome.accepted is not None
    assert outcome.accepted is not None
    kept = next(
        item for item in outcome.accepted.report.measurements if item.name == "speech_kept"
    )
    assert kept.value >= MIN_SPEECH_KEPT


def test_the_patch_explains_itself() -> None:
    context = _context((0.0, 4.0, False), (4.0, 7.0, True), (7.0, 10.0, False))

    outcome = _run(context, _plan())

    assert outcome.accepted is not None
    assert "cut silences longer than" in outcome.accepted.patch.rationale


def test_an_already_cut_timeline_is_left_alone() -> None:
    """Re-cutting needs a rule for the existing boundaries; there is no good default."""
    context = _context((0.0, 4.0, False), (4.0, 7.0, True), (7.0, 10.0, False))
    once = _run(context, _plan())

    twice = _run(context, once.plan)

    assert twice.accepted is None


def test_the_edit_is_reversible() -> None:
    """Like every patch: the retry loop depends on it."""
    plan = _plan()
    context = _context((0.0, 4.0, False), (4.0, 7.0, True), (7.0, 10.0, False))
    outcome = _run(context, plan)

    assert outcome.accepted is not None
    restored = apply_patch(outcome.plan, outcome.accepted.inverse).plan
    assert restored.model_dump_json() == plan.model_dump_json()


# --- against real media ----------------------------------------------------


@pytest.fixture(scope="session")
def gap_clip(media_dir: Path) -> Path:
    """Six seconds: tone, two seconds of exact silence, tone."""
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")

    path = media_dir / "silence-gap.mp4"
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
            "testsrc=size=320x240:rate=25:duration=6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=6",
            "-filter_complex",
            "[1:a]volume=enable='between(t,1.5,3.5)':volume=0[a]",
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


@pytest.mark.integration
@requires_ffmpeg
def test_real_silence_is_found_and_cut(gap_clip: Path, tmp_path: Path) -> None:
    """End to end: the context is backfilled and the gap disappears."""
    plan = straight_through(gap_clip)

    outcome = run_skill(RemoveSilence(), gap_clip, plan, cache=CacheStore(tmp_path / "cache"))

    assert outcome.accepted is not None
    assert outcome.plan.duration_s < plan.duration_s
    assert len(outcome.plan.timeline.clips()) == 2


@pytest.mark.integration
@requires_ffmpeg
def test_cuts_land_on_frame_boundaries(gap_clip: Path, tmp_path: Path) -> None:
    """An audio-derived cut point falls mid-frame unless it is snapped."""

    plan = straight_through(gap_clip)
    outcome = run_skill(RemoveSilence(), gap_clip, plan, cache=CacheStore(tmp_path / "cache"))

    rate = plan.timeline.frame_rate
    assert outcome.accepted is not None
    for clip in outcome.plan.timeline.clips():
        for boundary in (clip.source_in_s, clip.source_out_s):
            assert boundary == pytest.approx(
                frame_start_s(frame_index_at(boundary, rate), rate), abs=1e-6
            )


@pytest.mark.integration
@requires_ffmpeg
def test_the_cut_plan_renders(gap_clip: Path, tmp_path: Path) -> None:
    """The whole point: the edit has to survive all the way to a file."""
    plan_path = tmp_path / "plan.json"
    save_plan(straight_through(gap_clip), plan_path)

    applied = runner.invoke(app, ["apply", str(plan_path), "--skill", "edit.remove_silence"])
    assert applied.exit_code == 0

    out = tmp_path / "out.mp4"
    rendered = runner.invoke(app, ["render", str(plan_path), "-o", str(out)])

    assert rendered.exit_code == 0
    before, after = probe(gap_clip), probe(out)
    assert before.video.frame_count is not None
    assert after.video.frame_count is not None
    assert after.video.frame_count < before.video.frame_count
    assert after.has_audio


@pytest.mark.integration
@requires_ffmpeg
def test_the_skill_is_registered(gap_clip: Path) -> None:
    del gap_clip
    result = runner.invoke(app, ["skills"])

    assert "edit.remove_silence" in result.stdout
    assert "audio_activity" in result.stdout


def test_many_kept_spans_apply_without_a_false_overlap() -> None:
    """Regression: found end to end on a real 40-cut, 2-minute source.

    Cutting many alternating silences produces many clips laid end to end from
    independently quantised boundaries, which used to trip `Clip.overlaps` on
    float subtraction noise thousands of times smaller than a frame -- and the
    real fix is in `Clip.overlaps`, not in this skill; this pins the case that
    found it.
    """
    spans: list[tuple[float, float, bool]] = []
    t = 0.0
    for _ in range(20):
        spans.append((t, t + 3.0, False))
        spans.append((t + 3.0, t + 6.3, True))
        t += 6.3
    context = _context(*spans, duration_s=t)
    plan = _plan(duration_s=t)

    outcome = run_skill(RemoveSilence(), PRESENT, plan, SkillRequest(), context=context)

    assert outcome.accepted is not None
    assert len(outcome.plan.timeline.clips()) > 1
