"""Tests for reframing.

The failure this guards against is silent: a crop that asks for pixels outside
the frame produces a black bar, which is exactly the letterboxing the skill
exists to remove.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from vision_use.errors import VisionUseError
from vision_use.models.context import (
    AnalyzerRun,
    ContextSection,
    SubjectPoint,
    SubjectSpan,
    TimeSpan,
    VideoContext,
)
from vision_use.models.ids import clip_id, shot_id, source_id, track_id
from vision_use.models.media import FrameRate, MediaProbe, SourceInfo, VideoStreamInfo
from vision_use.models.plan import Clip, EditPlan, Effect, Source, Timeline, Track
from vision_use.planning import straight_through
from vision_use.render import render
from vision_use.skills import SkillOutcome, SkillRequest, apply_patch, run_skill
from vision_use.skills.reframe import (
    ASPECT_TOLERANCE,
    ASPECTS,
    MIN_SUBJECT_CONFIDENCE,
    Reframe,
    crop_window,
)
from vision_use.video.probe import probe

from tests.conftest import FFMPEG, requires_ffmpeg

PAL = FrameRate(num=25, den=1)
DIGEST = "b" * 64
PRESENT = Path(__file__)
VERTICAL = ASPECTS["vertical"]


def _context(centre_x: float | None = None, confidence: float = 1.0) -> VideoContext:
    """A probed context, optionally carrying a located subject.

    The subject section is *provenanced* as well as populated, so `has()` is
    true and the runtime does not try to backfill it by decoding -- these
    fixtures point at this test file, which is not media.
    """
    subject = (
        (
            SubjectSpan(
                shot_id=shot_id(0),
                span=TimeSpan(start_s=0.0, end_s=10.0),
                centre_x=centre_x,
                confidence=confidence,
            ),
        )
        if centre_x is not None
        else ()
    )
    return VideoContext(
        probe=MediaProbe(
            source=SourceInfo(
                path=PRESENT,
                sha256=DIGEST,
                size_bytes=1,
                container="mp4",
                duration_s=10.0,
            ),
            video=VideoStreamInfo(index=0, codec="h264", width=640, height=360, frame_rate=PAL),
        ),
        subject=subject,
        analyzers={
            "subject.energy": AnalyzerRun(
                name="subject.energy",
                version="1",
                params_hash="a" * 64,
                produced=("subject",),
            )
        },
    )


def _plan(width: int = 640, height: int = 360) -> EditPlan:
    source = Source(id=source_id(0), path=PRESENT, sha256=DIGEST, duration_s=10.0)
    return EditPlan(
        sources={source.id: source},
        timeline=Timeline(
            frame_rate=PAL,
            width=width,
            height=height,
            tracks=(
                Track(
                    id=track_id(0),
                    kind="video",
                    clips=(
                        Clip(
                            id=clip_id(0),
                            source_id=source.id,
                            source_in_s=0.0,
                            source_out_s=10.0,
                            timeline_start_s=0.0,
                        ),
                    ),
                ),
            ),
        ),
    )


def _run(plan: EditPlan, context: VideoContext | None = None, **kwargs: object) -> SkillOutcome:
    return run_skill(
        Reframe(**kwargs),  # type: ignore[arg-type]
        PRESENT,
        plan,
        SkillRequest(prompt="make this a short"),
        context=context if context is not None else _context(),
    )


# --- geometry --------------------------------------------------------------


def test_a_wide_source_keeps_its_full_height() -> None:
    """Cutting the sides takes as much picture as the target shape allows."""
    window = crop_window(640, 360, VERTICAL)

    assert window.height == 360
    assert window.width < 640


def test_a_tall_source_keeps_its_full_width() -> None:
    window = crop_window(360, 640, ASPECTS["landscape"])

    assert window.width == 360
    assert window.height < 640


def test_the_window_has_the_target_aspect() -> None:
    window = crop_window(640, 360, VERTICAL)

    assert abs(window.aspect - VERTICAL) <= ASPECT_TOLERANCE


def test_dimensions_are_even() -> None:
    """yuv420p subsamples chroma by two; an odd dimension cannot be encoded."""
    window = crop_window(641, 361, VERTICAL)

    assert window.width % 2 == 0
    assert window.height % 2 == 0


def test_focus_moves_the_window() -> None:
    left = crop_window(640, 360, VERTICAL, focus=0.2)
    right = crop_window(640, 360, VERTICAL, focus=0.8)

    assert left.x < right.x


def test_the_window_never_leaves_the_frame() -> None:
    """A crop reaching past the edge renders as a black bar."""
    for focus in (-1.0, 0.0, 0.5, 1.0, 2.0):
        window = crop_window(640, 360, VERTICAL, focus=focus)
        assert window.x >= 0
        assert window.x + window.width <= 640
        assert window.y + window.height <= 360


def test_a_square_target_is_supported() -> None:
    window = crop_window(640, 360, ASPECTS["square"])

    assert window.width == window.height


def test_a_degenerate_source_is_rejected() -> None:
    with pytest.raises(VisionUseError, match="reframe"):
        crop_window(1, 1, VERTICAL)


def test_a_degenerate_aspect_is_rejected() -> None:
    with pytest.raises(VisionUseError, match="positive"):
        crop_window(640, 360, 0.0)


# --- lifecycle -------------------------------------------------------------


def test_a_wide_plan_is_reframed() -> None:
    outcome = _run(_plan())

    assert outcome.accepted is not None
    achieved = outcome.plan.timeline.width / outcome.plan.timeline.height
    assert abs(achieved - VERTICAL) <= ASPECT_TOLERANCE


def test_a_plan_already_the_right_shape_is_left_alone() -> None:
    """Cropping what is already vertical would only lose picture."""
    outcome = _run(_plan(width=202, height=360))

    assert outcome.accepted is None
    assert outcome.attempts == ()


def test_the_centre_crop_is_tried_first() -> None:
    """A wrong guess about which side someone stands on is worse than centred."""
    outcome = _run(_plan())

    assert outcome.accepted is not None
    assert outcome.accepted.strategy.name == "vertical-centre"


def test_every_clip_gets_the_crop() -> None:
    outcome = _run(_plan())

    assert outcome.accepted is not None
    for clip in outcome.plan.timeline.clips():
        assert any(effect.type == "crop" for effect in clip.effects)


def test_the_canvas_is_resized_too() -> None:
    """Cropping alone would leave the picture letterboxed inside the old canvas."""
    plan = _plan()

    outcome = _run(plan)

    assert outcome.accepted is not None
    assert outcome.plan.timeline.width != plan.timeline.width


def test_the_edit_is_reversible() -> None:
    plan = _plan()

    outcome = _run(plan)

    assert outcome.accepted is not None
    restored = apply_patch(outcome.plan, outcome.accepted.inverse).plan
    assert restored.model_dump_json() == plan.model_dump_json()


def test_the_patch_explains_itself() -> None:
    outcome = _run(_plan())

    assert outcome.accepted is not None
    assert "crop to vertical" in outcome.accepted.patch.rationale


def test_a_square_request_is_honoured() -> None:
    outcome = _run(_plan(), aspect="square")

    assert outcome.accepted is not None
    timeline = outcome.plan.timeline
    assert abs(timeline.width / timeline.height - 1.0) <= ASPECT_TOLERANCE


def test_an_unknown_aspect_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown aspect"):
        Reframe(aspect="hexagonal")


def test_empty_focus_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Reframe(focus=())


def test_the_subject_section_is_required() -> None:
    """The crop needs the subject section, and says so rather than decoding."""
    assert Reframe().meta.requires == {ContextSection.PROBE, ContextSection.SUBJECT}


# --- against real media ----------------------------------------------------


@pytest.fixture(scope="session")
def wide_clip(media_dir: Path) -> Path:
    """A 16:9 clip to reframe."""
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")

    path = media_dir / "wide-16x9.mp4"
    if path.exists():
        return path

    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=25:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.mark.integration
@requires_ffmpeg
def test_the_rendered_file_is_vertical(wide_clip: Path, tmp_path: Path) -> None:
    """The whole point: the output file itself has to be the new shape."""
    outcome = run_skill(Reframe(), wide_clip, straight_through(wide_clip))
    assert outcome.accepted is not None
    out = tmp_path / "out.mp4"

    render(outcome.plan, out)

    result = probe(out)
    assert result.video.height > result.video.width
    assert abs(result.video.width / result.video.height - VERTICAL) <= ASPECT_TOLERANCE


@pytest.mark.integration
@requires_ffmpeg
def test_reframing_crops_rather_than_squashing(wide_clip: Path, tmp_path: Path) -> None:
    """A squashed frame would keep the full width; a crop keeps the height."""
    outcome = run_skill(Reframe(), wide_clip, straight_through(wide_clip))
    assert outcome.accepted is not None
    out = tmp_path / "out.mp4"

    render(outcome.plan, out)

    before, after = probe(wide_clip), probe(out)
    assert after.video.height == before.video.height
    assert after.video.width < before.video.width


@pytest.mark.integration
@requires_ffmpeg
def test_the_frame_count_is_unchanged(wide_clip: Path, tmp_path: Path) -> None:
    """Reframing changes the shape, never the timing."""
    outcome = run_skill(Reframe(), wide_clip, straight_through(wide_clip))
    assert outcome.accepted is not None
    out = tmp_path / "out.mp4"

    render(outcome.plan, out)

    assert probe(out).video.frame_count == probe(wide_clip).video.frame_count


# --- following the subject -------------------------------------------------


def _crop_x(outcome: SkillOutcome, index: int = 0) -> int:
    """Return the horizontal offset of a clip's crop."""
    effects = outcome.plan.timeline.clips()[index].effects
    crop = next(effect for effect in effects if effect.type == "crop")
    return int(crop.params["x"])  # type: ignore[arg-type]


def test_the_crop_follows_a_subject_on_the_right() -> None:
    """The whole point: a vertical crop must not assume the subject was centred."""
    outcome = _run(_plan(), context=_context(centre_x=0.8))

    # A 202-wide window in a 640-wide frame, centred on 0.8 -> 512 - 101 = 411,
    # clamped to the frame's right edge at 640 - 202 = 438.
    assert _crop_x(outcome) > 300


def test_the_crop_follows_a_subject_on_the_left() -> None:
    outcome = _run(_plan(), context=_context(centre_x=0.15))

    assert _crop_x(outcome) < 100


def test_the_two_sides_actually_differ() -> None:
    """A crop that lands in the same place either way is not following anything."""
    left = _crop_x(_run(_plan(), context=_context(centre_x=0.15)))
    right = _crop_x(_run(_plan(), context=_context(centre_x=0.85)))

    assert right - left > 200


def test_a_low_confidence_subject_is_ignored() -> None:
    """Confidence is only worth reporting if something acts on it.

    `testsrc` -- uniform motion, no subject -- scores about 0.32, and cropping
    to the centroid of that is worse than not trying.
    """
    outcome = _run(_plan(), context=_context(centre_x=0.9, confidence=0.2))

    centred = _crop_x(_run(_plan(), context=_context()))
    assert _crop_x(outcome) == centred


def test_no_subject_section_reframes_as_before() -> None:
    """A source with no locatable subject must still reframe."""
    outcome = _run(_plan(), context=_context())

    assert outcome.accepted is not None
    assert outcome.plan.timeline.width < outcome.plan.timeline.height


def test_the_subject_strategy_is_tried_first() -> None:
    """It is the one that is right when it applies; the fixed ones are fallbacks."""
    outcome = _run(_plan(), context=_context(centre_x=0.8))

    assert outcome.attempts[0].strategy.name.endswith("-subject")
    assert outcome.accepted is not None


def test_the_fixed_strategies_remain_available() -> None:
    """A rejected subject crop must leave the retry loop somewhere to go."""
    skill = Reframe()
    analysis = skill.analyse(_context(centre_x=0.8), _plan(), SkillRequest())

    names = [strategy.name for strategy in skill.plan(analysis)]

    assert names[0].endswith("-subject")
    assert any(name.endswith("-centre") for name in names)
    assert any(name.endswith("-left") for name in names)


def test_the_rationale_says_how_many_clips_were_located() -> None:
    """An agent choosing what to do next needs to know how much was guesswork."""
    outcome = _run(_plan(), context=_context(centre_x=0.8))

    assert outcome.accepted is not None
    assert "following the subject" in outcome.accepted.patch.rationale


def test_the_canvas_is_one_shape_for_every_clip() -> None:
    """Windows differ only in offset, so the timeline stays a single shape."""
    outcome = _run(_plan(), context=_context(centre_x=0.8))

    width = outcome.plan.timeline.width
    for clip in outcome.plan.timeline.clips():
        crop = next(effect for effect in clip.effects if effect.type == "crop")
        assert int(crop.params["width"]) == width  # type: ignore[arg-type]


# --- following the subject within a shot (M9) ------------------------------


def _context_with_track(*points: tuple[float, float]) -> VideoContext:
    """A context whose subject moves across the frame during one shot."""
    base = _context(centre_x=0.5, confidence=0.1)
    span = base.subject[0].model_copy(
        update={
            "track": tuple(SubjectPoint(t_s=t, centre_x=x, centre_y=0.5) for t, x in points)
        }
    )
    return base.model_copy(update={"subject": (span,)})


def _crop(outcome: SkillOutcome) -> Effect:
    effects = outcome.plan.timeline.clips()[0].effects
    return next(effect for effect in effects if effect.type == "crop")


def test_a_moving_subject_produces_keyframes() -> None:
    """The exit criterion of M9: the window follows rather than compromising."""
    outcome = _run(_plan(), context=_context_with_track((0.0, 0.1), (5.0, 0.9)))

    crop = _crop(outcome)
    assert crop.animated
    assert len(crop.animate["x"]) == 2


def test_the_keyframes_move_in_the_subject_s_direction() -> None:
    outcome = _run(_plan(), context=_context_with_track((0.0, 0.1), (5.0, 0.9)))

    frames = _crop(outcome).animate["x"]
    assert frames[-1].value > frames[0].value


def test_keyframes_never_leave_the_frame() -> None:
    """A crop asking for pixels outside the frame is a black bar."""
    outcome = _run(_plan(), context=_context_with_track((0.0, 0.0), (5.0, 1.0)))

    limit = 640 - outcome.plan.timeline.width
    for frame in _crop(outcome).animate["x"]:
        assert 0 <= frame.value <= limit


def test_the_static_value_is_where_the_animation_starts() -> None:
    """A reader ignoring `animate` must still get a sensible crop (D6)."""
    outcome = _run(_plan(), context=_context_with_track((0.0, 0.1), (5.0, 0.9)))

    crop = _crop(outcome)
    assert crop.params["x"] == crop.animate["x"][0].value


def test_following_is_tried_before_the_static_crops() -> None:
    """A window that moves beats the best fixed position for a walking subject."""
    outcome = _run(_plan(), context=_context_with_track((0.0, 0.1), (5.0, 0.9)))

    assert outcome.attempts[0].strategy.name.endswith("-follow")


def test_low_summary_confidence_does_not_block_following() -> None:
    """A subject who crosses frame has no good static position -- that is the point.

    The span's confidence is low precisely because there is no single place to
    put a fixed crop, which is exactly when a moving one is right.
    """
    context = _context_with_track((0.0, 0.1), (5.0, 0.9))
    assert (context.subject[0].confidence or 0.0) < MIN_SUBJECT_CONFIDENCE

    assert _crop(_run(_plan(), context=context)).animated


def test_a_clip_keeps_only_the_keyframes_it_covers() -> None:
    """A clip trimmed out of a longer shot must not inherit footage it never shows."""
    plan = _plan()
    track = _context_with_track((0.0, 0.1), (2.0, 0.5), (8.0, 0.9))
    trimmed = (
        plan.timeline.tracks[0]
        .clips[0]
        .model_copy(update={"source_in_s": 0.0, "source_out_s": 3.0})
    )
    plan = plan.model_copy(
        update={
            "timeline": plan.timeline.model_copy(
                update={
                    "tracks": (
                        plan.timeline.tracks[0].model_copy(update={"clips": (trimmed,)}),
                    )
                }
            )
        }
    )

    outcome = _run(plan, context=track)

    assert len(_crop(outcome).animate["x"]) == 2


def test_no_track_still_reframes_statically() -> None:
    """Every source without a usable track behaves exactly as it did before."""
    outcome = _run(_plan(), context=_context(centre_x=0.8))

    assert not _crop(outcome).animated
