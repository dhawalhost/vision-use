"""Tests for the EditPlan model and its structural invariants.

These invariants are load-bearing: a plan that violates one either renders
wrongly or fails halfway through an export, and both are far more expensive to
diagnose than a rejected construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from vision_use.models.ids import clip_id, source_id, track_id
from vision_use.models.media import FrameRate
from vision_use.models.plan import (
    Clip,
    EditPlan,
    Effect,
    Keyframe,
    RenderProfile,
    RenderSettings,
    Source,
    Timeline,
    Track,
    Transition,
)

DIGEST = "a" * 64
PAL = FrameRate(num=25, den=1)


def _source(index: int = 0, duration_s: float = 60.0) -> Source:
    return Source(
        id=source_id(index),
        path=Path(f"clip{index}.mp4"),
        sha256=DIGEST,
        duration_s=duration_s,
    )


def _clip(
    index: int,
    start: float,
    in_s: float = 0.0,
    out_s: float = 1.0,
    source: int = 0,
    **kwargs: object,
) -> Clip:
    return Clip(
        id=clip_id(index),
        source_id=source_id(source),
        source_in_s=in_s,
        source_out_s=out_s,
        timeline_start_s=start,
        **kwargs,
    )


def _plan(*clips: Clip, sources: tuple[Source, ...] = (), kind: str = "video") -> EditPlan:
    declared = sources or (_source(),)
    return EditPlan(
        sources={item.id: item for item in declared},
        timeline=Timeline(
            frame_rate=PAL,
            width=1920,
            height=1080,
            tracks=(Track(id=track_id(0), kind=kind, clips=clips),),
        ),
    )


# --- clip ------------------------------------------------------------------


def test_clip_duration_comes_from_the_source_range() -> None:
    assert _clip(0, start=5.0, in_s=2.0, out_s=6.0).duration_s == 4.0


def test_timeline_end_is_start_plus_duration() -> None:
    """Moving a clip must not change what it shows; these are separate axes."""
    clip = _clip(0, start=10.0, in_s=2.0, out_s=6.0)

    assert clip.timeline_end_s == 14.0


def test_reversed_source_range_rejected() -> None:
    with pytest.raises(ValidationError, match="reversed"):
        _clip(0, start=0.0, in_s=5.0, out_s=2.0)


def test_zero_length_clip_rejected() -> None:
    """A clip showing nothing is always a bug, never an intent."""
    with pytest.raises(ValidationError, match="empty or reversed"):
        _clip(0, start=0.0, in_s=2.0, out_s=2.0)


def test_negative_time_rejected() -> None:
    with pytest.raises(ValidationError):
        _clip(0, start=-1.0)


def test_clips_are_enabled_by_default() -> None:
    assert _clip(0, start=0.0).enabled


def test_disabled_clip_is_still_a_clip() -> None:
    """Auditioning an edit must not require discarding it."""
    assert not _clip(0, start=0.0, enabled=False).enabled


def test_overlap_detection() -> None:
    left = _clip(0, start=0.0, out_s=2.0)
    right = _clip(1, start=1.0, out_s=2.0)

    assert left.overlaps(right)


def test_touching_clips_do_not_overlap() -> None:
    """Back-to-back clips are the common case and must not be rejected."""
    left = _clip(0, start=0.0, out_s=2.0)
    right = _clip(1, start=2.0, out_s=2.0)

    assert not left.overlaps(right)


def test_touching_clips_survive_float_subtraction_noise() -> None:
    """Reproduces the bug found on real 40-cut media, with the exact clips it
    produced.

    `duration_s` is `source_out_s - source_in_s`: a plain subtraction of two
    independently quantised floats. When a skill lays many clips end to end --
    silence removal does this for every kept span -- that subtraction can land
    a few femtoseconds past where two clips were meant to touch exactly, and a
    strict `<` overlap check rejected the plan outright. These values are taken
    verbatim from the failing run, not constructed to make a point: a smaller,
    invented example did not reproduce the noise, which is itself the lesson --
    guessing at a float-precision bug is not the same as reproducing it.
    """
    left = Clip(
        id=clip_id(0),
        source_id=source_id(0),
        source_in_s=30.32,
        source_out_s=36.6,
        timeline_start_s=18.96,
    )
    right = Clip(
        id=clip_id(1),
        source_id=source_id(0),
        source_in_s=39.44,
        source_out_s=45.72,
        # `left.timeline_end_s` computes to 25.240000000000002, a hair above
        # this -- the exact noise that tripped the strict check.
        timeline_start_s=25.24,
    )

    assert not left.overlaps(right)


def test_a_real_overlap_below_one_frame_is_still_caught() -> None:
    """The tolerance is noise-sized, not frame-sized -- it must not hide a bug."""
    left = _clip(0, start=0.0, out_s=2.0)
    # A millisecond of real overlap: many orders of magnitude above the
    # tolerance, and still smaller than a frame at any ordinary rate.
    right = _clip(1, start=1.999, out_s=2.0)

    assert left.overlaps(right)


# --- track -----------------------------------------------------------------


def test_track_rejects_overlapping_clips() -> None:
    """Overlap on one track has no defined rendering; layering needs two tracks."""
    with pytest.raises(ValidationError, match="overlap"):
        Track(
            id=track_id(0),
            kind="video",
            clips=(_clip(0, start=0.0, out_s=3.0), _clip(1, start=1.0, out_s=3.0)),
        )


def test_track_rejects_unordered_clips() -> None:
    with pytest.raises(ValidationError, match="timeline order"):
        Track(
            id=track_id(0),
            kind="video",
            clips=(_clip(0, start=5.0), _clip(1, start=1.0)),
        )


def test_track_rejects_duplicate_clip_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate clip id"):
        Track(
            id=track_id(0),
            kind="video",
            clips=(_clip(0, start=0.0), _clip(0, start=5.0)),
        )


def test_track_duration_is_its_last_clip_end() -> None:
    track = Track(
        id=track_id(0),
        kind="video",
        clips=(_clip(0, start=0.0, out_s=2.0), _clip(1, start=5.0, out_s=3.0)),
    )

    assert track.duration_s == 8.0


def test_empty_track_has_zero_duration() -> None:
    assert Track(id=track_id(0), kind="video").duration_s == 0.0


def test_gaps_are_reported() -> None:
    """Gaps are legal but usually accidental, so they must be visible."""
    track = Track(
        id=track_id(0),
        kind="video",
        clips=(_clip(0, start=0.0, out_s=2.0), _clip(1, start=5.0, out_s=1.0)),
    )

    assert track.gaps() == [(2.0, 5.0)]


def test_leading_gap_is_reported() -> None:
    track = Track(id=track_id(0), kind="video", clips=(_clip(0, start=3.0, out_s=1.0),))

    assert track.gaps() == [(0.0, 3.0)]


def test_contiguous_track_has_no_gaps() -> None:
    track = Track(
        id=track_id(0),
        kind="video",
        clips=(_clip(0, start=0.0, out_s=2.0), _clip(1, start=2.0, out_s=2.0)),
    )

    assert track.gaps() == []


def test_touching_clips_do_not_produce_a_phantom_gap() -> None:
    """Regression: found on real 40-cut media.

    Clips meant to touch exactly can leave `cursor` a few femtoseconds short of
    the next clip's `timeline_start_s`. Reporting that as a gap is not just
    cosmetic: the render compiler used to fill every reported gap with a
    `color=...:d=0.0` lavfi source, and a zero-duration lavfi source never
    reaches EOF, so the render hung forever. `cursor` here is the *left* clip's
    `timeline_end_s` (56.71999999999999); the *right* clip's `timeline_start_s`
    (56.72) sits a few femtoseconds above it, which is the noise direction that
    trips a gap rather than an overlap -- the opposite pair from
    `test_touching_clips_survive_float_subtraction_noise`, and it needed its own
    numbers pulled from the same failing run.
    """
    track = Track(
        id=track_id(0),
        kind="video",
        clips=(
            Clip(
                id=clip_id(0),
                source_id=source_id(0),
                source_in_s=75.92,
                source_out_s=82.24,
                timeline_start_s=50.4,
            ),
            Clip(
                id=clip_id(1),
                source_id=source_id(0),
                source_in_s=85.04,
                source_out_s=91.36,
                timeline_start_s=56.72,
            ),
        ),
    )

    # The leading gap (0 to 50.4) is real; what must not appear is a *second*,
    # phantom one between the two clips.
    assert track.gaps() == [(0.0, 50.4)]


def test_invalid_track_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        Track(id=track_id(0), kind="subtitle")


# --- timeline --------------------------------------------------------------


def test_timeline_duration_spans_every_track() -> None:
    timeline = Timeline(
        frame_rate=PAL,
        width=1920,
        height=1080,
        tracks=(
            Track(id=track_id(0), kind="video", clips=(_clip(0, start=0.0, out_s=2.0),)),
            Track(id=track_id(1), kind="audio", clips=(_clip(1, start=0.0, out_s=9.0),)),
        ),
    )

    assert timeline.duration_s == 9.0


def test_timeline_rejects_duplicate_track_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate track id"):
        Timeline(
            frame_rate=PAL,
            width=1920,
            height=1080,
            tracks=(Track(id=track_id(0), kind="video"), Track(id=track_id(0), kind="audio")),
        )


def test_timeline_rejects_zero_dimensions() -> None:
    with pytest.raises(ValidationError):
        Timeline(frame_rate=PAL, width=0, height=1080)


# --- plan ------------------------------------------------------------------


def test_minimal_plan_is_valid() -> None:
    plan = _plan(_clip(0, start=0.0, out_s=2.0))

    assert plan.duration_s == 2.0
    assert plan.schema_version == "1"


def test_dangling_source_reference_rejected() -> None:
    """A reference discovered mid-export has already cost the user the render."""
    with pytest.raises(ValidationError, match="unknown source"):
        _plan(_clip(0, start=0.0, source=9))


def test_source_range_past_the_end_rejected() -> None:
    with pytest.raises(ValidationError, match="which is"):
        _plan(_clip(0, start=0.0, in_s=0.0, out_s=99.0), sources=(_source(duration_s=10.0),))


def test_source_key_must_match_its_id() -> None:
    """A mismatched key makes lookups silently miss."""
    with pytest.raises(ValidationError, match="must match"):
        EditPlan(
            sources={"src_0009": _source(0)},
            timeline=Timeline(frame_rate=PAL, width=640, height=480),
        )


def test_unused_sources_are_reported_not_rejected() -> None:
    """A plan under construction may declare a source before placing it."""
    plan = _plan(
        _clip(0, start=0.0, out_s=2.0, source=0),
        sources=(_source(0), _source(1)),
    )

    assert plan.unused_sources() == [source_id(1)]


def test_lookups_resolve() -> None:
    plan = _plan(_clip(0, start=0.0, out_s=2.0))

    assert plan.clip_by_id(clip_id(0)) is not None
    assert plan.track_by_id(track_id(0)) is not None
    assert plan.clip_by_id("clp_9999") is None


def test_empty_plan_has_zero_duration() -> None:
    plan = EditPlan(timeline=Timeline(frame_rate=PAL, width=640, height=480))

    assert plan.duration_s == 0.0


# --- effects and transitions -----------------------------------------------


def test_effects_hold_unknown_types() -> None:
    """A core lacking a plugin's effect must still round-trip its plan."""
    clip = _clip(0, start=0.0, effects=(Effect(type="some.plugin.blur", params={"px": 4}),))

    assert clip.effects[0].type == "some.plugin.blur"


def test_effect_stack_order_is_preserved() -> None:
    clip = _clip(0, start=0.0, effects=(Effect(type="a"), Effect(type="b")))

    assert [effect.type for effect in clip.effects] == ["a", "b"]


def test_empty_effect_type_rejected() -> None:
    with pytest.raises(ValidationError):
        Effect(type="")


def test_schema_hash_is_optional_but_validated() -> None:
    assert Effect(type="crop").schema_hash is None
    with pytest.raises(ValidationError):
        Effect(type="crop", schema_hash="short")


def test_transitions_attach_to_clips() -> None:
    clip = _clip(0, start=0.0, transition_in=Transition(type="crossfade", duration_s=0.5))

    assert clip.transition_in is not None
    assert clip.transition_in.duration_s == 0.5


# --- render settings -------------------------------------------------------


def test_proxy_is_smaller_and_faster_than_master() -> None:
    """The proxy/master split is what keeps the agent retry loop affordable."""
    settings = RenderSettings()

    assert settings.proxy.scale < settings.master.scale
    assert settings.proxy.crf > settings.master.crf


def test_scale_above_one_rejected() -> None:
    """A profile upscales nothing; the canvas is the maximum."""
    with pytest.raises(ValidationError):
        RenderProfile(scale=2.0)


def test_crf_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        RenderProfile(crf=99)


def test_loudnorm_defaults_to_untouched() -> None:
    assert RenderSettings().loudnorm_lufs is None


# --- serialization ---------------------------------------------------------


def test_plan_round_trips_through_json() -> None:
    original = _plan(
        _clip(
            0,
            start=0.0,
            out_s=2.0,
            effects=(Effect(type="crop", params={"x": 10}),),
            transition_out=Transition(type="fade", duration_s=0.25),
        )
    )

    assert EditPlan.model_validate_json(original.model_dump_json()) == original


def test_serialization_is_stable() -> None:
    """A plan is diffed and content-addressed; its encoding must not wobble."""
    plan = _plan(_clip(0, start=0.0, out_s=2.0))

    assert plan.model_dump_json() == plan.model_dump_json()


def test_frame_rate_survives_as_a_rational() -> None:
    """The plan carries the timebase the renderer must cut on."""
    plan = EditPlan(
        timeline=Timeline(frame_rate=FrameRate(num=30000, den=1001), width=640, height=480)
    )

    restored = EditPlan.model_validate_json(plan.model_dump_json())

    assert restored.timeline.frame_rate.num == 30000
    assert restored.timeline.frame_rate.den == 1001


# --- animated parameters (PLAN.md D6) --------------------------------------


def _animated(*frames: tuple[float, float]) -> Effect:
    """A crop whose `x` follows the given (time, value) pairs."""
    return Effect(
        type="crop",
        params={"x": 0, "y": 0, "width": 100, "height": 100},
        animate={"x": tuple(Keyframe(t_s=t, value=v) for t, v in frames)},
    )


def test_an_effect_is_static_by_default() -> None:
    """Every plan written before D6 must stay valid and stay still."""
    effect = Effect(type="crop", params={"x": 10})

    assert not effect.animated
    assert effect.value_at("x", 5.0) is None


def test_a_value_between_keyframes_is_interpolated() -> None:
    effect = _animated((0.0, 0.0), (2.0, 100.0))

    assert effect.value_at("x", 1.0) == 50.0


def test_a_value_before_the_first_keyframe_is_held() -> None:
    """ "What happens outside the animation" needs an obvious answer."""
    effect = _animated((1.0, 40.0), (2.0, 100.0))

    assert effect.value_at("x", 0.0) == 40.0


def test_a_value_after_the_last_keyframe_is_held() -> None:
    effect = _animated((0.0, 40.0), (1.0, 100.0))

    assert effect.value_at("x", 9.0) == 100.0


def test_a_value_exactly_on_a_keyframe_is_that_keyframe() -> None:
    effect = _animated((0.0, 0.0), (2.0, 100.0), (4.0, 0.0))

    assert effect.value_at("x", 2.0) == 100.0


def test_interpolation_spans_more_than_two_keyframes() -> None:
    effect = _animated((0.0, 0.0), (2.0, 100.0), (4.0, 0.0))

    assert effect.value_at("x", 3.0) == 50.0


def test_an_unanimated_parameter_returns_nothing() -> None:
    """`params` already holds its value; returning a number would hide that."""
    effect = _animated((0.0, 0.0), (1.0, 10.0))

    assert effect.value_at("y", 0.5) is None


# --- what a plan may not say -----------------------------------------------


def test_keyframes_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="out of order"):
        Effect(
            type="crop",
            params={"x": 0},
            animate={"x": (Keyframe(t_s=2.0, value=0), Keyframe(t_s=1.0, value=5))},
        )


def test_a_repeated_time_is_ambiguous() -> None:
    with pytest.raises(ValidationError, match="repeat a time"):
        Effect(
            type="crop",
            params={"x": 0},
            animate={"x": (Keyframe(t_s=1.0, value=0), Keyframe(t_s=1.0, value=5))},
        )


def test_an_empty_track_is_refused() -> None:
    with pytest.raises(ValidationError, match="no keyframes"):
        Effect(type="crop", params={"x": 0}, animate={"x": ()})


def test_animating_a_parameter_that_has_no_static_value_is_refused() -> None:
    """A reader ignoring `animate` must still produce the same shape of edit."""
    with pytest.raises(ValidationError, match="not in params"):
        Effect(type="crop", params={"y": 0}, animate={"x": (Keyframe(t_s=0.0, value=1),)})


def test_animating_a_non_numeric_parameter_is_refused() -> None:
    """Interpolating a string would be silently meaningless."""
    with pytest.raises(ValidationError, match="not numeric"):
        Effect(
            type="text",
            params={"body": "hello"},
            animate={"body": (Keyframe(t_s=0.0, value=1),)},
        )


def test_an_animated_effect_round_trips_through_json() -> None:
    effect = _animated((0.0, 0.0), (2.0, 100.0))

    assert Effect.model_validate_json(effect.model_dump_json()) == effect
