"""Tests for PlanPatch: typed operations, application, and inversion.

Inversion is the load-bearing property. The M3 retry loop tries a strategy and
abandons it on failure, so "undo" has to return the *exact* prior plan, not an
approximation of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from vision_use.errors import VisionUseError
from vision_use.models.ids import clip_id, source_id, track_id
from vision_use.models.media import FrameRate
from vision_use.models.plan import (
    Clip,
    EditPlan,
    Effect,
    Source,
    Timeline,
    Track,
    Transition,
)
from vision_use.skills import (
    AddClip,
    AddTrack,
    PlanPatch,
    PushEffect,
    RemoveClip,
    RemoveTrack,
    SetClipEnabled,
    SetClipRange,
    SetEffects,
    SetTransition,
    SkillRef,
    apply_patch,
)
from vision_use.skills.patch import Op

PAL = FrameRate(num=25, den=1)
DIGEST = "d" * 64
PRESENT = Path(__file__)


def _clip(
    index: int, start: float, in_s: float = 0.0, out_s: float = 1.0, **kw: object
) -> Clip:
    return Clip(
        id=clip_id(index),
        source_id=source_id(0),
        source_in_s=in_s,
        source_out_s=out_s,
        timeline_start_s=start,
        **kw,
    )


def _plan(*clips: Clip) -> EditPlan:
    source = Source(id=source_id(0), path=PRESENT, sha256=DIGEST, duration_s=60.0)
    return EditPlan(
        sources={source.id: source},
        timeline=Timeline(
            frame_rate=PAL,
            width=640,
            height=480,
            tracks=(Track(id=track_id(0), kind="video", clips=clips),),
        ),
    )


def _patch(*ops: Op) -> PlanPatch:
    return PlanPatch(ops=ops, rationale="test")


def _round_trips(plan: EditPlan, patch: PlanPatch) -> bool:
    """Whether applying then undoing ``patch`` restores ``plan`` exactly."""
    result = apply_patch(plan, patch)
    return result.undo().model_dump_json() == plan.model_dump_json()


# --- patch basics ----------------------------------------------------------


def test_empty_patch_changes_nothing() -> None:
    """A skill finding nothing to do is a legitimate outcome, not a failure."""
    plan = _plan(_clip(0, 0.0))
    patch = PlanPatch()

    assert patch.is_empty
    assert len(patch) == 0
    assert apply_patch(plan, patch).plan == plan


def test_patch_carries_its_rationale_and_author() -> None:
    """An edit nobody can explain is an edit nobody can trust."""
    patch = PlanPatch(
        ops=(SetClipEnabled(clip_id=clip_id(0), enabled=False),),
        rationale="the speaker is off-camera here",
        produced_by=SkillRef(name="edit.trim", version="1"),
    )

    assert patch.rationale
    assert patch.produced_by is not None
    assert patch.produced_by.name == "edit.trim"


def test_patch_round_trips_through_json() -> None:
    """Ops are discriminated by tag so a patch survives being stored."""
    patch = _patch(
        SetClipRange(clip_id=clip_id(0), source_in_s=1.0),
        PushEffect(
            clip_id=clip_id(0), effect=Effect(type="crop", params={"width": 8, "height": 8})
        ),
    )

    restored = PlanPatch.model_validate_json(patch.model_dump_json())

    assert restored == patch
    assert [op.op for op in restored.ops] == ["set_clip_range", "push_effect"]


# --- individual operations -------------------------------------------------


def test_set_clip_range_retrims() -> None:
    plan = _plan(_clip(0, 0.0, 0.0, 5.0))

    result = apply_patch(plan, _patch(SetClipRange(clip_id=clip_id(0), source_out_s=2.0)))

    assert result.plan.timeline.clips()[0].source_out_s == 2.0


def test_set_clip_range_leaves_omitted_fields_alone() -> None:
    """Trimming must not force a caller to restate where the clip sits."""
    plan = _plan(_clip(0, 3.0, 0.0, 5.0))

    result = apply_patch(plan, _patch(SetClipRange(clip_id=clip_id(0), source_in_s=1.0)))

    clip = result.plan.timeline.clips()[0]
    assert clip.source_in_s == 1.0
    assert clip.timeline_start_s == 3.0
    assert clip.source_out_s == 5.0


def test_set_clip_enabled_toggles() -> None:
    plan = _plan(_clip(0, 0.0))

    result = apply_patch(plan, _patch(SetClipEnabled(clip_id=clip_id(0), enabled=False)))

    assert not result.plan.timeline.clips()[0].enabled


def test_remove_clip() -> None:
    plan = _plan(_clip(0, 0.0), _clip(1, 2.0))

    result = apply_patch(plan, _patch(RemoveClip(clip_id=clip_id(0))))

    assert [clip.id for clip in result.plan.timeline.clips()] == [clip_id(1)]


def test_add_clip_keeps_the_track_in_timeline_order() -> None:
    """The model rejects unordered clips, so insertion must sort."""
    plan = _plan(_clip(0, 0.0), _clip(2, 4.0))

    result = apply_patch(plan, _patch(AddClip(track_id=track_id(0), clip=_clip(1, 2.0))))

    starts = [clip.timeline_start_s for clip in result.plan.timeline.clips()]
    assert starts == sorted(starts)


def test_push_effect_appends_to_the_stack() -> None:
    plan = _plan(_clip(0, 0.0, effects=(Effect(type="fade"),)))

    result = apply_patch(
        plan, _patch(PushEffect(clip_id=clip_id(0), effect=Effect(type="invert")))
    )

    assert [e.type for e in result.plan.timeline.clips()[0].effects] == ["fade", "invert"]


def test_set_effects_replaces_the_stack() -> None:
    """Reordering or clearing needs replacement; push alone cannot express it."""
    plan = _plan(_clip(0, 0.0, effects=(Effect(type="fade"), Effect(type="invert"))))

    result = apply_patch(plan, _patch(SetEffects(clip_id=clip_id(0), effects=())))

    assert result.plan.timeline.clips()[0].effects == ()


def test_set_transition() -> None:
    plan = _plan(_clip(0, 0.0), _clip(1, 1.0))
    fade = Transition(type="fade", duration_s=0.5)

    result = apply_patch(
        plan, _patch(SetTransition(clip_id=clip_id(1), side="in", transition=fade))
    )

    updated = result.plan.clip_by_id(clip_id(1))
    assert updated is not None
    assert updated.transition_in == fade


def test_add_and_remove_track() -> None:
    plan = _plan(_clip(0, 0.0))
    overlay = Track(id=track_id(1), kind="overlay")

    grown = apply_patch(plan, _patch(AddTrack(track=overlay))).plan
    assert len(grown.timeline.tracks) == 2

    shrunk = apply_patch(grown, _patch(RemoveTrack(track_id=track_id(1)))).plan
    assert len(shrunk.timeline.tracks) == 1


# --- inversion -------------------------------------------------------------


@pytest.mark.parametrize(
    "operation",
    [
        SetClipRange(clip_id=clip_id(0), source_in_s=0.5, source_out_s=0.9),
        SetClipEnabled(clip_id=clip_id(0), enabled=False),
        RemoveClip(clip_id=clip_id(0)),
        PushEffect(clip_id=clip_id(0), effect=Effect(type="invert")),
        SetEffects(clip_id=clip_id(0), effects=(Effect(type="invert"),)),
        SetTransition(
            clip_id=clip_id(0), side="out", transition=Transition(type="fade", duration_s=0.2)
        ),
    ],
)
def test_every_operation_round_trips(operation: Op) -> None:
    """Undo must restore the exact prior plan, or the retry loop is unsound."""
    plan = _plan(_clip(0, 0.0, 0.0, 2.0, effects=(Effect(type="fade"),)))

    assert _round_trips(plan, _patch(operation))


def test_add_clip_round_trips() -> None:
    plan = _plan(_clip(0, 0.0))

    assert _round_trips(plan, _patch(AddClip(track_id=track_id(0), clip=_clip(1, 3.0))))


def test_add_track_round_trips() -> None:
    plan = _plan(_clip(0, 0.0))

    assert _round_trips(plan, _patch(AddTrack(track=Track(id=track_id(1), kind="overlay"))))


def test_remove_track_round_trips_with_its_clips() -> None:
    """A removed track's clips must come back too, not just the empty lane."""
    plan = _plan(_clip(0, 0.0))

    assert _round_trips(plan, _patch(RemoveTrack(track_id=track_id(0))))


def test_multi_op_patch_round_trips() -> None:
    plan = _plan(_clip(0, 0.0, 0.0, 2.0), _clip(1, 3.0))

    patch = _patch(
        SetClipRange(clip_id=clip_id(0), source_out_s=1.0),
        SetClipEnabled(clip_id=clip_id(1), enabled=False),
        PushEffect(clip_id=clip_id(0), effect=Effect(type="invert")),
    )

    assert _round_trips(plan, patch)


def test_inverse_unwinds_in_reverse_order() -> None:
    """Undoing forwards would replay dependencies in the wrong order."""
    plan = _plan(_clip(0, 0.0))
    patch = _patch(
        AddClip(track_id=track_id(0), clip=_clip(1, 2.0)),
        RemoveClip(clip_id=clip_id(0)),
    )

    result = apply_patch(plan, patch)

    assert [op.op for op in result.inverse.ops] == ["add_clip", "remove_clip"]
    assert result.undo().model_dump_json() == plan.model_dump_json()


def test_inverse_is_itself_a_patch() -> None:
    """Undo is an ordinary patch, not a special path through the code."""
    plan = _plan(_clip(0, 0.0))

    inverse = apply_patch(
        plan, _patch(SetClipEnabled(clip_id=clip_id(0), enabled=False))
    ).inverse

    assert isinstance(inverse, PlanPatch)
    assert inverse.rationale.startswith("undo")


# --- failure handling ------------------------------------------------------


def test_unknown_clip_reports_a_remedy() -> None:
    plan = _plan(_clip(0, 0.0))

    with pytest.raises(VisionUseError, match="No clip"):
        apply_patch(plan, _patch(SetClipEnabled(clip_id="clp_9999", enabled=False)))


def test_unknown_track_reports_a_remedy() -> None:
    plan = _plan(_clip(0, 0.0))

    with pytest.raises(VisionUseError, match="No track"):
        apply_patch(plan, _patch(AddClip(track_id="trk_9999", clip=_clip(1, 2.0))))


def test_removing_a_missing_track_reports_a_remedy() -> None:
    plan = _plan(_clip(0, 0.0))

    with pytest.raises(VisionUseError, match="No track"):
        apply_patch(plan, _patch(RemoveTrack(track_id="trk_9999")))


def test_an_operation_producing_an_invalid_plan_is_rejected() -> None:
    """The model's invariants still hold; a patch cannot smuggle past them."""
    plan = _plan(_clip(0, 0.0, 0.0, 2.0), _clip(1, 2.0, 0.0, 2.0))

    with pytest.raises(VisionUseError, match="invalid"):
        # Moving the second clip on top of the first would overlap them.
        apply_patch(plan, _patch(SetClipRange(clip_id=clip_id(1), timeline_start_s=0.5)))


def test_a_failed_patch_leaves_the_original_untouched() -> None:
    """Half-applied edits are what make a retry loop unsound."""
    plan = _plan(_clip(0, 0.0, 0.0, 2.0))
    before = plan.model_dump_json()

    with pytest.raises(VisionUseError):
        apply_patch(
            plan,
            _patch(
                SetClipEnabled(clip_id=clip_id(0), enabled=False),
                SetClipEnabled(clip_id="clp_9999", enabled=False),
            ),
        )

    assert plan.model_dump_json() == before


def test_source_range_past_the_end_is_rejected() -> None:
    plan = _plan(_clip(0, 0.0))

    with pytest.raises(VisionUseError, match="invalid"):
        apply_patch(plan, _patch(SetClipRange(clip_id=clip_id(0), source_out_s=999.0)))
