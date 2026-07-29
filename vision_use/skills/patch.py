"""``PlanPatch`` -- the typed edit a skill returns.

A skill does not return a new plan. It returns a *patch*: an ordered list of
named operations against the plan it was given.

That distinction is the whole reason this module exists. A skill returning a
whole replacement plan cannot be reviewed (the diff is the entire document),
cannot be reverted (nothing records what changed), and cannot be validated
before application (there is no operation to check). Typed operations give all
three for free, and they let the runtime tell a caller *what a skill did* rather
than only *what came out*.

**Every operation is invertible.** Applying a patch returns its inverse, so undo
is a patch like any other rather than a special path through the code. That is
what makes the M3 retry loop sound: a rejected strategy is rolled back exactly,
not approximately.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from vision_use.errors import VisionUseError
from vision_use.models.plan import Clip, EditPlan, Effect, Source, Track, Transition


class SkillRef(BaseModel):
    """Which skill produced a patch, and at what version."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class _Op(BaseModel):
    """Common configuration for every operation."""

    model_config = ConfigDict(frozen=True)


class AddClip(_Op):
    """Insert a clip into a track, keeping the track in timeline order."""

    op: Literal["add_clip"] = "add_clip"
    track_id: str
    clip: Clip


class RemoveClip(_Op):
    """Remove a clip from whichever track holds it."""

    op: Literal["remove_clip"] = "remove_clip"
    clip_id: str


class SetClipRange(_Op):
    """Retrim a clip, move it, or both.

    Omitted fields are left alone, so trimming does not force a caller to
    restate where the clip sits.
    """

    op: Literal["set_clip_range"] = "set_clip_range"
    clip_id: str
    source_in_s: float | None = Field(default=None, ge=0)
    source_out_s: float | None = Field(default=None, ge=0)
    timeline_start_s: float | None = Field(default=None, ge=0)


class SetClipEnabled(_Op):
    """Enable or disable a clip without removing it."""

    op: Literal["set_clip_enabled"] = "set_clip_enabled"
    clip_id: str
    enabled: bool


class PushEffect(_Op):
    """Append an effect to a clip's stack."""

    op: Literal["push_effect"] = "push_effect"
    clip_id: str
    effect: Effect


class SetEffects(_Op):
    """Replace a clip's whole effect stack.

    The inverse of :class:`PushEffect` and the way a skill reorders or clears
    effects, which a push-only vocabulary could not express.
    """

    op: Literal["set_effects"] = "set_effects"
    clip_id: str
    effects: tuple[Effect, ...] = ()


class SetTransition(_Op):
    """Set or clear the transition on one side of a clip."""

    op: Literal["set_transition"] = "set_transition"
    clip_id: str
    side: Literal["in", "out"] = "in"
    transition: Transition | None = None


class AddSource(_Op):
    """Declare a new source the plan can draw from.

    Needed by any skill that introduces media the plan did not start with --
    a rendered caption, a generated title card, a fetched b-roll clip. Without
    it a skill could only rearrange what it was given.
    """

    op: Literal["add_source"] = "add_source"
    source: Source


class RemoveSource(_Op):
    """Remove a declared source.

    Fails if a clip still draws from it, because the plan model rejects
    dangling references -- which is the behaviour wanted: losing a source out
    from under a clip is never intended.
    """

    op: Literal["remove_source"] = "remove_source"
    source_id: str


class SetCanvas(_Op):
    """Change the timeline's output dimensions.

    Needed by any skill that changes the shape of the result rather than its
    content -- reframing 16:9 to 9:16, or rendering a square cut for a feed.
    Without it a skill could crop the picture but not the frame it lands in.
    """

    op: Literal["set_canvas"] = "set_canvas"
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class AddTrack(_Op):
    """Add an empty track, e.g. an overlay lane for captions."""

    op: Literal["add_track"] = "add_track"
    track: Track


class RemoveTrack(_Op):
    """Remove a track and everything on it."""

    op: Literal["remove_track"] = "remove_track"
    track_id: str


#: Every operation, discriminated by its ``op`` tag so a patch round-trips
#: through JSON without the caller having to know which class to construct.
Op = Annotated[
    AddClip
    | AddSource
    | SetCanvas
    | RemoveSource
    | RemoveClip
    | SetClipRange
    | SetClipEnabled
    | PushEffect
    | SetEffects
    | SetTransition
    | AddTrack
    | RemoveTrack,
    Field(discriminator="op"),
]


class PlanPatch(BaseModel):
    """An ordered, reviewable, invertible edit to a plan."""

    model_config = ConfigDict(frozen=True)

    ops: tuple[Op, ...] = ()
    rationale: str = Field(
        default="",
        description=(
            "Why the skill did this, in the skill's own words. Surfaced to the "
            "user and to logs -- an edit nobody can explain is an edit nobody "
            "can trust."
        ),
    )
    produced_by: SkillRef | None = None

    def __len__(self) -> int:
        """Return how many operations the patch carries."""
        return len(self.ops)

    @property
    def is_empty(self) -> bool:
        """Whether this patch would change nothing.

        A skill deciding there is nothing to do is a legitimate outcome, not a
        failure -- ``silence-removal`` on a clip with no silence should return
        an empty patch rather than inventing work.
        """
        return not self.ops


def _find_clip(plan: EditPlan, clip_id: str) -> tuple[Track, Clip]:
    """Return the track and clip for ``clip_id``.

    Raises:
        VisionUseError: If no track holds it.

    """
    for track in plan.timeline.tracks:
        for clip in track.clips:
            if clip.id == clip_id:
                return track, clip
    raise VisionUseError(
        f"No clip {clip_id!r} in this plan",
        remedy="check the clip id against `vision plan show`",
    )


def _replace_clip(plan: EditPlan, track_id: str, updated: Clip) -> EditPlan:
    """Return ``plan`` with one clip replaced, re-sorted into timeline order."""
    tracks = []
    for track in plan.timeline.tracks:
        if track.id != track_id:
            tracks.append(track)
            continue
        clips = [clip for clip in track.clips if clip.id != updated.id]
        clips.append(updated)
        clips.sort(key=lambda clip: clip.timeline_start_s)
        tracks.append(track.model_copy(update={"clips": tuple(clips)}))
    return plan.model_copy(
        update={"timeline": plan.timeline.model_copy(update={"tracks": tuple(tracks)})}
    )


def _apply_one(plan: EditPlan, operation: Op) -> tuple[EditPlan, Op]:
    """Apply one operation, returning the new plan and the inverse operation.

    One return per operation kind. A dispatch table would trade the exhaustive
    ``match`` -- which a type checker verifies covers every case -- for a dict
    lookup that it cannot, so the branch count is the better bargain.

    The inverse is built from the state *before* the change, which is why
    inversion happens here rather than in a separate pass -- afterwards, the
    information needed to undo is gone.
    """
    match operation:
        case AddClip():
            track = plan.track_by_id(operation.track_id)
            if track is None:
                raise VisionUseError(
                    f"No track {operation.track_id!r} to add a clip to",
                    remedy="add the track first, or target an existing one",
                )
            clips = sorted(
                [*track.clips, operation.clip], key=lambda clip: clip.timeline_start_s
            )
            updated = track.model_copy(update={"clips": tuple(clips)})
            return _swap_track(plan, updated), RemoveClip(clip_id=operation.clip.id)

        case RemoveClip():
            track, clip = _find_clip(plan, operation.clip_id)
            without = tuple(item for item in track.clips if item.id != clip.id)
            trimmed = track.model_copy(update={"clips": without})
            return _swap_track(plan, trimmed), AddClip(track_id=track.id, clip=clip)

        case SetClipRange():
            track, clip = _find_clip(plan, operation.clip_id)
            changes = {
                field: value
                for field, value in (
                    ("source_in_s", operation.source_in_s),
                    ("source_out_s", operation.source_out_s),
                    ("timeline_start_s", operation.timeline_start_s),
                )
                if value is not None
            }
            undo_range: Op = SetClipRange(
                clip_id=clip.id,
                source_in_s=clip.source_in_s,
                source_out_s=clip.source_out_s,
                timeline_start_s=clip.timeline_start_s,
            )
            return _replace_clip(plan, track.id, clip.model_copy(update=changes)), undo_range

        case SetClipEnabled():
            track, clip = _find_clip(plan, operation.clip_id)
            undo_enabled: Op = SetClipEnabled(clip_id=clip.id, enabled=clip.enabled)
            toggled = clip.model_copy(update={"enabled": operation.enabled})
            return _replace_clip(plan, track.id, toggled), undo_enabled

        case PushEffect():
            track, clip = _find_clip(plan, operation.clip_id)
            inverse = SetEffects(clip_id=clip.id, effects=clip.effects)
            stacked = clip.model_copy(update={"effects": (*clip.effects, operation.effect)})
            return _replace_clip(plan, track.id, stacked), inverse

        case SetEffects():
            track, clip = _find_clip(plan, operation.clip_id)
            undo_effects: Op = SetEffects(clip_id=clip.id, effects=clip.effects)
            replaced = clip.model_copy(update={"effects": operation.effects})
            return _replace_clip(plan, track.id, replaced), undo_effects

        case SetTransition():
            track, clip = _find_clip(plan, operation.clip_id)
            field = "transition_in" if operation.side == "in" else "transition_out"
            undo_transition: Op = SetTransition(
                clip_id=clip.id,
                side=operation.side,
                transition=getattr(clip, field),
            )
            swapped = clip.model_copy(update={field: operation.transition})
            return _replace_clip(plan, track.id, swapped), undo_transition

        case AddSource():
            if operation.source.id in plan.sources:
                raise VisionUseError(
                    f"Source {operation.source.id!r} is already declared",
                    remedy="use a different id, or remove the existing source first",
                )
            grown_sources = {**plan.sources, operation.source.id: operation.source}
            return (
                plan.model_copy(update={"sources": grown_sources}),
                RemoveSource(source_id=operation.source.id),
            )

        case RemoveSource():
            existing_source = plan.sources.get(operation.source_id)
            if existing_source is None:
                raise VisionUseError(
                    f"No source {operation.source_id!r} to remove",
                    remedy="check the source id against `vision plan show`",
                )
            kept_sources = {
                key: value for key, value in plan.sources.items() if key != operation.source_id
            }
            return (
                plan.model_copy(update={"sources": kept_sources}),
                AddSource(source=existing_source),
            )

        case SetCanvas():
            timeline = plan.timeline
            undo_canvas: Op = SetCanvas(width=timeline.width, height=timeline.height)
            resized = timeline.model_copy(
                update={"width": operation.width, "height": operation.height}
            )
            return plan.model_copy(update={"timeline": resized}), undo_canvas

        case AddTrack():
            tracks = (*plan.timeline.tracks, operation.track)
            grown = plan.model_copy(
                update={"timeline": plan.timeline.model_copy(update={"tracks": tracks})}
            )
            return grown, RemoveTrack(track_id=operation.track.id)

        case RemoveTrack():
            existing = plan.track_by_id(operation.track_id)
            if existing is None:
                raise VisionUseError(
                    f"No track {operation.track_id!r} to remove",
                    remedy="check the track id against `vision plan show`",
                )
            kept = tuple(item for item in plan.timeline.tracks if item.id != operation.track_id)
            shrunk = plan.model_copy(
                update={"timeline": plan.timeline.model_copy(update={"tracks": kept})}
            )
            return shrunk, AddTrack(track=existing)

    raise VisionUseError(  # pragma: no cover - the match above is exhaustive
        f"Unsupported operation {operation!r}",
        remedy="this is a bug; please report it",
    )


def _swap_track(plan: EditPlan, updated: Track) -> EditPlan:
    """Return ``plan`` with one track replaced in place."""
    tracks = tuple(
        updated if track.id == updated.id else track for track in plan.timeline.tracks
    )
    return plan.model_copy(
        update={"timeline": plan.timeline.model_copy(update={"tracks": tracks})}
    )


class PatchResult(BaseModel):
    """The outcome of applying a patch."""

    model_config = ConfigDict(frozen=True)

    plan: EditPlan
    inverse: PlanPatch

    def undo(self) -> EditPlan:
        """Return the plan as it was before the patch was applied."""
        return apply_patch(self.plan, self.inverse).plan


def apply_patch(plan: EditPlan, patch: PlanPatch) -> PatchResult:
    """Apply ``patch`` to ``plan``, returning the result and how to undo it.

    Operations apply in order. If any one fails -- an unknown clip, a resulting
    plan that violates the model's invariants -- **nothing is applied**: the
    caller gets an exception and the original plan, never a half-edited one.
    That is what lets the retry loop try a strategy and cleanly abandon it.

    Args:
        plan: The plan to edit.
        patch: The operations to apply.

    Returns:
        The new plan and the patch that reverses it.

    Raises:
        VisionUseError: If any operation cannot be applied, or if the result
            would be an invalid plan.

    """
    current = plan
    inverses: list[Op] = []

    for index, operation in enumerate(patch.ops):
        try:
            current, inverse = _apply_one(current, operation)
            # `model_copy` deliberately skips validation, so every operation is
            # re-validated here. Without this an operation could construct a
            # plan the model would have rejected -- overlapping clips, a range
            # past the end of its source -- and the failure would surface much
            # later, somewhere that cannot say which operation caused it.
            current = EditPlan.model_validate(current.model_dump())
        except VisionUseError:
            raise
        except ValueError as exc:
            raise VisionUseError(
                f"Operation {index} ({operation.op}) would make the plan invalid: {exc}",
                remedy="the patch was rejected; the plan is unchanged",
            ) from exc
        inverses.append(inverse)

    # Reversed, because undoing means unwinding in the opposite order.
    return PatchResult(
        plan=current,
        inverse=PlanPatch(
            ops=tuple(reversed(inverses)),
            rationale=f"undo: {patch.rationale}" if patch.rationale else "undo",
            produced_by=patch.produced_by,
        ),
    )
