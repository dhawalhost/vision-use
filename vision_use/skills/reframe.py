"""Reframing: turning a landscape source into a vertical one.

The request behind this is *"make this a Short"* -- take 16:9 footage and
produce 9:16 without the letterboxing that makes a clip look like an
afterthought. That means cropping a window out of the frame and shrinking the
canvas to match, not scaling the whole picture down.

**It follows the subject, per shot.** The `subject` section says where each
shot's activity is concentrated, and the crop window centres on that rather than
on the middle of the frame. A clip whose shot has no clear subject -- nothing
moving, or everything moving -- falls back to centre, which is what makes
`SubjectSpan.confidence` worth reporting at all.

The skill reads that section rather than detecting anything itself
(Principle 8), so improving subject location improves reframing everywhere and
costs this file no change.

**What it still does not do:** move the crop *within* a shot. Effect parameters
are static per clip, so a window that drifts with a walking subject is not
expressible in the plan model today. Fixed strategies -- centre, left, right --
remain available behind the subject-aware one, so the retry loop has somewhere
to go when a subject-led crop is rejected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from pydantic import JsonValue

from vision_use.errors import VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.context import ContextSection, VideoContext
from vision_use.models.plan import Clip, EditPlan, Effect, Keyframe
from vision_use.skills.patch import Op, PlanPatch, PushEffect, SetCanvas
from vision_use.skills.skill import (
    Analysis,
    Measurement,
    Score,
    SkillMeta,
    SkillRequest,
    Strategy,
    ValidationReport,
)

log = get_logger(__name__)

#: Aspect ratios a caller can ask for, by name. Width divided by height.
ASPECTS: dict[str, float] = {
    "vertical": 9 / 16,
    "square": 1.0,
    "landscape": 16 / 9,
}

#: Default target. Vertical is what "make this a Short" means.
DEFAULT_ASPECT: str = "vertical"

#: Horizontal focus offered as strategies, as a fraction of the frame width.
#: Centre first: it is right more often than not, and a wrong guess about which
#: side someone is standing on is more jarring than a centred crop.
DEFAULT_FOCUS: tuple[tuple[str, float], ...] = (
    ("centre", 0.5),
    ("left", 0.33),
    ("right", 0.67),
)

#: How close the output aspect must be to the requested one. Crop dimensions are
#: forced even for yuv420p, so an exact ratio is not always reachable; a frame
#: or two of slack is invisible and a wrong ratio is not.
ASPECT_TOLERANCE: float = 0.02

#: Smallest usable frame dimension. yuv420p subsamples chroma by two, so a
#: single-pixel edge cannot be encoded at all.
MIN_DIMENSION: int = 2

#: Confidence below which a located subject is ignored in favour of the centre.
#: Set above what a frame of uniform motion scores: `testsrc`, which changes
#: everywhere and contains no subject at all, reports around 0.32, and cropping
#: to the centroid of *that* is worse than not trying.
MIN_SUBJECT_CONFIDENCE: float = 0.5


@dataclass(frozen=True, slots=True)
class CropWindow:
    """The rectangle to keep, in source pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def aspect(self) -> float:
        """Width divided by height."""
        return self.width / self.height


def _param(params: Mapping[str, JsonValue], name: str) -> int:
    """Return an integer effect parameter, defaulting to zero.

    Effect params are loosely typed by design -- the plan model must be able to
    hold effects it has never heard of -- so a reader has to narrow them.
    """
    value = params.get(name, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _even(value: int) -> int:
    """Round down to an even number, which yuv420p requires."""
    return max(MIN_DIMENSION, value - (value % 2))


def crop_window(
    source_width: int,
    source_height: int,
    target_aspect: float,
    focus: float = 0.5,
) -> CropWindow:
    """Return the largest window of ``target_aspect`` that fits the source.

    Whichever dimension is the constraint is kept whole and the other is cut,
    so the crop always takes as much of the original picture as the target
    shape allows.

    ``focus`` positions the window horizontally as a fraction of the frame, and
    is clamped so the window never runs off the edge -- a crop that asks for
    pixels outside the frame is a black bar, which is the thing this skill
    exists to avoid.

    Args:
        source_width: Source frame width.
        source_height: Source frame height.
        target_aspect: Wanted width divided by height.
        focus: Horizontal centre of the window, 0.0 to 1.0.

    Returns:
        The window to crop.

    Raises:
        VisionUseError: If the source or the target aspect is degenerate.

    """
    if source_width < MIN_DIMENSION or source_height < MIN_DIMENSION:
        raise VisionUseError(
            f"Cannot reframe a {source_width}x{source_height} source",
            remedy="check the plan's timeline dimensions",
        )
    if target_aspect <= 0:
        raise VisionUseError(
            f"Target aspect must be positive, got {target_aspect}",
            remedy="choose one of: " + ", ".join(sorted(ASPECTS)),
        )

    if source_width / source_height > target_aspect:
        # Source is wider than wanted: keep the full height, cut the sides.
        height = _even(source_height)
        width = _even(round(height * target_aspect))
    else:
        # Source is taller than wanted: keep the full width, cut top and bottom.
        width = _even(source_width)
        height = _even(round(width / target_aspect))

    width = min(width, _even(source_width))
    height = min(height, _even(source_height))

    x = round(source_width * focus - width / 2)
    x = max(0, min(x, source_width - width))
    y = max(0, (source_height - height) // 2)
    return CropWindow(
        x=_even(x) if x else 0, y=_even(y) if y else 0, width=width, height=height
    )


#: Fewest track points worth animating a crop with. Two is enough to move.
_MIN_TRACK_POINTS = 2


def _subject_track(context: VideoContext, clip: Clip) -> tuple[tuple[float, float], ...]:
    """Return this clip's subject positions as (clip-local seconds, fraction).

    Empty when the section is absent or the covering span has no track, in
    which case the caller falls back to a fixed window.

    **Summary confidence deliberately does not gate this.** A subject who
    crosses the frame has no single place to put a static crop, so the span's
    confidence is low -- which is exactly the case a *moving* crop handles best.
    A track that barely varies degrades naturally into a static crop, so there
    is nothing to guard against.
    """
    if not context.has(ContextSection.SUBJECT):
        return ()

    midpoint = (clip.source_in_s + clip.source_out_s) / 2
    for span in context.subject:
        if not span.span.contains(midpoint):
            continue
        # Clip-local, which is the base keyframes use (PLAN.md D6), and only
        # the points this clip actually covers: a clip trimmed out of a longer
        # shot must not inherit keyframes for footage it does not show.
        return tuple(
            (point.t_s - clip.source_in_s, point.centre_x)
            for point in span.track
            if clip.source_in_s <= point.t_s <= clip.source_out_s
        )
    return ()


def _subject_focus(context: VideoContext, clip: Clip) -> float | None:
    """Return where this clip's subject sits horizontally, or ``None``.

    ``None`` means "no usable evidence" -- either the section was never
    populated or the located subject was too weak to trust -- and the caller
    should centre. Distinguishing that from `0.5` matters: a subject genuinely
    in the middle is not the same as no subject at all.
    """
    if not context.has(ContextSection.SUBJECT):
        return None

    # The clip's midpoint in *source* time, which is what subject spans are
    # measured against. A clip is a window onto the source, and its position on
    # the timeline says nothing about which shot it came from.
    midpoint = (clip.source_in_s + clip.source_out_s) / 2
    for span in context.subject:
        if span.span.contains(midpoint):
            if (span.confidence or 0.0) < MIN_SUBJECT_CONFIDENCE:
                return None
            return span.centre_x
    return None


def _first(frames: tuple[Keyframe, ...] | None, fallback: int) -> int:
    """Return the value an animation starts at, or ``fallback`` if there is none."""
    return int(frames[0].value) if frames else fallback


def _keyframes(
    source_width: int, window_width: int, points: tuple[tuple[float, float], ...]
) -> tuple[Keyframe, ...]:
    """Turn subject positions into crop offsets the plan can hold.

    Each point becomes the window's left edge, clamped so the crop never runs
    off the frame -- asking for pixels outside it is a black bar, which is the
    thing this skill exists to avoid.
    """
    limit = max(0, source_width - window_width)
    return tuple(
        Keyframe(
            t_s=round(max(0.0, t_s), 3),
            value=float(min(limit, max(0, round(source_width * fraction - window_width / 2)))),
        )
        for t_s, fraction in points
    )


@dataclass(frozen=True, slots=True)
class _Reframe:
    """What ``execute`` needs to carry out one focus.

    A window per clip rather than one for all of them: two shots can put the
    subject in different places, and a single window would be right for at most
    one of them. Every window is the same size -- only the horizontal offset
    differs -- so one canvas still fits them all.
    """

    windows: dict[str, CropWindow]
    canvas: CropWindow
    #: Keyframes for each clip whose crop follows the subject, in the form the
    #: plan model wants. Empty for the fixed strategies, which is what makes
    #: them the fallback: the same execute path serves both.
    tracks: dict[str, tuple[Keyframe, ...]] = field(default_factory=dict)


class Reframe:
    """Crops a plan to a different aspect ratio.

    Reads only the probe: the crop is geometric, and nothing about it needs the
    video decoded.
    """

    meta: ClassVar[SkillMeta] = SkillMeta(
        name="edit.reframe",
        version="2",
        summary="Crop to a different aspect ratio, e.g. 16:9 to 9:16.",
        # Subject as well as probe: the crop follows where the action is, and
        # the runtime backfills that section once for every skill that wants it
        # rather than this one growing its own detector (Principle 8).
        requires=frozenset({ContextSection.PROBE, ContextSection.SUBJECT}),
    )

    def __init__(
        self,
        aspect: str = DEFAULT_ASPECT,
        focus: tuple[tuple[str, float], ...] = DEFAULT_FOCUS,
    ) -> None:
        if aspect not in ASPECTS:
            raise ValueError(f"unknown aspect {aspect!r}; choose from {sorted(ASPECTS)}")
        if not focus:
            raise ValueError("at least one focus position is required")
        self._aspect = aspect
        self._focus = focus

    def analyse(
        self,
        context: VideoContext,
        plan: EditPlan,
        request: SkillRequest,  # noqa: ARG002 - the focus comes from strategies
    ) -> Analysis:
        """Measure the shape gap, and find where each clip's subject sits."""
        target = ASPECTS[self._aspect]
        current = plan.timeline.width / plan.timeline.height

        if abs(current - target) <= ASPECT_TOLERANCE:
            # Already the right shape; cropping would only lose picture.
            return Analysis(findings={"aspect": round(current, 4), "already": True})

        clips = [clip for clip in plan.timeline.clips() if clip.enabled]
        if not clips:
            return Analysis(findings={"clips": 0})

        focus = {clip.id: _subject_focus(context, clip) for clip in clips}
        tracks = {clip.id: _subject_track(context, clip) for clip in clips}
        located = sum(1 for value in focus.values() if value is not None)
        following = sum(1 for value in tracks.values() if len(value) >= _MIN_TRACK_POINTS)

        return Analysis(
            findings={
                "from": round(current, 4),
                "to": round(target, 4),
                "clips": len(clips),
                "subject_located": located,
                "subject_tracked": following,
            },
            detail=(plan.timeline.width, plan.timeline.height, focus, tracks),
        )

    def plan(self, analysis: Analysis) -> list[Strategy]:
        """Offer the subject-led crop first, then the fixed positions.

        Subject-led first because it is the one that is right when it applies;
        the fixed positions stay behind it so a rejected strategy still has
        somewhere to go, and so a source with no locatable subject reframes
        exactly as it did before this skill could see one.
        """
        if not isinstance(analysis.detail, tuple):
            return []
        width, height, focus, tracks = analysis.detail
        target = ASPECTS[self._aspect]
        canvas = crop_window(width, height, target, 0.5)

        strategies: list[Strategy] = []

        # Following the subject *within* each shot, when there is a track to
        # follow. Offered ahead of the static subject crop because a window that
        # moves keeps a walking subject in frame where one fixed position -- even
        # the best one -- cannot.
        if any(len(value) >= _MIN_TRACK_POINTS for value in tracks.values()):
            moving = sum(1 for value in tracks.values() if len(value) >= _MIN_TRACK_POINTS)
            strategies.append(
                Strategy(
                    name=f"{self._aspect}-follow",
                    rationale=(
                        f"crop to {self._aspect} ({target:.3f}), following the subject "
                        f"as it moves in {moving} of {len(tracks)} clip(s)"
                    ),
                    detail=_Reframe(
                        windows={
                            clip_id: crop_window(width, height, target, 0.5)
                            for clip_id in focus
                        },
                        canvas=canvas,
                        tracks={
                            clip_id: _keyframes(width, canvas.width, points)
                            for clip_id, points in tracks.items()
                            if len(points) >= _MIN_TRACK_POINTS
                        },
                    ),
                )
            )

        if any(value is not None for value in focus.values()):
            located = sum(1 for value in focus.values() if value is not None)
            strategies.append(
                Strategy(
                    name=f"{self._aspect}-subject",
                    rationale=(
                        f"crop to {self._aspect} ({target:.3f}), following the subject "
                        f"in {located} of {len(focus)} clip(s)"
                    ),
                    detail=_Reframe(
                        windows={
                            clip_id: crop_window(
                                width, height, target, 0.5 if value is None else value
                            )
                            for clip_id, value in focus.items()
                        },
                        canvas=canvas,
                    ),
                )
            )

        strategies.extend(
            Strategy(
                name=f"{self._aspect}-{name}",
                rationale=(f"crop to {self._aspect} ({target:.3f}), focused {name}"),
                detail=_Reframe(
                    windows={
                        clip_id: crop_window(width, height, target, fraction)
                        for clip_id in focus
                    },
                    canvas=canvas,
                ),
            )
            for name, fraction in self._focus
        )
        return strategies

    def execute(self, strategy: Strategy) -> PlanPatch:
        """Push a crop onto every clip and resize the canvas to match.

        Both are needed: cropping alone would leave the picture sitting inside
        the old canvas, which the renderer would letterbox back to the original
        shape -- the exact outcome this skill exists to prevent.

        Raises:
            VisionUseError: If the strategy did not come from this skill.

        """
        reframe = strategy.detail
        if not isinstance(reframe, _Reframe):
            raise VisionUseError(
                "This strategy did not come from this skill",
                remedy="pass a strategy returned by plan()",
            )

        canvas = reframe.canvas
        ops: list[Op] = [
            PushEffect(
                clip_id=clip_id,
                effect=Effect(
                    type="crop",
                    params={
                        # The static value stays the window's own offset, so a
                        # reader ignoring the animation still gets a sensible
                        # crop (PLAN.md D6).
                        "x": _first(reframe.tracks.get(clip_id), window.x),
                        "y": window.y,
                        "width": window.width,
                        "height": window.height,
                    },
                    animate=(
                        {"x": reframe.tracks[clip_id]} if clip_id in reframe.tracks else {}
                    ),
                ),
            )
            for clip_id, window in sorted(reframe.windows.items())
        ]
        # One canvas for every clip: the windows differ only in offset, so the
        # timeline stays a single shape and the compiler needs no special case.
        ops.append(SetCanvas(width=canvas.width, height=canvas.height))

        log.debug(
            "planned reframe",
            extra=bind(
                clips=len(reframe.windows),
                canvas=f"{canvas.width}x{canvas.height}",
                offsets=sorted({window.x for window in reframe.windows.values()}),
            ),
        )
        return PlanPatch(
            ops=tuple(ops), rationale=strategy.rationale, produced_by=self.meta.ref
        )

    def validate(
        self, before: EditPlan, after: EditPlan, context: VideoContext
    ) -> ValidationReport:
        """Check the canvas is the requested shape and the crop stayed inside."""
        del context
        target = ASPECTS[self._aspect]
        achieved = after.timeline.width / after.timeline.height

        inside = True
        for clip in after.timeline.clips():
            for effect in clip.effects:
                if effect.type != "crop":
                    continue
                right = _param(effect.params, "x") + _param(effect.params, "width")
                bottom = _param(effect.params, "y") + _param(effect.params, "height")
                inside &= right <= before.timeline.width
                inside &= bottom <= before.timeline.height

        even = after.timeline.width % 2 == 0 and after.timeline.height % 2 == 0

        return ValidationReport(
            measurements=(
                Measurement(
                    name="aspect",
                    value=round(achieved, 4),
                    passed=abs(achieved - target) <= ASPECT_TOLERANCE,
                    detail=f"wanted {target:.4f}, got {achieved:.4f}",
                ),
                Measurement(
                    name="crop_inside_frame",
                    value=float(inside),
                    passed=inside,
                    detail="the crop asks for pixels outside the source frame",
                ),
                Measurement(
                    name="even_dimensions",
                    value=float(even),
                    passed=even,
                    detail="yuv420p cannot encode an odd dimension",
                ),
            )
        )

    def score(self, report: ValidationReport) -> Score:
        """Prefer the attempt whose aspect lands closest to the request."""
        if not report.passed:
            return Score(value=-1.0, detail="rejected by validation")
        achieved = next(
            (item.value for item in report.measurements if item.name == "aspect"), 0.0
        )
        target = ASPECTS[self._aspect]
        return Score(value=-abs(achieved - target), detail=f"aspect {achieved:.4f}")
