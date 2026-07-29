"""Pre-rendering effects that a filtergraph cannot express.

Some effects need state across frames, or a model, or both: stabilisation,
object removal, upscaling. They cannot be a link in a filter chain, so the
compiler *bakes* them -- renders the affected span to an intermediate file
first, then treats that file as an ordinary input.

The result is content-addressed and cached on
``sha256(effect_type + version + params + source_digest + range)``. Baking is by
far the most expensive thing VisionUse does, and the retry loop re-renders
constantly, so an unchanged clip must bake exactly once ever.

The plan is never rewritten on disk. Baking substitutes clips in memory, on the
way into the graph, so the document the user edits stays the document they
wrote.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from vision_use.cache.store import CacheKey, CacheStore
from vision_use.effects.registry import EffectRegistry
from vision_use.effects.registry import registry as default_effects
from vision_use.errors import VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.media import FrameRate
from vision_use.models.plan import Clip, EditPlan, Effect, Source
from vision_use.progress import ProgressCallback, Reporter, Stage
from vision_use.video.transform import (
    INTERMEDIATE_SUFFIX,
    TRANSFORM_VERSION,
    FrameOp,
    transform_span,
)

log = get_logger(__name__)

#: Cache namespace for baked spans.
BAKE_NAMESPACE = "bake"


@dataclass(frozen=True, slots=True)
class _BakeRequest:
    """One clip's baking job. Bundles what would otherwise be six parameters."""

    clip: Clip
    source: Source
    baked_effects: list[Effect]
    rate: FrameRate


@dataclass(frozen=True, slots=True)
class BakedPlan:
    """A plan with every baked effect resolved to an intermediate file.

    Attributes:
        plan: The substituted plan, ready for the filtergraph compiler.
        baked: How many clips were baked.
        reused: How many of those came from the cache rather than being rendered.

    """

    plan: EditPlan
    baked: int = 0
    reused: int = 0


def _needs_baking(clip: Clip, effects: EffectRegistry) -> bool:
    """Whether ``clip`` carries any effect that must be pre-rendered."""
    return any(effects.get(effect.type).bakes for effect in clip.effects)


def bake_plan(
    plan: EditPlan,
    *,
    cache: CacheStore | None = None,
    effects: EffectRegistry | None = None,
    on_progress: ProgressCallback | None = None,
) -> BakedPlan:
    """Resolve every baked effect in ``plan``, returning a renderable plan.

    Clips carrying no baked effect are returned untouched, so a plan that needs
    no baking costs one pass over its clips and nothing else.

    Args:
        plan: The plan to prepare.
        cache: Store for intermediates. Defaults to the standard root.
        effects: Effect registry.
        on_progress: Called as frames are baked. Baking is the slowest thing
            here, so this is where silence is most likely to be mistaken for a
            hang.

    Returns:
        The substituted plan and what it cost.

    Raises:
        VisionUseError: If an effect declares that it bakes but does not
            implement the transformation, or if a span cannot be rendered.

    """
    registry = default_effects if effects is None else effects
    store = cache or CacheStore()
    report = Reporter(on_progress)

    if not any(_needs_baking(clip, registry) for clip in plan.timeline.clips()):
        return BakedPlan(plan=plan)

    sources = dict(plan.sources)
    tracks = []
    baked = reused = 0

    for track in plan.timeline.tracks:
        clips = []
        for clip in track.clips:
            if not clip.enabled or not _needs_baking(clip, registry):
                clips.append(clip)
                continue

            substitute, source, was_cached = _bake_clip(
                plan, clip, store, registry, len(sources), report
            )
            sources[source.id] = source
            clips.append(substitute)
            baked += 1
            reused += int(was_cached)
        tracks.append(track.model_copy(update={"clips": tuple(clips)}))

    prepared = plan.model_copy(
        update={
            "sources": sources,
            "timeline": plan.timeline.model_copy(update={"tracks": tuple(tracks)}),
        }
    )
    # Enforced here, after the intermediates are committed and the substituted
    # plan already points at them: baking is the only thing that writes at a
    # scale that fills a disk, and evicting before the plan was built could drop
    # an artefact this very render is about to read.
    store.enforce_budget(protect=[source.path for source in sources.values()])

    log.info("baked plan", extra=bind(clips=baked, from_cache=reused))
    return BakedPlan(plan=prepared, baked=baked, reused=reused)


def _bake_clip(
    plan: EditPlan,
    clip: Clip,
    store: CacheStore,
    effects: EffectRegistry,
    next_index: int,
    report: Reporter,
) -> tuple[Clip, Source, bool]:
    """Bake one clip's baked effects, returning its substitute and new source."""
    original = plan.sources[clip.source_id]
    baked_effects = [effect for effect in clip.effects if effects.get(effect.type).bakes]
    remaining = tuple(effect for effect in clip.effects if not effects.get(effect.type).bakes)

    key = store.key(
        BAKE_NAMESPACE,
        version=TRANSFORM_VERSION,
        inputs=[original.sha256],
        params={
            "range": [clip.source_in_s, clip.source_out_s],
            # The effect's `version` is in the key because nothing else in it
            # can see the implementation. A plugin that corrects its frame_op
            # would otherwise keep serving the artefact the broken one made,
            # since the source, range and parameters are all unchanged --
            # found exactly that way while building `vision-stabilize`.
            "effects": [
                {
                    "type": effect.type,
                    "params": dict(effect.params),
                    "version": effects.get(effect.type).version,
                }
                for effect in baked_effects
            ],
        },
        suffix=INTERMEDIATE_SUFFIX,
    )

    cached = store.get(key)
    was_cached = cached is not None
    if cached is None:
        cached = _render_intermediate(
            report,
            _BakeRequest(
                clip=clip,
                source=original,
                baked_effects=baked_effects,
                rate=plan.timeline.frame_rate,
            ),
            effects,
            store,
            key,
        )

    duration = round(clip.source_out_s - clip.source_in_s, 6)
    substitute_source = Source(
        # The intermediate is a new source, numbered after the declared ones so
        # it cannot collide with an id the plan already uses.
        id=f"src_{next_index:04d}",
        path=cached,
        sha256=original.sha256,
        duration_s=duration,
        # Intermediates are video only; the clip's audio still comes from the
        # original source, which the compiler handles because the substitute
        # declares no audio.
        has_audio=False,
    )
    substitute = clip.model_copy(
        update={
            "source_id": substitute_source.id,
            "source_in_s": 0.0,
            "source_out_s": duration,
            "effects": remaining,
        }
    )
    return substitute, substitute_source, was_cached


def _expected_frames(clip: Clip, rate: FrameRate) -> int:
    """Estimate how many frames a clip's span will yield.

    Exact for a constant-rate source, which is nearly all of them. A plan does
    not record whether its source is variable-rate -- that is a probe fact, and
    compiling stays free of media I/O -- so on a VFR source this is an estimate
    and `Progress.fraction` clamps at 1.0 rather than reporting 103%.
    """
    return max(1, round((clip.source_out_s - clip.source_in_s) * rate.decimal))


#: One step of a baked stack: a frame and its clip-local time in, a frame out.
_TimedOp = Callable[[NDArray[np.uint8], float], NDArray[np.uint8]]


def _params_at(effects: EffectRegistry, effect: Effect, t_s: float) -> BaseModel:
    """Return ``effect``'s parameters as they stand at clip-local ``t_s``.

    Validated rather than substituted straight into the model: the interpolated
    value has to satisfy the same constraints as an authored one -- an animation
    sweeping a crop's ``x`` negative is a bad plan, and finding that out at the
    frame it happens is better than producing a corrupt intermediate.
    """
    values = dict(effect.params)
    for name in effect.animate:
        current = effect.value_at(name, t_s)
        if current is not None:
            # Rounded to int when the static value was one: a crop offset of
            # 12.5 pixels is not a thing, and the params model would reject it.
            values[name] = round(current) if isinstance(effect.params[name], int) else current
    return effects.validate(effect.model_copy(update={"params": values}))


def _timed(
    operation: Callable[[NDArray[np.uint8], BaseModel], NDArray[np.uint8]],
    effects: EffectRegistry,
    effect: Effect,
) -> _TimedOp:
    """Adapt an animated frame op to the timed signature the stack uses."""

    def apply(frame: NDArray[np.uint8], t_s: float) -> NDArray[np.uint8]:
        return operation(frame, _params_at(effects, effect, t_s))

    return apply


def _untimed(operation: FrameOp) -> _TimedOp:
    """Adapt a static frame op to the timed signature, ignoring the time."""

    def apply(frame: NDArray[np.uint8], t_s: float) -> NDArray[np.uint8]:
        del t_s
        return operation(frame)

    return apply


def _compose(effects: EffectRegistry, baked_effects: list[Effect], rate: FrameRate) -> FrameOp:
    """Return one callable applying every baked effect in stack order.

    Animated effects are handed their interpolated parameters for each frame;
    static ones are built once, as before. Time is derived from the frame index
    rather than a decoded timestamp because the intermediate is written at a
    fixed rate, and the two agree by construction for the constant-rate sources
    this path handles.

    Raises:
        VisionUseError: If an effect declares that it bakes but has no
            implementation -- a plugin bug worth naming plainly.

    """
    operations: list[_TimedOp] = []
    for effect in baked_effects:
        definition = effects.get(effect.type)
        try:
            if effect.animated and definition.animatable:
                animated = definition.frame_op_animated(effects.validate(effect))
                operations.append(_timed(animated, effects, effect))
            else:
                operations.append(_untimed(definition.frame_op(effects.validate(effect))))
        except NotImplementedError as exc:
            raise VisionUseError(
                f"Effect {definition.effect_type!r} declares that it bakes but "
                "does not implement frame_op()",
                remedy="implement frame_op(), or set bakes = False",
            ) from exc

    fps = rate.decimal
    index = 0

    def apply_all(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply every baked effect to one frame, in stack order."""
        nonlocal index
        t_s = index / fps
        index += 1
        for operation in operations:
            frame = operation(frame, t_s)
        return frame

    return apply_all


def _render_intermediate(
    report: Reporter,
    request: _BakeRequest,
    effects: EffectRegistry,
    store: CacheStore,
    key: CacheKey,
) -> Path:
    """Render one clip's baked span into the cache and return its path."""
    clip, source = request.clip, request.source
    rate: FrameRate = request.rate
    operation = _compose(effects, request.baked_effects, rate)

    log.info(
        "baking clip",
        extra=bind(
            clip=clip.id,
            effects=[effect.type for effect in request.baked_effects],
            span=f"{clip.source_in_s}-{clip.source_out_s}",
        ),
    )

    # Frames expected, so a caller can draw a bar that fills rather than a
    # number that counts upwards.
    expected = _expected_frames(clip, rate)

    with store.writing(key) as tmp:
        written = transform_span(
            source.path,
            tmp,
            operation,
            rate,
            span=(clip.source_in_s, clip.source_out_s),
            on_frame=lambda done: report(Stage.BAKE, done, expected, clip.id),
        )
    if not written:
        raise VisionUseError(
            f"Baking {clip.id} produced no frames",
            remedy="check the clip's source range covers decodable frames",
        )
    return store.path_for(key)
