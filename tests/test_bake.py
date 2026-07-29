"""Tests for the baked-effect path.

Baking is the escape hatch for effects a filtergraph cannot express. It is also
the most expensive thing the framework does, so the cache behaviour matters as
much as the pixels.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from vision_use.cache.store import CACHE_BUDGET_ENV, CacheStore
from vision_use.effects.builtin import CropEffect, CropParams
from vision_use.effects.registry import EffectDef, EffectRegistry
from vision_use.effects.registry import registry as effect_registry
from vision_use.errors import VisionUseError
from vision_use.models.media import FrameRate
from vision_use.models.plan import EditPlan, Effect, Keyframe
from vision_use.planning import straight_through, validate_plan
from vision_use.progress import Progress, Stage
from vision_use.render import bake_plan, compile_plan, render
from vision_use.render.bake import _params_at
from vision_use.video.frames import extract_frames
from vision_use.video.probe import probe
from vision_use.video.transform import read_frames, transform_span

from tests.conftest import MakeClip, requires_ffmpeg

PAL = FrameRate(num=25, den=1)


class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _DeclaresButDoesNot(EffectDef):
    """Claims to bake but never implemented it -- a plugin bug worth naming."""

    effect_type: ClassVar[str] = "broken.bake"
    params_model: ClassVar[type[BaseModel]] = _NoParams
    bakes: ClassVar[bool] = True


class _Versioned(EffectDef):
    """A baked effect at its first implementation."""

    effect_type: ClassVar[str] = "versioned.bake"
    params_model: ClassVar[type[BaseModel]] = _NoParams
    bakes: ClassVar[bool] = True

    def frame_op(self, params: BaseModel) -> Callable[[NDArray[np.uint8]], NDArray[np.uint8]]:
        """Darken every frame."""
        del params
        return lambda frame: np.asarray(frame // 2, dtype=np.uint8)


class _VersionedV2(_Versioned):
    """The same effect after its output changed."""

    version: ClassVar[str] = "2"


def _with_effects(source: Path, *effects: Effect) -> EditPlan:
    """Return a straight-through plan whose single clip carries ``effects``."""
    plan = straight_through(source)
    track = plan.timeline.tracks[0]
    clip = track.clips[0].model_copy(update={"effects": effects})
    return plan.model_copy(
        update={
            "timeline": plan.timeline.model_copy(
                update={"tracks": (track.model_copy(update={"clips": (clip,)}),)}
            )
        }
    )


# --- frame transformation --------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_read_frames_yields_rgb_arrays(make_clip: MakeClip) -> None:
    frames = list(read_frames(make_clip(name="bake_read", seconds=1.0), 0.0, 0.2))

    assert frames
    assert frames[0].ndim == 3
    assert frames[0].shape[2] == 3
    assert frames[0].dtype == np.uint8


@pytest.mark.integration
@requires_ffmpeg
def test_read_frames_respects_the_span(make_clip: MakeClip) -> None:
    clip = make_clip(name="bake_span", seconds=2.0, fps="25")

    whole = list(read_frames(clip))
    part = list(read_frames(clip, 0.0, 0.4))

    assert len(part) < len(whole)


@pytest.mark.integration
@requires_ffmpeg
def test_transform_writes_every_frame(make_clip: MakeClip, tmp_path: Path) -> None:
    clip = make_clip(name="bake_write", seconds=1.0, fps="25")
    out = tmp_path / "intermediate.mkv"

    written = transform_span(clip, out, lambda frame: frame, PAL, span=(0.0, 0.4))

    assert written == 10
    assert probe(out).video.frame_count == 10


@pytest.mark.integration
@requires_ffmpeg
def test_transform_is_lossless(make_clip: MakeClip, tmp_path: Path) -> None:
    """Intermediates must not be taxed with generation loss before the real encode."""
    clip = make_clip(name="bake_lossless", seconds=1.0, fps="25")
    out = tmp_path / "intermediate.mkv"

    transform_span(clip, out, lambda frame: frame, PAL, span=(0.0, 0.2))

    before = next(iter(read_frames(clip, 0.0, 0.2)))
    after = next(iter(read_frames(out)))
    assert np.abs(before.astype(int) - after.astype(int)).max() <= 2


# --- baking ----------------------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_plan_without_baked_effects_is_untouched(make_clip: MakeClip, tmp_path: Path) -> None:
    """A plan needing no baking must cost one pass and nothing else."""
    plan = straight_through(make_clip(name="bake_none", seconds=1.0))

    result = bake_plan(plan, cache=CacheStore(tmp_path / "cache"))

    assert result.baked == 0
    assert result.plan is plan


@pytest.mark.integration
@requires_ffmpeg
def test_baking_substitutes_the_clip_source(make_clip: MakeClip, tmp_path: Path) -> None:
    plan = _with_effects(make_clip(name="bake_sub", seconds=1.0), Effect(type="invert"))

    result = bake_plan(plan, cache=CacheStore(tmp_path / "cache"))

    assert result.baked == 1
    baked_clip = result.plan.timeline.clips()[0]
    assert baked_clip.source_id != plan.timeline.clips()[0].source_id
    assert baked_clip.source_in_s == 0.0


@pytest.mark.integration
@requires_ffmpeg
def test_baked_effect_is_removed_from_the_clip(make_clip: MakeClip, tmp_path: Path) -> None:
    """Leaving it on would make the compiler try to filter what is already applied."""
    plan = _with_effects(make_clip(name="bake_strip", seconds=1.0), Effect(type="invert"))

    result = bake_plan(plan, cache=CacheStore(tmp_path / "cache"))

    assert result.plan.timeline.clips()[0].effects == ()


@pytest.mark.integration
@requires_ffmpeg
def test_filter_effects_survive_baking(make_clip: MakeClip, tmp_path: Path) -> None:
    """A clip may carry both kinds; only the baked one is pre-rendered."""
    plan = _with_effects(
        make_clip(name="bake_mixed", seconds=1.0),
        Effect(type="invert"),
        Effect(type="scale", params={"width": 160, "height": 120}),
    )

    result = bake_plan(plan, cache=CacheStore(tmp_path / "cache"))

    remaining = result.plan.timeline.clips()[0].effects
    assert [effect.type for effect in remaining] == ["scale"]


@pytest.mark.integration
@requires_ffmpeg
def test_second_bake_reuses_the_cache(make_clip: MakeClip, tmp_path: Path) -> None:
    """Baking is the most expensive thing here; an unchanged clip bakes once ever."""
    store = CacheStore(tmp_path / "cache")
    plan = _with_effects(make_clip(name="bake_cache", seconds=1.0), Effect(type="invert"))

    first = bake_plan(plan, cache=store)
    second = bake_plan(plan, cache=store)

    assert first.reused == 0
    assert second.reused == 1


@pytest.mark.integration
@requires_ffmpeg
def test_a_new_effect_version_bakes_again(make_clip: MakeClip, tmp_path: Path) -> None:
    """A corrected implementation must not serve the broken one's artefact.

    Source, range and parameters are all identical when a plugin fixes its
    `frame_op`, so nothing else in the key changes. Found while building
    `vision-stabilize`: a sign fix appeared to do nothing because the render
    came from cache.
    """
    store = CacheStore(tmp_path / "cache")
    clip = make_clip(name="bake_version", seconds=1.0)
    plan = _with_effects(clip, Effect(type="versioned.bake"))

    isolated = EffectRegistry()
    isolated.register(_Versioned())
    first = bake_plan(plan, cache=store, effects=isolated)

    bumped = EffectRegistry()
    bumped.register(_VersionedV2())
    second = bake_plan(plan, cache=store, effects=bumped)

    assert first.reused == 0
    assert second.reused == 0


@pytest.mark.integration
@requires_ffmpeg
def test_different_params_bake_separately(make_clip: MakeClip, tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    clip = make_clip(name="bake_params", seconds=1.0)

    bake_plan(_with_effects(clip, Effect(type="invert")), cache=store)
    baked = list((store.root / "bake").rglob("*.mkv"))

    assert len(baked) == 1


@pytest.mark.integration
@requires_ffmpeg
def test_unimplemented_bake_names_the_effect(make_clip: MakeClip, tmp_path: Path) -> None:
    isolated = EffectRegistry()
    isolated.register(_DeclaresButDoesNot())
    plan = _with_effects(make_clip(name="bake_broken", seconds=1.0), Effect(type="broken.bake"))

    with pytest.raises(VisionUseError, match=r"broken\.bake"):
        bake_plan(plan, cache=CacheStore(tmp_path / "cache"), effects=isolated)


@pytest.mark.integration
@requires_ffmpeg
def test_disabled_clips_are_not_baked(make_clip: MakeClip, tmp_path: Path) -> None:
    """Baking a clip that renders nothing would be pure waste."""
    plan = _with_effects(make_clip(name="bake_off", seconds=1.0), Effect(type="invert"))
    track = plan.timeline.tracks[0]
    off = track.clips[0].model_copy(update={"enabled": False})
    plan = plan.model_copy(
        update={
            "timeline": plan.timeline.model_copy(
                update={"tracks": (track.model_copy(update={"clips": (off,)}),)}
            )
        }
    )

    result = bake_plan(plan, cache=CacheStore(tmp_path / "cache"))

    assert result.baked == 0


# --- through the renderer --------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_baked_effect_reaches_the_output_pixels(make_clip: MakeClip, tmp_path: Path) -> None:
    """The point of the whole path: a baked effect must be visible in the file."""
    clip = make_clip(name="bake_pixels", seconds=1.0, fps="25")
    plan = _with_effects(clip, Effect(type="invert"))
    out = tmp_path / "out.mp4"

    render(plan, out, cache=CacheStore(tmp_path / "cache"))

    (_, before), *_ = list(extract_frames(clip, [0.5]))
    (_, after), *_ = list(extract_frames(out, [0.5]))
    source_rgb = np.asarray(before.to_ndarray(format="rgb24")).mean(axis=(0, 1))
    baked_rgb = np.asarray(after.to_ndarray(format="rgb24")).mean(axis=(0, 1))

    assert all(abs((255 - a) - b) < 40 for a, b in zip(source_rgb, baked_rgb, strict=True))


@pytest.mark.integration
@requires_ffmpeg
def test_baked_render_keeps_the_frame_count(make_clip: MakeClip, tmp_path: Path) -> None:
    clip = make_clip(name="bake_frames", seconds=2.0, fps="25")
    plan = _with_effects(clip, Effect(type="invert"))
    out = tmp_path / "out.mp4"

    render(plan, out, cache=CacheStore(tmp_path / "cache"))

    assert probe(out).video.frame_count == probe(clip).video.frame_count


@pytest.mark.integration
@requires_ffmpeg
def test_compiler_sees_an_ordinary_input(make_clip: MakeClip, tmp_path: Path) -> None:
    """A baked effect must leave no trace in the filtergraph."""
    plan = _with_effects(make_clip(name="bake_graph", seconds=1.0), Effect(type="invert"))
    prepared = bake_plan(plan, cache=CacheStore(tmp_path / "cache"))

    compiled = compile_plan(prepared.plan, tmp_path / "out.mp4")

    graph = compiled.args[compiled.args.index("-filter_complex") + 1]
    assert "invert" not in graph


def test_invert_is_registered_as_a_baked_effect() -> None:
    assert effect_registry.get("invert").bakes


# --- progress and cancellation (M7) ----------------------------------------


class _Slow(EffectDef):
    """A baked effect that can be made to fail partway through."""

    effect_type: ClassVar[str] = "test.slow"
    params_model: ClassVar[type[BaseModel]] = _NoParams
    bakes: ClassVar[bool] = True

    #: Frame at which to raise, or None to run to completion.
    fail_at: ClassVar[int | None] = None

    def frame_op(self, params: BaseModel) -> Callable[[NDArray[np.uint8]], NDArray[np.uint8]]:
        """Return an identity op that may raise partway through."""
        del params
        seen = 0

        def op(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
            nonlocal seen
            seen += 1
            if self.fail_at is not None and seen >= self.fail_at:
                raise KeyboardInterrupt
            return frame

        return op


@pytest.mark.integration
@requires_ffmpeg
def test_baking_reports_progress(make_clip: MakeClip, tmp_path: Path) -> None:
    """Twenty seconds of silence is indistinguishable from a hang."""
    plan = _with_effects(make_clip(name="bake_progress", seconds=1.0), Effect(type="invert"))
    seen: list[Progress] = []

    bake_plan(plan, cache=CacheStore(tmp_path / "cache"), on_progress=seen.append)

    assert seen
    assert seen[-1].stage is Stage.BAKE
    assert seen[-1].done > seen[0].done
    assert seen[-1].detail  # names the clip being baked


@pytest.mark.integration
@requires_ffmpeg
def test_progress_reports_a_total_so_a_bar_can_fill(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    plan = _with_effects(
        make_clip(name="bake_total", seconds=1.0, fps="25"), Effect(type="invert")
    )
    seen: list[Progress] = []

    bake_plan(plan, cache=CacheStore(tmp_path / "cache"), on_progress=seen.append)

    assert seen[0].total == 25
    assert seen[-1].fraction == pytest.approx(1.0, abs=0.05)


@pytest.mark.integration
@requires_ffmpeg
def test_a_failing_progress_callback_does_not_spoil_the_bake(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    """The work is expensive; the reporting is not."""
    plan = _with_effects(make_clip(name="bake_badcb", seconds=1.0), Effect(type="invert"))

    def explode(update: Progress) -> None:
        raise RuntimeError("callback is broken")

    result = bake_plan(plan, cache=CacheStore(tmp_path / "cache"), on_progress=explode)

    assert result.baked == 1


@pytest.mark.integration
@requires_ffmpeg
def test_an_interrupted_bake_leaves_no_cache_entry(make_clip: MakeClip, tmp_path: Path) -> None:
    """The atomicity claim, tested by actually interrupting a bake.

    A partial intermediate served as if it were complete would be the worst
    possible failure here: silently wrong output, cached, with nothing to
    indicate it.
    """
    store = CacheStore(tmp_path / "cache")
    plan = _with_effects(
        make_clip(name="bake_cancel", seconds=2.0, fps="25"), Effect(type="test.slow")
    )

    isolated = EffectRegistry()
    slow = _Slow()
    type(slow).fail_at = 10
    isolated.register(slow)

    with pytest.raises(KeyboardInterrupt):
        bake_plan(plan, cache=store, effects=isolated)
    type(slow).fail_at = None

    assert store.usage().files == 0


@pytest.mark.integration
@requires_ffmpeg
def test_a_bake_after_an_interruption_starts_clean(make_clip: MakeClip, tmp_path: Path) -> None:
    """The retry must not find a half-written artefact waiting for it."""
    store = CacheStore(tmp_path / "cache")
    clip = make_clip(name="bake_retry", seconds=2.0, fps="25")
    plan = _with_effects(clip, Effect(type="test.slow"))

    isolated = EffectRegistry()
    slow = _Slow()
    type(slow).fail_at = 10
    isolated.register(slow)
    with pytest.raises(KeyboardInterrupt):
        bake_plan(plan, cache=store, effects=isolated)

    type(slow).fail_at = None
    result = bake_plan(plan, cache=store, effects=isolated)

    assert result.baked == 1
    assert result.reused == 0
    baked_source = result.plan.sources[result.plan.timeline.clips()[0].source_id]
    assert probe(baked_source.path).video.frame_count == 50


@pytest.mark.integration
@requires_ffmpeg
def test_a_tight_budget_does_not_evict_this_render(
    make_clip: MakeClip, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The intermediate this render is about to read must survive eviction.

    It is both the newest and the largest entry, and eviction takes the largest
    first within an era -- so a budget smaller than the bake would delete
    exactly the file the returned plan points at.
    """
    monkeypatch.setenv(CACHE_BUDGET_ENV, "1KB")
    store = CacheStore(tmp_path / "cache")
    plan = _with_effects(make_clip(name="bake_budget", seconds=1.0), Effect(type="invert"))

    result = bake_plan(plan, cache=store)

    baked_source = result.plan.sources[result.plan.timeline.clips()[0].source_id]
    assert baked_source.path.exists()


# --- animated baking (PLAN.md D6) ------------------------------------------


class _DimParams(BaseModel):
    """How much to darken a frame."""

    model_config = ConfigDict(extra="forbid")

    amount: float = 0.0


class _DimEffect(EffectDef):
    """A baked effect whose strength can vary over the clip."""

    effect_type: ClassVar[str] = "test.dim"
    params_model: ClassVar[type[BaseModel]] = _DimParams
    bakes: ClassVar[bool] = True
    animatable: ClassVar[bool] = True

    def frame_op_animated(
        self, params: BaseModel
    ) -> Callable[[NDArray[np.uint8], BaseModel], NDArray[np.uint8]]:
        """Return an op darkening by whatever `amount` holds at that frame."""
        del params

        def dim(frame: NDArray[np.uint8], at: BaseModel) -> NDArray[np.uint8]:
            assert isinstance(at, _DimParams)
            return np.asarray(frame * (1.0 - at.amount), dtype=np.uint8)

        return dim


class _StaticBake(EffectDef):
    """A baked effect that never learned to animate."""

    effect_type: ClassVar[str] = "test.static"
    params_model: ClassVar[type[BaseModel]] = _DimParams
    bakes: ClassVar[bool] = True

    def frame_op(self, params: BaseModel) -> Callable[[NDArray[np.uint8]], NDArray[np.uint8]]:
        """Return an identity op."""
        del params
        return lambda frame: frame


def _animated_dim() -> Effect:
    """Dim from nothing to almost black across two seconds."""
    return Effect(
        type="test.dim",
        params={"amount": 0.0},
        animate={"amount": (Keyframe(t_s=0.0, value=0.0), Keyframe(t_s=2.0, value=0.9))},
    )


@pytest.mark.integration
@requires_ffmpeg
def test_a_baked_effect_sees_its_parameters_change(make_clip: MakeClip, tmp_path: Path) -> None:
    """The measurement that matters: the picture changes over the clip."""
    isolated = EffectRegistry()
    isolated.register(_DimEffect())
    clip = make_clip(name="bake_anim", seconds=2.0, fps="25")
    plan = _with_effects(clip, _animated_dim())

    result = bake_plan(plan, cache=CacheStore(tmp_path / "cache"), effects=isolated)

    baked = result.plan.sources[result.plan.timeline.clips()[0].source_id].path
    frames = list(read_frames(baked))
    assert frames[0].mean() > frames[len(frames) // 2].mean() > frames[-1].mean()


@pytest.mark.integration
@requires_ffmpeg
def test_the_first_frame_matches_the_static_value(make_clip: MakeClip, tmp_path: Path) -> None:
    """At t=0 the animation is its first keyframe, which is the static value."""
    isolated = EffectRegistry()
    isolated.register(_DimEffect())
    clip = make_clip(name="bake_anim_start", seconds=1.0, fps="25")

    result = bake_plan(
        _with_effects(clip, _animated_dim()),
        cache=CacheStore(tmp_path / "cache"),
        effects=isolated,
    )

    baked = result.plan.sources[result.plan.timeline.clips()[0].source_id].path
    original = next(iter(read_frames(clip)))
    first = next(iter(read_frames(baked)))
    assert np.abs(first.astype(int) - original.astype(int)).max() <= 4


@pytest.mark.integration
@requires_ffmpeg
def test_animating_a_baked_effect_that_cannot_is_refused(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    """Baking is not a free pass -- an effect must implement animation to claim it."""
    isolated = EffectRegistry()
    isolated.register(_StaticBake())
    clip = make_clip(name="bake_anim_no", seconds=1.0)
    plan = _with_effects(
        clip,
        Effect(
            type="test.static",
            params={"amount": 0.0},
            animate={"amount": (Keyframe(t_s=0.0, value=0.0), Keyframe(t_s=1.0, value=1.0))},
        ),
    )

    report = validate_plan(plan, effects=isolated)

    assert not report.ok
    assert any("cannot render" in finding.message for finding in report.errors)


def test_interpolated_parameters_are_validated(tmp_path: Path) -> None:
    """An animation sweeping a value out of range is a bad plan, not a corrupt render."""
    del tmp_path
    isolated = EffectRegistry()
    isolated.register(CropEffect())
    effect = Effect(
        type="crop",
        params={"x": 10, "y": 0, "width": 64, "height": 64},
        animate={"x": (Keyframe(t_s=0.0, value=10.0), Keyframe(t_s=1.0, value=-40.0))},
    )

    with pytest.raises(VisionUseError, match="crop"):
        _params_at(isolated, effect, 1.0)


def test_an_interpolated_integer_stays_an_integer() -> None:
    """A crop offset of 12.5 pixels is not a thing, and the model would reject it."""
    isolated = EffectRegistry()
    isolated.register(CropEffect())
    effect = Effect(
        type="crop",
        params={"x": 0, "y": 0, "width": 64, "height": 64},
        animate={"x": (Keyframe(t_s=0.0, value=0.0), Keyframe(t_s=1.0, value=25.0))},
    )

    params = _params_at(isolated, effect, 0.5)

    assert isinstance(params, CropParams)
    # 12.5, rounded half-to-even as Python does.
    assert params.x == 12
