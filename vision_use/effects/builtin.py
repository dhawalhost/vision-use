"""Built-in effects.

A deliberately small set: enough to exercise the registry, the plan model, and
the M2.4 compiler without pre-judging what skills will need. Each carries a
Pydantic parameter model, which is what the additive-only rule in ``PLAN.md``
§ D4 is checked against.

Adding a parameter here means adding an *optional* field with a default. A
change that is not additive means a new type -- ``crop`` becomes ``crop.v2`` --
and the old one stays registered and deprecated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from vision_use.effects.registry import EffectDef, registry
from vision_use.errors import VisionUseError


class CropParams(BaseModel):
    """Rectangular crop, in pixels, relative to the source frame."""

    model_config = ConfigDict(extra="forbid")

    x: Annotated[int, Field(ge=0)] = 0
    y: Annotated[int, Field(ge=0)] = 0
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]


class ScaleParams(BaseModel):
    """Resize to explicit dimensions."""

    model_config = ConfigDict(extra="forbid")

    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]


class FadeParams(BaseModel):
    """Fade the picture in from, or out to, black."""

    model_config = ConfigDict(extra="forbid")

    direction: Literal["in", "out"] = "in"
    duration_s: Annotated[float, Field(gt=0)] = 0.5
    start_s: Annotated[float, Field(ge=0)] = 0.0


class VolumeParams(BaseModel):
    """Scale audio level by a linear gain factor."""

    model_config = ConfigDict(extra="forbid")

    gain: Annotated[float, Field(ge=0)] = 1.0


class CropEffect(EffectDef):
    """Crop a clip to a rectangle."""

    effect_type: ClassVar[str] = "crop"
    params_model: ClassVar[type[BaseModel]] = CropParams
    # `crop` takes an expression for x and y, so a moving crop costs nothing
    # beyond the pass the render already makes -- no bake, no intermediate.
    animatable: ClassVar[bool] = True

    def compile_filter(self, params: BaseModel) -> str:
        """Return the ``crop`` filter for these parameters."""
        assert isinstance(params, CropParams)
        return f"crop={params.width}:{params.height}:{params.x}:{params.y}"

    def compile_animated_filter(self, params: BaseModel, animate: Mapping[str, str]) -> str:
        """Return the ``crop`` filter with animated offsets as expressions.

        Only ``x`` and ``y`` animate. Animating ``width`` or ``height`` would
        change the frame size mid-clip, which no encoder accepts and which the
        timeline's single canvas could not describe anyway -- so it is refused
        rather than silently ignored.

        Raises:
            VisionUseError: If a parameter that cannot animate is animated.

        """
        assert isinstance(params, CropParams)
        fixed = set(animate) - {"x", "y"}
        if fixed:
            raise VisionUseError(
                f"crop cannot animate {', '.join(sorted(fixed))}: the frame size is "
                "fixed for a clip's whole span",
                remedy="animate x or y, or split the clip and use two crops",
            )
        x = animate.get("x", str(params.x))
        y = animate.get("y", str(params.y))
        # Quoted because an expression contains commas, which otherwise end the
        # filter's argument list.
        return f"crop={params.width}:{params.height}:'{x}':'{y}'"


class ScaleEffect(EffectDef):
    """Scale a clip to explicit dimensions."""

    effect_type: ClassVar[str] = "scale"
    params_model: ClassVar[type[BaseModel]] = ScaleParams

    def compile_filter(self, params: BaseModel) -> str:
        """Return the ``scale`` filter for these parameters."""
        assert isinstance(params, ScaleParams)
        return f"scale={params.width}:{params.height}"


class FadeEffect(EffectDef):
    """Fade a clip in or out."""

    effect_type: ClassVar[str] = "fade"
    params_model: ClassVar[type[BaseModel]] = FadeParams

    def compile_filter(self, params: BaseModel) -> str:
        """Return the ``fade`` filter for these parameters.

        Times are relative to the start of the clip, not the timeline: the
        filter runs inside the clip's own chain, after ``setpts`` has rebased
        its timestamps to zero.
        """
        assert isinstance(params, FadeParams)
        return f"fade=t={params.direction}:st={params.start_s}:d={params.duration_s}"


class VolumeEffect(EffectDef):
    """Adjust a clip's audio level."""

    effect_type: ClassVar[str] = "volume"
    params_model: ClassVar[type[BaseModel]] = VolumeParams
    applies_to: ClassVar[str] = "audio"

    def compile_filter(self, params: BaseModel) -> str:
        """Return the ``volume`` filter for these parameters."""
        assert isinstance(params, VolumeParams)
        return f"volume={params.gain}"


class InvertParams(BaseModel):
    """Invert the picture. Takes no parameters."""

    model_config = ConfigDict(extra="forbid")


class InvertEffect(EffectDef):
    """Invert a clip's colours, frame by frame.

    ffmpeg has a `negate` filter, so this is not the cheapest way to invert an
    image. It exists as the reference *baked* effect: the smallest thing that
    exercises the whole pre-render path end to end, so that stabilisation and
    object removal have a proven road to follow rather than a design sketch.
    """

    effect_type: ClassVar[str] = "invert"
    params_model: ClassVar[type[BaseModel]] = InvertParams
    bakes: ClassVar[bool] = True

    def frame_op(
        self,
        params: BaseModel,  # noqa: ARG002 - inversion takes no parameters
    ) -> Callable[[NDArray[np.uint8]], NDArray[np.uint8]]:
        """Return a callable inverting every channel of a frame."""

        def invert(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
            return np.asarray(255 - frame, dtype=np.uint8)

        return invert


#: The effects this repository ships. Named separately from the registry, which
#: also holds whatever plugins the installation happens to have -- the schema
#: guard covers what this repository is responsible for, not the environment.
BUILTIN_EFFECT_TYPES: tuple[str, ...] = (
    CropEffect.effect_type,
    ScaleEffect.effect_type,
    FadeEffect.effect_type,
    VolumeEffect.effect_type,
    InvertEffect.effect_type,
)


def register_builtins() -> None:
    """Register the built-in effects on the default registry.

    Idempotent, so importing this module twice is harmless.
    """
    for definition in (
        CropEffect(),
        ScaleEffect(),
        FadeEffect(),
        VolumeEffect(),
        InvertEffect(),
    ):
        registry.register(definition, replace=True)


register_builtins()
