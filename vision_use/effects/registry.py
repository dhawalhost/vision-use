"""Effect definitions, registration, and schema compatibility.

An *effect* is a named, parameterised transformation a clip can carry. The
registry is what turns the loosely-typed ``Effect`` entries in an ``EditPlan``
into validated parameters and, in M2.4, filtergraph fragments.

This module also enforces ``PLAN.md`` § D4, which is the reason it looks the way
it does:

**Schemas are additive-only. The type is the version.** An effect's parameter
schema may only gain optional fields with defaults. Anything else -- removing a
field, renaming one, changing a type, making an optional field required -- is a
*new effect type* (``crop`` becomes ``crop.v2``), with the old one kept and
marked deprecated.

The consequence is that there is no version field to resolve and no migration
code. A stored plan naming ``crop`` loads against ``crop``, whose schema can by
construction still read it, and the plan is never rewritten on load.

Why this rather than migrations: by Principle 3 most effects will be
third-party, and any scheme whose correctness depends on plugin authors writing
migrations correctly fails silently and corrupts stored plans. Additive-only is
mechanically checkable, and :func:`breaking_changes` is the check.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ValidationError

from vision_use.errors import ConfigurationError, VisionUseError
from vision_use.hashing import canonical_json, hash_params
from vision_use.log import bind, get_logger
from vision_use.models.plan import Effect

log = get_logger(__name__)


class EffectDef:
    """Definition of one effect type. Subclass this to add an effect.

    A base class rather than a Protocol so that the two deprecation fields have
    defaults: an author declaring a new effect should have to write ``type`` and
    ``params_model`` and nothing else.

    Attributes:
        effect_type: Registry key, matching the value stored in ``Effect.type``.
        params_model: Pydantic model validating this effect's parameters. Its
            JSON schema is what the additive-only rule is checked against.
        version: Bumped when the effect's *output* changes for identical
            parameters. Mixed into the bake cache key, so a corrected
            implementation does not keep serving the artefact the old one
            produced. Parameters and their schema are already in that key;
            this covers the code, which nothing else can see.
        deprecated_since: Version in which this type was superseded, if it was.
        superseded_by: The type that replaced it, so a warning can point
            somewhere useful rather than just complaining.

    """

    # Named `effect_type` rather than `type`: an attribute called `type` in a
    # class body shadows the builtin, so the annotation on the line below would
    # resolve to this string instead of `builtins.type`. The plan model's field
    # is still called `type` -- only the definition class differs.
    effect_type: ClassVar[str] = ""
    # Annotated without a default: assigning one here makes a type checker read
    # the attribute as a type alias rather than a class variable. Subclasses
    # repeat the annotation for the same reason.
    params_model: ClassVar[type[BaseModel]]
    version: ClassVar[str] = "1"
    deprecated_since: ClassVar[str | None] = None
    superseded_by: ClassVar[str | None] = None

    #: Which track kind this effect applies to. The compiler uses it to keep
    #: an audio filter out of a video chain, where ffmpeg would reject it with
    #: an error that does not name the effect.
    applies_to: ClassVar[str] = "video"

    #: Whether this effect can render animated parameters at all. A filter
    #: effect declaring this implements `compile_animated_filter`; a baked one
    #: implements `frame_op_animated`. Opt-in rather than assumed, because an
    #: effect that ignored animation would render a plan that says the crop
    #: moves as one that does not -- silently wrong, which validation refuses.
    animatable: ClassVar[bool] = False

    #: Whether this effect must be *baked* -- rendered to an intermediate file
    #: before the main graph runs -- because it cannot be expressed as a filter.
    #: Stabilisation, object removal and upscaling are all in this category:
    #: each needs state across frames, or a model, or both.
    #:
    #: A baked effect is still an ordinary entry in a clip's effect stack. Only
    #: its compilation differs, which is what keeps the plan model free of
    #: special cases (``PLAN.md`` § Effects that ffmpeg cannot express).
    bakes: ClassVar[bool] = False

    def frame_op(self, params: BaseModel) -> Callable[[NDArray[np.uint8]], NDArray[np.uint8]]:
        """Return a per-frame transformation for a baked effect.

        Called once; the returned callable is then applied to every frame of
        the clip's span, in order. Returning a closure rather than a method
        lets an implementation carry state -- a motion model, a warm network --
        across the frames of one bake without leaking it between clips.

        Raises:
            NotImplementedError: If the effect does not bake.

        """
        raise NotImplementedError(f"effect {self.effect_type!r} does not implement frame_op()")

    def frame_op_animated(
        self, params: BaseModel
    ) -> Callable[[NDArray[np.uint8], BaseModel], NDArray[np.uint8]]:
        """Return a per-frame transformation that also receives its parameters.

        The bake loop interpolates this effect's animated parameters at each
        frame's time and passes the result as the second argument, so an
        implementation never touches keyframes or timestamps itself.

        Like :meth:`frame_op`, the returned callable is created once and may
        carry state across the clip's frames; unlike it, the parameters can
        differ from frame to frame.

        Args:
            params: The static parameters, as a starting point.

        Raises:
            NotImplementedError: If the effect does not animate while baking.

        """
        raise NotImplementedError(
            f"effect {self.effect_type!r} does not implement frame_op_animated()"
        )

    def compile_animated_filter(self, params: BaseModel, animate: Mapping[str, str]) -> str:
        """Return this effect's filter with animated parameters as expressions.

        Args:
            params: The validated parameters, holding each value's static form.
            animate: Animated parameter names mapped to ffmpeg time
                expressions, already built from the plan's keyframes.

        Returns:
            One filtergraph link, as :meth:`compile_filter` returns.

        Raises:
            NotImplementedError: If the effect does not declare ``animatable``.

        """
        raise NotImplementedError(
            f"effect {self.effect_type!r} does not render animation as a filter"
        )

    def compile_filter(self, params: BaseModel) -> str:
        """Return this effect's ffmpeg filter fragment.

        Args:
            params: The validated parameter model, as returned by
                :meth:`EffectRegistry.validate`.

        Returns:
            One filtergraph link, e.g. ``"crop=100:50:0:0"``. Chained with
            commas into the clip's filter chain by the compiler.

        Raises:
            NotImplementedError: If the effect has not implemented compilation.

        """
        raise NotImplementedError(
            f"effect {self.effect_type!r} does not implement compile_filter()"
        )


def schema_of(definition: EffectDef) -> dict[str, Any]:
    """Return the JSON schema of an effect's parameters."""
    return dict(definition.params_model.model_json_schema())


def schema_hash(definition: EffectDef) -> str:
    """Return a stable digest of an effect's parameter schema.

    Stored on ``Effect.schema_hash`` and always mixed into the render cache key,
    so an additive change that alters output cannot serve a stale artefact.
    """
    return hash_params({"type": definition.effect_type, "schema": schema_of(definition)})


def _required(schema: Mapping[str, Any]) -> set[str]:
    """Return the required property names declared by ``schema``."""
    required = schema.get("required", [])
    return set(required) if isinstance(required, list) else set()


def _properties(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the property map declared by ``schema``."""
    properties = schema.get("properties", {})
    return dict(properties) if isinstance(properties, dict) else {}


def breaking_changes(golden: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Return the ways ``current`` breaks compatibility with ``golden``.

    A change is breaking when a plan valid against ``golden`` would no longer be
    valid, or would mean something different, against ``current``:

    * a property was removed -- stored plans still carry it
    * a property changed type -- the stored value no longer fits
    * a property became required -- plans written without it stop loading

    Adding an optional property is not breaking, which is exactly the latitude
    the additive-only rule grants.

    Returns:
        Human-readable descriptions, empty when the change is additive.

    """
    problems: list[str] = []

    old_props, new_props = _properties(golden), _properties(current)
    for name, old in old_props.items():
        new = new_props.get(name)
        if new is None:
            problems.append(f"property {name!r} was removed")
            continue
        if old.get("type") != new.get("type"):
            problems.append(
                f"property {name!r} changed type from {old.get('type')!r} "
                f"to {new.get('type')!r}"
            )

    newly_required = _required(current) - _required(golden)
    problems.extend(f"property {name!r} became required" for name in sorted(newly_required))
    return problems


class EffectRegistry:
    """Effect definitions, addressable by type."""

    def __init__(self) -> None:
        self._by_type: dict[str, EffectDef] = {}

    def register(self, definition: EffectDef, *, replace: bool = False) -> None:
        """Add ``definition`` to the registry.

        Raises:
            ConfigurationError: If the type is empty, already registered, or
                claims to be superseded by a type that is not registered.

        """
        if not definition.effect_type:
            raise ConfigurationError(
                "An effect must declare a non-empty type",
                remedy="set the 'type' class attribute",
            )
        if definition.effect_type in self._by_type and not replace:
            raise ConfigurationError(
                f"An effect of type {definition.effect_type!r} is already registered",
                remedy="rename one, or pass replace=True to override",
            )

        self._by_type[definition.effect_type] = definition
        log.debug(
            "registered effect",
            extra=bind(
                type=definition.effect_type, deprecated=bool(definition.deprecated_since)
            ),
        )

    def get(self, effect_type: str) -> EffectDef:
        """Return the definition for ``effect_type``.

        Raises:
            ConfigurationError: If nothing is registered under that type.

        """
        try:
            return self._by_type[effect_type]
        except KeyError:
            known = ", ".join(sorted(self._by_type)) or "none"
            raise ConfigurationError(
                f"No effect registered as {effect_type!r}",
                remedy=f"registered effects: {known}",
            ) from None

    def validate(self, effect: Effect) -> BaseModel:
        """Validate ``effect``'s parameters against its declared schema.

        A deprecated effect still validates -- old plans must keep working, that
        being the whole point of D4 -- but logs a warning naming its
        replacement.

        Returns:
            The parsed parameter model.

        Raises:
            ConfigurationError: If the effect type is unknown.
            VisionUseError: If the parameters do not satisfy the schema.

        """
        definition = self.get(effect.type)

        if definition.deprecated_since:
            log.warning(
                "effect is deprecated",
                extra=bind(
                    type=effect.type,
                    since=definition.deprecated_since,
                    use=definition.superseded_by or "no replacement declared",
                ),
            )

        try:
            parsed: BaseModel = definition.params_model.model_validate(effect.params)
        except ValidationError as exc:
            raise VisionUseError(
                f"Invalid parameters for effect {effect.type!r}: "
                f"{exc.error_count()} problem(s)",
                remedy=f"check the schema: {canonical_json(schema_of(definition))[:200]}",
            ) from exc
        return parsed

    def bind_schema_hash(self, effect: Effect) -> Effect:
        """Return ``effect`` with its ``schema_hash`` filled in.

        Called when an effect is added to a plan, so the plan records the schema
        shape its parameters were written against.
        """
        return effect.model_copy(update={"schema_hash": schema_hash(self.get(effect.type))})

    def schemas(self) -> dict[str, dict[str, Any]]:
        """Return every registered effect's schema, keyed by type.

        This is what the golden snapshot stores and compares against.
        """
        return {name: schema_of(self._by_type[name]) for name in sorted(self._by_type)}

    def deprecated(self) -> dict[str, str | None]:
        """Return deprecated types mapped to their replacements."""
        return {
            name: definition.superseded_by
            for name, definition in sorted(self._by_type.items())
            if definition.deprecated_since
        }

    def __len__(self) -> int:
        """Return how many effects are registered."""
        return len(self._by_type)

    def __contains__(self, effect_type: object) -> bool:
        """Whether an effect is registered under this type."""
        return isinstance(effect_type, str) and effect_type in self._by_type


#: Process-wide default registry. Built-in effects register themselves on
#: import; consumers needing isolation construct their own.
registry = EffectRegistry()
