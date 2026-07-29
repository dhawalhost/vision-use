"""Tests for the effect registry and the D4 additive-only rule.

The golden-snapshot test at the bottom is the mechanical enforcement of
`PLAN.md` § D4. It is the reason plugin authors never have to write migrations,
and the reason a stored plan from a year ago still loads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field
from vision_use.effects.builtin import BUILTIN_EFFECT_TYPES, CropParams
from vision_use.effects.registry import (
    EffectDef,
    EffectRegistry,
    breaking_changes,
    registry,
    schema_hash,
    schema_of,
)
from vision_use.errors import ConfigurationError, VisionUseError
from vision_use.models.plan import Effect

GOLDEN_PATH = Path(__file__).parent / "schemas.golden.json"


class SampleParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(default=1.0, ge=0)


class SampleEffect(EffectDef):
    """A registrable effect for wiring tests."""

    effect_type: ClassVar[str] = "sample"
    params_model: ClassVar[type[BaseModel]] = SampleParams


class DeprecatedEffect(EffectDef):
    """An effect kept alive so old plans keep loading."""

    effect_type: ClassVar[str] = "sample.old"
    params_model: ClassVar[type[BaseModel]] = SampleParams
    deprecated_since: ClassVar[str | None] = "0.2.0"
    superseded_by: ClassVar[str | None] = "sample"


# --- registration ----------------------------------------------------------


def test_register_and_retrieve() -> None:
    reg = EffectRegistry()
    reg.register(SampleEffect())

    assert reg.get("sample").effect_type == "sample"
    assert "sample" in reg
    assert len(reg) == 1


def test_duplicate_registration_rejected() -> None:
    reg = EffectRegistry()
    reg.register(SampleEffect())

    with pytest.raises(ConfigurationError, match="already registered"):
        reg.register(SampleEffect())


def test_duplicate_allowed_with_replace() -> None:
    reg = EffectRegistry()
    reg.register(SampleEffect())
    reg.register(SampleEffect(), replace=True)

    assert len(reg) == 1


def test_empty_type_rejected() -> None:
    class Nameless(EffectDef):
        params_model: ClassVar[type[BaseModel]] = SampleParams

    with pytest.raises(ConfigurationError, match="non-empty type"):
        EffectRegistry().register(Nameless())


def test_unknown_type_lists_what_is_registered() -> None:
    reg = EffectRegistry()
    reg.register(SampleEffect())

    with pytest.raises(ConfigurationError) as excinfo:
        reg.get("absent")

    assert "sample" in (excinfo.value.remedy or "")


def test_builtins_are_registered() -> None:
    for name in ("crop", "scale", "fade", "volume"):
        assert name in registry


# --- validation ------------------------------------------------------------


def test_valid_params_parse() -> None:
    parsed = registry.validate(Effect(type="crop", params={"width": 100, "height": 50}))

    assert isinstance(parsed, CropParams)
    assert parsed.width == 100


def test_defaults_are_applied() -> None:
    parsed = registry.validate(Effect(type="crop", params={"width": 10, "height": 10}))

    assert isinstance(parsed, CropParams)
    assert parsed.x == 0


def test_missing_required_param_rejected() -> None:
    with pytest.raises(VisionUseError, match="Invalid parameters"):
        registry.validate(Effect(type="crop", params={"width": 100}))


def test_out_of_range_param_rejected() -> None:
    with pytest.raises(VisionUseError):
        registry.validate(Effect(type="crop", params={"width": 0, "height": 10}))


def test_unknown_param_rejected() -> None:
    """extra='forbid' turns a typo into an error instead of a silent no-op."""
    with pytest.raises(VisionUseError):
        registry.validate(Effect(type="crop", params={"width": 1, "height": 1, "wdith": 2}))


def test_validation_error_points_at_the_schema() -> None:
    with pytest.raises(VisionUseError) as excinfo:
        registry.validate(Effect(type="crop", params={}))

    assert "width" in (excinfo.value.remedy or "")


def test_unknown_effect_type_rejected() -> None:
    with pytest.raises(ConfigurationError):
        registry.validate(Effect(type="nope.not.registered"))


def test_deprecated_effect_still_validates() -> None:
    """Old plans must keep working -- that is the entire point of D4."""
    reg = EffectRegistry()
    reg.register(DeprecatedEffect())

    assert reg.validate(Effect(type="sample.old", params={"amount": 2.0}))


def test_deprecated_effects_are_listed_with_replacements() -> None:
    reg = EffectRegistry()
    reg.register(SampleEffect())
    reg.register(DeprecatedEffect())

    assert reg.deprecated() == {"sample.old": "sample"}


# --- schema hashing --------------------------------------------------------


def test_schema_hash_is_stable() -> None:
    assert schema_hash(SampleEffect()) == schema_hash(SampleEffect())


def test_schema_hash_differs_between_effects() -> None:
    assert schema_hash(SampleEffect()) != schema_hash(DeprecatedEffect())


def test_bind_schema_hash_stamps_the_effect() -> None:
    """A plan records the schema shape its params were written against."""
    stamped = registry.bind_schema_hash(Effect(type="crop", params={"width": 10, "height": 10}))

    assert stamped.schema_hash is not None
    assert len(stamped.schema_hash) == 64


# --- the additive-only rule ------------------------------------------------


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"properties": properties, "required": required or []}


def test_identical_schemas_are_compatible() -> None:
    schema = _schema({"a": {"type": "integer"}})

    assert breaking_changes(schema, schema) == []


def test_adding_an_optional_property_is_allowed() -> None:
    """This is the one kind of change D4 permits in place."""
    before = _schema({"a": {"type": "integer"}})
    after = _schema({"a": {"type": "integer"}, "b": {"type": "string"}})

    assert breaking_changes(before, after) == []


def test_removing_a_property_is_breaking() -> None:
    before = _schema({"a": {"type": "integer"}, "b": {"type": "string"}})
    after = _schema({"a": {"type": "integer"}})

    assert breaking_changes(before, after) == ["property 'b' was removed"]


def test_changing_a_type_is_breaking() -> None:
    before = _schema({"a": {"type": "integer"}})
    after = _schema({"a": {"type": "string"}})

    (problem,) = breaking_changes(before, after)
    assert "changed type" in problem


def test_making_a_property_required_is_breaking() -> None:
    """Plans written without it would stop loading."""
    before = _schema({"a": {"type": "integer"}})
    after = _schema({"a": {"type": "integer"}}, required=["a"])

    assert breaking_changes(before, after) == ["property 'a' became required"]


def test_relaxing_a_requirement_is_not_breaking() -> None:
    before = _schema({"a": {"type": "integer"}}, required=["a"])
    after = _schema({"a": {"type": "integer"}})

    assert breaking_changes(before, after) == []


def test_multiple_problems_are_all_reported() -> None:
    before = _schema({"a": {"type": "integer"}, "b": {"type": "string"}})
    after = _schema({"a": {"type": "string"}}, required=["a"])

    assert len(breaking_changes(before, after)) == 3


# --- golden snapshot -------------------------------------------------------


def test_golden_snapshot_exists() -> None:
    assert GOLDEN_PATH.exists(), (
        "tests/schemas.golden.json is missing; regenerate it with "
        '`python -c "import json,pathlib;'
        "from vision_use.effects import registry;"
        "pathlib.Path('tests/schemas.golden.json').write_text("
        'json.dumps(registry.schemas(), indent=2, sort_keys=True)+chr(10))"`'
    )


def test_no_effect_schema_changed_incompatibly() -> None:
    """The D4 enforcement. A non-additive schema change fails CI here.

    If this fails, the fix is almost never to update the snapshot. It is to
    introduce a new effect type -- `crop` becomes `crop.v2` -- and leave the old
    one registered and deprecated, so plans written against it keep loading.
    Updating the snapshot instead silently breaks every stored plan.
    """
    golden = json.loads(GOLDEN_PATH.read_text())
    current = registry.schemas()

    problems: list[str] = []
    for effect_type, old in golden.items():
        new = current.get(effect_type)
        if new is None:
            problems.append(f"{effect_type}: the effect type was removed entirely")
            continue
        problems.extend(f"{effect_type}: {issue}" for issue in breaking_changes(old, new))

    assert not problems, "non-additive effect schema change:\n  " + "\n  ".join(problems)


def test_new_effects_do_not_need_a_snapshot_update_to_pass() -> None:
    """Adding an effect type is additive; only changing an existing one is not."""
    golden = json.loads(GOLDEN_PATH.read_text())
    extended = {**golden, "brand.new": _schema({"z": {"type": "integer"}})}

    problems = [
        issue
        for effect_type, old in golden.items()
        for issue in breaking_changes(old, extended[effect_type])
    ]

    assert problems == []


def test_snapshot_covers_every_builtin_effect() -> None:
    """A new built-in must be added to the snapshot, or it is never guarded.

    Built-ins, not everything registered: a developer with a plugin installed
    would otherwise fail a test about this repository's own schemas.
    """
    golden = json.loads(GOLDEN_PATH.read_text())

    assert set(BUILTIN_EFFECT_TYPES) == set(golden)
    assert set(golden) <= set(registry.schemas())


def test_schema_of_returns_a_json_schema() -> None:
    schema = schema_of(SampleEffect())

    assert "properties" in schema
    assert "amount" in schema["properties"]
