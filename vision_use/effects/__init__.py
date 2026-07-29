"""Effect definitions and the registry that validates them.

Importing this package registers the built-in effects. The registry is also
where ``PLAN.md`` § D4 is enforced: effect parameter schemas are additive-only,
and :func:`~vision_use.effects.registry.breaking_changes` is the mechanical
check that keeps them that way.
"""

# Imported for its side effect: it populates the default registry. Placed after
# the imports above so the modules it depends on are already loaded.
from vision_use.effects import builtin as _builtin  # noqa: F401
from vision_use.effects.registry import (
    EffectDef,
    EffectRegistry,
    breaking_changes,
    registry,
    schema_hash,
    schema_of,
)

__all__ = [
    "EffectDef",
    "EffectRegistry",
    "breaking_changes",
    "registry",
    "schema_hash",
    "schema_of",
]
