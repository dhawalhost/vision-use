"""Skills: the unit of capability in VisionUse.

A skill reads the shared ``VideoContext``, decides what should change, and
returns a ``PlanPatch`` against an ``EditPlan``. It never touches pixels --
that is the renderer's job, and it happens once, at export.

This is where the two halves of the framework meet: analysis produced a context
with stable IDs, and editing consumes a plan that describes what to do with
them.
"""

# Imported for its side effect: it populates the default registry. Placed after
# the imports above so the modules it depends on are already loaded.
from vision_use.skills import builtin as _builtin  # noqa: F401
from vision_use.skills.patch import (
    AddClip,
    AddSource,
    AddTrack,
    PatchResult,
    PlanPatch,
    PushEffect,
    RemoveClip,
    RemoveSource,
    RemoveTrack,
    SetClipEnabled,
    SetClipRange,
    SetEffects,
    SetTransition,
    SkillRef,
    apply_patch,
)
from vision_use.skills.pipeline import (
    Attempt,
    RetryPolicy,
    SkillOutcome,
    resolvable,
    resolve_context,
    run_skill,
)
from vision_use.skills.registry import SkillRegistry, default_registry, registry
from vision_use.skills.skill import (
    Analysis,
    Measurement,
    Score,
    Skill,
    SkillMeta,
    SkillRequest,
    Strategy,
    ValidationReport,
)

__all__ = [
    "AddClip",
    "AddSource",
    "AddTrack",
    "Analysis",
    "Attempt",
    "Measurement",
    "PatchResult",
    "PlanPatch",
    "PushEffect",
    "RemoveClip",
    "RemoveSource",
    "RemoveTrack",
    "RetryPolicy",
    "Score",
    "SetCanvas",
    "SetClipEnabled",
    "SetClipRange",
    "SetEffects",
    "SetTransition",
    "Skill",
    "SkillMeta",
    "SkillOutcome",
    "SkillRef",
    "SkillRegistry",
    "SkillRequest",
    "Strategy",
    "ValidationReport",
    "apply_patch",
    "default_registry",
    "registry",
    "resolvable",
    "resolve_context",
    "run_skill",
]
