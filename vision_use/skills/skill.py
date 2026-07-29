"""The skill contract.

A skill is the unit of capability. Everything VisionUse can do to a video is a
skill, and adding one requires no change to the runtime.

The lifecycle is the one in ``SKILL_SPEC.md``:

``analyse`` → ``plan`` → ``execute`` → ``validate`` → ``score``

with one change from the original ``VISION.md`` sketch, and it is the change the
whole framework is built around: **``execute`` returns a patch, not a video.**

Two rules carry most of the weight:

**A skill declares what context it needs.** ``SkillMeta.requires`` names the
sections it reads, and the runtime backfills them once, for everyone. A skill
that quietly runs its own decoder because the context lacks something has turned
a gap in the context engine into duplicated work for every other skill --
exactly what Principle 8 exists to prevent.

**A skill offers several strategies.** ``plan`` returns candidates in order of
expected quality, so a rejected first attempt has somewhere to go. Retrying the
same computation and hoping is not a retry loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from vision_use.models.context import SectionName, VideoContext
from vision_use.models.plan import EditPlan
from vision_use.skills.patch import PlanPatch, SkillRef


class SkillRequest(BaseModel):
    """What the caller asked for.

    ``prompt`` is the user's own words, kept verbatim: a skill may use it to
    choose a strategy, and it is what the rationale on the resulting patch is
    ultimately answering.
    """

    model_config = ConfigDict(frozen=True)

    prompt: str = ""
    params: dict[str, JsonValue] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SkillMeta:
    """Everything the runtime needs to know about a skill without running it.

    Attributes:
        name: Dotted identifier, e.g. ``"edit.silence_removal"``.
        version: Bumped when the skill's output changes for identical input.
        summary: One line, shown in ``vision skills`` and to an agent choosing
            between skills.
        requires: Context sections the skill reads. The runtime guarantees
            these are populated before ``analyse`` is called. A plugin's own
            dotted section name (``"stabilize.motion"``) is as valid here as a
            built-in one, and is backfilled the same way (PLAN.md § D5).
        produces: Sections the skill's patch will affect. Advisory today;
            it is what a future scheduler would use to order skills.

    """

    name: str
    version: str
    summary: str = ""
    requires: frozenset[SectionName] = field(default_factory=frozenset)
    produces: frozenset[SectionName] = field(default_factory=frozenset)

    @property
    def ref(self) -> SkillRef:
        """Return the skill's identity, for stamping onto a patch."""
        return SkillRef(name=self.name, version=self.version)


@dataclass(frozen=True, slots=True)
class Analysis:
    """What a skill learned from the context and the current plan.

    Deliberately open: what a skill needs to carry from ``analyse`` to ``plan``
    is its own business. ``findings`` exists so the runtime can log something
    useful without understanding it.
    """

    findings: dict[str, JsonValue] = field(default_factory=dict)
    detail: object | None = None


@dataclass(frozen=True, slots=True)
class Strategy:
    """One candidate approach, with a name the user could recognise.

    Attributes:
        name: Short identifier, e.g. ``"aggressive"``.
        rationale: Why this approach, in a sentence.
        detail: Whatever ``execute`` needs to carry it out.

    """

    name: str
    rationale: str = ""
    detail: object | None = None


@dataclass(frozen=True, slots=True)
class Measurement:
    """One named check with a pass/fail verdict.

    A bare number cannot be judged by a runtime that does not know the skill;
    a verdict can, which is what lets the retry loop reject a strategy without
    understanding what it measured.
    """

    name: str
    value: float
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """A skill's judgement of its own work (Principle 7)."""

    measurements: tuple[Measurement, ...] = ()

    @property
    def passed(self) -> bool:
        """Whether every measurement passed."""
        return all(item.passed for item in self.measurements)

    @property
    def failures(self) -> tuple[Measurement, ...]:
        """The measurements that did not pass."""
        return tuple(item for item in self.measurements if not item.passed)


@dataclass(frozen=True, slots=True)
class Score:
    """A single comparable number for one attempt.

    Comparable *within* a skill, not across skills: the runtime uses it to pick
    a winner among that skill's strategies, and nothing else. Two skills'
    scores mean different things and are never compared.
    """

    value: float
    detail: str = ""


@runtime_checkable
class Skill(Protocol):
    """The contract every skill implements.

    Implementations are ordinary classes carrying a :class:`SkillMeta`. Nothing
    is registered by inheritance, so a plugin's skill is indistinguishable from
    a built-in one.
    """

    meta: ClassVar[SkillMeta]

    def analyse(self, context: VideoContext, plan: EditPlan, request: SkillRequest) -> Analysis:
        """Read the context and the current plan. Decide nothing yet.

        Read-only, and must not re-analyse the video: if something is missing,
        declare it in ``meta.requires`` instead.
        """
        ...

    def plan(self, analysis: Analysis) -> list[Strategy]:
        """Return candidate approaches, best first.

        Returning several is what makes the retry loop meaningful -- when
        validation rejects the first, the runtime has somewhere to go.
        Returning none means the skill has nothing to do, which is a legitimate
        outcome.
        """
        ...

    def execute(self, strategy: Strategy) -> PlanPatch:
        """Carry out one strategy and return the edit it implies.

        Pure with respect to media: it reads what ``analyse`` gathered and
        writes operations. It does not encode, and it does not write files
        outside the cache.
        """
        ...

    def validate(
        self, before: EditPlan, after: EditPlan, context: VideoContext
    ) -> ValidationReport:
        """Judge the skill's own work by comparing the plan before and after."""
        ...

    def score(self, report: ValidationReport) -> Score:
        """Collapse a report into one comparable number for this skill."""
        ...
