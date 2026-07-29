"""Running a skill through its lifecycle.

This is the loop ``VISION.md`` describes:

    analyse → plan → execute → validate → retry → return the best result

with the two things that sketch left out, both of which matter more than they
look:

**A budget.** An auto-retry loop over video without a cap is a runaway-compute
bug wearing a feature's clothes. :class:`RetryPolicy` bounds both the number of
attempts and the wall time, and the pipeline stops when either runs out.

**Rollback.** A rejected strategy is undone exactly, using the inverse patch
that ``apply_patch`` already returns, so the next strategy starts from the plan
the skill was given rather than from wherever the last attempt left things.

When no strategy passes validation, the pipeline returns **the highest-scoring
attempt** rather than nothing (``VISION.md`` § Validation Philosophy). A caller
that wanted strictness can check :attr:`SkillOutcome.accepted`; a caller that
wanted a result gets the best one available, with the failures attached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from vision_use.cache.store import CacheStore
from vision_use.context.registry import AnalyzerRegistry
from vision_use.context.runner import analyze
from vision_use.errors import VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.context import ContextSection, SectionName, VideoContext
from vision_use.models.plan import EditPlan
from vision_use.skills.patch import PlanPatch, apply_patch
from vision_use.skills.skill import (
    Score,
    Skill,
    SkillRequest,
    Strategy,
    ValidationReport,
)

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounds on how hard the pipeline may try.

    Attributes:
        max_attempts: How many strategies to try. One means no retry.
        max_wall_seconds: Give up once this much time has passed, checked
            between attempts. A single very slow attempt is not interrupted --
            the pipeline does not own the skill's execution -- but no further
            one begins.

    """

    max_attempts: int = 3
    max_wall_seconds: float = 300.0

    def __post_init__(self) -> None:
        """Reject a policy that could never run an attempt."""
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if self.max_wall_seconds <= 0:
            raise ValueError(f"max_wall_seconds must be positive, got {self.max_wall_seconds}")


@dataclass(frozen=True, slots=True)
class Attempt:
    """One strategy, tried."""

    strategy: Strategy
    patch: PlanPatch
    plan: EditPlan
    inverse: PlanPatch
    report: ValidationReport
    score: Score

    @property
    def passed(self) -> bool:
        """Whether the skill's own validation accepted this attempt."""
        return self.report.passed


@dataclass(frozen=True, slots=True)
class SkillOutcome:
    """What running a skill produced.

    Attributes:
        plan: The resulting plan. The original when nothing was attempted.
        accepted: The attempt that passed validation, if any.
        attempts: Every attempt made, in order.
        exhausted: True when the budget ran out before a strategy passed.

    """

    plan: EditPlan
    accepted: Attempt | None = None
    attempts: tuple[Attempt, ...] = ()
    exhausted: bool = False

    @property
    def changed(self) -> bool:
        """Whether the plan was actually edited."""
        return self.accepted is not None or bool(self.attempts)

    @property
    def best(self) -> Attempt | None:
        """The highest-scoring attempt, passing or not."""
        if not self.attempts:
            return None
        return max(self.attempts, key=lambda attempt: attempt.score.value)


@dataclass(slots=True)
class _Budget:
    """Tracks what the policy has left."""

    policy: RetryPolicy
    started: float = field(default_factory=time.monotonic)
    used: int = 0

    def allows_another(self) -> bool:
        """Whether one more attempt is permitted."""
        if self.used >= self.policy.max_attempts:
            return False
        return time.monotonic() - self.started < self.policy.max_wall_seconds

    @property
    def elapsed_s(self) -> float:
        """How long the pipeline has been running."""
        return time.monotonic() - self.started


def resolve_context(
    skill: Skill,
    source: Path,
    *,
    context: VideoContext | None = None,
    cache: CacheStore | None = None,
    analyzers: AnalyzerRegistry | None = None,
) -> VideoContext:
    """Return a context satisfying ``skill.meta.requires``.

    A skill declares what it reads and the runtime backfills it -- once, for
    everyone. This is the mechanism behind Principle 8: a skill that quietly ran
    its own decoder because the context lacked something would turn a gap in the
    context engine into duplicated work for every other skill.

    An existing context is reused when it already carries the required sections,
    and topped up otherwise.

    Args:
        skill: The skill whose requirements to satisfy.
        source: Media file to analyse.
        context: A context to reuse, if one is already to hand.
        cache: Store for analyzer results.
        analyzers: Analyzer registry.

    Returns:
        A context with every required section populated.

    Raises:
        VisionUseError: If a required section has no registered producer.

    """
    wanted = sorted(skill.meta.requires, key=str)
    if context is not None and not context.missing(wanted):
        return context

    missing = context.missing(wanted) if context is not None else wanted
    if missing:
        log.info(
            "backfilling context for skill",
            extra=bind(
                skill=skill.meta.name,
                sections=[str(section) for section in missing],
            ),
        )
    return analyze(source, wanted, cache=cache, registry=analyzers)


def run_skill(
    skill: Skill,
    source: Path,
    plan: EditPlan,
    request: SkillRequest | None = None,
    *,
    context: VideoContext | None = None,
    cache: CacheStore | None = None,
    analyzers: AnalyzerRegistry | None = None,
    policy: RetryPolicy | None = None,
) -> SkillOutcome:
    """Run ``skill`` against ``plan``, returning the best result it achieved.

    Strategies are tried in the order the skill offered them, best first. Each
    attempt applies its patch to the *original* plan, so a rejected strategy
    cannot contaminate the next one.

    Args:
        skill: The skill to run.
        source: Media the plan draws from, used to backfill context.
        plan: The plan to edit.
        request: What the caller asked for.
        context: A context to reuse instead of analysing.
        cache: Store for analyzer results.
        analyzers: Analyzer registry.
        policy: Retry bounds. Defaults to three attempts in five minutes.

    Returns:
        The outcome, including every attempt made.

    Raises:
        VisionUseError: If context cannot be resolved, or the skill raises.

    """
    budget = _Budget(policy=policy or RetryPolicy())
    asked = request or SkillRequest()

    resolved = resolve_context(skill, source, context=context, cache=cache, analyzers=analyzers)
    analysis = skill.analyse(resolved, plan, asked)
    strategies = skill.plan(analysis)

    if not strategies:
        log.info("skill found nothing to do", extra=bind(skill=skill.meta.name))
        return SkillOutcome(plan=plan)

    attempts: list[Attempt] = []

    for strategy in strategies:
        if not budget.allows_another():
            log.warning(
                "retry budget exhausted",
                extra=bind(
                    skill=skill.meta.name,
                    attempts=budget.used,
                    elapsed_s=round(budget.elapsed_s, 2),
                ),
            )
            return SkillOutcome(
                plan=_best_plan(plan, attempts),
                attempts=tuple(attempts),
                exhausted=True,
            )

        budget.used += 1
        attempt = _try(skill, plan, resolved, strategy)
        if attempt is None:
            continue
        attempts.append(attempt)

        if attempt.passed:
            log.info(
                "skill accepted a strategy",
                extra=bind(
                    skill=skill.meta.name,
                    strategy=strategy.name,
                    attempt=budget.used,
                    score=attempt.score.value,
                ),
            )
            return SkillOutcome(plan=attempt.plan, accepted=attempt, attempts=tuple(attempts))

        log.info(
            "strategy rejected by its own validation",
            extra=bind(
                skill=skill.meta.name,
                strategy=strategy.name,
                failures=[item.name for item in attempt.report.failures],
            ),
        )

    return SkillOutcome(
        plan=_best_plan(plan, attempts),
        attempts=tuple(attempts),
        exhausted=bool(attempts),
    )


def _try(
    skill: Skill, plan: EditPlan, context: VideoContext, strategy: Strategy
) -> Attempt | None:
    """Execute one strategy and judge it, or return ``None`` if it cannot apply.

    A patch that will not apply is a bug in the skill, not a reason to abandon
    the run: the next strategy may well be fine, so it is logged and skipped.
    """
    patch = skill.execute(strategy)
    if patch.is_empty:
        log.debug(
            "strategy produced an empty patch",
            extra=bind(skill=skill.meta.name, strategy=strategy.name),
        )
        return None

    try:
        applied = apply_patch(plan, patch)
    except VisionUseError as exc:
        log.warning(
            "strategy produced a patch that will not apply",
            extra=bind(skill=skill.meta.name, strategy=strategy.name, detail=exc.message),
        )
        return None

    report = skill.validate(plan, applied.plan, context)
    return Attempt(
        strategy=strategy,
        patch=patch,
        plan=applied.plan,
        inverse=applied.inverse,
        report=report,
        score=skill.score(report),
    )


def _best_plan(original: EditPlan, attempts: list[Attempt]) -> EditPlan:
    """Return the highest-scoring attempt's plan, or the original if none.

    ``VISION.md`` asks the framework to "return the highest scoring result"
    rather than fail outright, so a caller that wanted *something* gets the best
    available. :attr:`SkillOutcome.accepted` is how a caller that wanted
    strictness tells the difference.
    """
    if not attempts:
        return original
    return max(attempts, key=lambda attempt: attempt.score.value).plan


def resolvable(skill: Skill, analyzers: AnalyzerRegistry) -> list[SectionName]:
    """Return the sections ``skill`` needs that nothing can produce.

    Lets a caller check up front whether a skill is usable at all, rather than
    discovering it after paying for an analysis.
    """
    return [
        section
        for section in sorted(skill.meta.requires, key=str)
        if section != ContextSection.PROBE and analyzers.producer_of(section) is None
    ]
