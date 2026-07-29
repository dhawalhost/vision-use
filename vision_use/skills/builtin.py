"""Built-in skills.

Currently one: a reference implementation that exercises every stage of the
lifecycle without touching media. It exists so that the pipeline can be tested
end to end without ffmpeg, and so a plugin author has a complete, working skill
to read before writing their own.

The real skills arrive in M4: `silence-removal`, `reframe`, `captions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from vision_use.errors import VisionUseError
from vision_use.models.context import ContextSection, VideoContext
from vision_use.models.plan import EditPlan
from vision_use.skills.captions import Captions
from vision_use.skills.patch import PlanPatch, SetClipEnabled
from vision_use.skills.reframe import Reframe
from vision_use.skills.registry import SkillRegistry, registry
from vision_use.skills.silence import RemoveSilence
from vision_use.skills.skill import (
    Analysis,
    Measurement,
    Score,
    SkillMeta,
    SkillRequest,
    Strategy,
    ValidationReport,
)


@dataclass(frozen=True, slots=True)
class _Cut:
    """What `execute` needs to carry out one threshold strategy."""

    threshold: float
    durations: dict[str, float]


class DisableEmptyClips:
    """Disables clips whose source range is shorter than a threshold.

    A deliberately trivial skill: it needs no media, no model, and no new
    dependency, so it can prove the lifecycle wiring without dragging anything
    else in. It is still a *real* edit -- runt clips are a genuine artefact of
    automated cutting -- so it is not a stub either.
    """

    meta: ClassVar[SkillMeta] = SkillMeta(
        name="edit.disable_runts",
        version="1",
        summary="Disable clips too short to be worth showing.",
        requires=frozenset({ContextSection.PROBE}),
    )

    #: Thresholds offered as successive strategies, strictest first. Offering
    #: several is what gives the retry loop somewhere to go.
    THRESHOLDS: ClassVar[tuple[float, ...]] = (0.5, 0.25, 0.1)

    def analyse(self, context: VideoContext, plan: EditPlan, request: SkillRequest) -> Analysis:
        """Record every clip's duration."""
        del context, request
        durations = {clip.id: clip.duration_s for clip in plan.timeline.clips()}
        return Analysis(
            findings={"clips": len(durations)},
            detail=durations,
        )

    def plan(self, analysis: Analysis) -> list[Strategy]:
        """Offer one strategy per threshold, strictest first."""
        if not isinstance(analysis.detail, dict) or not analysis.detail:
            return []
        durations: dict[str, float] = analysis.detail
        return [
            Strategy(
                name=f"under-{threshold}s",
                rationale=f"disable clips shorter than {threshold}s",
                detail=_Cut(threshold=threshold, durations=durations),
            )
            for threshold in self.THRESHOLDS
        ]

    def execute(self, strategy: Strategy) -> PlanPatch:
        """Disable every clip below this strategy's threshold."""
        cut = strategy.detail
        if not isinstance(cut, _Cut):
            raise VisionUseError(
                "This strategy did not come from this skill",
                remedy="pass a strategy returned by plan()",
            )
        threshold, durations = cut.threshold, cut.durations
        ops = tuple(
            SetClipEnabled(clip_id=clip_id, enabled=False)
            for clip_id, duration in sorted(durations.items())
            if duration < threshold
        )
        return PlanPatch(
            ops=ops,
            rationale=strategy.rationale,
            produced_by=self.meta.ref,
        )

    def validate(
        self, before: EditPlan, after: EditPlan, context: VideoContext
    ) -> ValidationReport:
        """Check the edit left something to render and did not remove anything."""
        del context
        remaining = sum(1 for clip in after.timeline.clips() if clip.enabled)
        return ValidationReport(
            measurements=(
                Measurement(
                    name="clips_remaining",
                    value=float(remaining),
                    passed=remaining > 0,
                    detail="the render would be empty" if not remaining else "",
                ),
                Measurement(
                    name="clips_preserved",
                    value=float(len(after.timeline.clips())),
                    passed=len(after.timeline.clips()) == len(before.timeline.clips()),
                    detail="disabling must not remove clips",
                ),
            )
        )

    def score(self, report: ValidationReport) -> Score:
        """Prefer the attempt that leaves the most usable clips."""
        remaining = next(
            (item.value for item in report.measurements if item.name == "clips_remaining"),
            0.0,
        )
        return Score(value=remaining if report.passed else -1.0)


def register_builtins(target: SkillRegistry | None = None) -> None:
    """Register the built-in skills on ``target``, or the default registry."""
    target = registry if target is None else target
    target.register(DisableEmptyClips(), replace=True)
    target.register(RemoveSilence(), replace=True)
    target.register(Captions(), replace=True)
    target.register(Reframe(), replace=True)


register_builtins()
