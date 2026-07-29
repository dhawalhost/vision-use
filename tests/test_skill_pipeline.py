"""Tests for the skill registry and the lifecycle pipeline.

The pipeline's job is to run a skill's own decisions and then hold it to its own
standard: try strategies in order, stop when validation passes, stay inside a
budget, and never let a rejected attempt contaminate the next one.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from typer.testing import CliRunner
from vision_use.cli.main import app
from vision_use.context.registry import AnalyzerRegistry
from vision_use.errors import ConfigurationError
from vision_use.models.context import ContextSection, VideoContext
from vision_use.models.ids import clip_id, source_id, track_id
from vision_use.models.media import (
    FrameRate,
    MediaProbe,
    SourceInfo,
    VideoStreamInfo,
)
from vision_use.models.plan import Clip, EditPlan, Source, Timeline, Track
from vision_use.planning import load_plan, save_plan, straight_through
from vision_use.skills import (
    Analysis,
    Measurement,
    PlanPatch,
    RetryPolicy,
    Score,
    SetClipEnabled,
    SkillMeta,
    SkillRegistry,
    SkillRequest,
    Strategy,
    ValidationReport,
    default_registry,
    run_skill,
)
from vision_use.skills.builtin import DisableEmptyClips
from vision_use.skills.pipeline import resolvable
from vision_use.skills.registry import registry as default_skills

from tests.conftest import MakeClip, requires_ffmpeg

PAL = FrameRate(num=25, den=1)
DIGEST = "e" * 64
PRESENT = Path(__file__)
runner = CliRunner()


def _context() -> VideoContext:
    return VideoContext(
        probe=MediaProbe(
            source=SourceInfo(
                path=PRESENT,
                sha256=DIGEST,
                size_bytes=1,
                container="mp4",
                duration_s=60.0,
            ),
            video=VideoStreamInfo(index=0, codec="h264", width=640, height=480, frame_rate=PAL),
        )
    )


def _plan(*durations: float) -> EditPlan:
    source = Source(id=source_id(0), path=PRESENT, sha256=DIGEST, duration_s=600.0)
    clips = []
    cursor = 0.0
    for index, duration in enumerate(durations):
        clips.append(
            Clip(
                id=clip_id(index),
                source_id=source.id,
                source_in_s=0.0,
                source_out_s=duration,
                timeline_start_s=cursor,
            )
        )
        cursor += duration
    return EditPlan(
        sources={source.id: source},
        timeline=Timeline(
            frame_rate=PAL,
            width=640,
            height=480,
            tracks=(Track(id=track_id(0), kind="video", clips=tuple(clips)),),
        ),
    )


class _Recorder:
    """A skill whose behaviour each test dictates, to drive the pipeline."""

    meta: ClassVar[SkillMeta] = SkillMeta(
        name="test.recorder", version="1", requires=frozenset()
    )

    def __init__(
        self,
        strategies: list[Strategy] | None = None,
        passes: list[bool] | None = None,
        scores: list[float] | None = None,
    ) -> None:
        self._strategies = strategies if strategies is not None else [Strategy(name="only")]
        self._passes = passes or []
        self._scores = scores or []
        self.executed: list[str] = []

    def analyse(self, context: VideoContext, plan: EditPlan, request: SkillRequest) -> Analysis:
        del context, request
        return Analysis(findings={"clips": len(plan.timeline.clips())})

    def plan(self, analysis: Analysis) -> list[Strategy]:
        del analysis
        return list(self._strategies)

    def execute(self, strategy: Strategy) -> PlanPatch:
        self.executed.append(strategy.name)
        return PlanPatch(
            ops=(SetClipEnabled(clip_id=clip_id(0), enabled=False),),
            rationale=strategy.name,
            produced_by=self.meta.ref,
        )

    def validate(
        self, before: EditPlan, after: EditPlan, context: VideoContext
    ) -> ValidationReport:
        del before, after, context
        index = len(self.executed) - 1
        passed = self._passes[index] if index < len(self._passes) else True
        return ValidationReport(
            measurements=(Measurement(name="check", value=1.0, passed=passed),)
        )

    def score(self, report: ValidationReport) -> Score:
        del report
        index = len(self.executed) - 1
        return Score(value=self._scores[index] if index < len(self._scores) else 1.0)


# --- registry --------------------------------------------------------------


def test_register_and_retrieve() -> None:
    registry = SkillRegistry()
    skill = _Recorder()
    registry.register(skill)

    assert registry.get("test.recorder") is skill
    assert "test.recorder" in registry
    assert len(registry) == 1


def test_duplicate_registration_rejected() -> None:
    """Two plugins claiming one name must fail loudly, not by import order."""
    registry = default_registry([_Recorder()])

    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register(_Recorder())


def test_unknown_skill_lists_what_is_registered() -> None:
    registry = default_registry([_Recorder()])

    with pytest.raises(ConfigurationError) as excinfo:
        registry.get("absent")

    assert "test.recorder" in (excinfo.value.remedy or "")


def test_requiring_finds_skills_by_section() -> None:
    """Answers 'what becomes possible if I transcribe this?'"""
    registry = default_registry([DisableEmptyClips()])

    assert registry.requiring(ContextSection.PROBE)
    assert not registry.requiring(ContextSection.TRANSCRIPT)


def test_builtin_skill_is_registered() -> None:
    assert "edit.disable_runts" in default_skills


# --- lifecycle -------------------------------------------------------------


def test_no_strategies_means_nothing_to_do() -> None:
    """A skill finding no work is a legitimate outcome, not a failure."""
    skill = _Recorder(strategies=[])

    outcome = run_skill(skill, PRESENT, _plan(1.0), context=_context())

    assert outcome.accepted is None
    assert outcome.attempts == ()
    assert not outcome.changed


def test_first_passing_strategy_is_accepted() -> None:
    skill = _Recorder(strategies=[Strategy(name="a"), Strategy(name="b")])

    outcome = run_skill(skill, PRESENT, _plan(1.0, 1.0), context=_context())

    assert outcome.accepted is not None
    assert outcome.accepted.strategy.name == "a"
    assert skill.executed == ["a"]


def test_a_rejected_strategy_falls_through_to_the_next() -> None:
    """Several strategies are what make the retry loop meaningful."""
    skill = _Recorder(strategies=[Strategy(name="a"), Strategy(name="b")], passes=[False, True])

    outcome = run_skill(skill, PRESENT, _plan(1.0, 1.0), context=_context())

    assert skill.executed == ["a", "b"]
    assert outcome.accepted is not None
    assert outcome.accepted.strategy.name == "b"


def test_each_attempt_starts_from_the_original_plan() -> None:
    """A rejected attempt must not contaminate the next one."""
    plan = _plan(1.0, 1.0)
    skill = _Recorder(strategies=[Strategy(name="a"), Strategy(name="b")], passes=[False, True])

    outcome = run_skill(skill, PRESENT, plan, context=_context())

    # Both attempts disabled the same clip; if the second had built on the
    # first, its patch would have failed or double-applied.
    assert outcome.accepted is not None
    disabled = [c for c in outcome.accepted.plan.timeline.clips() if not c.enabled]
    assert len(disabled) == 1


def test_every_attempt_is_recorded() -> None:
    skill = _Recorder(
        strategies=[Strategy(name="a"), Strategy(name="b"), Strategy(name="c")],
        passes=[False, False, True],
    )

    outcome = run_skill(skill, PRESENT, _plan(1.0, 1.0), context=_context())

    assert [a.strategy.name for a in outcome.attempts] == ["a", "b", "c"]


def test_all_rejected_returns_the_highest_scoring_attempt() -> None:
    """VISION.md asks for the best available result, not an outright failure."""
    skill = _Recorder(
        strategies=[Strategy(name="a"), Strategy(name="b")],
        passes=[False, False],
        scores=[0.2, 0.9],
    )

    outcome = run_skill(skill, PRESENT, _plan(1.0, 1.0), context=_context())

    assert outcome.accepted is None
    assert outcome.exhausted
    assert outcome.best is not None
    assert outcome.best.strategy.name == "b"
    assert outcome.plan == outcome.best.plan


def test_attempt_carries_its_own_inverse() -> None:
    """Rollback uses the inverse the patch application already produced."""
    plan = _plan(1.0, 1.0)
    skill = _Recorder()

    outcome = run_skill(skill, PRESENT, plan, context=_context())

    assert outcome.accepted is not None
    assert outcome.accepted.inverse.ops


# --- budget ----------------------------------------------------------------


def test_attempt_budget_is_enforced() -> None:
    """An unbounded retry loop over video is a runaway-compute bug."""
    skill = _Recorder(
        strategies=[Strategy(name=str(index)) for index in range(5)],
        passes=[False] * 5,
    )

    outcome = run_skill(
        skill,
        PRESENT,
        _plan(1.0, 1.0),
        context=_context(),
        policy=RetryPolicy(max_attempts=2),
    )

    assert len(skill.executed) == 2
    assert outcome.exhausted


def test_a_single_attempt_policy_never_retries() -> None:
    skill = _Recorder(strategies=[Strategy(name="a"), Strategy(name="b")], passes=[False, True])

    run_skill(
        skill,
        PRESENT,
        _plan(1.0, 1.0),
        context=_context(),
        policy=RetryPolicy(max_attempts=1),
    )

    assert skill.executed == ["a"]


def test_time_budget_stops_further_attempts() -> None:
    skill = _Recorder(
        strategies=[Strategy(name="a"), Strategy(name="b")], passes=[False, False]
    )

    outcome = run_skill(
        skill,
        PRESENT,
        _plan(1.0, 1.0),
        context=_context(),
        policy=RetryPolicy(max_attempts=5, max_wall_seconds=0.000001),
    )

    assert len(skill.executed) <= 1
    assert outcome.exhausted


@pytest.mark.parametrize(("attempts", "seconds"), [(0, 10.0), (-1, 10.0), (3, 0.0), (3, -1.0)])
def test_impossible_policies_are_rejected(attempts: int, seconds: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        RetryPolicy(max_attempts=attempts, max_wall_seconds=seconds)


# --- context resolution ----------------------------------------------------


def test_a_satisfied_context_is_reused() -> None:
    """Re-analysing what is already to hand is exactly what Principle 8 forbids."""
    context = _context()
    skill = _Recorder()

    outcome = run_skill(skill, Path("/nonexistent.mp4"), _plan(1.0, 1.0), context=context)

    # A missing file would have failed had the pipeline tried to analyse it.
    assert outcome.accepted is not None


def test_unproducible_requirement_is_reported_up_front() -> None:
    """Better to know before paying for an analysis."""

    class NeedsTranscript(_Recorder):
        meta: ClassVar[SkillMeta] = SkillMeta(
            name="test.needs_transcript",
            version="1",
            requires=frozenset({ContextSection.TRANSCRIPT}),
        )

    assert resolvable(NeedsTranscript(), AnalyzerRegistry()) == [ContextSection.TRANSCRIPT]


def test_probe_is_never_reported_as_unresolvable() -> None:
    assert resolvable(DisableEmptyClips(), AnalyzerRegistry()) == []


# --- the reference skill ---------------------------------------------------


def test_reference_skill_disables_runts() -> None:
    outcome = run_skill(DisableEmptyClips(), PRESENT, _plan(2.0, 0.2), context=_context())

    assert outcome.accepted is not None
    enabled = {c.id: c.enabled for c in outcome.accepted.plan.timeline.clips()}
    assert enabled[clip_id(0)]
    assert not enabled[clip_id(1)]


def test_reference_skill_leaves_a_clean_plan_alone() -> None:
    """Nothing to disable means an empty patch, which the pipeline skips."""
    outcome = run_skill(DisableEmptyClips(), PRESENT, _plan(2.0, 3.0), context=_context())

    assert outcome.accepted is None
    assert outcome.attempts == ()


def test_reference_skill_refuses_to_empty_the_timeline() -> None:
    """Its own validation rejects a strategy that would leave nothing to render."""
    outcome = run_skill(DisableEmptyClips(), PRESENT, _plan(0.05, 0.05), context=_context())

    assert outcome.accepted is None
    assert outcome.attempts
    assert all(not attempt.passed for attempt in outcome.attempts)


def test_reference_skill_stamps_its_identity_on_the_patch() -> None:
    outcome = run_skill(DisableEmptyClips(), PRESENT, _plan(2.0, 0.2), context=_context())

    assert outcome.accepted is not None
    produced_by = outcome.accepted.patch.produced_by
    assert produced_by is not None
    assert produced_by.name == "edit.disable_runts"


# --- CLI -------------------------------------------------------------------
#
# These go through the real entry point, which has no context to reuse and so
# must probe the source. That needs actual media, unlike the unit tests above.


def _media_plan(clip: Path, *durations: float) -> EditPlan:
    """A plan over real media, with one clip per duration."""
    base = straight_through(clip)
    track = base.timeline.tracks[0]
    source_id_ = next(iter(base.sources))
    clips = []
    cursor = 0.0
    for index, duration in enumerate(durations):
        clips.append(
            Clip(
                id=clip_id(index),
                source_id=source_id_,
                source_in_s=cursor,
                source_out_s=cursor + duration,
                timeline_start_s=cursor,
            )
        )
        cursor += duration
    return base.model_copy(
        update={
            "timeline": base.timeline.model_copy(
                update={"tracks": (track.model_copy(update={"clips": tuple(clips)}),)}
            )
        }
    )


def test_cli_skills_lists_the_registry() -> None:
    result = runner.invoke(app, ["skills"])

    assert result.exit_code == 0
    assert "edit.disable_runts" in result.stdout
    assert "probe" in result.stdout


@pytest.mark.integration
@requires_ffmpeg
def test_cli_apply_edits_the_plan(make_clip: MakeClip, tmp_path: Path) -> None:
    target = tmp_path / "plan.json"
    save_plan(_media_plan(make_clip(name="apply_a", seconds=3.0), 2.0, 0.2), target)

    result = runner.invoke(app, ["apply", str(target), "--skill", "edit.disable_runts"])

    assert result.exit_code == 0
    edited = load_plan(target).clip_by_id(clip_id(1))
    assert edited is not None
    assert not edited.enabled


@pytest.mark.integration
@requires_ffmpeg
def test_cli_apply_reports_the_rationale(make_clip: MakeClip, tmp_path: Path) -> None:
    """An edit nobody can explain is an edit nobody can trust."""
    target = tmp_path / "plan.json"
    save_plan(_media_plan(make_clip(name="apply_b", seconds=3.0), 2.0, 0.2), target)

    result = runner.invoke(app, ["apply", str(target), "--skill", "edit.disable_runts"])

    assert "disable clips shorter than" in result.stdout


@pytest.mark.integration
@requires_ffmpeg
def test_cli_apply_dry_run_writes_nothing(make_clip: MakeClip, tmp_path: Path) -> None:
    target = tmp_path / "plan.json"
    save_plan(_media_plan(make_clip(name="apply_c", seconds=3.0), 2.0, 0.2), target)
    before = target.read_text()

    result = runner.invoke(
        app, ["apply", str(target), "--skill", "edit.disable_runts", "--dry-run"]
    )

    assert result.exit_code == 0
    assert target.read_text() == before


@pytest.mark.integration
@requires_ffmpeg
def test_cli_apply_can_write_elsewhere(make_clip: MakeClip, tmp_path: Path) -> None:
    original = tmp_path / "plan.json"
    save_plan(_media_plan(make_clip(name="apply_d", seconds=3.0), 2.0, 0.2), original)
    target = tmp_path / "edited.json"

    runner.invoke(
        app, ["apply", str(original), "--skill", "edit.disable_runts", "-o", str(target)]
    )

    assert load_plan(target) != load_plan(original)


@pytest.mark.integration
@requires_ffmpeg
def test_cli_apply_exits_nonzero_when_nothing_passes(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    """Every clip is a runt, so disabling any leaves nothing to render."""
    target = tmp_path / "plan.json"
    save_plan(_media_plan(make_clip(name="apply_e", seconds=3.0), 0.05), target)

    result = runner.invoke(app, ["apply", str(target), "--skill", "edit.disable_runts"])

    assert result.exit_code == 1
    assert "No strategy passed" in result.stdout


@pytest.mark.integration
@requires_ffmpeg
def test_cli_apply_reports_why_each_strategy_was_rejected(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    target = tmp_path / "plan.json"
    save_plan(_media_plan(make_clip(name="apply_f", seconds=3.0), 0.05), target)

    result = runner.invoke(app, ["apply", str(target), "--skill", "edit.disable_runts"])

    assert "rejected" in result.stdout
    assert "the render would be empty" in result.stdout


@pytest.mark.integration
@requires_ffmpeg
def test_cli_apply_says_when_there_is_nothing_to_do(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    target = tmp_path / "plan.json"
    save_plan(_media_plan(make_clip(name="apply_g", seconds=6.0), 2.0, 3.0), target)

    result = runner.invoke(app, ["apply", str(target), "--skill", "edit.disable_runts"])

    assert "nothing to do" in result.stdout


@pytest.mark.integration
@requires_ffmpeg
def test_cli_apply_unknown_skill_lists_the_registry(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    target = tmp_path / "plan.json"
    save_plan(_media_plan(make_clip(name="apply_h", seconds=3.0), 2.0), target)

    result = runner.invoke(app, ["apply", str(target), "--skill", "no.such.skill"])

    assert result.exit_code == 1
    assert "edit.disable_runts" in result.stdout + result.stderr


@pytest.mark.integration
@requires_ffmpeg
def test_cli_apply_respects_the_attempt_budget(make_clip: MakeClip, tmp_path: Path) -> None:
    target = tmp_path / "plan.json"
    save_plan(_media_plan(make_clip(name="apply_i", seconds=3.0), 0.05), target)

    result = runner.invoke(
        app,
        ["apply", str(target), "--skill", "edit.disable_runts", "--max-attempts", "1"],
    )

    assert result.stdout.count("rejected") == 1
