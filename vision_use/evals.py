"""Scored regression baselines for skills.

Every skill validates its own work, which catches an edit that is wrong *today*.
Nothing so far catches an edit that is quietly *worse than last month* -- a
threshold nudged, a padding constant tweaked, a grouping rule "improved". Those
changes pass every unit test, because every unit test was written to allow them.

This harness records what each skill actually measured on a fixed fixture and
compares the next run against it. A change is not automatically a failure, but
it is never silent: the baseline is a statement about behaviour, and changing
behaviour means changing the statement deliberately.

**Any movement beyond tolerance is reported, in either direction.** Deciding
which way is "better" for a given metric would need the harness to understand
each skill, which it deliberately does not. A number that moved is a number a
human should look at once.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from vision_use.cache.store import CacheStore
from vision_use.errors import VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.context import VideoContext
from vision_use.models.plan import EditPlan
from vision_use.skills.pipeline import run_skill
from vision_use.skills.skill import Skill

log = get_logger(__name__)

#: Relative movement a measurement may make before it is reported. Encoders and
#: analyzers vary slightly between platforms, so an exact match would fail on
#: someone else's machine for reasons that have nothing to do with the skill.
DEFAULT_TOLERANCE: float = 0.02


@dataclass(frozen=True, slots=True)
class EvalResult:
    """What one skill measured on one fixture."""

    case: str
    skill: str
    accepted: bool
    strategy: str
    measurements: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        """Return a JSON-safe record, for the baseline file."""
        return {
            "skill": self.skill,
            "accepted": self.accepted,
            "strategy": self.strategy,
            "measurements": {
                name: round(value, 4) for name, value in sorted(self.measurements.items())
            },
        }


@dataclass(frozen=True, slots=True)
class Regression:
    """One way a result differs from its baseline."""

    case: str
    detail: str


def evaluate(
    case: str,
    skill: Skill,
    source: Path,
    plan: EditPlan,
    *,
    context: VideoContext | None = None,
    cache: CacheStore | None = None,
) -> EvalResult:
    """Run ``skill`` against ``plan`` and record what it measured.

    Args:
        case: Name of this fixture, used as the baseline key.
        skill: The skill to exercise.
        source: Media the plan draws from.
        plan: The plan to edit.
        context: A prepared context, if the fixture supplies one.
        cache: Store for analyzer and render artefacts.

    Returns:
        The measurements the skill's own validation produced.

    """
    outcome = run_skill(skill, source, plan, context=context, cache=cache)
    best = outcome.accepted or outcome.best

    measurements: dict[str, float] = {}
    strategy = ""
    if best is not None:
        strategy = best.strategy.name
        measurements = {item.name: float(item.value) for item in best.report.measurements}

    log.debug(
        "evaluated",
        extra=bind(case=case, skill=skill.meta.name, accepted=outcome.accepted is not None),
    )
    return EvalResult(
        case=case,
        skill=skill.meta.name,
        accepted=outcome.accepted is not None,
        strategy=strategy,
        measurements=measurements,
    )


def compare(
    result: EvalResult,
    baseline: Mapping[str, object],
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[Regression]:
    """Return the ways ``result`` differs from its recorded baseline.

    Four kinds of difference are reported:

    * the skill stopped accepting a strategy it used to accept -- the clearest
      regression there is
    * the skill now accepts a *different* strategy. Not necessarily worse, but
      always worth a human deciding: the measurements can be identical while
      the approach has changed completely, which is how a behaviour change hides
      from a numeric baseline
    * a measurement moved beyond ``tolerance``
    * a measurement the baseline recorded is now missing, which usually means a
      validation check was dropped

    A *new* measurement is not a difference: adding a check is how a skill gets
    stricter, and it should not need a baseline update to be allowed.
    """
    problems: list[Regression] = []

    was_accepted = bool(baseline.get("accepted", False))
    if was_accepted and not result.accepted:
        problems.append(
            Regression(
                case=result.case,
                detail="the skill no longer accepts any strategy on this fixture",
            )
        )

    was_strategy = baseline.get("strategy")
    if isinstance(was_strategy, str) and result.accepted and was_strategy != result.strategy:
        problems.append(
            Regression(
                case=result.case,
                detail=f"strategy changed: {was_strategy!r} -> {result.strategy!r}",
            )
        )

    recorded = baseline.get("measurements", {})
    if not isinstance(recorded, dict):
        return problems

    for name, expected in recorded.items():
        if name not in result.measurements:
            problems.append(
                Regression(
                    case=result.case, detail=f"measurement {name!r} is no longer reported"
                )
            )
            continue

        actual = result.measurements[name]
        reference = abs(float(expected)) or 1.0
        drift = abs(actual - float(expected)) / reference
        if drift > tolerance:
            problems.append(
                Regression(
                    case=result.case,
                    detail=(
                        f"{name}: {expected} -> {actual:g} "
                        f"({drift:.1%} movement, tolerance {tolerance:.0%})"
                    ),
                )
            )

    return problems


def load_baseline(path: Path) -> dict[str, Mapping[str, object]]:
    """Read the recorded baselines.

    Raises:
        VisionUseError: If the file exists but cannot be parsed -- a corrupt
            baseline must not be silently treated as an empty one, or every
            regression would pass unnoticed.

    """
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VisionUseError(
            f"Could not read the eval baseline at {path}",
            remedy="fix the JSON, or regenerate it with `python -m tests.test_evals --update`",
        ) from exc
    return dict(loaded) if isinstance(loaded, dict) else {}


def save_baseline(results: list[EvalResult], path: Path) -> None:
    """Write ``results`` as the new baseline, sorted for a readable diff."""
    payload = {
        result.case: result.to_json() for result in sorted(results, key=lambda r: r.case)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("wrote eval baseline", extra=bind(path=str(path), cases=len(results)))
