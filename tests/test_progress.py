"""Tests for progress reporting.

The contract worth pinning is that reporting is *never* fatal: the work is
expensive and the reporting is not, so a callback that raises must not destroy
a bake that was otherwise about to succeed.
"""

from __future__ import annotations

import pytest
from vision_use.progress import Progress, Reporter, Stage

# --- the report ------------------------------------------------------------


def test_fraction_is_the_completed_share() -> None:
    assert Progress(stage=Stage.BAKE, done=5, total=20).fraction == 0.25


def test_fraction_is_unknown_without_a_total() -> None:
    """A variable-rate source has no reliable frame count; claiming one lies."""
    assert Progress(stage=Stage.BAKE, done=5).fraction is None


def test_fraction_is_unknown_for_a_zero_total() -> None:
    assert Progress(stage=Stage.BAKE, done=0, total=0).fraction is None


def test_fraction_never_exceeds_one() -> None:
    """The total is an estimate; 103% would be worse than clamping."""
    assert Progress(stage=Stage.BAKE, done=30, total=25).fraction == 1.0


# --- the reporter ----------------------------------------------------------


def test_updates_reach_the_callback() -> None:
    seen: list[Progress] = []

    Reporter(seen.append)(Stage.BAKE, 3, 10, "cl_0000")

    assert seen == [Progress(stage=Stage.BAKE, done=3, total=10, detail="cl_0000")]


def test_no_callback_is_not_an_error() -> None:
    """Not asking for progress is the common case."""
    Reporter()(Stage.RENDER, 1)


def test_a_failing_callback_does_not_raise() -> None:
    def explode(update: Progress) -> None:
        raise RuntimeError("broken")

    Reporter(explode)(Stage.BAKE, 1)


def test_a_failing_callback_is_dropped_not_retried() -> None:
    """It would raise every frame, and the log would outweigh the render."""
    calls = 0

    def explode(update: Progress) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("broken")

    reporter = Reporter(explode)
    for index in range(5):
        reporter(Stage.BAKE, index)

    assert calls == 1
    assert not reporter.active


def test_one_bad_callback_does_not_disable_another() -> None:
    """The failure flag is per operation, not global."""
    seen: list[Progress] = []
    broken = Reporter(lambda update: (_ for _ in ()).throw(RuntimeError("broken")))
    working = Reporter(seen.append)

    broken(Stage.BAKE, 1)
    working(Stage.BAKE, 1)

    assert not broken.active
    assert working.active
    assert len(seen) == 1


@pytest.mark.parametrize("stage", list(Stage))
def test_every_stage_is_reportable(stage: Stage) -> None:
    seen: list[Progress] = []

    Reporter(seen.append)(stage, 1)

    assert seen[0].stage is stage
