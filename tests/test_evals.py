"""Golden-fixture regression tests for the M4 skills.

Every skill validates its own work, which catches an edit that is wrong today.
This catches an edit that is quietly *worse than last month* -- the kind of
change that passes every unit test because every unit test was written to allow
it.

When a case here fails, the fix is usually not to regenerate the baseline. It is
to decide whether the behaviour change was intended, and only then to record it:

    uv run python -m tests.test_evals --update
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from vision_use.cache.store import CacheStore
from vision_use.errors import VisionUseError
from vision_use.evals import (
    EvalResult,
    compare,
    evaluate,
    load_baseline,
    save_baseline,
)
from vision_use.models.context import (
    AnalyzerRun,
    ContextSection,
    TimeSpan,
    Transcript,
    VideoContext,
    Word,
)
from vision_use.models.ids import word_id
from vision_use.planning import straight_through
from vision_use.skills.captions import Captions
from vision_use.skills.reframe import Reframe
from vision_use.skills.silence import RemoveSilence
from vision_use.video.probe import probe

from tests.conftest import FFMPEG, requires_ffmpeg

BASELINE = Path(__file__).parent / "evals.golden.json"


def _build(path: Path, args: list[str]) -> Path:
    """Synthesise a fixture clip, unless it is already there."""
    if path.exists():
        return path
    assert FFMPEG is not None
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args, str(path)],
        check=True,
    )
    return path


@pytest.fixture(scope="session")
def fixtures(media_dir: Path) -> dict[str, Path]:
    """The fixed clips every eval case runs against.

    Synthesised rather than committed, like all media here -- but *fixed* in
    content, which is what makes a baseline meaningful.
    """
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")

    return {
        # Six seconds: tone, two seconds of exact silence, tone.
        "gap": _build(
            media_dir / "eval-gap.mp4",
            [
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x240:rate=25:duration=6",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=6",
                "-filter_complex",
                "[1:a]volume=enable='between(t,1.5,3.5)':volume=0[a]",
                "-map",
                "0:v",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ],
        ),
        # A 16:9 clip with nothing else interesting about it.
        "wide": _build(
            media_dir / "eval-wide.mp4",
            [
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=640x360:rate=25:duration=3",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
            ],
        ),
    }


def _spoken_context(source: Path) -> VideoContext:
    """A context with a fixed transcript.

    Real ASR is not used here on purpose: a model update would move every
    caption measurement at once, and the baseline would be recording Whisper's
    behaviour rather than this skill's.
    """
    words = [
        ("our", 0.4, 0.7),
        ("pricing", 0.7, 1.2),
        ("is", 1.2, 1.4),
        ("simple", 1.4, 2.0),
        ("fifty", 2.9, 3.3),
        ("dollars", 3.3, 3.8),
        ("per", 3.8, 4.0),
        ("month", 4.0, 4.5),
    ]
    return VideoContext(
        probe=probe(source),
        transcript=Transcript(
            language="en",
            words=tuple(
                Word(
                    id=word_id(index),
                    span=TimeSpan(start_s=start, end_s=end),
                    text=text,
                )
                for index, (text, start, end) in enumerate(words)
            ),
        ),
        analyzers={
            "transcript.fixed": AnalyzerRun(
                name="transcript.fixed",
                version="1",
                params_hash="0" * 64,
                produced=(ContextSection.TRANSCRIPT,),
            )
        },
    )


def run_cases(fixtures: dict[str, Path], cache_root: Path) -> list[EvalResult]:
    """Run every eval case and return their measurements."""
    store = CacheStore(cache_root)
    gap, wide = fixtures["gap"], fixtures["wide"]

    return [
        evaluate(
            "silence/gap",
            RemoveSilence(),
            gap,
            straight_through(gap),
            cache=store,
        ),
        evaluate(
            "captions/spoken",
            Captions(cache=store),
            gap,
            straight_through(gap),
            context=_spoken_context(gap),
            cache=store,
        ),
        evaluate(
            "reframe/wide-to-vertical",
            Reframe(),
            wide,
            straight_through(wide),
            cache=store,
        ),
    ]


# --- the harness itself ----------------------------------------------------


def test_a_matching_result_reports_nothing() -> None:
    result = EvalResult(
        case="x", skill="s", accepted=True, strategy="a", measurements={"m": 1.0}
    )

    assert compare(result, result.to_json()) == []


def test_losing_acceptance_is_a_regression() -> None:
    """The clearest regression there is: it used to work and now does not."""
    before = EvalResult(case="x", skill="s", accepted=True, strategy="a").to_json()
    after = EvalResult(case="x", skill="s", accepted=False, strategy="")

    (problem,) = compare(after, before)

    assert "no longer accepts" in problem.detail


def test_a_moved_measurement_is_reported() -> None:
    before = EvalResult(
        case="x", skill="s", accepted=True, strategy="a", measurements={"m": 1.0}
    ).to_json()
    after = EvalResult(
        case="x", skill="s", accepted=True, strategy="a", measurements={"m": 2.0}
    )

    (problem,) = compare(after, before)

    assert "m: 1.0 -> 2" in problem.detail


def test_small_movement_is_tolerated() -> None:
    """Encoders vary between platforms; an exact match would fail elsewhere."""
    before = EvalResult(
        case="x", skill="s", accepted=True, strategy="a", measurements={"m": 1.0}
    ).to_json()
    after = EvalResult(
        case="x", skill="s", accepted=True, strategy="a", measurements={"m": 1.005}
    )

    assert compare(after, before) == []


def test_a_dropped_measurement_is_reported() -> None:
    """Usually means a validation check was quietly removed."""
    before = EvalResult(
        case="x", skill="s", accepted=True, strategy="a", measurements={"m": 1.0}
    ).to_json()
    after = EvalResult(case="x", skill="s", accepted=True, strategy="a")

    (problem,) = compare(after, before)

    assert "no longer reported" in problem.detail


def test_a_new_measurement_is_not_a_regression() -> None:
    """Adding a check is how a skill gets stricter; it needs no permission."""
    before = EvalResult(
        case="x", skill="s", accepted=True, strategy="a", measurements={"m": 1.0}
    ).to_json()
    after = EvalResult(
        case="x",
        skill="s",
        accepted=True,
        strategy="a",
        measurements={"m": 1.0, "extra": 9.0},
    )

    assert compare(after, before) == []


def test_a_zero_baseline_does_not_divide_by_zero() -> None:
    before = EvalResult(
        case="x", skill="s", accepted=True, strategy="a", measurements={"m": 0.0}
    ).to_json()
    after = EvalResult(
        case="x", skill="s", accepted=True, strategy="a", measurements={"m": 0.5}
    )

    assert compare(after, before)


def test_a_missing_baseline_file_is_empty(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "absent.json") == {}


def test_a_corrupt_baseline_is_an_error(tmp_path: Path) -> None:
    """Treating it as empty would let every regression pass unnoticed."""
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(VisionUseError, match="baseline"):
        load_baseline(broken)


def test_the_baseline_round_trips(tmp_path: Path) -> None:
    results = [
        EvalResult(case="b", skill="s", accepted=True, strategy="x", measurements={"m": 2.0}),
        EvalResult(case="a", skill="s", accepted=False, strategy="", measurements={}),
    ]
    target = tmp_path / "golden.json"

    save_baseline(results, target)

    loaded = load_baseline(target)
    assert sorted(loaded) == ["a", "b"]


# --- the fixtures themselves -----------------------------------------------


def test_the_baseline_is_committed() -> None:
    assert BASELINE.exists(), (
        "tests/evals.golden.json is missing; generate it with "
        "`uv run python -m tests.test_evals --update`"
    )


@pytest.mark.integration
@requires_ffmpeg
def test_no_skill_regressed(fixtures: dict[str, Path], tmp_path: Path) -> None:
    """The M4 skills still behave as they did when the baseline was recorded.

    If this fails, decide whether the change was intended *before* regenerating
    the baseline. A baseline updated reflexively records the bug.
    """
    baseline = load_baseline(BASELINE)
    results = run_cases(fixtures, tmp_path / "cache")

    problems = [
        problem
        for result in results
        for problem in compare(result, baseline.get(result.case, {}))
    ]

    assert not problems, "skill behaviour changed:\n  " + "\n  ".join(
        f"{problem.case}: {problem.detail}" for problem in problems
    )


@pytest.mark.integration
@requires_ffmpeg
def test_every_case_is_in_the_baseline(fixtures: dict[str, Path], tmp_path: Path) -> None:
    """A case missing from the baseline is a case nothing is guarding."""
    baseline = load_baseline(BASELINE)
    results = run_cases(fixtures, tmp_path / "cache")

    assert {result.case for result in results} <= set(baseline)


@pytest.mark.integration
@requires_ffmpeg
def test_evaluation_is_repeatable(fixtures: dict[str, Path], tmp_path: Path) -> None:
    """A baseline is worthless if the same input scores differently twice."""
    first = run_cases(fixtures, tmp_path / "a")
    second = run_cases(fixtures, tmp_path / "b")

    assert [r.to_json() for r in first] == [r.to_json() for r in second]


if __name__ == "__main__":  # pragma: no cover - developer entry point
    import tempfile

    if "--update" not in sys.argv:
        sys.exit("usage: python -m tests.test_evals --update")

    with tempfile.TemporaryDirectory() as work:
        root = Path(work)
        built = {
            "gap": _build(
                root / "eval-gap.mp4",
                [
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=320x240:rate=25:duration=6",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=6",
                    "-filter_complex",
                    "[1:a]volume=enable='between(t,1.5,3.5)':volume=0[a]",
                    "-map",
                    "0:v",
                    "-map",
                    "[a]",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                ],
            ),
            "wide": _build(
                root / "eval-wide.mp4",
                [
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=640x360:rate=25:duration=3",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                ],
            ),
        }
        save_baseline(run_cases(built, root / "cache"), BASELINE)


def test_a_changed_strategy_is_reported() -> None:
    """Measurements can be identical while the approach changed completely.

    That is how a behaviour change hides from a numeric baseline, which is
    exactly what happened when `reframe` started following the subject: every
    measurement held and the guard stayed green.
    """
    result = EvalResult(
        case="x", skill="s", accepted=True, strategy="follow", measurements={"m": 1.0}
    )

    problems = compare(
        result, {"accepted": True, "strategy": "centre", "measurements": {"m": 1.0}}
    )

    assert [problem.detail for problem in problems] == [
        "strategy changed: 'centre' -> 'follow'"
    ]


def test_an_unchanged_strategy_reports_nothing() -> None:
    result = EvalResult(case="x", skill="s", accepted=True, strategy="centre", measurements={})

    assert compare(result, {"accepted": True, "strategy": "centre", "measurements": {}}) == []
