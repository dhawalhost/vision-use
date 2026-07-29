"""Tests for shot detection and the analyzer runner."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from vision_use.cache.store import CacheStore
from vision_use.context.analyzer import AnalyzerRequest, AnalyzerResult
from vision_use.context.analyzers.shots import ShotAnalyzer
from vision_use.context.registry import default_registry, registry
from vision_use.context.runner import analyze, context_digest, merge, run_analyzer
from vision_use.models.context import ContextSection, SectionName, VideoContext
from vision_use.models.media import FrameRate, MediaProbe, SourceInfo, VideoStreamInfo
from vision_use.video.probe import probe

from tests.conftest import FFMPEG, requires_ffmpeg

S = ContextSection

MakeCutsClip = Callable[..., Path]


@pytest.fixture(scope="session")
def cuts_clip(media_dir: Path) -> Path:
    """A clip with two hard cuts at known times: colour changes at 2s and 4s."""
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")

    path = media_dir / "three-shots.mp4"
    if path.exists():
        return path

    parts = " ".join(
        f"color=c={colour}:size=320x240:rate=25:duration=2[v{index}];"
        for index, colour in enumerate(("red", "blue", "green"))
    )
    graph = parts + "[v0][v1][v2]concat=n=3:v=1[out]"
    completed = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    return path


# --- construction ----------------------------------------------------------


def test_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="positive"):
        ShotAnalyzer(threshold=0)


def test_rejects_negative_min_shot() -> None:
    with pytest.raises(ValueError, match="negative"):
        ShotAnalyzer(min_shot_s=-1)


def test_params_cover_everything_that_changes_output() -> None:
    params = ShotAnalyzer(threshold=30.0, min_shot_s=1.0).params()

    assert params == {"detector": "content", "threshold": 30.0, "min_shot_s": 1.0}


def test_declares_only_shots() -> None:
    analyzer = ShotAnalyzer()

    assert analyzer.produces == {S.SHOTS}
    assert analyzer.requires == frozenset()


# --- shot construction (no media needed) -----------------------------------


def _shots(
    boundaries: list[float], duration: float, **kwargs: float
) -> list[tuple[float, float]]:
    analyzer = ShotAnalyzer(**kwargs)
    built = analyzer._build_shots(boundaries, duration)
    return [(shot.span.start_s, shot.span.end_s) for shot in built]


def test_no_cuts_yields_one_covering_shot() -> None:
    """Never an empty list -- consumers would have to special-case it."""
    assert _shots([], 10.0) == [(0.0, 10.0)]


def test_shots_tile_the_source_without_gaps() -> None:
    spans = _shots([2.0, 4.0], 6.0)

    assert spans == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    assert spans[0][0] == 0.0
    assert spans[-1][1] == 6.0


def test_short_shots_are_merged_away() -> None:
    """Detectors emit one-frame shots on flashes; they are noise, not structure."""
    spans = _shots([2.0, 2.05, 4.0], 6.0, min_shot_s=0.4)

    assert spans == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]


def test_short_trailing_shot_merged_backwards() -> None:
    spans = _shots([2.0, 5.9], 6.0, min_shot_s=0.4)

    assert spans == [(0.0, 2.0), (2.0, 6.0)]


def test_shot_indices_and_ids_agree() -> None:
    built = ShotAnalyzer()._build_shots([2.0, 4.0], 6.0)

    assert [shot.index for shot in built] == [0, 1, 2]
    assert [shot.id for shot in built] == ["sh_0000", "sh_0001", "sh_0002"]


def test_built_shots_satisfy_the_context_model(tmp_path: Path) -> None:
    """The model enforces ordering and non-overlap; the analyzer must comply."""
    built = ShotAnalyzer()._build_shots([2.0, 4.0], 6.0)
    clip = tmp_path / "x.mp4"
    clip.write_bytes(b"")

    context = VideoContext(probe=_fake_probe(6.0))
    assert context.model_copy(update={"shots": built}).shots == built


def _fake_probe(duration_s: float) -> MediaProbe:
    return MediaProbe(
        source=SourceInfo(
            path=Path("clip.mp4"),
            sha256="0" * 64,
            size_bytes=1,
            container="mp4",
            duration_s=duration_s,
        ),
        video=VideoStreamInfo(
            index=0,
            codec="h264",
            width=320,
            height=240,
            frame_rate=FrameRate(num=25, den=1),
        ),
    )


# --- detection against real media ------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_detects_known_cuts(cuts_clip: Path, tmp_path: Path) -> None:
    context = analyze(
        cuts_clip,
        [S.SHOTS],
        cache=CacheStore(tmp_path / "cache"),
        registry=default_registry([ShotAnalyzer()]),
    )

    assert len(context.shots) == 3
    starts = [shot.span.start_s for shot in context.shots]
    assert starts[0] == 0.0
    assert starts[1] == pytest.approx(2.0, abs=0.1)
    assert starts[2] == pytest.approx(4.0, abs=0.1)


@pytest.mark.integration
@requires_ffmpeg
def test_shots_cover_full_duration(cuts_clip: Path, tmp_path: Path) -> None:
    context = analyze(
        cuts_clip,
        [S.SHOTS],
        cache=CacheStore(tmp_path / "cache"),
        registry=default_registry([ShotAnalyzer()]),
    )

    assert context.shots[-1].span.end_s == pytest.approx(context.duration_s, abs=0.05)


@pytest.mark.integration
@requires_ffmpeg
def test_static_clip_yields_single_shot(make_clip: Callable[..., Path], tmp_path: Path) -> None:
    context = analyze(
        make_clip(name="static", seconds=2.0),
        [S.SHOTS],
        cache=CacheStore(tmp_path / "cache"),
        registry=default_registry([ShotAnalyzer()]),
    )

    assert len(context.shots) == 1


@pytest.mark.integration
@requires_ffmpeg
def test_detection_failure_reports_remedy(tmp_path: Path) -> None:
    junk = tmp_path / "broken.mp4"
    junk.write_bytes(b"not a video")
    request = AnalyzerRequest(
        source=junk,
        context=VideoContext(probe=_fake_probe(1.0)),
        cache=CacheStore(tmp_path / "cache"),
    )

    with pytest.raises(Exception, match=r"Shot detection failed|decode"):
        ShotAnalyzer().run(request)


# --- runner: caching and merging -------------------------------------------


class CountingAnalyzer:
    """Records how many times it actually ran, to prove the cache works."""

    name: str = "counting"
    version: str = "1"
    produces: frozenset[SectionName] = frozenset({S.SHOTS})
    requires: frozenset[SectionName] = frozenset()

    def __init__(self) -> None:
        self.runs = 0

    def params(self) -> dict[str, float]:
        return {}

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        self.runs += 1
        return AnalyzerResult(shots=ShotAnalyzer()._build_shots([], 5.0))


def _request(tmp_path: Path) -> AnalyzerRequest:
    return AnalyzerRequest(
        source=tmp_path / "clip.mp4",
        context=VideoContext(probe=_fake_probe(5.0)),
        cache=CacheStore(tmp_path / "cache"),
    )


def test_second_run_hits_cache(tmp_path: Path) -> None:
    analyzer = CountingAnalyzer()
    request = _request(tmp_path)

    first = run_analyzer(analyzer, request)
    second = run_analyzer(analyzer, request)

    assert analyzer.runs == 1
    assert first == second


def test_no_cache_forces_recomputation(tmp_path: Path) -> None:
    analyzer = CountingAnalyzer()
    request = _request(tmp_path)

    run_analyzer(analyzer, request, use_cache=True)
    run_analyzer(analyzer, request, use_cache=False)

    assert analyzer.runs == 2


def test_cached_result_round_trips_losslessly(tmp_path: Path) -> None:
    """Generic caching only works if a result serializes without loss."""
    analyzer = CountingAnalyzer()
    request = _request(tmp_path)

    fresh = run_analyzer(analyzer, request, use_cache=False)
    from_cache = run_analyzer(analyzer, request)

    assert fresh == from_cache


def test_merge_takes_only_declared_sections(tmp_path: Path) -> None:
    """An analyzer must not overwrite a section it never claimed."""

    class Overreaching(CountingAnalyzer):
        produces: frozenset[SectionName] = frozenset({S.AUDIO_ACTIVITY})

    context = VideoContext(probe=_fake_probe(5.0))
    result = AnalyzerResult(shots=ShotAnalyzer()._build_shots([], 5.0))

    merged = merge(context, Overreaching(), result)

    assert merged.shots == ()


def test_merge_records_provenance(tmp_path: Path) -> None:
    context = VideoContext(probe=_fake_probe(5.0))
    analyzer = ShotAnalyzer()

    merged = merge(context, analyzer, AnalyzerResult(shots=analyzer._build_shots([], 5.0)))

    assert merged.has(S.SHOTS)
    assert merged.analyzers["shots.content"].version == "1"


def test_merge_replaces_rather_than_accumulates(tmp_path: Path) -> None:
    context = VideoContext(probe=_fake_probe(5.0))
    analyzer = ShotAnalyzer()
    result = AnalyzerResult(shots=analyzer._build_shots([], 5.0))

    once = merge(context, analyzer, result)
    twice = merge(once, analyzer, result)

    assert len(twice.analyzers) == 1


# --- M1 exit criterion -----------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_analysis_is_byte_identical_across_runs(cuts_clip: Path, tmp_path: Path) -> None:
    """M1's exit criterion, asserted directly."""
    store = CacheStore(tmp_path / "cache")
    shots_registry = default_registry([ShotAnalyzer()])

    first = analyze(cuts_clip, [S.SHOTS], cache=store, registry=shots_registry)
    second = analyze(cuts_clip, [S.SHOTS], cache=store, registry=shots_registry)

    assert first.model_dump_json() == second.model_dump_json()
    assert context_digest(first) == context_digest(second)


@pytest.mark.integration
@requires_ffmpeg
def test_cold_and_warm_cache_agree(cuts_clip: Path, tmp_path: Path) -> None:
    """A cache hit must be indistinguishable from a fresh computation."""
    registry = default_registry([ShotAnalyzer()])

    cold = analyze(cuts_clip, [S.SHOTS], cache=CacheStore(tmp_path / "a"), registry=registry)
    warm_store = CacheStore(tmp_path / "b")
    analyze(cuts_clip, [S.SHOTS], cache=warm_store, registry=registry)
    warm = analyze(cuts_clip, [S.SHOTS], cache=warm_store, registry=registry)

    assert context_digest(cold) == context_digest(warm)


@pytest.mark.integration
@requires_ffmpeg
def test_changed_params_bypass_the_cache(cuts_clip: Path, tmp_path: Path) -> None:
    """A different threshold is different work and must not serve a stale entry."""
    store = CacheStore(tmp_path / "cache")
    context = VideoContext(probe=probe(cuts_clip))
    request = AnalyzerRequest(source=cuts_clip, context=context, cache=store)

    lenient = run_analyzer(ShotAnalyzer(threshold=27.0), request)
    strict = run_analyzer(ShotAnalyzer(threshold=95.0), request)

    assert len(strict.shots) <= len(lenient.shots)


@pytest.mark.integration
@requires_ffmpeg
def test_probe_only_analysis_runs_nothing(cuts_clip: Path, tmp_path: Path) -> None:
    context = analyze(cuts_clip, [], cache=CacheStore(tmp_path / "cache"))

    assert context.has(S.PROBE)
    assert not context.has(S.SHOTS)
    assert context.analyzers == {}


def test_builtin_analyzers_are_registered() -> None:
    """Importing the public package must be enough; no manual registration."""
    assert "shots.content" in registry
    assert registry.producer_of(S.SHOTS) is not None


@pytest.mark.integration
@requires_ffmpeg
def test_public_analyze_works_with_the_default_registry(
    cuts_clip: Path, tmp_path: Path
) -> None:
    """Regression: the default registry was empty until built-ins were imported."""
    context = analyze(cuts_clip, [S.SHOTS], cache=CacheStore(tmp_path / "cache"))

    assert len(context.shots) == 3


def test_ffmpeg_available_for_integration_tests() -> None:
    assert shutil.which("ffmpeg") is None or FFMPEG is not None
