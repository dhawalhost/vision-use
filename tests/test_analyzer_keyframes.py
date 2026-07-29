"""Tests for frame extraction, JPEG encoding, and the keyframe analyzer."""

from __future__ import annotations

from pathlib import Path

import pytest
from vision_use.cache.store import CacheStore
from vision_use.context.analyzer import AnalyzerRequest
from vision_use.context.analyzers.keyframes import KeyframeAnalyzer
from vision_use.context.analyzers.shots import ShotAnalyzer
from vision_use.context.registry import AnalyzerRegistry, default_registry, registry
from vision_use.context.runner import analyze, context_digest
from vision_use.errors import ProbeError
from vision_use.models.context import ContextSection, Shot, TimeSpan, VideoContext
from vision_use.models.ids import shot_id
from vision_use.video.frames import encode_jpeg, extract_frames, scaled_size
from vision_use.video.probe import probe

from tests.conftest import MakeClip, requires_ffmpeg

S = ContextSection

#: JPEG files always begin with this marker.
JPEG_MAGIC = b"\xff\xd8\xff"


def _shot(index: int, start: float, end: float) -> Shot:
    return Shot(id=shot_id(index), index=index, span=TimeSpan(start_s=start, end_s=end))


# --- scaling ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "max_edge", "expected"),
    [
        (1920, 1080, 640, (640, 360)),
        (1080, 1920, 640, (360, 640)),
        (320, 240, 640, (320, 240)),
        (640, 640, 640, (640, 640)),
    ],
)
def test_scaled_size(width: int, height: int, max_edge: int, expected: tuple[int, int]) -> None:
    assert scaled_size(width, height, max_edge) == expected


def test_scaling_never_enlarges() -> None:
    assert scaled_size(100, 50, 640) == (100, 50)


def test_extreme_aspect_never_scales_to_zero() -> None:
    """A very wide, very short source would otherwise reach zero height."""
    _, height = scaled_size(10_000, 5, 640)

    assert height >= 1


def test_scaled_size_rejects_bad_max_edge() -> None:
    with pytest.raises(ValueError, match="positive"):
        scaled_size(100, 100, 0)


# --- construction ----------------------------------------------------------


@pytest.mark.parametrize("position", [-0.1, 1.1])
def test_rejects_position_outside_range(position: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        KeyframeAnalyzer(position=position)


def test_rejects_bad_quality() -> None:
    with pytest.raises(ValueError, match="quality"):
        KeyframeAnalyzer(quality=0)


def test_rejects_bad_max_edge() -> None:
    with pytest.raises(ValueError, match="positive"):
        KeyframeAnalyzer(max_edge=0)


def test_params_cover_everything_that_changes_output() -> None:
    params = KeyframeAnalyzer(position=0.25, max_edge=320, quality=70).params()

    assert params["position"] == 0.25
    assert params["max_edge"] == 320
    assert params["quality"] == 70
    assert "encoder" in params


def test_requires_shots() -> None:
    """Keyframes are per shot, so the dependency must be declared, not assumed."""
    analyzer = KeyframeAnalyzer()

    assert analyzer.produces == {S.KEYFRAMES}
    assert analyzer.requires == {S.SHOTS}


# --- sample position -------------------------------------------------------


def test_midpoint_is_the_default() -> None:
    """The first frame of a shot is often mid-transition and unrepresentative."""
    assert KeyframeAnalyzer().sample_time(_shot(0, 2.0, 6.0)) == 4.0


def test_position_zero_samples_the_start() -> None:
    assert KeyframeAnalyzer(position=0.0).sample_time(_shot(0, 2.0, 6.0)) == 2.0


def test_position_one_stays_inside_the_shot() -> None:
    """Sampling at exactly the end would grab the first frame of the next shot."""
    sampled = KeyframeAnalyzer(position=1.0).sample_time(_shot(0, 2.0, 6.0))

    assert sampled < 6.0


def test_zero_length_shot_samples_its_start() -> None:
    assert KeyframeAnalyzer().sample_time(_shot(0, 3.0, 3.0)) == 3.0


# --- encoding --------------------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_encodes_a_real_jpeg(make_clip: MakeClip) -> None:
    (_, frame), *_ = list(extract_frames(make_clip(), [0.5]))

    payload, width, height = encode_jpeg(frame)

    assert payload.startswith(JPEG_MAGIC)
    assert (width, height) == (320, 240)


@pytest.mark.integration
@requires_ffmpeg
def test_encoding_is_deterministic(make_clip: MakeClip) -> None:
    """Cache correctness depends on identical input producing identical bytes."""
    (_, frame), *_ = list(extract_frames(make_clip(), [0.5]))

    assert encode_jpeg(frame)[0] == encode_jpeg(frame)[0]


@pytest.mark.integration
@requires_ffmpeg
def test_lower_quality_is_smaller(make_clip: MakeClip) -> None:
    """Guards the quality-to-quantiser mapping, which runs backwards."""
    (_, frame), *_ = list(extract_frames(make_clip(), [0.5]))

    assert len(encode_jpeg(frame, quality=20)[0]) < len(encode_jpeg(frame, quality=95)[0])


@pytest.mark.integration
@requires_ffmpeg
def test_max_edge_downscales(make_clip: MakeClip) -> None:
    (_, frame), *_ = list(extract_frames(make_clip(), [0.5]))

    _, width, height = encode_jpeg(frame, max_edge=64)

    assert max(width, height) == 64


# --- frame extraction ------------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_extracts_one_frame_per_target(make_clip: MakeClip) -> None:
    frames = list(extract_frames(make_clip(seconds=2.0), [0.2, 1.0, 1.5]))

    assert [t for t, _ in frames] == [0.2, 1.0, 1.5]


@pytest.mark.integration
@requires_ffmpeg
def test_target_past_the_end_returns_the_last_frame(make_clip: MakeClip) -> None:
    """A boundary from container duration can land past the last decodable frame."""
    frames = list(extract_frames(make_clip(name="short", seconds=2.0), [99.0]))

    assert len(frames) == 1


def test_no_targets_yields_nothing(tmp_path: Path) -> None:
    assert list(extract_frames(tmp_path / "absent.mp4", [])) == []


def test_unopenable_source_raises(tmp_path: Path) -> None:
    junk = tmp_path / "broken.mp4"
    junk.write_bytes(b"not a video")

    with pytest.raises(ProbeError):
        list(extract_frames(junk, [0.0]))


# --- analyzer against real media -------------------------------------------


def _keyframe_registry() -> AnalyzerRegistry:
    return default_registry([ShotAnalyzer(), KeyframeAnalyzer()])


@pytest.mark.integration
@requires_ffmpeg
def test_one_keyframe_per_shot(make_clip: MakeClip, tmp_path: Path) -> None:
    context = analyze(
        make_clip(name="kf", seconds=2.0),
        [S.KEYFRAMES],
        cache=CacheStore(tmp_path / "cache"),
        registry=_keyframe_registry(),
    )

    assert len(context.keyframes) == len(context.shots)


@pytest.mark.integration
@requires_ffmpeg
def test_shots_are_resolved_as_a_prerequisite(make_clip: MakeClip, tmp_path: Path) -> None:
    """Requesting only KEYFRAMES must pull SHOTS in automatically."""
    context = analyze(
        make_clip(name="kf2", seconds=2.0),
        [S.KEYFRAMES],
        cache=CacheStore(tmp_path / "cache"),
        registry=_keyframe_registry(),
    )

    assert context.has(S.SHOTS)
    assert context.has(S.KEYFRAMES)


@pytest.mark.integration
@requires_ffmpeg
def test_keyframes_reference_their_shots(make_clip: MakeClip, tmp_path: Path) -> None:
    context = analyze(
        make_clip(name="kf3", seconds=2.0),
        [S.KEYFRAMES],
        cache=CacheStore(tmp_path / "cache"),
        registry=_keyframe_registry(),
    )

    shot_ids = {shot.id for shot in context.shots}
    for keyframe in context.keyframes:
        assert keyframe.shot_id in shot_ids


@pytest.mark.integration
@requires_ffmpeg
def test_bytes_land_in_the_cache_not_the_context(make_clip: MakeClip, tmp_path: Path) -> None:
    """Inlining stills would blow past a model's context window."""
    store = CacheStore(tmp_path / "cache")
    context = analyze(
        make_clip(name="kf4", seconds=2.0),
        [S.KEYFRAMES],
        cache=store,
        registry=_keyframe_registry(),
    )

    cached = list((store.root / "keyframe.jpeg").rglob("*.jpg"))
    assert len(cached) == len(context.keyframes)
    assert cached[0].read_bytes().startswith(JPEG_MAGIC)
    assert JPEG_MAGIC.hex() not in context.model_dump_json()


@pytest.mark.integration
@requires_ffmpeg
def test_analysis_is_byte_identical_across_runs(make_clip: MakeClip, tmp_path: Path) -> None:
    clip = make_clip(name="kf5", seconds=2.0)
    store = CacheStore(tmp_path / "cache")
    kf_registry = _keyframe_registry()

    first = analyze(clip, [S.KEYFRAMES], cache=store, registry=kf_registry)
    second = analyze(clip, [S.KEYFRAMES], cache=store, registry=kf_registry)

    assert first.model_dump_json() == second.model_dump_json()
    assert context_digest(first) == context_digest(second)


@pytest.mark.integration
@requires_ffmpeg
def test_settings_change_the_cache_key(make_clip: MakeClip, tmp_path: Path) -> None:
    """A different max_edge is different work and must not serve a stale still."""
    clip = make_clip(name="kf6", seconds=2.0)
    store = CacheStore(tmp_path / "cache")
    context = analyze(clip, [S.SHOTS], cache=store, registry=default_registry([ShotAnalyzer()]))
    request = AnalyzerRequest(source=clip, context=context, cache=store)

    big = KeyframeAnalyzer(max_edge=320).run(request)
    small = KeyframeAnalyzer(max_edge=64).run(request)

    assert big.keyframes[0].cache_key != small.keyframes[0].cache_key
    assert big.keyframes[0].width > small.keyframes[0].width


@pytest.mark.integration
@requires_ffmpeg
def test_no_shots_warns_rather_than_failing(make_clip: MakeClip, tmp_path: Path) -> None:
    clip = make_clip(name="kf7", seconds=2.0)
    request = AnalyzerRequest(
        source=clip,
        context=VideoContext(probe=probe(clip)),
        cache=CacheStore(tmp_path / "cache"),
    )

    result = KeyframeAnalyzer().run(request)

    assert result.keyframes == ()
    assert result.warnings


@pytest.mark.integration
@requires_ffmpeg
def test_context_accepts_the_produced_cross_references(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    """The model rejects dangling refs; the analyzer must produce valid ones."""
    context = analyze(
        make_clip(name="kf8", seconds=2.0),
        [S.KEYFRAMES],
        cache=CacheStore(tmp_path / "cache"),
        registry=_keyframe_registry(),
    )

    for keyframe in context.keyframes:
        assert context.shot_by_id(keyframe.shot_id or "") is not None


def test_registered_as_a_builtin() -> None:
    assert "keyframes.per_shot" in registry
    assert registry.producer_of(S.KEYFRAMES) is not None
