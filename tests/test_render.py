"""Tests for the filtergraph compiler and the renderer.

The compiler tests need no media: a filtergraph is a string, and most of what
can go wrong is visible in it. The render tests do need ffmpeg, and they assert
frame counts rather than eyeballing duration -- a render that is one frame short
is exactly the failure this whole timebase design exists to prevent.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from pydantic import BaseModel
from vision_use.effects.registry import EffectDef, EffectRegistry
from vision_use.effects.registry import registry as default_effects
from vision_use.errors import VisionUseError
from vision_use.models.ids import clip_id, source_id, track_id
from vision_use.models.media import FrameRate
from vision_use.models.plan import (
    Clip,
    EditPlan,
    Effect,
    RenderProfile,
    Source,
    Timeline,
    Track,
    Transition,
)
from vision_use.render import CompiledRender, compile_plan, render
from vision_use.render.compiler import _Canvas, _GraphBuilder
from vision_use.video.frames import extract_frames
from vision_use.video.probe import probe

from tests.conftest import FFMPEG, MakeClip, requires_ffmpeg

PAL = FrameRate(num=25, den=1)
NTSC = FrameRate(num=30000, den=1001)
DIGEST = "b" * 64


def _plan(
    *clips: Clip,
    rate: FrameRate = PAL,
    path: Path = Path("in.mp4"),
    duration_s: float = 60.0,
    kind: str = "video",
    has_audio: bool = True,
) -> EditPlan:
    source = Source(
        id=source_id(0),
        path=path,
        sha256=DIGEST,
        duration_s=duration_s,
        has_audio=has_audio,
    )
    return EditPlan(
        sources={source.id: source},
        timeline=Timeline(
            frame_rate=rate,
            width=640,
            height=480,
            tracks=(Track(id=track_id(0), kind=kind, clips=clips),),
        ),
    )


def _clip(index: int, start: float, in_s: float, out_s: float, **kwargs: object) -> Clip:
    return Clip(
        id=clip_id(index),
        source_id=source_id(0),
        source_in_s=in_s,
        source_out_s=out_s,
        timeline_start_s=start,
        **kwargs,
    )


def _graph(compiled: CompiledRender) -> str:
    """Return the filtergraph string from a compiled invocation."""
    return compiled.args[compiled.args.index("-filter_complex") + 1]


# --- compilation -----------------------------------------------------------


def test_single_clip_compiles() -> None:
    compiled = compile_plan(_plan(_clip(0, 0.0, 1.0, 3.0)), Path("out.mp4"))

    graph = _graph(compiled)
    assert "trim=start=1.0:end=3.0" in graph
    assert "concat=n=1:v=1:a=1" in graph


def test_timestamps_are_rebased_per_clip() -> None:
    """Without setpts the second clip keeps its source timing and lands late."""
    compiled = compile_plan(_plan(_clip(0, 0.0, 1.0, 3.0)), Path("out.mp4"))

    assert "setpts=PTS-STARTPTS" in _graph(compiled)


def test_every_clip_is_normalised_to_the_output_rate() -> None:
    """concat requires agreeing rates; sources routinely disagree."""
    compiled = compile_plan(
        _plan(_clip(0, 0.0, 0.0, 1.0), _clip(1, 1.0, 2.0, 3.0)), Path("out.mp4")
    )

    assert _graph(compiled).count("fps=25") == 2


def test_every_clip_gets_square_pixels() -> None:
    """concat also requires matching SAR; anamorphic sources otherwise fail."""
    compiled = compile_plan(_plan(_clip(0, 0.0, 0.0, 1.0)), Path("out.mp4"))

    assert "setsar=1" in _graph(compiled)


def test_ntsc_rate_reaches_the_graph_exactly() -> None:
    compiled = compile_plan(_plan(_clip(0, 0.0, 0.0, 1.0), rate=NTSC), Path("out.mp4"))

    assert "fps=30000/1001" in _graph(compiled)


def test_gap_becomes_a_generated_black_segment() -> None:
    compiled = compile_plan(
        _plan(_clip(0, 0.0, 0.0, 1.0), _clip(1, 3.0, 0.0, 1.0)), Path("out.mp4")
    )

    assert "color=c=black" in " ".join(compiled.args)
    assert compiled.segments == 3


def test_leading_gap_is_filled() -> None:
    compiled = compile_plan(_plan(_clip(0, 2.0, 0.0, 1.0)), Path("out.mp4"))

    assert compiled.segments == 2


def test_a_phantom_gap_never_reaches_the_graph() -> None:
    """Regression: found on real 40-cut media, and it hung the render.

    Two clips meant to touch exactly can leave `cursor` a few femtoseconds
    short of the next clip's `timeline_start_s`. Compiling that as a real gap
    used to emit `color=...:d=0.0` -- a zero-duration lavfi source that ffmpeg
    never reaches EOF on, so the render ran forever rather than failing.
    `add_gap` now refuses a non-positive duration outright, and the caller
    that used to trigger it is exercised here with the exact clips that hung.
    """
    clips = (
        Clip(
            id=clip_id(0),
            source_id=source_id(0),
            source_in_s=75.92,
            source_out_s=82.24,
            timeline_start_s=50.4,
        ),
        Clip(
            id=clip_id(1),
            source_id=source_id(0),
            source_in_s=85.04,
            source_out_s=91.36,
            timeline_start_s=56.72,
        ),
    )

    compiled = compile_plan(_plan(*clips, duration_s=120.0), Path("out.mp4"))

    assert "d=0.0" not in " ".join(compiled.args)
    # The leading gap (0 to 50.4) is real and legitimately fills a segment;
    # what must not appear is a *second*, phantom one between the two clips.
    assert compiled.segments == 3


def test_the_gap_builder_refuses_a_non_positive_duration() -> None:
    """The defensive guard, tested directly rather than only through a caller.

    A hang is too expensive a way to discover a caller stopped checking.
    """
    builder = _GraphBuilder(
        plan=_plan(_clip(0, 0.0, 0.0, 1.0)),
        canvas=_Canvas(width=640, height=360, fps="25"),
        effects=default_effects,
    )

    with pytest.raises(VisionUseError, match="Refusing to insert"):
        builder.add_gap(0.0)


def test_input_slots_stay_correct_across_generated_gaps() -> None:
    """Regression: a lavfi input is four arguments, a file input two.

    Deriving the stream index from the argument-list length misnumbers every
    input after the first gap, which silently renders the wrong source.
    """
    compiled = compile_plan(
        _plan(_clip(0, 0.0, 0.0, 1.0), _clip(1, 3.0, 2.0, 3.0)), Path("out.mp4")
    )

    graph = _graph(compiled)
    # Input 0 is the file. The gap contributes a colour input and a silence
    # input, so the indices must keep advancing past both.
    assert "[0:v]" in graph
    assert "[1:v]" in graph
    assert "[2:a]" in graph
    # Every generated stream label is referenced exactly once by concat.
    for label in ("gv1", "ga1"):
        assert graph.count(f"[{label}]") == 2


def test_one_input_per_source_even_with_many_clips() -> None:
    compiled = compile_plan(
        _plan(_clip(0, 0.0, 0.0, 1.0), _clip(1, 1.0, 2.0, 3.0), _clip(2, 2.0, 4.0, 5.0)),
        Path("out.mp4"),
    )

    assert compiled.args.count("-i") == 1


def test_disabled_clips_are_skipped() -> None:
    compiled = compile_plan(
        _plan(_clip(0, 0.0, 0.0, 1.0), _clip(1, 1.0, 2.0, 3.0, enabled=False)),
        Path("out.mp4"),
    )

    assert compiled.segments == 1


def test_all_disabled_reports_a_remedy() -> None:
    with pytest.raises(VisionUseError, match="disabled"):
        compile_plan(_plan(_clip(0, 0.0, 0.0, 1.0, enabled=False)), Path("out.mp4"))


def test_no_video_track_reports_a_remedy() -> None:
    with pytest.raises(VisionUseError, match="no video clips"):
        compile_plan(_plan(_clip(0, 0.0, 0.0, 1.0), kind="audio"), Path("out.mp4"))


def test_empty_plan_reports_a_remedy() -> None:
    plan = EditPlan(timeline=Timeline(frame_rate=PAL, width=640, height=480))

    with pytest.raises(VisionUseError, match="no video clips"):
        compile_plan(plan, Path("out.mp4"))


# --- effects in the graph --------------------------------------------------


def test_effects_appear_in_clip_order() -> None:
    clip = _clip(
        0,
        0.0,
        0.0,
        1.0,
        effects=(
            Effect(type="crop", params={"width": 100, "height": 50}),
            Effect(type="scale", params={"width": 200, "height": 100}),
        ),
    )
    graph = _graph(compile_plan(_plan(clip), Path("out.mp4")))

    assert graph.index("crop=100:50:0:0") < graph.index("scale=200:100")


def test_audio_effects_stay_out_of_the_video_chain() -> None:
    """ffmpeg would reject a volume filter in a video chain unhelpfully."""
    clip = _clip(0, 0.0, 0.0, 1.0, effects=(Effect(type="volume", params={"gain": 2.0}),))
    graph = _graph(compile_plan(_plan(clip), Path("out.mp4")))

    video_chain = next(line for line in graph.split(";") if "[v0]" in line)
    audio_chain = next(line for line in graph.split(";") if "[a0]" in line)
    assert "volume=" not in video_chain
    assert "volume=2.0" in audio_chain


def test_video_effects_stay_out_of_the_audio_chain() -> None:
    clip = _clip(
        0, 0.0, 0.0, 1.0, effects=(Effect(type="crop", params={"width": 8, "height": 8}),)
    )
    graph = _graph(compile_plan(_plan(clip), Path("out.mp4")))

    audio_chain = next(line for line in graph.split(";") if "[a0]" in line)
    assert "crop=" not in audio_chain


def test_audio_is_normalised_to_one_rate_and_format() -> None:
    """concat requires agreement; sources disagree on rate, format and layout."""
    graph = _graph(compile_plan(_plan(_clip(0, 0.0, 0.0, 1.0)), Path("out.mp4")))

    assert "aresample=48000" in graph
    assert "channel_layouts=stereo" in graph


def test_one_concat_carries_both_streams() -> None:
    """Concatenating separately lets video and audio drift into lip-sync error."""
    graph = _graph(
        compile_plan(_plan(_clip(0, 0.0, 0.0, 1.0), _clip(1, 1.0, 2.0, 3.0)), Path("o.mp4"))
    )

    assert graph.count("concat=") == 1
    assert "[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]" in graph


def test_silent_source_still_contributes_an_audio_leg() -> None:
    """concat demands the same stream count from every input."""
    source = Source(
        id=source_id(0),
        path=Path("silent.mp4"),
        sha256=DIGEST,
        duration_s=60.0,
        has_audio=False,
    )
    plan = EditPlan(
        sources={source.id: source},
        timeline=Timeline(
            frame_rate=PAL,
            width=640,
            height=480,
            tracks=(Track(id=track_id(0), kind="video", clips=(_clip(0, 0.0, 0.0, 1.0),)),),
        ),
    )
    graph = _graph(compile_plan(plan, Path("out.mp4")))

    assert "anullsrc" in " ".join(compile_plan(plan, Path("out.mp4")).args)
    assert "[a0]" in graph


def test_gap_contributes_silence_as_well_as_black() -> None:
    args = " ".join(
        compile_plan(
            _plan(_clip(0, 0.0, 0.0, 1.0), _clip(1, 3.0, 0.0, 1.0)), Path("out.mp4")
        ).args
    )

    assert "color=c=black" in args
    assert "anullsrc" in args


def test_invalid_effect_params_are_rejected_before_rendering() -> None:
    clip = _clip(0, 0.0, 0.0, 1.0, effects=(Effect(type="crop", params={"width": 10}),))

    with pytest.raises(VisionUseError, match="Invalid parameters"):
        compile_plan(_plan(clip), Path("out.mp4"))


class _NoParams(BaseModel):
    """An effect that takes no parameters."""


class _Uncompilable(EffectDef):
    """Registered but with no filter implementation, to prove the error path."""

    effect_type: ClassVar[str] = "not.renderable"
    params_model: ClassVar[type[BaseModel]] = _NoParams


def test_uncompilable_effect_names_the_clip() -> None:
    isolated = EffectRegistry()
    isolated.register(_Uncompilable())
    clip = _clip(0, 0.0, 0.0, 1.0, effects=(Effect(type="not.renderable"),))

    with pytest.raises(VisionUseError, match="clp_0000"):
        compile_plan(_plan(clip), Path("out.mp4"), effects=isolated)


# --- profiles --------------------------------------------------------------


def test_proxy_profile_shrinks_the_canvas() -> None:
    plan = _plan(_clip(0, 0.0, 0.0, 1.0))

    compiled = compile_plan(plan, Path("out.mp4"), profile=plan.render.proxy)

    assert (compiled.width, compiled.height) == (320, 240)


def test_scaled_dimensions_stay_even() -> None:
    """yuv420p subsamples chroma by two; an odd dimension will not encode."""
    plan = _plan(_clip(0, 0.0, 0.0, 1.0))

    compiled = compile_plan(plan, Path("out.mp4"), profile=RenderProfile(scale=0.333))

    assert compiled.width % 2 == 0
    assert compiled.height % 2 == 0


def test_profile_settings_reach_the_command() -> None:
    compiled = compile_plan(
        _plan(_clip(0, 0.0, 0.0, 1.0)),
        Path("out.mp4"),
        profile=RenderProfile(crf=30, preset="ultrafast"),
    )

    assert "30" in compiled.args
    assert "ultrafast" in compiled.args


# --- rendering against real media ------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_render_is_frame_accurate(make_clip: MakeClip, tmp_path: Path) -> None:
    """The whole timebase design exists so this holds exactly, not approximately."""
    clip = make_clip(name="render_a", seconds=4.0, fps="25")
    source_probe = probe(clip)
    plan = _plan(
        _clip(0, 0.0, 1.0, 2.0),
        _clip(1, 1.0, 3.0, 4.0),
        path=clip,
        duration_s=source_probe.source.duration_s,
    )
    out = tmp_path / "out.mp4"

    render(plan, out)

    assert probe(out).video.frame_count == 50


@pytest.mark.integration
@requires_ffmpeg
def test_render_preserves_an_ntsc_rate(make_clip: MakeClip, tmp_path: Path) -> None:
    clip = make_clip(name="render_ntsc", seconds=3.0, fps="30000/1001")
    plan = _plan(
        _clip(0, 0.0, 0.0, 2.0),
        rate=NTSC,
        path=clip,
        duration_s=probe(clip).source.duration_s,
    )
    out = tmp_path / "out.mp4"

    render(plan, out)

    result = probe(out)
    assert result.video.frame_rate.num == 30000
    assert result.video.frame_rate.den == 1001


@pytest.mark.integration
@requires_ffmpeg
def test_gap_renders_black_at_the_right_time(make_clip: MakeClip, tmp_path: Path) -> None:
    """A gap is filled explicitly rather than left for ffmpeg to improvise."""
    clip = make_clip(name="render_gap", seconds=4.0, fps="25")
    plan = _plan(
        _clip(0, 0.0, 0.0, 1.0),
        _clip(1, 2.0, 2.0, 3.0),
        path=clip,
        duration_s=probe(clip).source.duration_s,
    )
    out = tmp_path / "out.mp4"

    render(plan, out)

    assert probe(out).video.frame_count == 75
    (_, frame), *_ = list(extract_frames(out, [1.5]))
    assert np.asarray(frame.to_ndarray(format="rgb24")).mean() < 5


@pytest.mark.integration
@requires_ffmpeg
def test_proxy_render_is_smaller(make_clip: MakeClip, tmp_path: Path) -> None:
    """The proxy path is what makes the agent's retry loop affordable."""
    clip = make_clip(name="render_proxy", seconds=2.0, fps="25")
    plan = _plan(_clip(0, 0.0, 0.0, 1.0), path=clip, duration_s=probe(clip).source.duration_s)

    master = render(plan, tmp_path / "master.mp4")
    proxy = render(plan, tmp_path / "proxy.mp4", proxy=True)

    assert proxy.width < master.width
    assert proxy.profile == "proxy"


@pytest.mark.integration
@requires_ffmpeg
def test_render_applies_an_effect(make_clip: MakeClip, tmp_path: Path) -> None:
    clip = make_clip(name="render_fx", seconds=2.0, fps="25")
    plan = _plan(
        _clip(
            0,
            0.0,
            0.0,
            1.0,
            effects=(Effect(type="scale", params={"width": 160, "height": 120}),),
        ),
        path=clip,
        duration_s=probe(clip).source.duration_s,
    )
    out = tmp_path / "out.mp4"

    render(plan, out)

    # The clip is scaled down then letterboxed back onto the 640x480 canvas.
    assert probe(out).video.width == 640


@pytest.mark.integration
@requires_ffmpeg
def test_render_failure_surfaces_ffmpeg_message(tmp_path: Path) -> None:
    missing = tmp_path / "absent.mp4"
    plan = _plan(_clip(0, 0.0, 0.0, 1.0), path=missing, duration_s=10.0)

    with pytest.raises(VisionUseError, match="ffmpeg failed"):
        render(plan, tmp_path / "out.mp4")


@pytest.mark.integration
@requires_ffmpeg
def test_render_produces_an_audio_track(make_clip: MakeClip, tmp_path: Path) -> None:
    """A cut must carry its sound, at one normalised rate and layout."""
    clip = make_clip(name="render_audio", seconds=4.0, fps="25", with_audio=True)
    plan = _plan(
        _clip(0, 0.0, 0.0, 1.0),
        _clip(1, 1.0, 2.0, 3.0),
        path=clip,
        duration_s=probe(clip).source.duration_s,
    )
    out = tmp_path / "out.mp4"

    render(plan, out)

    result = probe(out)
    assert result.has_audio
    assert result.audio is not None
    assert result.audio.sample_rate == 48_000
    assert result.audio.channels == 2


@pytest.mark.integration
@requires_ffmpeg
def test_silent_source_renders_with_generated_silence(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    """Without a generated audio leg, concat fails on mismatched stream counts."""
    clip = make_clip(name="render_silent", seconds=2.0, fps="25", with_audio=False)
    source = Source(
        id=source_id(0),
        path=clip,
        sha256=DIGEST,
        duration_s=probe(clip).source.duration_s,
        has_audio=False,
    )
    plan = EditPlan(
        sources={source.id: source},
        timeline=Timeline(
            frame_rate=PAL,
            width=640,
            height=480,
            tracks=(Track(id=track_id(0), kind="video", clips=(_clip(0, 0.0, 0.0, 1.0),)),),
        ),
    )
    out = tmp_path / "out.mp4"

    render(plan, out)

    result = probe(out)
    assert result.has_audio
    assert result.video.frame_count == 25


@pytest.mark.integration
@requires_ffmpeg
def test_audio_and_video_stay_the_same_length_across_a_gap(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    """One concat for both streams is what keeps them from drifting apart."""
    clip = make_clip(name="render_sync", seconds=4.0, fps="25", with_audio=True)
    plan = _plan(
        _clip(0, 0.0, 0.0, 1.0),
        _clip(1, 2.0, 2.0, 3.0),
        path=clip,
        duration_s=probe(clip).source.duration_s,
    )
    out = tmp_path / "out.mp4"

    render(plan, out)

    result = probe(out)
    assert result.video.frame_count == 75
    assert result.source.duration_s == pytest.approx(3.0, abs=0.1)


# --- transitions -----------------------------------------------------------


def _fade(duration_s: float = 0.5) -> Transition:
    return Transition(type="fade", duration_s=duration_s)


def test_transition_switches_the_join_to_xfade() -> None:
    """concat butts segments together; a crossfade has to overlap them."""
    graph = _graph(
        compile_plan(
            _plan(
                _clip(0, 0.0, 0.0, 1.0),
                _clip(1, 1.0, 2.0, 3.0, transition_in=_fade()),
            ),
            Path("out.mp4"),
        )
    )

    assert "xfade=transition=fade" in graph
    assert "acrossfade=d=0.5" in graph


def test_a_plan_without_transitions_still_uses_concat() -> None:
    """The simple path stays simple; xfade is only paid for when asked for."""
    graph = _graph(
        compile_plan(_plan(_clip(0, 0.0, 0.0, 1.0), _clip(1, 1.0, 2.0, 3.0)), Path("o.mp4"))
    )

    assert "xfade" not in graph
    assert "concat=n=2" in graph


def test_transition_shortens_the_output() -> None:
    """A crossfade consumes its duration from the boundary it spans."""
    compiled = compile_plan(
        _plan(_clip(0, 0.0, 0.0, 1.0), _clip(1, 1.0, 2.0, 3.0, transition_in=_fade(0.4))),
        Path("out.mp4"),
    )

    assert compiled.duration_s == pytest.approx(1.6)


def test_transition_offset_accounts_for_earlier_transitions() -> None:
    """xfade offsets index the accumulated stream, which each fade has shortened."""
    graph = _graph(
        compile_plan(
            _plan(
                _clip(0, 0.0, 0.0, 2.0),
                _clip(1, 2.0, 2.0, 4.0, transition_in=_fade(0.5)),
                _clip(2, 4.0, 4.0, 6.0, transition_in=_fade(0.5)),
            ),
            Path("out.mp4"),
        )
    )

    # First fade starts at 2.0 - 0.5. After it the stream is 3.5s long, so the
    # second starts at 3.5 - 0.5, not at 4.0 - 0.5.
    assert "offset=1.5" in graph
    assert "offset=3.0" in graph


def test_transition_out_describes_the_same_boundary() -> None:
    graph = _graph(
        compile_plan(
            _plan(
                _clip(0, 0.0, 0.0, 1.0, transition_out=_fade()),
                _clip(1, 1.0, 2.0, 3.0),
            ),
            Path("out.mp4"),
        )
    )

    assert "xfade" in graph


def test_transition_in_wins_over_transition_out() -> None:
    """Both ends describing one boundary would be ambiguous; the later clip wins."""
    graph = _graph(
        compile_plan(
            _plan(
                _clip(0, 0.0, 0.0, 1.0, transition_out=_fade(0.9)),
                _clip(1, 1.0, 2.0, 3.0, transition_in=_fade(0.2)),
            ),
            Path("out.mp4"),
        )
    )

    assert "duration=0.2" in graph
    assert "duration=0.9" not in graph


def test_first_clip_transition_in_is_ignored() -> None:
    """There is nothing before it to fade from; that is a `fade` effect."""
    graph = _graph(
        compile_plan(_plan(_clip(0, 0.0, 0.0, 1.0, transition_in=_fade())), Path("o.mp4"))
    )

    assert "xfade" not in graph


def test_transition_longer_than_the_footage_is_rejected() -> None:
    with pytest.raises(VisionUseError, match="longer than the footage"):
        compile_plan(
            _plan(
                _clip(0, 0.0, 0.0, 0.3),
                _clip(1, 0.3, 1.0, 2.0, transition_in=_fade(2.0)),
            ),
            Path("out.mp4"),
        )


def test_mixed_joins_keep_untransitioned_boundaries_as_cuts() -> None:
    graph = _graph(
        compile_plan(
            _plan(
                _clip(0, 0.0, 0.0, 1.0),
                _clip(1, 1.0, 2.0, 3.0),
                _clip(2, 2.0, 4.0, 5.0, transition_in=_fade(0.3)),
            ),
            Path("out.mp4"),
        )
    )

    assert "concat=n=2" in graph
    assert "xfade" in graph


@pytest.mark.integration
@requires_ffmpeg
def test_crossfade_renders_and_shortens(make_clip: MakeClip, tmp_path: Path) -> None:
    clip = make_clip(name="render_xfade", seconds=4.0, fps="25")
    plan = _plan(
        _clip(0, 0.0, 0.0, 1.0),
        _clip(1, 1.0, 2.0, 3.0, transition_in=_fade(0.4)),
        path=clip,
        duration_s=probe(clip).source.duration_s,
    )
    out = tmp_path / "out.mp4"

    render(plan, out)

    # 2.0s of footage minus a 0.4s overlap.
    assert probe(out).video.frame_count == 40


@pytest.mark.integration
@requires_ffmpeg
def test_crossfade_actually_blends_the_pictures(tmp_path: Path) -> None:
    """Mid-transition the frame must be neither clip, but a mix of both."""
    assert FFMPEG is not None
    source = tmp_path / "two-colours.mp4"
    graph = (
        "color=c=red:size=160x120:rate=25:duration=1[a];"
        "color=c=blue:size=160x120:rate=25:duration=1[b];[a][b]concat=n=2:v=1[out]"
    )
    subprocess.run(
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
            str(source),
        ],
        check=True,
    )

    plan = _plan(
        _clip(0, 0.0, 0.0, 1.0),
        _clip(1, 1.0, 1.0, 2.0, transition_in=_fade(0.4)),
        path=source,
        duration_s=2.0,
        has_audio=False,
    )
    out = tmp_path / "out.mp4"
    render(plan, out)

    (_, frame), *_ = list(extract_frames(out, [0.8]))
    red, green, blue = np.asarray(frame.to_ndarray(format="rgb24")).mean(axis=(0, 1))
    assert 60 < red < 190
    assert 60 < blue < 190
    assert green < 60
