"""Tests for caption rendering and overlay compositing.

Text is rasterised in-process rather than by ffmpeg because `drawtext` and
`subtitles` are absent from many builds -- including the one this was developed
against. These tests guard both halves: that the image is right, and that it
reaches the output frames.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from vision_use.errors import VisionUseError
from vision_use.hashing import hash_file
from vision_use.models.ids import clip_id, source_id, track_id
from vision_use.models.plan import Clip, EditPlan, Source, Track
from vision_use.planning import straight_through
from vision_use.render import compile_plan, render
from vision_use.video.frames import extract_frames
from vision_use.video.probe import probe
from vision_use.video.text import CaptionStyle, _load_font, render_caption, wrap

from tests.conftest import MakeClip, requires_ffmpeg

PNG_MAGIC = b"\x89PNG"


def _decode(payload: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(payload)).convert("RGBA"))


# --- rendering -------------------------------------------------------------


def test_renders_a_png_the_size_of_the_canvas() -> None:
    """Full-canvas output means compositing is a plain origin overlay."""
    payload = render_caption("hello", 640, 360)

    assert payload.startswith(PNG_MAGIC)
    assert _decode(payload).shape == (360, 640, 4)


def test_rendering_is_deterministic() -> None:
    """Caption images are content-addressed like every other artefact."""
    assert render_caption("hello", 320, 240) == render_caption("hello", 320, 240)


def test_different_text_renders_differently() -> None:
    assert render_caption("hello", 320, 240) != render_caption("goodbye", 320, 240)


def test_empty_text_yields_a_transparent_image() -> None:
    """A caption with nothing to say must not draw an empty plate."""
    decoded = _decode(render_caption("   ", 320, 240))

    assert decoded[:, :, 3].max() == 0


def test_the_caption_sits_near_the_bottom() -> None:
    decoded = _decode(render_caption("hello", 320, 240))
    rows = np.where(decoded[:, :, 3].max(axis=1) > 0)[0]

    assert rows.min() > 240 * 0.6


def test_the_text_has_a_plate_behind_it() -> None:
    """White text on bright footage is unreadable without one."""
    decoded = _decode(render_caption("hello", 320, 240))
    opaque = decoded[:, :, 3] > 0

    # Far more pixels are covered by the plate than by the glyphs themselves.
    glyphs = (decoded[:, :, 0] > 200) & opaque
    assert int(opaque.sum()) > int(glyphs.sum()) * 3


def test_style_changes_the_result() -> None:
    plain = render_caption("hello", 320, 240)
    styled = render_caption(
        "hello", 320, 240, CaptionStyle(fill=(255, 0, 0, 255), font_scale=0.1)
    )

    assert plain != styled


def test_a_degenerate_canvas_is_rejected() -> None:
    with pytest.raises(VisionUseError, match="canvas"):
        render_caption("hello", 0, 240)


def test_a_missing_font_reports_a_remedy() -> None:
    with pytest.raises(VisionUseError, match="font"):
        render_caption("hello", 320, 240, CaptionStyle(font_path=Path("/nope.ttf")))


def test_style_key_is_json_safe() -> None:
    """It feeds a cache key, so it must serialise deterministically."""
    assert json.dumps(CaptionStyle().key(), sort_keys=True)


# --- wrapping --------------------------------------------------------------


def test_short_text_stays_on_one_line() -> None:
    font = _load_font(CaptionStyle(), 360)

    assert wrap("hello", font, 10_000) == ["hello"]


def test_long_text_wraps() -> None:
    """A caption that overflows the frame is worse than one that wraps."""
    font = _load_font(CaptionStyle(), 360)

    lines = wrap("the quick brown fox jumps over the lazy dog", font, 120)

    assert len(lines) > 1


def test_wrapping_preserves_every_word() -> None:
    font = _load_font(CaptionStyle(), 360)
    text = "the quick brown fox jumps over the lazy dog"

    lines = wrap(text, font, 120)

    assert " ".join(lines).split() == text.split()


def test_wrapping_is_measured_not_counted() -> None:
    """Capitals are far wider than lowercase at the same character count."""
    font = _load_font(CaptionStyle(), 360)

    narrow = wrap("iiii iiii iiii iiii", font, 100)
    wide = wrap("WWWW WWWW WWWW WWWW", font, 100)

    assert len(wide) >= len(narrow)


def test_empty_text_wraps_to_nothing() -> None:
    font = _load_font(CaptionStyle(), 360)

    assert wrap("   ", font, 500) == []


# --- compositing -----------------------------------------------------------


def _with_overlay(base: EditPlan, image: Path, start_s: float, end_s: float) -> EditPlan:
    """Return ``base`` with one overlay clip covering ``[start_s, end_s)``."""
    source = Source(
        id=source_id(1),
        path=image,
        sha256=hash_file(image),
        duration_s=end_s - start_s,
        has_audio=False,
    )
    track = Track(
        id=track_id(1),
        kind="overlay",
        clips=(
            Clip(
                id=clip_id(1),
                source_id=source.id,
                source_in_s=0.0,
                source_out_s=end_s - start_s,
                timeline_start_s=start_s,
            ),
        ),
    )
    return base.model_copy(
        update={
            "sources": {**base.sources, source.id: source},
            "timeline": base.timeline.model_copy(
                update={"tracks": (*base.timeline.tracks, track)}
            ),
        }
    )


@pytest.mark.integration
@requires_ffmpeg
def test_an_overlay_reaches_the_filtergraph(make_clip: MakeClip, tmp_path: Path) -> None:
    clip = make_clip(name="ov_graph", seconds=3.0)
    image = tmp_path / "caption.png"
    image.write_bytes(render_caption("hello", 320, 240))
    plan = _with_overlay(straight_through(clip), image, 1.0, 2.0)

    compiled = compile_plan(plan, tmp_path / "out.mp4")

    graph = compiled.args[compiled.args.index("-filter_complex") + 1]
    assert "overlay=0:0" in graph
    assert "between(t,1.0,2.0)" in graph


@pytest.mark.integration
@requires_ffmpeg
def test_a_plan_without_overlays_gains_no_overlay_filter(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    """The simple path stays simple."""
    plan = straight_through(make_clip(name="ov_none", seconds=2.0))

    compiled = compile_plan(plan, tmp_path / "out.mp4")

    assert "overlay=" not in compiled.args[compiled.args.index("-filter_complex") + 1]


@pytest.mark.integration
@requires_ffmpeg
def test_the_overlay_appears_only_within_its_span(make_clip: MakeClip, tmp_path: Path) -> None:
    """`enable=between(...)` is what makes an overlay timed rather than global."""
    clip = make_clip(name="ov_timed", seconds=4.0, fps="25")
    image = tmp_path / "caption.png"
    image.write_bytes(render_caption("CAPTION", 320, 240))
    plan = _with_overlay(straight_through(clip), image, 1.0, 3.0)
    out = tmp_path / "out.mp4"

    render(plan, out)

    def band(t: float) -> float:
        (_, frame), *_ = list(extract_frames(out, [t]))
        pixels = np.asarray(frame.to_ndarray(format="rgb24"))
        low = pixels[int(pixels.shape[0] * 0.82) : int(pixels.shape[0] * 0.95)]
        return float(low.mean())

    outside, inside = band(0.4), band(2.0)
    # The caption's dark plate pulls the band's brightness down.
    assert inside < outside


@pytest.mark.integration
@requires_ffmpeg
def test_overlays_do_not_change_the_frame_count(make_clip: MakeClip, tmp_path: Path) -> None:
    """An overlay sits on top; it must not extend or shorten the timeline."""
    clip = make_clip(name="ov_frames", seconds=3.0, fps="25")
    image = tmp_path / "caption.png"
    image.write_bytes(render_caption("hello", 320, 240))

    plain = tmp_path / "plain.mp4"
    render(straight_through(clip), plain)
    with_overlay = tmp_path / "overlaid.mp4"
    render(_with_overlay(straight_through(clip), image, 1.0, 2.0), with_overlay)

    assert probe(with_overlay).video.frame_count == probe(plain).video.frame_count


@pytest.mark.integration
@requires_ffmpeg
def test_a_muted_overlay_track_is_skipped(make_clip: MakeClip, tmp_path: Path) -> None:
    clip = make_clip(name="ov_muted", seconds=2.0)
    image = tmp_path / "caption.png"
    image.write_bytes(render_caption("hello", 320, 240))
    plan = _with_overlay(straight_through(clip), image, 0.5, 1.5)
    tracks = tuple(
        track.model_copy(update={"muted": True}) if track.kind == "overlay" else track
        for track in plan.timeline.tracks
    )
    muted = plan.model_copy(
        update={"timeline": plan.timeline.model_copy(update={"tracks": tracks})}
    )

    compiled = compile_plan(muted, tmp_path / "out.mp4")

    assert "overlay=" not in compiled.args[compiled.args.index("-filter_complex") + 1]
