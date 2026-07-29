"""Tests for contact sheet tiling, labelling, and the context bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner
from vision_use.cache.store import CacheStore
from vision_use.cli.main import app
from vision_use.context.runner import analyze
from vision_use.context.sheet import ContactSheet, SheetTile, build_sheet
from vision_use.errors import UnsupportedMediaError, VisionUseError
from vision_use.models.context import ContextSection, VideoContext
from vision_use.video.contact_sheet import (
    DEFAULT_GAP,
    _draw_label,
    _resize_nearest,
    build_contact_sheet,
    decode_jpeg,
)
from vision_use.video.probe import probe

from tests.conftest import MakeClip, requires_ffmpeg

runner = CliRunner()
S = ContextSection
JPEG_MAGIC = b"\xff\xd8\xff"


def _solid(width: int = 64, height: int = 48, value: int = 200) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


# --- resizing --------------------------------------------------------------


def test_resize_produces_requested_shape() -> None:
    assert _resize_nearest(_solid(64, 48), 32, 24).shape == (24, 32, 3)


def test_resize_preserves_a_solid_colour() -> None:
    resized = _resize_nearest(_solid(64, 48, value=123), 16, 12)

    assert np.all(resized == 123)


def test_resize_is_deterministic() -> None:
    """Sheets are content-addressed, so resampling must not vary."""
    source = np.arange(64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3)

    assert np.array_equal(_resize_nearest(source, 20, 15), _resize_nearest(source, 20, 15))


def test_resize_can_upscale() -> None:
    assert _resize_nearest(_solid(8, 8), 32, 32).shape == (32, 32, 3)


# --- labelling -------------------------------------------------------------


def test_label_marks_pixels() -> None:
    tile = _solid(96, 64, value=10)
    _draw_label(tile, "0")

    assert tile.max() == 255


def test_label_is_confined_to_the_top_left() -> None:
    """A label covering the frame would defeat the point of showing the frame."""
    tile = _solid(200, 150, value=10)
    _draw_label(tile, "7")

    assert tile[100:, 100:].max() == 10


def test_different_digits_render_differently() -> None:
    """If glyphs collided, tile numbers would be unusable."""
    first, second = _solid(96, 64, 10), _solid(96, 64, 10)
    _draw_label(first, "1")
    _draw_label(second, "8")

    assert not np.array_equal(first, second)


def test_multi_digit_labels_are_wider() -> None:
    single, double = _solid(160, 64, 10), _solid(160, 64, 10)
    _draw_label(single, "9")
    _draw_label(double, "12")

    assert int((double == 255).sum()) != int((single == 255).sum())


def test_empty_label_leaves_the_tile_untouched() -> None:
    tile = _solid(96, 64, value=10)
    before = tile.copy()
    _draw_label(tile, "")

    assert np.array_equal(tile, before)


def test_unknown_characters_do_not_raise() -> None:
    """A non-digit degrades to a gap rather than crashing the sheet."""
    tile = _solid(96, 64, value=10)
    _draw_label(tile, "x")

    assert tile.shape == (64, 96, 3)


def test_label_survives_a_tiny_tile() -> None:
    """Clipping, not an IndexError, when the tile is smaller than the glyph."""
    tile = _solid(6, 6, value=10)
    _draw_label(tile, "5")

    assert tile.shape == (6, 6, 3)


# --- tiling ----------------------------------------------------------------


def test_single_tile_sheet() -> None:
    payload, width, height = build_contact_sheet([("0", _solid())], tile_width=64)

    assert payload.startswith(JPEG_MAGIC)
    assert width > 0
    assert height > 0


def test_grid_geometry_matches_the_layout() -> None:
    images = [(str(i), _solid(64, 48)) for i in range(5)]

    _, width, height = build_contact_sheet(images, columns=4, tile_width=100, gap=DEFAULT_GAP)

    # Five tiles at four across is two rows, the second partly empty.
    assert width == 4 * 100 + 5 * DEFAULT_GAP
    assert height == 2 * 75 + 3 * DEFAULT_GAP


def test_fewer_images_than_columns_shrinks_the_grid() -> None:
    """Two tiles must not render as a four-wide row with two empty cells."""
    _, narrow, _ = build_contact_sheet(
        [("0", _solid()), ("1", _solid())], columns=4, tile_width=100
    )
    _, wide, _ = build_contact_sheet(
        [(str(i), _solid()) for i in range(4)], columns=4, tile_width=100
    )

    assert narrow < wide


def test_tiling_is_deterministic() -> None:
    images = [(str(i), _solid(64, 48, value=10 * i)) for i in range(4)]

    assert build_contact_sheet(images)[0] == build_contact_sheet(images)[0]


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        build_contact_sheet([])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"columns": 0}, "columns"), ({"tile_width": 0}, "tile_width"), ({"gap": -1}, "gap")],
)
def test_invalid_layout_rejected(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_contact_sheet([("0", _solid())], **kwargs)


# --- decoding --------------------------------------------------------------


def test_decode_round_trips_a_sheet() -> None:
    payload, width, height = build_contact_sheet([("0", _solid(64, 48))], tile_width=64)

    decoded = decode_jpeg(payload)

    assert decoded.shape == (height, width, 3)


def test_decode_rejects_garbage() -> None:
    with pytest.raises(UnsupportedMediaError):
        decode_jpeg(b"not an image")


# --- manifest --------------------------------------------------------------


def test_describe_lists_every_tile() -> None:
    sheet = ContactSheet(
        path=Path("sheet.jpg"),
        width=100,
        height=100,
        columns=2,
        tiles=(
            SheetTile(index=0, shot_id="sh_0000", keyframe_id="kf_0000", t_s=1.0),
            SheetTile(index=1, shot_id="sh_0001", keyframe_id="kf_0001", t_s=3.0),
        ),
    )

    described = sheet.describe()

    assert "tile 0: sh_0000 at 1.00s" in described
    assert "tile 1: sh_0001 at 3.00s" in described


# --- against real media ----------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_builds_a_sheet_from_a_context(make_clip: MakeClip, tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    context = analyze(make_clip(name="sheet_a", seconds=2.0), [S.KEYFRAMES], cache=store)
    target = tmp_path / "sheet.jpg"

    sheet = build_sheet(context, target, cache=store)

    assert target.read_bytes().startswith(JPEG_MAGIC)
    assert len(sheet.tiles) == len(context.keyframes)


@pytest.mark.integration
@requires_ffmpeg
def test_tiles_map_back_to_shots(make_clip: MakeClip, tmp_path: Path) -> None:
    """The manifest is what turns "tile 3" into a cut; it must resolve."""
    store = CacheStore(tmp_path / "cache")
    context = analyze(make_clip(name="sheet_b", seconds=2.0), [S.KEYFRAMES], cache=store)

    sheet = build_sheet(context, tmp_path / "sheet.jpg", cache=store)

    for tile in sheet.tiles:
        assert context.shot_by_id(tile.shot_id) is not None
        assert context.keyframe_by_id(tile.keyframe_id) is not None


@pytest.mark.integration
@requires_ffmpeg
def test_context_without_keyframes_reports_a_remedy(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    clip = make_clip(name="sheet_c", seconds=2.0)
    context = VideoContext(probe=probe(clip))

    with pytest.raises(VisionUseError) as excinfo:
        build_sheet(context, tmp_path / "sheet.jpg", cache=CacheStore(tmp_path / "cache"))

    assert "analyze" in (excinfo.value.remedy or "")


@pytest.mark.integration
@requires_ffmpeg
def test_missing_cached_image_reports_a_remedy(make_clip: MakeClip, tmp_path: Path) -> None:
    """A cleared cache must produce an actionable error, not a stack trace."""
    store = CacheStore(tmp_path / "cache")
    context = analyze(make_clip(name="sheet_d", seconds=2.0), [S.KEYFRAMES], cache=store)
    store.clear("keyframe.jpeg")

    with pytest.raises(VisionUseError) as excinfo:
        build_sheet(context, tmp_path / "sheet.jpg", cache=store)

    assert "no-cache" in (excinfo.value.remedy or "")


@pytest.mark.integration
@requires_ffmpeg
def test_cli_writes_a_sheet(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="sheet_e", seconds=2.0)
    target = tmp_path / "out.jpg"

    result = runner.invoke(app, ["contact-sheet", str(clip), "-o", str(target), "--describe"])

    assert result.exit_code == 0
    assert target.read_bytes().startswith(JPEG_MAGIC)
    assert "tile 0" in result.stdout
