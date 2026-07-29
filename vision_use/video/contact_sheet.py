"""Contact sheets: the whole video as one image.

This is the other half of how a model perceives video. A shot list gives
structure and per-shot stills give content, but handing a model twenty separate
images costs twenty attachments and loses the ordering. One tiled sheet with
numbered tiles lets it answer "which shot shows the pricing slide" in a single
look, and the accompanying manifest maps tile numbers back to shot IDs.

**Tiles are numbered, and the numbers are burned in.** A model cannot reliably
count grid positions, so an unlabelled sheet produces confident wrong answers
about which shot it is describing. The label has to be *in the pixels*.

**The font is a 3x5 bitmap defined here.** Burning text would normally mean
Pillow, or ffmpeg's drawtext and therefore a libfreetype-enabled build. Digits
are the only glyphs needed, ten of them; a small table is cheaper than either
dependency and cannot vary between platforms -- which matters, because the sheet
is content-addressed and must be byte-reproducible.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Final, cast

import av
import numpy as np
from av.container import InputContainer
from av.video.frame import VideoFrame

from vision_use.errors import UnsupportedMediaError
from vision_use.log import bind, get_logger
from vision_use.video.frames import DEFAULT_QUALITY, encode_jpeg

log = get_logger(__name__)

#: Bumped when sheet layout or rendering changes, so cached sheets from an
#: older implementation are not served.
SHEET_VERSION: Final = "1"

#: Default tile width in pixels. Wide enough for a model to read a slide, small
#: enough that twenty of them stay inside a sensible attachment size.
DEFAULT_TILE_WIDTH: Final = 320

#: Default columns. Four across keeps a 16:9 sheet roughly square up to ~20
#: shots, which is where most short-form footage lands.
DEFAULT_COLUMNS: Final = 4

#: Padding between tiles, in pixels.
DEFAULT_GAP: Final = 4

#: Sheet background, mid-grey: visible against both dark and bright footage.
_BACKGROUND: Final = 32

#: 3x5 bitmap glyphs for the digits. Each string is one row, "#" is ink.
_GLYPHS: Final[dict[str, tuple[str, ...]]] = {
    "0": ("###", "# #", "# #", "# #", "###"),
    "1": (" # ", "## ", " # ", " # ", "###"),
    "2": ("###", "  #", "###", "#  ", "###"),
    "3": ("###", "  #", "###", "  #", "###"),
    "4": ("# #", "# #", "###", "  #", "  #"),
    "5": ("###", "#  ", "###", "  #", "###"),
    "6": ("###", "#  ", "###", "# #", "###"),
    "7": ("###", "  #", "  #", "  #", "  #"),
    "8": ("###", "# #", "###", "# #", "###"),
    "9": ("###", "# #", "###", "  #", "###"),
}

_GLYPH_WIDTH: Final = 3
_GLYPH_HEIGHT: Final = 5

#: Pixels per glyph cell. At scale 3 a label is 15px tall, legible after a
#: model's own downscaling without dominating the tile.
DEFAULT_LABEL_SCALE: Final = 3

#: Margin around a label's background plate, in pixels.
_LABEL_PAD: Final = 3


def decode_jpeg(payload: bytes) -> np.ndarray:
    """Decode JPEG bytes into an RGB array of shape ``(height, width, 3)``.

    Raises:
        UnsupportedMediaError: If the bytes are not a decodable image.

    """
    try:
        # av.open is typed as returning either direction; read mode always
        # yields an InputContainer.
        container = cast("InputContainer", av.open(io.BytesIO(payload), format="mjpeg"))
        with container:
            for frame in container.decode(video=0):
                return np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
    except av.FFmpegError as exc:
        raise UnsupportedMediaError(
            "Could not decode a cached keyframe",
            remedy="clear the cache with `vision cache clear` and re-analyse",
        ) from exc

    raise UnsupportedMediaError(
        "Cached keyframe contained no image",
        remedy="clear the cache with `vision cache clear` and re-analyse",
    )


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize ``image`` by nearest-neighbour sampling.

    Nearest neighbour rather than an interpolating resize because it needs no
    imaging library and is exactly reproducible: a contact sheet participates in
    the content-addressed cache, so a resampler whose output varies with SIMD
    width or library version would break that.
    """
    source_h, source_w = image.shape[:2]
    rows = (np.arange(height) * source_h // height).clip(0, source_h - 1)
    cols = (np.arange(width) * source_w // width).clip(0, source_w - 1)
    return cast("np.ndarray", image[rows][:, cols])


def _draw_label(tile: np.ndarray, text: str, scale: int = DEFAULT_LABEL_SCALE) -> None:
    """Burn ``text`` into the top-left of ``tile``, in place.

    Only characters present in :data:`_GLYPHS` are drawn; anything else advances
    the cursor as a space, so an unexpected character degrades to a gap rather
    than raising.
    """
    if not text:
        return

    glyph_w = _GLYPH_WIDTH * scale
    glyph_h = _GLYPH_HEIGHT * scale
    spacing = scale
    plate_w = len(text) * (glyph_w + spacing) - spacing + 2 * _LABEL_PAD
    plate_h = glyph_h + 2 * _LABEL_PAD

    height, width = tile.shape[:2]
    plate_w = min(plate_w, width)
    plate_h = min(plate_h, height)

    # Dark plate behind the digits so they stay readable over bright footage.
    tile[:plate_h, :plate_w] = (tile[:plate_h, :plate_w] // 4).astype(np.uint8)

    cursor = _LABEL_PAD
    for character in text:
        glyph = _GLYPHS.get(character)
        if glyph is not None:
            for row_index, row in enumerate(glyph):
                for col_index, cell in enumerate(row):
                    if cell != "#":
                        continue
                    top = _LABEL_PAD + row_index * scale
                    left = cursor + col_index * scale
                    if top + scale > height or left + scale > width:
                        continue
                    tile[top : top + scale, left : left + scale] = 255
        cursor += glyph_w + spacing


def build_contact_sheet(
    images: Sequence[tuple[str, np.ndarray]],
    *,
    columns: int = DEFAULT_COLUMNS,
    tile_width: int = DEFAULT_TILE_WIDTH,
    gap: int = DEFAULT_GAP,
    quality: int = DEFAULT_QUALITY,
) -> tuple[bytes, int, int]:
    """Tile ``images`` into a single labelled JPEG.

    Args:
        images: ``(label, rgb_array)`` pairs in display order. The label is
            burned into the tile; it is normally the tile's index.
        columns: Tiles per row.
        tile_width: Width of each tile in pixels; height follows the aspect
            ratio of the first image so the grid stays regular.
        gap: Padding between tiles.
        quality: JPEG quality for the finished sheet.

    Returns:
        ``(jpeg_bytes, width, height)`` for the sheet.

    Raises:
        ValueError: If ``images`` is empty or the layout parameters are invalid.

    """
    if not images:
        raise ValueError("a contact sheet needs at least one image")
    if columns < 1:
        raise ValueError(f"columns must be at least 1, got {columns}")
    if tile_width < 1:
        raise ValueError(f"tile_width must be at least 1, got {tile_width}")
    if gap < 0:
        raise ValueError(f"gap must not be negative, got {gap}")

    first = images[0][1]
    source_h, source_w = first.shape[:2]
    tile_height = max(1, round(tile_width * source_h / source_w))

    used_columns = min(columns, len(images))
    rows = -(-len(images) // used_columns)

    sheet_w = used_columns * tile_width + (used_columns + 1) * gap
    sheet_h = rows * tile_height + (rows + 1) * gap
    sheet = np.full((sheet_h, sheet_w, 3), _BACKGROUND, dtype=np.uint8)

    for index, (label, image) in enumerate(images):
        row, column = divmod(index, used_columns)
        tile = _resize_nearest(image, tile_width, tile_height).copy()
        _draw_label(tile, label)

        top = gap + row * (tile_height + gap)
        left = gap + column * (tile_width + gap)
        sheet[top : top + tile_height, left : left + tile_width] = tile

    payload, width, height = encode_jpeg(
        VideoFrame.from_ndarray(sheet, format="rgb24"),
        quality=quality,
        max_edge=max(sheet_w, sheet_h),
    )
    log.debug(
        "built contact sheet",
        extra=bind(tiles=len(images), columns=used_columns, size=f"{width}x{height}"),
    )
    return payload, width, height
