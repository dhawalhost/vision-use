"""Rendering caption text to images.

**Why this exists rather than ffmpeg's `drawtext`.** Text rendering in ffmpeg
needs the `drawtext` filter (libfreetype) or `subtitles` (libass), and neither is
guaranteed: the Homebrew build this was developed against has *neither*, and
distribution builds vary widely. A framework whose captions work on one machine
and not the next is not usable, so text is rasterised in-process and composited
with `overlay`, which every build has.

Pillow does the rasterising. It is the mature choice for the job, its bundled
default font means captions work with no font files to locate or ship, and a
caller who wants their own typeface passes a path.

Everything here is deterministic: the same text and style produce the same
bytes, because the result is content-addressed and cached like every other
derived artefact.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image, ImageDraw, ImageFont

from vision_use.errors import VisionUseError
from vision_use.log import bind, get_logger

log = get_logger(__name__)

#: Pillow's `load_default` and `truetype` return different classes, both of
#: which expose the metrics used here.
_Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

#: Bumped when rendered output changes for identical input, so cached caption
#: images from an older implementation are not served.
TEXT_VERSION: Final = "1"

#: Fraction of the canvas height a caption's text occupies by default.
DEFAULT_FONT_SCALE: Final = 0.055

#: Fraction of the canvas height the caption baseline sits above the bottom.
DEFAULT_MARGIN_SCALE: Final = 0.08

#: Padding around the text inside its plate, as a fraction of the font size.
_PAD_SCALE: Final = 0.4


@dataclass(frozen=True, slots=True)
class CaptionStyle:
    """How a caption looks.

    Attributes:
        font_scale: Text height as a fraction of the canvas height.
        margin_scale: Distance from the bottom, as a fraction of canvas height.
        fill: Text colour, RGBA.
        plate: Colour of the plate behind the text, RGBA. A fully transparent
            plate leaves the text unbacked, which is unreadable over bright
            footage -- the default is a semi-opaque black.
        font_path: A TrueType file to use instead of Pillow's bundled font.
        max_width_scale: Longest line as a fraction of canvas width, before
            wrapping.

    """

    font_scale: float = DEFAULT_FONT_SCALE
    margin_scale: float = DEFAULT_MARGIN_SCALE
    fill: tuple[int, int, int, int] = (255, 255, 255, 255)
    plate: tuple[int, int, int, int] = (0, 0, 0, 168)
    font_path: Path | None = None
    max_width_scale: float = 0.9

    def key(self) -> dict[str, object]:
        """Return a JSON-safe description, for cache keys."""
        return {
            "font_scale": self.font_scale,
            "margin_scale": self.margin_scale,
            "fill": list(self.fill),
            "plate": list(self.plate),
            "font": str(self.font_path) if self.font_path else "default",
            "max_width_scale": self.max_width_scale,
        }


def _load_font(style: CaptionStyle, canvas_height: int) -> _Font:
    """Return the font to draw with, at a size derived from the canvas.

    Raises:
        VisionUseError: If a requested font file cannot be loaded.

    """
    size = max(8, int(canvas_height * style.font_scale))
    if style.font_path is None:
        return ImageFont.load_default(size=size)
    try:
        return ImageFont.truetype(str(style.font_path), size=size)
    except OSError as exc:
        raise VisionUseError(
            f"Could not load the font at {style.font_path}",
            remedy="check the path, or omit it to use the bundled font",
        ) from exc


def wrap(text: str, font: _Font, max_width: int) -> list[str]:
    """Split ``text`` into lines that fit within ``max_width`` pixels.

    Wrapping is measured, not guessed at a character count: a caption of
    capitals is far wider than the same number of lowercase letters, and a
    caption that overflows the frame is worse than one that wraps.
    """
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.getbbox(candidate)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_caption(
    text: str,
    width: int,
    height: int,
    style: CaptionStyle | None = None,
) -> bytes:
    """Render ``text`` as a transparent PNG the size of the canvas.

    The image is full-canvas rather than cropped to the text, so compositing is
    a plain overlay at the origin and the caption's position is baked in. That
    costs a few kilobytes and removes a whole class of alignment bug.

    Args:
        text: The caption. Wrapped to fit; empty text yields a blank image.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        style: Appearance. Defaults to white on a semi-opaque plate.

    Returns:
        PNG bytes with an alpha channel.

    Raises:
        VisionUseError: If the canvas is degenerate or the font cannot load.

    """
    if width < 1 or height < 1:
        raise VisionUseError(
            f"Cannot render a caption onto a {width}x{height} canvas",
            remedy="check the plan's timeline dimensions",
        )

    chosen = style or CaptionStyle()
    font = _load_font(chosen, height)
    lines = wrap(text, font, int(width * chosen.max_width_scale))

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if not lines:
        return _encode(canvas)

    draw = ImageDraw.Draw(canvas)
    line_height = _line_height(font)
    pad = int(line_height * _PAD_SCALE)

    block_height = line_height * len(lines)
    bottom = height - int(height * chosen.margin_scale)
    top = bottom - block_height

    widest = max(font.getbbox(line)[2] for line in lines)
    plate_left = (width - widest) // 2 - pad
    plate_right = (width + widest) // 2 + pad
    draw.rectangle(
        (plate_left, top - pad, plate_right, bottom + pad),
        fill=chosen.plate,
    )

    for index, line in enumerate(lines):
        line_width = font.getbbox(line)[2]
        draw.text(
            ((width - line_width) // 2, top + index * line_height),
            line,
            font=font,
            fill=chosen.fill,
        )

    log.debug("rendered caption", extra=bind(lines=len(lines), size=f"{width}x{height}"))
    return _encode(canvas)


def _line_height(font: _Font) -> int:
    """Return the vertical advance for one line of ``font``.

    `load_default` and `truetype` return different classes; only the latter
    exposes `getmetrics`, so the bitmap fallback is measured instead.
    """
    if isinstance(font, ImageFont.FreeTypeFont):
        ascent, descent = font.getmetrics()
        return int(ascent + descent)
    bbox = font.getbbox("Ag")
    return int(bbox[3] - bbox[1]) + 2


def _encode(image: Image.Image) -> bytes:
    """Encode an image as a deterministic PNG."""
    buffer = io.BytesIO()
    # optimize=False and a fixed compress_level keep the bytes reproducible;
    # the optimiser's choices have varied between Pillow releases.
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()
