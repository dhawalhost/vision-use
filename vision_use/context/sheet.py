"""Building a contact sheet from an analysed context.

Bridges :mod:`vision_use.video.contact_sheet`, which knows about pixels, and
:class:`~vision_use.models.context.VideoContext`, which knows about shots. The
result is one image plus a manifest that maps each numbered tile back to the
shot and timestamp it came from -- so a model's answer about "tile 3" can be
turned into a cut.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from vision_use.cache.store import CacheStore
from vision_use.errors import VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.context import ContextSection, VideoContext
from vision_use.video.contact_sheet import (
    DEFAULT_COLUMNS,
    DEFAULT_TILE_WIDTH,
    SHEET_VERSION,
    build_contact_sheet,
    decode_jpeg,
)

log = get_logger(__name__)


class SheetTile(BaseModel):
    """One tile of a contact sheet, and what it came from."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Tile number, as burned into the image.")
    shot_id: str
    keyframe_id: str
    t_s: float = Field(ge=0)


class ContactSheet(BaseModel):
    """A rendered contact sheet and the manifest a model needs to use it."""

    model_config = ConfigDict(frozen=True)

    path: Path = Field(description="Where the sheet image was written.")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    columns: int = Field(gt=0)
    tiles: tuple[SheetTile, ...] = ()

    def describe(self) -> str:
        """Return a text index pairing tile numbers with shots and timestamps.

        Handed to a model alongside the image. Without it the model can see the
        frames but cannot name what it is looking at in terms the framework
        understands.
        """
        lines = [
            f"Contact sheet: {len(self.tiles)} tiles, {self.columns} per row, "
            "numbered left to right, top to bottom."
        ]
        lines.extend(
            f"  tile {tile.index}: {tile.shot_id} at {tile.t_s:.2f}s" for tile in self.tiles
        )
        return "\n".join(lines)


def build_sheet(
    context: VideoContext,
    output: Path,
    *,
    cache: CacheStore | None = None,
    columns: int = DEFAULT_COLUMNS,
    tile_width: int = DEFAULT_TILE_WIDTH,
) -> ContactSheet:
    """Render ``context``'s keyframes into a labelled contact sheet.

    Args:
        context: An analysed context carrying keyframes.
        output: Path to write the sheet image to.
        cache: Store holding the keyframe images. Defaults to the standard root.
        columns: Tiles per row.
        tile_width: Tile width in pixels.

    Returns:
        The rendered sheet and its manifest.

    Raises:
        VisionUseError: If the context has no keyframes, or a keyframe's cached
            image is missing.

    """
    if not context.has(ContextSection.KEYFRAMES) or not context.keyframes:
        raise VisionUseError(
            "This context has no keyframes to tile",
            remedy="run `vision analyze --with keyframes` first",
        )

    store = cache or CacheStore()
    images: list[tuple[str, np.ndarray]] = []
    tiles: list[SheetTile] = []

    for index, keyframe in enumerate(context.keyframes):
        path = _locate(store, keyframe.cache_key)
        if path is None:
            raise VisionUseError(
                f"Cached image for {keyframe.id} is missing",
                remedy="re-run `vision analyze --with keyframes --no-cache`",
            )
        images.append((str(index), decode_jpeg(path.read_bytes())))
        tiles.append(
            SheetTile(
                index=index,
                shot_id=keyframe.shot_id or "",
                keyframe_id=keyframe.id,
                t_s=keyframe.t_s,
            )
        )

    payload, width, height = build_contact_sheet(images, columns=columns, tile_width=tile_width)

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    except OSError as exc:
        raise VisionUseError(
            f"Could not write the contact sheet to {output}",
            remedy="check the directory exists and is writable",
        ) from exc

    log.info(
        "wrote contact sheet",
        extra=bind(path=str(output), tiles=len(tiles), size=f"{width}x{height}"),
    )
    return ContactSheet(
        path=output,
        width=width,
        height=height,
        columns=min(columns, len(tiles)),
        tiles=tuple(tiles),
    )


def _locate(store: CacheStore, digest: str) -> Path | None:
    """Find a cached keyframe image by its digest.

    The keyframe records only the digest, not the full key, so the namespace
    directory is searched. Entries are content-addressed, so at most one file
    can match.
    """
    candidates = list((store.root / "keyframe.jpeg").rglob(f"{digest}.jpg"))
    return candidates[0] if candidates else None


#: Cache namespace for rendered sheets, exposed so callers can clear just these.
SHEET_NAMESPACE = f"contact_sheet.{SHEET_VERSION}"
