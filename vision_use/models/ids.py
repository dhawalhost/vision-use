"""Stable identifiers for context entities.

IDs are how an agent addresses part of a video: "trim to ``sh_0003``", "cut from
``w_000412``". For that to work across runs, across skills, and across a stored
context reloaded a week later, the ID assigned to a given entity must depend
only on that entity's position in chronological order -- never on iteration
order, dictionary ordering, or a random suffix.

The zero-padded numeric form is deliberate: it sorts lexicographically into
chronological order, which makes a serialized context readable and diffable.
"""

from __future__ import annotations

import re
from typing import Final

#: Digits in the numeric part of each ID kind. Words get more because a
#: one-hour talk contains roughly ten thousand of them.
_WIDTH_SHORT: Final = 4
_WIDTH_LONG: Final = 6

SHOT_PREFIX: Final = "sh"
KEYFRAME_PREFIX: Final = "kf"
WORD_PREFIX: Final = "w"
SEGMENT_PREFIX: Final = "sg"
SPEAKER_PREFIX: Final = "spk"

#: Edit-plan entities. Kept in the same scheme as context entities so one
#: regular expression validates any ID the framework produces, but numbered
#: independently -- a plan's clips have nothing to do with a context's shots.
SOURCE_PREFIX: Final = "src"
TRACK_PREFIX: Final = "trk"
CLIP_PREFIX: Final = "clp"

#: Matches any well-formed VisionUse entity ID.
ID_PATTERN: Final = re.compile(r"^(sh|kf|w|sg|spk|src|trk|clp)_\d+$")


def _format(prefix: str, index: int, width: int) -> str:
    """Return ``prefix_0001`` for ``index``, widening past ``width`` if needed.

    Raises:
        ValueError: If ``index`` is negative.

    """
    if index < 0:
        raise ValueError(f"entity index must not be negative, got {index}")
    return f"{prefix}_{index:0{width}d}"


def shot_id(index: int) -> str:
    """Return the ID of the shot at chronological position ``index``."""
    return _format(SHOT_PREFIX, index, _WIDTH_SHORT)


def keyframe_id(index: int) -> str:
    """Return the ID of the keyframe at chronological position ``index``."""
    return _format(KEYFRAME_PREFIX, index, _WIDTH_SHORT)


def word_id(index: int) -> str:
    """Return the ID of the transcript word at chronological position ``index``."""
    return _format(WORD_PREFIX, index, _WIDTH_LONG)


def segment_id(index: int) -> str:
    """Return the ID of the transcript segment at chronological position ``index``."""
    return _format(SEGMENT_PREFIX, index, _WIDTH_SHORT)


def speaker_id(index: int) -> str:
    """Return the ID of the speaker at first-appearance position ``index``.

    Speakers are numbered by when they first speak rather than by the label the
    diarizer happened to emit, which is not stable between runs.
    """
    return _format(SPEAKER_PREFIX, index, _WIDTH_SHORT)


def is_valid(value: str) -> bool:
    """Whether ``value`` is a well-formed entity ID."""
    return ID_PATTERN.match(value) is not None


def source_id(index: int) -> str:
    """Return the ID of the source at declaration position ``index``."""
    return _format(SOURCE_PREFIX, index, _WIDTH_SHORT)


def track_id(index: int) -> str:
    """Return the ID of the track at position ``index``."""
    return _format(TRACK_PREFIX, index, _WIDTH_SHORT)


def clip_id(index: int) -> str:
    """Return the ID of the clip at position ``index`` within its track."""
    return _format(CLIP_PREFIX, index, _WIDTH_SHORT)
