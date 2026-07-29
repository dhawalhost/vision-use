"""Time and frame conversion.

Two distinct problems live here.

**Quantisation.** ``VideoContext`` must serialise identically on every run --
that is M1's exit criterion and the precondition for content addressing. Raw
analyzer output is full-precision binary floating point whose last digits vary
with library version and CPU, so every time written into a context is rounded
to a fixed decimal precision first.

**Frame mapping.** Converting between seconds and frame indices must use the
exact rational frame rate, never its float approximation. At 30000/1001 the
float error reaches a whole frame after roughly 30 minutes, which is the
difference between a cut landing on the intended frame and one frame late.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from typing import Final

from vision_use.models.media import FrameRate

#: Decimal places retained for any time stored in a context or plan.
#: Microsecond resolution: finer than any container timebase in practice, and
#: far finer than a single frame at any realistic rate.
TIME_PRECISION: Final = 6

_QUANTUM: Final = Decimal(1).scaleb(-TIME_PRECISION)

#: Half of one quantisation step, as an exact rational.
#:
#: A time that has been through :func:`quantize_s` may sit up to this far below
#: the frame boundary it was derived from. Flooring such a value would name the
#: *previous* frame, so every seconds-to-frame conversion nudges by this amount
#: first. Without it, ``frame_index_at(frame_start_s(n)) == n`` fails for any
#: rate whose frame duration is not an exact multiple of a microsecond -- which
#: includes every NTSC rate, precisely the case this module exists to get right.
#:
#: The nudge is far smaller than a frame (1.5e-5 frames at 30 fps), so it can
#: never move a genuinely mid-frame timestamp into the next frame.
_TIME_TOLERANCE: Final = Fraction(1, 2 * 10**TIME_PRECISION)


def quantize_s(seconds: float) -> float:
    """Round ``seconds`` to :data:`TIME_PRECISION` decimal places.

    Banker's rounding is used so that repeatedly quantising an already
    quantised value is a no-op and so that rounding bias does not accumulate
    across a long list of timestamps.

    Every timestamp entering a serialized document passes through here.
    """
    return float(Decimal(seconds).quantize(_QUANTUM, rounding=ROUND_HALF_EVEN))


def frame_index_at(seconds: float, rate: FrameRate) -> int:
    """Return the index of the frame displayed at ``seconds``.

    Uses exact rational arithmetic, then floors: the frame displayed at time
    *t* is the last frame whose presentation time does not exceed *t*.
    ``seconds`` is first nudged by :data:`_TIME_TOLERANCE` so that a quantised
    frame boundary maps back to its own frame rather than the previous one.

    Raises:
        ValueError: If ``seconds`` is negative.

    """
    if seconds < 0:
        raise ValueError(f"time must not be negative, got {seconds}")
    return math.floor((Fraction(seconds) + _TIME_TOLERANCE) * rate.fraction)


def frame_start_s(index: int, rate: FrameRate) -> float:
    """Return the exact presentation time of frame ``index``, quantised.

    Raises:
        ValueError: If ``index`` is negative.

    """
    if index < 0:
        raise ValueError(f"frame index must not be negative, got {index}")
    return quantize_s(float(index * rate.frame_duration_s))


def snap_to_frame(seconds: float, rate: FrameRate) -> float:
    """Return the presentation time of the frame displayed at ``seconds``.

    Used when a time derived from audio -- a word boundary, a silence edge --
    must become a video cut point. Without snapping, a cut falls mid-frame and
    the encoder resolves it arbitrarily.
    """
    return frame_start_s(frame_index_at(seconds, rate), rate)


def duration_frames(seconds: float, rate: FrameRate) -> int:
    """Return how many whole frames fit in ``seconds``.

    Carries the same quantisation tolerance as :func:`frame_index_at`, so a
    duration read back from a serialized document yields the frame count it
    was built from.
    """
    if seconds < 0:
        raise ValueError(f"duration must not be negative, got {seconds}")
    return math.floor((Fraction(seconds) + _TIME_TOLERANCE) * rate.fraction)
