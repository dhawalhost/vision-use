"""Compiling keyframes into ffmpeg time expressions.

An animated parameter is a piecewise-linear function of clip-local time, and
several ffmpeg filters accept an expression rather than a number for exactly
that. `crop` takes one for `x` and `y`, so a moving crop costs nothing beyond
the single pass the render already makes.

**Clip-local time comes free.** The compiler emits `setpts=PTS-STARTPTS` before
any effect, so `t` inside a clip's chain is already measured from the clip's
own start -- the same base `PLAN.md` § D6 chose for keyframes. The two agree by
construction rather than by conversion.

The generated expression is nested `if()`s: held flat before the first keyframe,
linear between each pair, held flat after the last. That mirrors
`Effect.value_at` exactly, which matters because the baked path uses that method
and the two must not disagree about what the animation means.
"""

from __future__ import annotations

from vision_use.errors import VisionUseError
from vision_use.models.plan import Keyframe

#: Decimal places kept in a generated expression. Enough for sub-pixel
#: positioning, short enough that a filtergraph stays readable when something
#: goes wrong and a human has to look at it.
_PRECISION = 4


def _number(value: float) -> str:
    """Render a number for an ffmpeg expression, without scientific notation.

    `1e-05` is valid Python and not valid in a filtergraph, so formatting is
    fixed-point deliberately.
    """
    return f"{value:.{_PRECISION}f}".rstrip("0").rstrip(".") or "0"


def time_expression(frames: tuple[Keyframe, ...]) -> str:
    """Return an ffmpeg expression giving a parameter's value at time ``t``.

    Args:
        frames: Keyframes in time order, as validated by :class:`Effect`.

    Returns:
        An expression usable wherever a filter accepts one, e.g. ``crop``'s
        ``x``. Held flat outside the keyframes, linear between them.

    Raises:
        VisionUseError: If there are no keyframes. A caller reaching here with
            an empty track has skipped validation the model already does.

    """
    if not frames:
        raise VisionUseError(
            "Cannot build a time expression from no keyframes",
            remedy="give the parameter at least one keyframe, or do not animate it",
        )

    if len(frames) == 1:
        # A single keyframe is a constant, and emitting `if(lt(t,T),V,V)` would
        # only make a filtergraph harder to read for no behavioural difference.
        return _number(frames[0].value)

    # Built from the last segment backwards, so each `if` wraps the rest as its
    # else branch and the whole thing nests without a second pass.
    expression = _number(frames[-1].value)
    for earlier, later in zip(reversed(frames[:-1]), reversed(frames[1:]), strict=True):
        span = later.t_s - earlier.t_s
        if span <= 0:  # pragma: no cover - repeated times are rejected by the model
            continue
        slope = (later.value - earlier.value) / span
        segment = (
            f"{_number(earlier.value)}+{_number(slope)}*(t-{_number(earlier.t_s)})"
            if slope >= 0
            else f"{_number(earlier.value)}-{_number(abs(slope))}*(t-{_number(earlier.t_s)})"
        )
        expression = f"if(lt(t,{_number(later.t_s)}),{segment},{expression})"

    # Before the first keyframe the first value holds. Without this, `t` below
    # the first keyframe would extrapolate the first segment backwards, which
    # is a different -- and wrong -- answer.
    first = frames[0]
    return f"if(lt(t,{_number(first.t_s)}),{_number(first.value)},{expression})"
