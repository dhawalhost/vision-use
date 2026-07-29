"""Tests for compiling keyframes into ffmpeg time expressions.

The property that matters most is agreement: the filtergraph path and
`Effect.value_at` must mean the same thing by the animation, or a clip renders
differently depending on whether its effect happened to bake.
"""

from __future__ import annotations

import pytest
from vision_use.effects.animation import time_expression
from vision_use.effects.builtin import CropEffect, CropParams
from vision_use.errors import VisionUseError
from vision_use.models.plan import Effect, Keyframe


def _frames(*pairs: tuple[float, float]) -> tuple[Keyframe, ...]:
    return tuple(Keyframe(t_s=t, value=v) for t, v in pairs)


def _evaluate(expression: str, t: float) -> float:
    """Evaluate an ffmpeg expression the way ffmpeg would.

    A small `if(lt(a,b),x,y)` interpreter: enough for what this module emits,
    and it means the tests check the *expression*, not a Python reimplementation
    of the intent behind it.
    """
    text = expression
    while "if(lt(" in text:
        start = text.rindex("if(lt(")
        depth, index = 0, start + 2
        while True:
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    break
            index += 1

        condition, rest = text[start + 6 : index].split(")", 1)
        left, right = condition.split(",")

        depth, parts, current = 0, [], ""
        for char in rest[1:]:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if char == "," and depth == 0:
                parts.append(current)
                current = ""
                continue
            current += char
        parts.append(current)

        branch = f"(({parts[0]}) if ({left})<({right}) else ({parts[1]}))"
        text = f"{text[:start]}{branch}{text[index + 1 :]}"
    return float(eval(text, {"__builtins__": {}}, {"t": t}))


# --- the expression --------------------------------------------------------


def test_a_single_keyframe_is_a_constant() -> None:
    """Wrapping it in an `if` would only make a filtergraph harder to read."""
    assert time_expression(_frames((1.0, 42.0))) == "42"


def test_no_keyframes_is_refused() -> None:
    with pytest.raises(VisionUseError, match="no keyframes"):
        time_expression(())


def test_the_expression_has_no_scientific_notation() -> None:
    """`1e-05` is valid Python and not valid in a filtergraph."""
    assert "e-" not in time_expression(_frames((0.0, 0.00001), (1.0, 2.0)))


@pytest.mark.parametrize("t", [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 9.0])
def test_the_expression_agrees_with_value_at(t: float) -> None:
    """The filtergraph and the baked path must not disagree about the animation."""
    frames = _frames((0.0, 10.0), (2.0, 110.0), (4.0, 60.0))
    effect = Effect(type="crop", params={"x": 10}, animate={"x": frames})

    expected = effect.value_at("x", t)
    assert expected is not None
    assert _evaluate(time_expression(frames), t) == pytest.approx(expected, abs=1e-4)


def test_a_falling_segment_is_expressed() -> None:
    """A negative slope must not produce `+-5*(t-...)`."""
    expression = time_expression(_frames((0.0, 100.0), (2.0, 0.0)))

    assert "+-" not in expression
    assert _evaluate(expression, 1.0) == pytest.approx(50.0)


# --- crop ------------------------------------------------------------------


def test_crop_renders_an_animated_offset() -> None:
    filtergraph = CropEffect().compile_animated_filter(
        CropParams(x=0, y=0, width=100, height=50), {"x": "if(lt(t,1),0,50)"}
    )

    assert filtergraph.startswith("crop=100:50:")
    # Quoted, because an expression contains commas that would otherwise end
    # the filter's argument list.
    assert "'if(lt(t,1),0,50)'" in filtergraph


def test_crop_keeps_static_parameters_static() -> None:
    filtergraph = CropEffect().compile_animated_filter(
        CropParams(x=7, y=9, width=100, height=50), {"x": "0"}
    )

    assert filtergraph.endswith(":'0':'9'")


def test_crop_refuses_to_animate_its_size() -> None:
    """A frame that changes size mid-clip is not encodable, so say so."""
    with pytest.raises(VisionUseError, match="frame size is"):
        CropEffect().compile_animated_filter(CropParams(width=100, height=50), {"width": "100"})


def test_crop_declares_that_it_animates() -> None:
    assert CropEffect.animatable
