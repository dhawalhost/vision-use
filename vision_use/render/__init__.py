"""Turning an ``EditPlan`` into a file.

The renderer is the only component in VisionUse that encodes, and it runs
exactly once, at export. Everything upstream edits a document.

Effects a filtergraph cannot express are *baked* first -- rendered to a cached
intermediate that the compiler then treats as an ordinary input, so the plan
model needs no special case for them.
"""

from vision_use.render.bake import BakedPlan, bake_plan
from vision_use.render.compiler import CompiledRender, compile_plan
from vision_use.render.renderer import RenderResult, render

__all__ = [
    "BakedPlan",
    "CompiledRender",
    "RenderResult",
    "bake_plan",
    "compile_plan",
    "render",
]
