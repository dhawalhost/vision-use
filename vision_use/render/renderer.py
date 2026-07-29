"""Running a compiled plan through ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vision_use.cache.store import CacheStore
from vision_use.effects.registry import EffectRegistry
from vision_use.errors import DependencyError, VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.plan import EditPlan, RenderProfile
from vision_use.progress import ProgressCallback
from vision_use.render.bake import bake_plan
from vision_use.render.compiler import CompiledRender, compile_plan

log = get_logger(__name__)

#: Longest a single render may run before it is abandoned. A hung ffmpeg is
#: otherwise indistinguishable from a slow one, and the retry loop would wait
#: forever rather than trying the next strategy.
DEFAULT_TIMEOUT_S: Final = 3600


@dataclass(frozen=True, slots=True)
class RenderResult:
    """What a completed render produced."""

    path: Path
    width: int
    height: int
    duration_s: float
    profile: str


def render(
    plan: EditPlan,
    output: Path,
    *,
    proxy: bool = False,
    effects: EffectRegistry | None = None,
    cache: CacheStore | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    on_progress: ProgressCallback | None = None,
) -> RenderResult:
    """Render ``plan`` to ``output``.

    Args:
        plan: The plan to render.
        output: Destination file.
        proxy: Render at proxy scale rather than master. The agent loop uses
            this; export does not.
        effects: Effect registry.
        cache: Store for baked intermediates.
        timeout_s: Abandon the render after this long.
        on_progress: Called as baking proceeds and when the encode starts.
            ffmpeg's own progress is not parsed: the encode is the fast part,
            and baking is where the minutes go.

    Returns:
        A description of what was written.

    Raises:
        DependencyError: If ffmpeg is not installed.
        VisionUseError: If the plan cannot be compiled, or ffmpeg fails.

    """
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise DependencyError(
            "Rendering requires ffmpeg, which is not on PATH",
            remedy="install ffmpeg (`brew install ffmpeg`, `apt install ffmpeg`)",
        )

    # Effects a filtergraph cannot express are rendered to cached intermediates
    # first; the compiler then sees ordinary inputs and needs no special case.
    prepared = bake_plan(plan, cache=cache, effects=effects, on_progress=on_progress)
    if prepared.baked:
        log.info(
            "resolved baked effects",
            extra=bind(clips=prepared.baked, from_cache=prepared.reused),
        )

    profile: RenderProfile = plan.render.proxy if proxy else plan.render.master
    compiled: CompiledRender = compile_plan(
        prepared.plan, output, profile=profile, effects=effects
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "rendering",
        extra=bind(
            output=str(output),
            profile="proxy" if proxy else "master",
            size=f"{compiled.width}x{compiled.height}",
            segments=compiled.segments,
        ),
    )

    try:
        completed = subprocess.run(
            [binary, *compiled.args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VisionUseError(
            f"Rendering exceeded {timeout_s}s and was abandoned",
            remedy="raise timeout_s, or render at proxy scale while iterating",
        ) from exc
    except OSError as exc:
        raise VisionUseError(
            "Could not start ffmpeg",
            remedy="check that ffmpeg is executable (`vision doctor`)",
        ) from exc

    if completed.returncode != 0:
        # ffmpeg's own last line names the actual problem; the whole log is
        # pages of stream metadata that buries it.
        detail = (completed.stderr or "").strip().splitlines()
        raise VisionUseError(
            f"ffmpeg failed: {detail[-1] if detail else 'no error output'}",
            remedy="run with --verbose to see the full filtergraph",
        )

    return RenderResult(
        path=output,
        width=compiled.width,
        height=compiled.height,
        duration_s=compiled.duration_s,
        profile="proxy" if proxy else "master",
    )
