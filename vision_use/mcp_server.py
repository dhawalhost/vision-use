"""MCP server: VisionUse as a tool an agent can call.

**Five generic tools, not one per skill.** ``VISION.md`` sketched
``stabilize_video``, ``score_video`` and so on, which grows one tool per
capability and reaches forty as the ecosystem fills in. An agent choosing
between forty tools chooses worse than one choosing between five and then
naming a skill, and every new plugin would need the server to change. A
registry behind ``list_skills``/``apply_skill`` does not.

**Tools return summaries and paths, never whole documents.** A ``VideoContext``
for a ten-minute video does not belong in a model's context window, and a plan
is meant to be handed to ``render``, not read aloud. Every tool here returns
what an agent needs to decide the next call, plus the path to the artefact it
produced.

Optional: the MCP SDK lives behind the ``mcp`` extra, since the library and CLI
need nothing from it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.context import Context
else:
    # `Context` is imported at *runtime*, not just for typing: the SDK reads a
    # tool's annotations with `eval_str=True` to build its schema, so a name
    # that exists only under TYPE_CHECKING makes registration fail. Guarded
    # because the SDK is an optional extra -- without it this module still
    # imports, and `build_server` is what refuses with an actionable message.
    #
    # The runtime fallback lives in this `else`, not a bare `try/except`,
    # deliberately: mypy only ever statically analyses the `if TYPE_CHECKING`
    # branch above, so it never sees this assignment and never has to
    # reconcile its type against the try-import's. A `# type: ignore` on that
    # assignment used to be needed in one installed configuration (`mcp`
    # present) and flagged as unused in the other (`mcp` absent, which is
    # exactly what CI's lint job runs) -- an ignore whose correctness depends
    # on which optional extra happens to be installed is not a real fix.
    try:
        from mcp.server.mcpserver.context import Context
    except ImportError:
        Context = Any

from vision_use.cache.store import CacheStore
from vision_use.context.registry import resolve_section_names
from vision_use.context.runner import analyze
from vision_use.errors import DependencyError, VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.context import ContextSection, SectionName, VideoContext
from vision_use.models.plan import EditPlan
from vision_use.planning import load_plan, save_plan, straight_through, validate_plan
from vision_use.plugins import ensure_loaded
from vision_use.progress import Progress
from vision_use.render import RenderResult
from vision_use.render import render as render_plan
from vision_use.skills.pipeline import RetryPolicy, run_skill
from vision_use.skills.registry import registry as skill_registry
from vision_use.skills.skill import SkillRequest

log = get_logger(__name__)

#: Sections analysed when a caller does not say, matching the CLI's default:
#: transcript is excluded because it needs the `asr` extra and is by far the
#: slowest.
DEFAULT_SECTIONS: tuple[ContextSection, ...] = (
    ContextSection.SHOTS,
    ContextSection.KEYFRAMES,
    ContextSection.AUDIO_ACTIVITY,
)

#: Words returned per `query_context` match, so a transcript search answers with
#: usable quotations rather than the whole script.
_MATCH_LIMIT = 20


def _sections(names: list[str] | None) -> list[SectionName]:
    """Resolve section names, defaulting to the cheap ones.

    A plugin's section (`"stabilize.motion"`) resolves here too, so an agent can
    ask for one without the server knowing it exists.

    Raises:
        VisionUseError: If a name is not a known section.

    """
    if not names:
        return list(DEFAULT_SECTIONS)
    return resolve_section_names(names)


def analyze_tool(source: str, sections: list[str] | None = None) -> dict[str, Any]:
    """Analyse a video into the shared context.

    Args:
        source: Path to the media file.
        sections: Which sections to populate. Defaults to shots, keyframes and
            audio activity. Add "transcript" for speech, which is slower.

    Returns:
        A summary of what was found, plus the path the context was written to.

    """
    path = Path(source)
    context = analyze(path, _sections(sections), cache=CacheStore())

    target = path.with_suffix(".context.json")
    target.write_text(context.model_dump_json(indent=2), encoding="utf-8")

    return {
        "context_path": str(target),
        "duration_s": round(context.duration_s, 3),
        "resolution": f"{context.probe.video.display_width}x"
        f"{context.probe.video.display_height}",
        "frame_rate": str(context.probe.video.frame_rate),
        "has_audio": context.probe.has_audio,
        "shots": len(context.shots),
        "keyframes": len(context.keyframes),
        "silent_spans": sum(1 for span in context.audio_activity if span.is_silence),
        "words": len(context.transcript.words) if context.transcript else 0,
        # Named, not dumped: this is how an agent learns a plugin contributed
        # something it can then ask `query_context` about.
        "sections": sorted(context.extra),
        "analyzers": sorted(context.analyzers),
    }


def query_context_tool(
    context_path: str,
    find: str | None = None,
    at_s: float | None = None,
    section: str | None = None,
) -> dict[str, Any]:
    """Ask a question of an analysed context without loading the whole thing.

    Args:
        context_path: Path written by `analyze`.
        find: Text to locate in the transcript. Returns the matching words with
            their exact timings, which is what a cut needs.
        at_s: A moment to describe -- which shot covers it, and what is said.
        section: A plugin's section to read, by its dotted name
            (`"stabilize.motion"`). `sections` in the reply lists what this
            context carries, so an agent can discover one it has never seen.

    Returns:
        Only what was asked for.

    """
    context = VideoContext.model_validate_json(Path(context_path).read_text(encoding="utf-8"))
    answer: dict[str, Any] = {"duration_s": round(context.duration_s, 3)}

    # Always listed, never dumped: naming the sections is cheap and is how an
    # agent finds out a plugin contributed something. The payload itself is
    # returned only when asked for, since a plugin's section can be large.
    if context.extra:
        answer["sections"] = sorted(context.extra)

    if section is not None:
        if not context.has(section):
            raise VisionUseError(
                f"This context has no section {section!r}",
                remedy=(
                    "analyse with that section first"
                    + (
                        f"; it carries: {', '.join(sorted(context.extra))}"
                        if context.extra
                        else ""
                    )
                ),
            )
        answer["section"] = {"name": section, "value": context.section(section)}

    if find:
        answer["matches"] = [
            {
                "start_s": match[0].span.start_s,
                "end_s": match[-1].span.end_s,
                "word_ids": [word.id for word in match],
                "text": " ".join(word.text for word in match),
            }
            for match in list(context.find_text(find))[:_MATCH_LIMIT]
        ]

    if at_s is not None:
        shot = context.shot_at(at_s)
        answer["at"] = {
            "t_s": at_s,
            "shot_id": shot.id if shot else None,
            "shot_span": [shot.span.start_s, shot.span.end_s] if shot else None,
            "said": context.text_between(at_s, at_s + 5.0) or None,
        }

    return answer


def list_skills_tool() -> dict[str, Any]:
    """List the skills this installation can run, and what each needs."""
    return {
        "skills": [
            {
                "name": skill.meta.name,
                "version": skill.meta.version,
                "summary": skill.meta.summary,
                "requires": sorted(str(item) for item in skill.meta.requires),
            }
            for skill in skill_registry.all()
        ]
    }


def plan_tool(source: str, output: str | None = None) -> dict[str, Any]:
    """Create a straight-through plan: the identity edit, ready to patch.

    Args:
        source: Path to the media file.
        output: Where to write the plan. Defaults beside the source.

    Returns:
        The plan's path and shape.

    """
    path = Path(source)
    plan = straight_through(path)
    target = Path(output) if output else path.with_suffix(".plan.json")
    save_plan(plan, target)

    return {
        "plan_path": str(target),
        "duration_s": round(plan.duration_s, 3),
        "canvas": f"{plan.timeline.width}x{plan.timeline.height}",
        "clips": len(plan.timeline.clips()),
    }


def apply_skill_tool(
    plan_path: str,
    skill: str,
    prompt: str = "",
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Run a skill against a plan, editing it in place.

    Nothing is rendered: the result is an edited plan, which `render` turns into
    a file when the agent is ready.

    Args:
        plan_path: Path to the plan JSON.
        skill: Skill name, from `list_skills`.
        prompt: What the user asked for, in their own words.
        max_attempts: How many strategies the skill may try.

    Returns:
        What the skill did, including every strategy it tried and why any were
        rejected -- an agent needs the rejections to decide what to try next.

    """
    target = Path(plan_path)
    plan = load_plan(target)
    chosen = skill_registry.get(skill)

    source = next((item.path for item in plan.sources.values()), None)
    if source is None:
        raise VisionUseError(
            "This plan declares no sources, so there is nothing to analyse",
            remedy="build a plan with the `plan` tool first",
        )

    outcome = run_skill(
        chosen,
        source,
        plan,
        SkillRequest(prompt=prompt),
        cache=CacheStore(),
        policy=RetryPolicy(max_attempts=max_attempts),
    )

    if outcome.accepted is not None:
        save_plan(outcome.plan, target)

    return {
        "plan_path": str(target),
        "accepted": outcome.accepted is not None,
        "rationale": outcome.accepted.patch.rationale if outcome.accepted else None,
        "duration_s": round(outcome.plan.duration_s, 3),
        "attempts": [
            {
                "strategy": attempt.strategy.name,
                "passed": attempt.passed,
                "failures": [item.name for item in attempt.report.failures],
            }
            for attempt in outcome.attempts
        ],
        "written": outcome.accepted is not None,
    }


async def render_tool(
    plan_path: str,
    output: str | None = None,
    proxy: bool = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Render a plan to a video file, reporting progress as it goes.

    Args:
        plan_path: Path to the plan JSON.
        output: Where to write the video. Defaults beside the plan.
        proxy: Render at reduced scale, which is much faster and is what an
            agent should use while iterating.
        ctx: Injected by the server. Used to emit protocol-level progress, so
            an agent sees a long bake advancing rather than a silent call.

    Returns:
        What was written.

    Raises:
        VisionUseError: If validation finds something that would spoil the
            render -- reported before ffmpeg is invoked, not after.

    """
    source = Path(plan_path)
    plan = load_plan(source)

    report = validate_plan(plan)
    if not report.ok:
        raise VisionUseError(
            "This plan has problems that would spoil the render: "
            + "; ".join(item.message for item in report.errors),
            remedy="fix them, or edit the plan and try again",
        )

    target = Path(output) if output else source.with_suffix(".mp4")
    result = await _render_off_the_loop(plan, target, proxy=proxy, ctx=ctx)

    return {
        "output_path": str(result.path),
        "resolution": f"{result.width}x{result.height}",
        "duration_s": round(result.duration_s, 3),
        "profile": result.profile,
        "warnings": [item.message for item in report.warnings],
    }


#: Shortest gap between progress notifications, in seconds. Baking reports every
#: frame -- twenty-five a second at PAL -- and forwarding all of them would spend
#: more of the transport on progress than on the work.
_PROGRESS_INTERVAL_S = 0.5


async def _render_off_the_loop(
    plan: EditPlan, target: Path, *, proxy: bool, ctx: Context | None
) -> RenderResult:
    """Run the render in a worker thread, forwarding progress as it arrives.

    **In a thread because rendering is blocking work.** Baking decodes frames
    and ffmpeg is a subprocess; running either on the event loop stalls the whole
    server, so an agent could not so much as cancel until the render finished.

    Progress crosses back with `call_soon_threadsafe`, which is the only safe way
    to touch an asyncio primitive from another thread.
    """
    loop = asyncio.get_running_loop()
    updates: asyncio.Queue[Progress | None] = asyncio.Queue()

    def on_progress(update: Progress) -> None:
        """Hand one update back to the loop. Runs on the render thread."""
        loop.call_soon_threadsafe(updates.put_nowait, update)

    async def forward() -> None:
        """Drain updates and report at most one per interval."""
        last = 0.0
        while True:
            update = await updates.get()
            if update is None:
                return
            now = loop.time()
            if ctx is None or now - last < _PROGRESS_INTERVAL_S:
                continue
            last = now
            # `total` may be None on a source whose frame count is unknown;
            # the protocol allows that and a client shows a spinner instead.
            await ctx.report_progress(
                float(update.done),
                float(update.total) if update.total else None,
                update.detail or str(update.stage),
            )

    pump = asyncio.create_task(forward())
    try:
        return await asyncio.to_thread(
            render_plan, plan, target, proxy=proxy, cache=CacheStore(), on_progress=on_progress
        )
    finally:
        # Always stops the pump, including when the render raised.
        loop.call_soon_threadsafe(updates.put_nowait, None)
        await pump


#: The tools, in the order an agent would normally reach for them.
TOOLS = (
    analyze_tool,
    query_context_tool,
    list_skills_tool,
    plan_tool,
    apply_skill_tool,
    render_tool,
)


def build_server(name: str = "vision-use") -> MCPServer:
    """Construct the MCP server with every tool registered.

    Raises:
        DependencyError: If the ``mcp`` extra is not installed.

    """
    try:
        # Imported lazily: the library and CLI need nothing from the MCP SDK,
        # and a top-level import would make the whole package depend on it.
        from mcp.server.mcpserver import MCPServer  # noqa: PLC0415
    except ImportError as exc:
        raise DependencyError(
            "The MCP server requires the 'mcp' extra, which is not installed",
            remedy="install it with `uv sync --extra mcp`",
        ) from exc

    # Same reasoning as the CLI: discovery is explicit, at the entry point.
    ensure_loaded()

    server = MCPServer(
        name=name,
        instructions=(
            "Analyse and edit video. Typical flow: `analyze` a source, `plan` it, "
            "`apply_skill` one or more times, then `render` -- with proxy=true "
            "while iterating. Nothing is encoded until you call `render`."
        ),
    )

    for tool in TOOLS:
        server.add_tool(tool, name=tool.__name__.removesuffix("_tool"))

    log.info("built MCP server", extra=bind(tools=len(TOOLS)))
    return server


def main() -> None:  # pragma: no cover - process entry point
    """Run the server on stdio."""
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
