"""Tests for the MCP server.

The tools are a thin adapter over the library, so what is worth testing is the
*shape of the surface*: that an agent can get from a file to a rendered edit,
that nothing returns a document too large to reason about, and that failures
come back as messages an agent can act on.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from vision_use.errors import ConfigurationError, VisionUseError
from vision_use.mcp_server import (
    DEFAULT_SECTIONS,
    TOOLS,
    _sections,
    analyze_tool,
    apply_skill_tool,
    build_server,
    list_skills_tool,
    plan_tool,
    query_context_tool,
    render_tool,
)
from vision_use.models.context import AnalyzerRun, ContextSection, VideoContext
from vision_use.models.media import FrameRate, MediaProbe, SourceInfo, VideoStreamInfo

from tests.conftest import MakeClip, requires_ffmpeg

# --- the surface -----------------------------------------------------------


def test_the_server_exposes_every_tool() -> None:
    tools = asyncio.run(build_server().list_tools())

    assert {tool.name for tool in tools} == {
        "analyze",
        "query_context",
        "list_skills",
        "plan",
        "apply_skill",
        "render",
    }
    assert len(tools) == len(TOOLS)


def test_the_surface_stays_small() -> None:
    """One tool per skill would reach forty as the ecosystem fills in."""
    tools = asyncio.run(build_server().list_tools())

    assert len(tools) <= 8


def test_every_tool_is_described() -> None:
    """An agent picks a tool from its description; a blank one is unusable."""
    for tool in asyncio.run(build_server().list_tools()):
        assert tool.description


def test_tool_schemas_name_their_arguments() -> None:
    tools = {tool.name: tool for tool in asyncio.run(build_server().list_tools())}

    assert "source" in tools["analyze"].input_schema["properties"]
    assert "skill" in tools["apply_skill"].input_schema["properties"]
    assert "proxy" in tools["render"].input_schema["properties"]


def test_skills_are_listed_from_the_registry() -> None:
    """A new plugin must not require the server to change."""
    listed = list_skills_tool()["skills"]

    names = {skill["name"] for skill in listed}
    assert "edit.remove_silence" in names
    assert all("requires" in skill for skill in listed)


# --- section resolution ----------------------------------------------------


def test_sections_default_to_the_cheap_ones() -> None:
    """Transcript needs an optional extra and is far slower; opt in explicitly."""
    assert _sections(None) == list(DEFAULT_SECTIONS)
    assert ContextSection.TRANSCRIPT not in DEFAULT_SECTIONS


def test_sections_can_be_named() -> None:
    assert _sections(["shots"]) == [ContextSection.SHOTS]


def test_an_unknown_section_lists_the_valid_ones() -> None:
    with pytest.raises(VisionUseError) as excinfo:
        _sections(["bogus"])

    assert "shots" in (excinfo.value.remedy or "")


# --- the workflow ----------------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_analyze_returns_a_summary_not_the_context(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A whole context does not belong in a model's context window."""
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="mcp_a", seconds=2.0)

    result = analyze_tool(str(clip))

    assert "shots" in result
    assert "context_path" in result
    # The summary is small; the document it points at is not.
    assert len(json.dumps(result)) < 600
    assert Path(result["context_path"]).stat().st_size > len(json.dumps(result))


@pytest.mark.integration
@requires_ffmpeg
def test_plan_then_apply_then_render(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The flow an agent actually follows, end to end."""
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="mcp_flow", seconds=3.0)

    planned = plan_tool(str(clip), output=str(tmp_path / "plan.json"))
    assert planned["clips"] == 1

    applied = apply_skill_tool(planned["plan_path"], "edit.reframe", prompt="make it vertical")
    assert applied["accepted"]

    rendered = asyncio.run(
        render_tool(planned["plan_path"], output=str(tmp_path / "out.mp4"), proxy=True)
    )
    assert Path(rendered["output_path"]).exists()
    assert rendered["profile"] == "proxy"


@pytest.mark.integration
@requires_ffmpeg
def test_apply_reports_every_attempt(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An agent needs the rejections to decide what to try next."""
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="mcp_attempts", seconds=3.0)
    planned = plan_tool(str(clip), output=str(tmp_path / "plan.json"))

    applied = apply_skill_tool(planned["plan_path"], "edit.reframe")

    assert applied["attempts"]
    assert "passed" in applied["attempts"][0]


@pytest.mark.integration
@requires_ffmpeg
def test_apply_leaves_the_plan_alone_when_nothing_is_accepted(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A skill with nothing to do must not rewrite the file."""
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="mcp_noop", seconds=3.0)
    planned = plan_tool(str(clip), output=str(tmp_path / "plan.json"))
    before = Path(planned["plan_path"]).read_text()

    applied = apply_skill_tool(planned["plan_path"], "edit.disable_runts")

    assert not applied["accepted"]
    assert Path(planned["plan_path"]).read_text() == before


@pytest.mark.integration
@requires_ffmpeg
def test_query_context_finds_nothing_without_a_transcript(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="mcp_query", seconds=2.0)
    analysed = analyze_tool(str(clip))

    answer = query_context_tool(analysed["context_path"], find="anything")

    assert answer["matches"] == []


@pytest.mark.integration
@requires_ffmpeg
def test_query_context_describes_a_moment(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="mcp_at", seconds=2.0)
    analysed = analyze_tool(str(clip))

    answer = query_context_tool(analysed["context_path"], at_s=1.0)

    assert answer["at"]["t_s"] == 1.0
    assert answer["at"]["shot_id"]


@pytest.mark.integration
@requires_ffmpeg
def test_render_refuses_an_invalid_plan(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Failing before ffmpeg starts beats failing after it has spent minutes."""
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="mcp_invalid", seconds=2.0)
    planned = plan_tool(str(clip), output=str(tmp_path / "plan.json"))

    payload = json.loads(Path(planned["plan_path"]).read_text())
    key = next(iter(payload["sources"]))
    payload["sources"][key]["path"] = "/nope/gone.mp4"
    Path(planned["plan_path"]).write_text(json.dumps(payload))

    with pytest.raises(VisionUseError, match="spoil the render"):
        asyncio.run(render_tool(planned["plan_path"]))


@pytest.mark.integration
@requires_ffmpeg
def test_an_unknown_skill_lists_what_is_available(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="mcp_unknown", seconds=2.0)
    planned = plan_tool(str(clip), output=str(tmp_path / "plan.json"))

    with pytest.raises(ConfigurationError) as excinfo:
        apply_skill_tool(planned["plan_path"], "no.such.skill")

    assert "edit.remove_silence" in (excinfo.value.remedy or "")


# --- plugin sections (PLAN.md D5) ------------------------------------------


def _context_with_section(tmp_path: Path) -> str:
    """Write a context carrying a section the server has never heard of."""
    context = VideoContext(
        probe=MediaProbe(
            source=SourceInfo(
                path=tmp_path / "clip.mp4",
                sha256="e" * 64,
                size_bytes=1,
                container="mp4",
                duration_s=4.0,
            ),
            video=VideoStreamInfo(
                index=0,
                codec="h264",
                width=320,
                height=240,
                frame_rate=FrameRate(num=25, den=1),
            ),
        ),
        extra={"demo.motion": {"mean_px": 4.5, "shaky": True}},
        analyzers={
            "demo.motion": AnalyzerRun(
                name="demo.motion",
                version="1",
                params_hash="f" * 64,
                produced=("demo.motion",),
            )
        },
    )
    target = tmp_path / "with_section.context.json"
    target.write_text(context.model_dump_json(), encoding="utf-8")
    return str(target)


def test_query_lists_plugin_sections(tmp_path: Path) -> None:
    """Naming them is how an agent discovers a section it has never seen."""
    answer = query_context_tool(_context_with_section(tmp_path))

    assert answer["sections"] == ["demo.motion"]


def test_query_returns_a_plugin_section(tmp_path: Path) -> None:
    answer = query_context_tool(_context_with_section(tmp_path), section="demo.motion")

    assert answer["section"]["value"] == {"mean_px": 4.5, "shaky": True}


def test_a_missing_section_names_what_is_available(tmp_path: Path) -> None:
    with pytest.raises(VisionUseError) as excinfo:
        query_context_tool(_context_with_section(tmp_path), section="nobody.wrote.this")

    assert "demo.motion" in (excinfo.value.remedy or "")


def test_a_plugin_section_is_not_dumped_unless_asked(tmp_path: Path) -> None:
    """A plugin's payload can be large; the surface stays summary-sized."""
    answer = query_context_tool(_context_with_section(tmp_path))

    assert "section" not in answer


# --- progress over the protocol (M8) ---------------------------------------


class _RecordingContext:
    """Stands in for the server-injected context, recording what it is told."""

    def __init__(self) -> None:
        self.reports: list[tuple[float, float | None, str | None]] = []

    async def report_progress(
        self, progress: float, total: float | None = None, message: str | None = None
    ) -> None:
        self.reports.append((progress, total, message))


@pytest.mark.integration
@requires_ffmpeg
def test_render_reports_progress_to_the_client(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An agent driving a long bake should see it advancing, not silence."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("vision_use.mcp_server._PROGRESS_INTERVAL_S", 0.0)
    clip = make_clip(name="mcp_progress", seconds=2.0)
    planned = plan_tool(str(clip), output=str(tmp_path / "plan.json"))
    _bake_effect(Path(planned["plan_path"]))

    ctx = _RecordingContext()
    asyncio.run(
        render_tool(
            planned["plan_path"],
            output=str(tmp_path / "out.mp4"),
            proxy=True,
            ctx=cast(Any, ctx),
        )
    )

    assert ctx.reports
    assert ctx.reports[-1][0] > ctx.reports[0][0]


@pytest.mark.integration
@requires_ffmpeg
def test_progress_is_rate_limited(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Baking reports every frame; forwarding all of them would flood the transport."""
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="mcp_throttle", seconds=2.0, fps="25")
    planned = plan_tool(str(clip), output=str(tmp_path / "plan.json"))
    _bake_effect(Path(planned["plan_path"]))

    ctx = _RecordingContext()
    asyncio.run(
        render_tool(
            planned["plan_path"],
            output=str(tmp_path / "out.mp4"),
            proxy=True,
            ctx=cast(Any, ctx),
        )
    )

    # Fifty frames are baked; the half-second interval keeps reports far below that.
    assert len(ctx.reports) < 10


@pytest.mark.integration
@requires_ffmpeg
def test_rendering_does_not_block_the_event_loop(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A render on the loop would stall the server -- an agent could not even cancel."""
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="mcp_loop", seconds=2.0)
    planned = plan_tool(str(clip), output=str(tmp_path / "plan.json"))
    _bake_effect(Path(planned["plan_path"]))

    async def drive() -> int:
        ticks = 0

        async def heartbeat() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        await render_tool(planned["plan_path"], output=str(tmp_path / "out.mp4"), proxy=True)
        beat.cancel()
        return ticks

    assert asyncio.run(drive()) > 0


def _bake_effect(plan_path: Path) -> None:
    """Add a baked effect, so the render has progress worth reporting."""
    payload = json.loads(plan_path.read_text())
    clip = payload["timeline"]["tracks"][0]["clips"][0]
    clip["effects"] = [{"type": "invert", "params": {}}]
    plan_path.write_text(json.dumps(payload))
