"""``vision`` command line entry point.

``render`` and ``plan`` land with the timeline in M2, and ``apply`` with the
skill SDK in M3.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from vision_use import __version__
from vision_use.cache.store import CacheStore, parse_size
from vision_use.cli.doctor import Check, Status, run_checks, worst_status
from vision_use.context.registry import available_sections, registry, resolve_section_names
from vision_use.context.runner import analyze as analyze_source
from vision_use.context.sheet import build_sheet
from vision_use.errors import VisionUseError
from vision_use.log import LogFormat, configure_logging
from vision_use.models.context import ContextSection, SectionName, VideoContext
from vision_use.models.plan import EditPlan
from vision_use.planning import Severity, load_plan, save_plan, straight_through, validate_plan
from vision_use.plugins import ensure_loaded as ensure_plugins
from vision_use.progress import Progress as ProgressUpdate
from vision_use.progress import ProgressCallback
from vision_use.render import render as render_plan
from vision_use.scaffold import scaffold_plugin
from vision_use.skills import RetryPolicy, SkillOutcome, SkillRequest, run_skill
from vision_use.skills.registry import registry as skill_registry
from vision_use.video.probe import probe as probe_media

app = typer.Typer(
    name="vision",
    help="AI video intelligence and editing framework.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
err_console = Console(stderr=True)

#: Shots listed in the human-readable summary before it truncates. Beyond this
#: the table stops being readable and `--json` is the right tool.
_MAX_LISTED_SHOTS = 12

_STATUS_MARK: dict[Status, str] = {
    Status.OK: "[green]ok[/green]",
    Status.WARN: "[yellow]warn[/yellow]",
    Status.FAIL: "[red]fail[/red]",
}


class LogStyle(StrEnum):
    """How log records are rendered on stderr."""

    HUMAN = "human"
    JSON = "json"


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Emit debug-level logs.")
    ] = False,
    log_format: Annotated[
        LogStyle, typer.Option("--log-format", help="Log rendering style.")
    ] = LogStyle.HUMAN,
) -> None:
    """Configure logging before any subcommand runs."""
    fmt: LogFormat = "json" if log_format is LogStyle.JSON else "human"
    configure_logging(level=logging.DEBUG if verbose else logging.INFO, fmt=fmt)
    # Third-party capabilities are loaded here rather than at package import.
    # `plugins` imports the registries, so discovering during their own import
    # would re-enter a half-initialised module the moment a plugin imports
    # VisionUse back -- and a library user gets no third-party code running
    # merely because they imported the package.
    ensure_plugins()


@contextmanager
def _progress_bar(label: str) -> Iterator[ProgressCallback]:
    """Yield a callback that draws a live bar on stderr.

    On stderr rather than stdout so that `--json` output stays machine-readable
    while long work still shows something. Rich suppresses the bar entirely when
    stderr is not a terminal, so piping to a file does not fill it with escape
    codes.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=err_console,
        transient=True,
    ) as bar:
        task = bar.add_task(label, total=None)

        def report(update: ProgressUpdate) -> None:
            bar.update(
                task,
                completed=update.done,
                total=update.total,
                description=f"{label} {update.detail}".strip(),
            )

        yield report


@app.command()
def version() -> None:
    """Print the installed VisionUse version."""
    console.print(__version__)


@app.command()
def doctor(
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable results.")
    ] = False,
) -> None:
    """Check that this environment can run VisionUse.

    Exits non-zero when any check fails, so CI can gate on it.
    """
    checks = run_checks()
    overall = worst_status(checks)

    if as_json:
        console.print_json(
            json.dumps(
                {
                    "status": overall.value,
                    "checks": [
                        {
                            "name": check.name,
                            "status": check.status.value,
                            "detail": check.detail,
                            "remedy": check.remedy,
                        }
                        for check in checks
                    ],
                }
            )
        )
    else:
        _render_checks(checks, overall)

    if overall is Status.FAIL:
        raise typer.Exit(code=1)


def _render_checks(checks: list[Check], overall: Status) -> None:
    """Render doctor results as a table plus a summary line."""
    table = Table(title="VisionUse environment", show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("status")
    table.add_column("detail", overflow="fold")

    for check in checks:
        table.add_row(check.name, _STATUS_MARK[check.status], check.detail)

    console.print(table)

    for check in checks:
        if check.status is not Status.OK and check.remedy:
            console.print(f"  [dim]{check.name}:[/dim] {check.remedy}")

    if overall is Status.FAIL:
        console.print("\n[red]Environment is not ready.[/red]")
    elif overall is Status.WARN:
        console.print("\n[yellow]Environment is usable with warnings.[/yellow]")
    else:
        console.print("\n[green]Environment is ready.[/green]")


@app.command()
def inspect(
    video: Annotated[Path, typer.Argument(help="Media file to probe.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the full probe as JSON.")
    ] = False,
    skip_hash: Annotated[
        bool,
        typer.Option("--skip-hash", help="Skip the source SHA-256 (faster, not cacheable)."),
    ] = False,
) -> None:
    """Report container and stream properties without decoding frames."""
    try:
        result = probe_media(video, compute_hash=not skip_hash)
    except VisionUseError as exc:
        _fail(exc)

    if as_json:
        console.print_json(result.model_dump_json())
        return

    table = Table(title=str(video), show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value", overflow="fold")

    stream = result.video
    table.add_row("container", result.source.container)
    table.add_row("duration", f"{result.source.duration_s:.3f} s")
    table.add_row("video codec", stream.codec)
    table.add_row("stored size", f"{stream.width}x{stream.height}")
    if stream.rotation:
        table.add_row("rotation", f"{stream.rotation} deg")
        table.add_row("display size", f"{stream.display_width}x{stream.display_height}")
    table.add_row("frame rate", f"{stream.frame_rate} ({stream.frame_rate.decimal:.3f} fps)")
    if stream.variable_frame_rate:
        table.add_row("frame timing", "[yellow]variable[/yellow]")
    if stream.frame_count is not None:
        suffix = " (estimated)" if stream.frame_count_estimated else ""
        table.add_row("frames", f"{stream.frame_count}{suffix}")
    table.add_row("pixel format", stream.pix_fmt or "unknown")

    if result.audio is not None:
        audio = result.audio
        table.add_row("audio", f"{audio.codec} {audio.sample_rate} Hz, {audio.channels} ch")
    else:
        table.add_row("audio", "[dim]none[/dim]")

    if not skip_hash:
        table.add_row("sha256", result.source.sha256)

    console.print(table)


#: Sections analysed when the caller does not say. Transcript is excluded
#: deliberately: it needs the optional `asr` extra, downloads model weights, and
#: is by far the slowest analyzer. Opting into it should be explicit.
DEFAULT_SECTIONS: tuple[ContextSection, ...] = (
    ContextSection.SHOTS,
    ContextSection.KEYFRAMES,
    ContextSection.AUDIO_ACTIVITY,
)


def _parse_sections(raw: str | None, *, everything: bool) -> list[SectionName]:
    """Resolve the requested sections from the command line.

    Plugin sections are named like any other (`--with stabilize.motion`), since
    to a caller a plugin's section is exactly as real as a built-in one.

    Raises:
        VisionUseError: If a name is not a known section. The message lists the
            valid names, since a typo here is the most likely mistake.

    """
    if everything:
        return available_sections()
    if raw is None:
        return list(DEFAULT_SECTIONS)

    return resolve_section_names(part.strip() for part in raw.split(",") if part.strip())


@app.command()
def analyze(
    video: Annotated[Path, typer.Argument(help="Media file to analyse.")],
    with_sections: Annotated[
        str | None,
        typer.Option(
            "--with",
            help="Comma-separated sections, e.g. 'shots,transcript'. Defaults to "
            "shots, keyframes and audio activity.",
        ),
    ] = None,
    everything: Annotated[
        bool, typer.Option("--all", help="Analyse every available section.")
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the context JSON to this path."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full context as JSON to stdout.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Recompute everything, ignoring cached results.")
    ] = False,
) -> None:
    """Analyse a video into a shared VideoContext.

    The context is what every skill reads instead of re-deriving the same facts,
    and what an agent addresses when it says "the shot at 0:42" or "where he
    says pricing".
    """
    try:
        sections = _parse_sections(with_sections, everything=everything)
        context = analyze_source(
            video,
            sections,
            cache=CacheStore(enabled=not no_cache),
            use_cache=not no_cache,
        )
    except VisionUseError as exc:
        _fail(exc)

    if output is not None:
        try:
            output.write_text(context.model_dump_json(indent=2), encoding="utf-8")
        except OSError:
            _fail(
                VisionUseError(
                    f"Could not write {output}",
                    remedy="check the directory exists and is writable",
                )
            )

    if as_json:
        console.print_json(context.model_dump_json())
        return

    _render_context(context, video)
    if output is not None:
        console.print(f"\n[dim]Context written to[/dim] {output}")


def _describe_section(payload: object) -> str:
    """Summarise a plugin's section without pretending to understand it."""
    if isinstance(payload, dict):
        return (
            ", ".join(f"{key}={value}" for key, value in sorted(payload.items()))
            or "[dim]empty[/dim]"
        )
    if isinstance(payload, list):
        return f"{len(payload)} items"
    return str(payload)


def _render_context(context: VideoContext, source: Path) -> None:
    """Render a human summary of what the analysis found."""
    table = Table(title=str(source), show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value", overflow="fold")

    stream = context.probe.video
    table.add_row("duration", f"{context.duration_s:.3f} s")
    table.add_row(
        "video", f"{stream.display_width}x{stream.display_height} @ {stream.frame_rate} fps"
    )

    if context.has(ContextSection.SHOTS):
        table.add_row("shots", str(len(context.shots)))
    if context.has(ContextSection.KEYFRAMES):
        table.add_row("keyframes", f"{len(context.keyframes)} (cached)")
    if context.has(ContextSection.AUDIO_ACTIVITY):
        silent = sum(1 for span in context.audio_activity if span.is_silence)
        quiet_s = sum(
            span.span.duration_s for span in context.audio_activity if span.is_silence
        )
        table.add_row(
            "audio activity",
            f"{len(context.audio_activity)} spans, {silent} silent ({quiet_s:.1f} s)",
        )
    if context.has(ContextSection.TRANSCRIPT) and context.transcript is not None:
        transcript = context.transcript
        table.add_row(
            "transcript",
            f"{len(transcript.words)} words, {len(transcript.segments)} segments"
            f" [{transcript.language or 'unknown'}]",
        )

    # Plugin sections, rendered generically: this command cannot know what any
    # of them mean, so it shows the shape and lets `--json` show the rest.
    for name in sorted(context.extra):
        table.add_row(name, _describe_section(context.extra[name]))

    table.add_row("analyzers", ", ".join(sorted(context.analyzers)) or "[dim]none[/dim]")
    console.print(table)

    if context.has(ContextSection.SHOTS) and context.shots:
        shots = Table(title="Shots", show_lines=False)
        shots.add_column("id", style="bold")
        shots.add_column("start")
        shots.add_column("end")
        shots.add_column("duration")
        for shot in context.shots[:_MAX_LISTED_SHOTS]:
            shots.add_row(
                shot.id,
                f"{shot.span.start_s:.2f}",
                f"{shot.span.end_s:.2f}",
                f"{shot.span.duration_s:.2f}",
            )
        console.print(shots)
        if len(context.shots) > _MAX_LISTED_SHOTS:
            console.print(
                f"[dim]... and {len(context.shots) - _MAX_LISTED_SHOTS} more; "
                f"use --json for all[/dim]"
            )


@app.command("contact-sheet")
def contact_sheet(
    video: Annotated[Path, typer.Argument(help="Media file to build a sheet for.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write the sheet image."),
    ] = None,
    columns: Annotated[int, typer.Option("--columns", help="Tiles per row.")] = 4,
    tile_width: Annotated[
        int, typer.Option("--tile-width", help="Tile width in pixels.")
    ] = 320,
    describe: Annotated[
        bool,
        typer.Option("--describe", help="Print the tile-to-shot index for a model."),
    ] = False,
) -> None:
    """Tile a video's keyframes into one labelled image.

    This is what an agent looks at to see a whole video at once. Tiles are
    numbered in the image itself, because a model cannot reliably count grid
    positions; `--describe` prints the index that maps those numbers back to
    shots and timestamps.
    """
    target = output or video.with_suffix(".sheet.jpg")
    try:
        context = analyze_source(video, [ContextSection.KEYFRAMES], cache=CacheStore())
        sheet = build_sheet(context, target, columns=columns, tile_width=tile_width)
    except VisionUseError as exc:
        _fail(exc)

    console.print(
        f"Wrote {sheet.width}x{sheet.height} sheet with {len(sheet.tiles)} tiles "
        f"to [bold]{sheet.path}[/bold]"
    )
    if describe:
        console.print()
        console.print(sheet.describe())


@app.command()
def sections() -> None:
    """List the context sections this installation can produce."""
    table = Table(title="Context sections", show_header=True)
    table.add_column("section", style="bold")
    table.add_column("producer")
    table.add_column("requires")

    # Built-ins first, then whatever plugins contribute -- a plugin's section is
    # as real to a caller as a built-in one, and `--with` accepts both.
    for section in [ContextSection.PROBE, *available_sections()]:
        if section is ContextSection.PROBE:
            table.add_row(str(section), "[dim]always present[/dim]", "")
            continue
        producer = registry.producer_of(section)
        table.add_row(
            str(section),
            producer.name if producer else "[red]none registered[/red]",
            ", ".join(sorted(str(item) for item in producer.requires)) if producer else "",
        )

    console.print(table)


plan_app = typer.Typer(help="Create and inspect edit plans.")
app.add_typer(plan_app, name="plan")


@plan_app.command("new")
def plan_new(
    video: Annotated[Path, typer.Argument(help="Media file to build a plan around.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write the plan JSON."),
    ] = None,
) -> None:
    """Create a straight-through plan for a video.

    The identity edit: one clip covering the whole file, canvas and frame rate
    taken from the source. It renders back unchanged, and it is the document
    every skill starts patching.
    """
    try:
        plan = straight_through(video)
    except VisionUseError as exc:
        _fail(exc)

    target = output or video.with_suffix(".plan.json")
    try:
        save_plan(plan, target)
    except VisionUseError as exc:
        _fail(exc)

    console.print(
        f"Wrote a {plan.duration_s:.2f}s plan "
        f"({plan.timeline.width}x{plan.timeline.height} @ "
        f"{plan.timeline.frame_rate} fps) to [bold]{target}[/bold]"
    )


@plan_app.command("show")
def plan_show(
    plan_path: Annotated[Path, typer.Argument(help="Plan JSON to describe.")],
) -> None:
    """Summarise a plan's tracks and clips."""
    try:
        plan = load_plan(plan_path)
    except VisionUseError as exc:
        _fail(exc)

    table = Table(title=str(plan_path), show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value", overflow="fold")
    table.add_row("duration", f"{plan.duration_s:.3f} s")
    table.add_row(
        "canvas",
        f"{plan.timeline.width}x{plan.timeline.height} @ {plan.timeline.frame_rate} fps",
    )
    table.add_row("sources", str(len(plan.sources)))
    table.add_row("tracks", str(len(plan.timeline.tracks)))
    console.print(table)

    for track in plan.timeline.tracks:
        clips = Table(title=f"{track.id} ({track.kind})")
        clips.add_column("clip", style="bold")
        clips.add_column("timeline")
        clips.add_column("source range")
        clips.add_column("effects")
        for clip in track.clips:
            clips.add_row(
                clip.id if clip.enabled else f"[dim]{clip.id} (off)[/dim]",
                f"{clip.timeline_start_s:.2f} - {clip.timeline_end_s:.2f}",
                f"{clip.source_in_s:.2f} - {clip.source_out_s:.2f}",
                ", ".join(effect.type for effect in clip.effects) or "-",
            )
        console.print(clips)


@app.command()
def validate(
    plan_path: Annotated[Path, typer.Argument(help="Plan JSON to check.")],
) -> None:
    """Check a plan for problems that would spoil a render.

    Structural impossibilities are already rejected when a plan loads. This
    reports what is legal but usually accidental: gaps, unused sources, missing
    files, effects no registry knows.

    Exits non-zero when anything would stop a render.
    """
    try:
        plan = load_plan(plan_path)
    except VisionUseError as exc:
        _fail(exc)

    report = validate_plan(plan)
    if not report.findings:
        console.print("[green]No problems found.[/green]")
        return

    for finding in report.findings:
        colour = "red" if finding.severity is Severity.ERROR else "yellow"
        console.print(f"[{colour}]{finding.severity.value}:[/{colour}] {finding.message}")
        if finding.remedy:
            console.print(f"  [dim]{finding.remedy}[/dim]")

    console.print(f"\n{len(report.errors)} error(s), {len(report.warnings)} warning(s).")
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def render(
    plan_path: Annotated[Path, typer.Argument(help="Plan JSON to render.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write the rendered file."),
    ] = None,
    proxy: Annotated[
        bool,
        typer.Option("--proxy", help="Render at proxy scale, for fast iteration."),
    ] = False,
    skip_validation: Annotated[
        bool,
        typer.Option("--skip-validation", help="Render even if validation reports errors."),
    ] = False,
) -> None:
    """Render a plan to a video file.

    Validation runs first: an error found in a second beats one found after
    ffmpeg has spent minutes on a render that was never going to be right.
    """
    try:
        plan = load_plan(plan_path)
    except VisionUseError as exc:
        _fail(exc)

    report = validate_plan(plan)
    if not report.ok and not skip_validation:
        for finding in report.errors:
            err_console.print(f"[red]error:[/red] {finding.message}")
            if finding.remedy:
                err_console.print(f"  [dim]{finding.remedy}[/dim]")
        err_console.print("[dim]pass --skip-validation to render anyway[/dim]")
        raise typer.Exit(code=1)

    target = output or plan_path.with_suffix(".mp4")
    try:
        with _progress_bar("baking") as on_progress:
            result = render_plan(plan, target, proxy=proxy, on_progress=on_progress)
    except VisionUseError as exc:
        _fail(exc)

    console.print(
        f"Rendered {result.duration_s:.2f}s at {result.width}x{result.height} "
        f"({result.profile}) to [bold]{result.path}[/bold]"
    )


plugin_app = typer.Typer(help="Scaffold and inspect plugins.")
app.add_typer(plugin_app, name="plugin")


@plugin_app.command("new")
def plugin_new(
    name: Annotated[str, typer.Argument(help="Distribution name, e.g. my-plugin.")],
    into: Annotated[
        Path | None,
        typer.Option("--into", help="Directory to create the package in."),
    ] = None,
) -> None:
    """Scaffold a working plugin package.

    What it writes already installs, already registers with VisionUse, and
    already passes its own test — including one asserting the skill is
    discoverable, which is the part documentation cannot prove.
    """
    try:
        names, written = scaffold_plugin(name, into or Path.cwd())
    except VisionUseError as exc:
        _fail(exc)

    console.print(f"Created [bold]{names.dist}[/bold] with {len(written)} files:")
    for path in written:
        console.print(f"  [dim]{path}[/dim]")
    console.print(
        f"\nNext: [bold]cd {names.dist} && uv sync && uv run vision plugins[/bold]\n"
        f"[dim]{names.skill} should appear in `vision skills` once installed.[/dim]"
    )


@app.command()
def plugins() -> None:
    """List third-party skills, analyzers and effects, and any that failed.

    Exits non-zero when a plugin could not be loaded, so a broken install is
    visible to CI without taking the framework down at import.
    """
    report = ensure_plugins()

    if not report.loaded and not report.failed:
        console.print("[dim]No plugins installed.[/dim]")
        return

    if report.loaded:
        table = Table(title="Plugins", show_header=True)
        table.add_column("plugin", style="bold")
        table.add_column("kind")
        table.add_column("from", overflow="fold")
        for item in report.loaded:
            table.add_row(item.name, item.group.rsplit(".", 1)[-1], item.module)
        console.print(table)

    for failure in report.failed:
        err_console.print(
            f"[red]failed:[/red] {failure.name} "
            f"({failure.group.rsplit('.', 1)[-1]}) — {failure.reason}"
        )

    if report.failed:
        raise typer.Exit(code=1)


@app.command()
def skills() -> None:
    """List the skills this installation can run."""
    table = Table(title="Skills", show_header=True)
    table.add_column("skill", style="bold")
    table.add_column("version")
    table.add_column("requires")
    table.add_column("summary", overflow="fold")

    available = skill_registry.all()
    if not available:
        console.print("[dim]No skills registered.[/dim]")
        return

    for skill in available:
        meta = skill.meta
        table.add_row(
            meta.name,
            meta.version,
            ", ".join(sorted(str(item) for item in meta.requires)) or "-",
            meta.summary or "-",
        )
    console.print(table)


@app.command()
def apply(
    plan_path: Annotated[Path, typer.Argument(help="Plan JSON to edit.")],
    skill_name: Annotated[str, typer.Option("--skill", "-s", help="Which skill to run.")],
    prompt: Annotated[
        str, typer.Option("--prompt", help="What you want, in your own words.")
    ] = "",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write the edited plan."),
    ] = None,
    max_attempts: Annotated[
        int, typer.Option("--max-attempts", help="How many strategies to try.")
    ] = 3,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report what would change without writing."),
    ] = False,
) -> None:
    """Run a skill against a plan.

    The skill reads the shared context, proposes strategies, and returns a patch
    for each. The first whose own validation passes is kept. Nothing is
    rendered: the result is an edited plan, which `vision render` turns into a
    file when you are ready.
    """
    try:
        plan = load_plan(plan_path)
        skill = skill_registry.get(skill_name)
        source = _plan_source(plan)
        outcome = run_skill(
            skill,
            source,
            plan,
            SkillRequest(prompt=prompt),
            cache=CacheStore(),
            policy=RetryPolicy(max_attempts=max_attempts),
        )
    except VisionUseError as exc:
        _fail(exc)

    _render_outcome(skill_name, outcome)

    if outcome.accepted is None:
        raise typer.Exit(code=1)
    if dry_run:
        console.print("[dim]--dry-run: nothing written[/dim]")
        return

    target = output or plan_path
    try:
        save_plan(outcome.plan, target)
    except VisionUseError as exc:
        _fail(exc)
    console.print(f"Wrote the edited plan to [bold]{target}[/bold]")


def _plan_source(plan: EditPlan) -> Path:
    """Return the media a plan draws from, for backfilling context.

    Raises:
        VisionUseError: If the plan declares no sources.

    """
    for source in plan.sources.values():
        return source.path
    raise VisionUseError(
        "This plan declares no sources, so there is nothing to analyse",
        remedy="build a plan with `vision plan new`",
    )


def _render_outcome(skill_name: str, outcome: SkillOutcome) -> None:
    """Report what a skill did, or why it did nothing."""
    if not outcome.attempts:
        console.print(f"[yellow]{skill_name} found nothing to do.[/yellow]")
        return

    for attempt in outcome.attempts:
        mark = "[green]accepted[/green]" if attempt.passed else "[yellow]rejected[/yellow]"
        console.print(
            f"{mark} [bold]{attempt.strategy.name}[/bold] "
            f"({len(attempt.patch)} op(s), score {attempt.score.value:g})"
        )
        for measurement in attempt.report.failures:
            console.print(f"    [dim]{measurement.name}: {measurement.detail}[/dim]")

    if outcome.accepted is not None:
        console.print(f"\n[green]{outcome.accepted.patch.rationale}[/green]")
        return

    console.print(
        f"\n[red]No strategy passed its own validation[/red]"
        f"{' (retry budget exhausted)' if outcome.exhausted else ''}."
    )


cache_app = typer.Typer(help="Inspect and manage the content-addressed cache.")
app.add_typer(cache_app, name="cache")


@cache_app.command("path")
def cache_path() -> None:
    """Print the cache root directory."""
    console.print(str(CacheStore().root))


@cache_app.command("status")
def cache_status(
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable usage.")
    ] = False,
) -> None:
    """Report what the cache is holding, by producer.

    Worth looking at on real footage: a baked effect stores a *lossless*
    intermediate, so ten minutes of 640x360 is around half a gigabyte and a long
    1080p source is far more.
    """
    usage = CacheStore().usage()

    if as_json:
        console.print_json(
            json.dumps(
                {
                    "root": str(CacheStore().root),
                    "files": usage.files,
                    "bytes": usage.bytes,
                    "namespaces": [
                        {"namespace": item.namespace, "files": item.files, "bytes": item.bytes}
                        for item in usage.namespaces
                    ],
                }
            )
        )
        return

    if not usage.namespaces:
        console.print("[dim]Cache is empty.[/dim]")
        return

    table = Table(title=f"Cache at {CacheStore().root}", show_header=True)
    table.add_column("producer", style="bold")
    table.add_column("files", justify="right")
    table.add_column("size", justify="right")
    for item in usage.namespaces:
        table.add_row(item.namespace, str(item.files), _human_bytes(item.bytes))
    table.add_row(
        "[bold]total[/bold]",
        f"[bold]{usage.files}[/bold]",
        f"[bold]{_human_bytes(usage.bytes)}[/bold]",
    )
    console.print(table)


@cache_app.command("prune")
def cache_prune(
    budget: Annotated[
        str,
        typer.Option("--budget", help="Ceiling to bring the cache under, e.g. 20GB."),
    ],
) -> None:
    """Evict least-recently-used entries until the cache fits a budget.

    Least recently used by *access* time: an artefact currently being rendered
    against is the most expensive one to lose, and it is also the one being read.
    """
    try:
        limit = parse_size(budget)
        result = CacheStore().prune(limit)
    except VisionUseError as exc:
        _fail(exc)

    console.print(
        f"Removed {result.removed} file(s), freeing {_human_bytes(result.freed_bytes)}. "
        f"Cache now {_human_bytes(result.remaining_bytes)}."
    )


#: One binary step, matching what `du` reports.
_STEP = 1024


def _human_bytes(value: int) -> str:
    """Render a byte count the way a person reads it."""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < _STEP or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= _STEP
    return f"{size:.1f} TB"  # pragma: no cover - unreachable, the loop returns


@cache_app.command("clear")
def cache_clear(
    namespace: Annotated[
        str | None,
        typer.Option("--namespace", help="Clear only this producer's entries."),
    ] = None,
) -> None:
    """Delete cached artefacts."""
    try:
        removed = CacheStore().clear(namespace)
    except VisionUseError as exc:
        _fail(exc)
    console.print(f"Removed {removed} cached file(s).")


def _fail(exc: VisionUseError) -> NoReturn:
    """Render a framework error and exit non-zero."""
    err_console.print(f"[red]error:[/red] {exc.message}")
    if exc.remedy:
        err_console.print(f"[dim]hint:[/dim] {exc.remedy}")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
