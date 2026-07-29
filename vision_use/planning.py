"""Creating and checking edit plans.

Two things live here, both sitting between the pure ``EditPlan`` model and the
outside world:

**Building a starting plan.** A skill patches a plan; something has to produce
the first one. :func:`straight_through` probes a source and returns a plan that
renders it back unchanged -- the identity edit, and the base every skill starts
from.

**Advisory validation.** The model already rejects anything structurally
impossible (overlaps, dangling references, out-of-range ranges). What is left is
*legal but usually accidental*: a gap in the middle of a timeline, a declared
source nothing draws from, an effect no registry knows how to render. Those are
reported rather than raised, because each one is occasionally deliberate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from vision_use.effects.registry import EffectRegistry
from vision_use.effects.registry import registry as default_effects
from vision_use.errors import ConfigurationError, VisionUseError
from vision_use.models.ids import clip_id, source_id, track_id
from vision_use.models.media import FrameRate
from vision_use.models.plan import (
    Clip,
    EditPlan,
    Effect,
    RenderSettings,
    Source,
    Timeline,
    Track,
)
from vision_use.timebase import quantize_s
from vision_use.video.probe import probe


class Severity(StrEnum):
    """How much a finding should worry the caller."""

    WARNING = "warning"
    ERROR = "error"


#: Tolerance when comparing a keyframe time against a clip's length. A keyframe
#: landing exactly on the final frame is legitimate, and float arithmetic on
#: trimmed spans should not turn that into a warning.
_TIME_SLACK_S = 1e-6


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing worth knowing about a plan."""

    severity: Severity
    message: str
    remedy: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Everything advisory validation found."""

    findings: tuple[Finding, ...] = field(default=())

    @property
    def ok(self) -> bool:
        """Whether the plan is free of errors. Warnings do not block a render."""
        return not any(item.severity is Severity.ERROR for item in self.findings)

    @property
    def errors(self) -> tuple[Finding, ...]:
        """Findings that will stop a render."""
        return tuple(item for item in self.findings if item.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """Findings worth a look that will not stop a render."""
        return tuple(item for item in self.findings if item.severity is Severity.WARNING)


def straight_through(
    source: Path,
    *,
    render: RenderSettings | None = None,
) -> EditPlan:
    """Return a plan that renders ``source`` back unchanged.

    The identity edit: one video track, one clip covering the whole file, canvas
    and frame rate taken from the source. It is what ``vision plan new``
    produces and what every skill starts patching.

    Rotation is applied to the canvas, so portrait footage stored landscape with
    a rotation flag produces a portrait plan rather than a sideways one.

    Args:
        source: Media file to build a plan around.
        render: Render settings. Defaults to the standard proxy/master pair.

    Returns:
        A valid, renderable plan.

    Raises:
        VisionUseError: If the source cannot be probed.

    """
    probed = probe(source)
    duration = quantize_s(probed.source.duration_s)
    if duration <= 0:
        raise VisionUseError(
            f"{source.name} reports no duration, so there is nothing to place",
            remedy="check the file is complete (`vision inspect`)",
        )

    declared = Source(
        id=source_id(0),
        path=source.resolve(),
        sha256=probed.source.sha256,
        duration_s=duration,
        has_audio=probed.has_audio,
    )
    clip = Clip(
        id=clip_id(0),
        source_id=declared.id,
        source_in_s=0.0,
        source_out_s=duration,
        timeline_start_s=0.0,
    )
    return EditPlan(
        sources={declared.id: declared},
        timeline=Timeline(
            frame_rate=probed.video.frame_rate,
            width=probed.video.display_width,
            height=probed.video.display_height,
            tracks=(Track(id=track_id(0), kind="video", clips=(clip,)),),
        ),
        render=render or RenderSettings(),
    )


def _animation_findings(clip: Clip, effect: Effect, effects: EffectRegistry) -> list[Finding]:
    """Return what is wrong with one effect's animation, if anything.

    Split out of `validate_plan` because that function is a list of independent
    checks, and a reader looking for "what do we say about animation" should
    find it in one place.
    """
    if not effect.animated:
        return []

    findings: list[Finding] = []

    # An effect that cannot render animation would render statically and say
    # nothing, which is silently wrong -- the worst outcome available. Baking is
    # not a free pass: a baked effect animates only if it implements
    # `frame_op_animated`, which is what declaring `animatable` means.
    definition = effects.get(effect.type) if effect.type in effects else None
    if definition is not None and not definition.animatable:
        findings.append(
            Finding(
                Severity.ERROR,
                f"clip {clip.id} animates {', '.join(sorted(effect.animate))} on "
                f"{effect.type!r}, which cannot render animated parameters",
                "remove the animation, or use an effect that supports it",
            )
        )

    findings.extend(
        Finding(
            Severity.WARNING,
            f"clip {clip.id} animates {name} until {frames[-1].t_s:.3f}s but is only "
            f"{clip.duration_s:.3f}s long, so the last keyframes never apply",
            "shorten the animation, or lengthen the clip",
        )
        for name, frames in sorted(effect.animate.items())
        if frames[-1].t_s > clip.duration_s + _TIME_SLACK_S
    )
    return findings


def validate_plan(plan: EditPlan, *, effects: EffectRegistry | None = None) -> ValidationReport:
    """Check ``plan`` for things that are legal but probably unintended.

    Structural impossibilities are already rejected by the model, so anything
    reported here is a judgement call rather than a contradiction.

    Args:
        plan: The plan to check.
        effects: Registry used to resolve effect types.

    Returns:
        Findings, worst first is not guaranteed -- check :attr:`~ValidationReport.ok`.

    """
    # Never `effects or default_effects`: an empty registry is falsy, so a
    # caller passing their own would silently validate against the global one.
    registry = default_effects if effects is None else effects
    findings: list[Finding] = []

    if not plan.timeline.tracks:
        findings.append(
            Finding(
                Severity.ERROR,
                "the plan has no tracks",
                "add a video track with at least one clip",
            )
        )

    for track in plan.timeline.tracks:
        if not track.clips:
            findings.append(Finding(Severity.WARNING, f"track {track.id} has no clips"))
        for start, end in track.gaps():
            findings.append(
                Finding(
                    Severity.WARNING,
                    f"track {track.id} has a gap from {start:.3f}s to {end:.3f}s",
                    "gaps render as black and silence; close it if unintended",
                )
            )

    for unused in plan.unused_sources():
        findings.append(
            Finding(
                Severity.WARNING,
                f"source {unused} is declared but no clip uses it",
                "remove it, or place a clip that draws from it",
            )
        )

    for source in plan.sources.values():
        if not source.path.exists():
            findings.append(
                Finding(
                    Severity.ERROR,
                    f"source {source.id} is missing: {source.path}",
                    "restore the file, or point the plan at its new location",
                )
            )

    for clip in plan.timeline.clips():
        for effect in clip.effects:
            findings.extend(_animation_findings(clip, effect, registry))
            try:
                registry.validate(effect)
            except ConfigurationError:
                findings.append(
                    Finding(
                        Severity.ERROR,
                        f"clip {clip.id} uses unknown effect {effect.type!r}",
                        "install the plugin providing it, or remove the effect",
                    )
                )
            except VisionUseError as exc:
                findings.append(
                    Finding(
                        Severity.ERROR,
                        f"clip {clip.id} has invalid parameters for {effect.type!r}",
                        exc.remedy,
                    )
                )

    if not any(clip.enabled for clip in plan.timeline.clips()):
        findings.append(
            Finding(
                Severity.ERROR,
                "every clip is disabled, so the render would be empty",
                "enable at least one clip",
            )
        )

    return ValidationReport(findings=tuple(findings))


def load_plan(path: Path) -> EditPlan:
    """Read and validate a plan from JSON.

    Raises:
        VisionUseError: If the file is missing, unreadable, or not a valid plan.

    """
    try:
        return EditPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VisionUseError(
            f"Could not read the plan at {path}",
            remedy="check the path exists and is readable",
        ) from exc
    except ValueError as exc:
        raise VisionUseError(
            f"{path.name} is not a valid edit plan: {exc}",
            remedy="regenerate it with `vision plan new`, or fix the JSON by hand",
        ) from exc


def save_plan(plan: EditPlan, path: Path) -> None:
    """Write ``plan`` to ``path`` as indented JSON.

    Raises:
        VisionUseError: If the file cannot be written.

    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        raise VisionUseError(
            f"Could not write the plan to {path}",
            remedy="check the directory exists and is writable",
        ) from exc


#: Frame rate used when a plan is built without a source to take one from.
FALLBACK_RATE = FrameRate(num=25, den=1)
