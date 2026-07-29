"""Compiling an ``EditPlan`` into ffmpeg arguments.

This is the only component that encodes, and it does so once, at export. Every
skill upstream has been editing a document; here that document becomes a
filtergraph and a command line.

The shape of the graph, per clip:

```
[N:v] trim=start=IN:end=OUT      -- take the wanted span of the source
      setpts=PTS-STARTPTS        -- rebase its timestamps to zero
      <effect chain>             -- the clip's effect stack, in order
      scale=W:H:decrease         -- fit the canvas without distorting
      pad=W:H:centre             -- letterbox whatever is left
      setsar=1                   -- square pixels, so concat accepts it
      fps=RATE                   -- exact output cadence
```

and the matching audio chain:

```
[N:a] atrim=start=IN:end=OUT     -- the same span of the source's audio
      asetpts=PTS-STARTPTS       -- rebase
      <audio effect chain>
      aresample=RATE             -- one sample rate for the whole graph
      aformat=...                -- one sample format and layout
```

then video and audio are interleaved and joined by a single ``concat``.

Five details are load-bearing:

**`fps` on every clip, not once at the end.** ``concat`` requires its inputs to
agree on frame rate, and sources routinely do not. Normalising per clip is what
makes a cut land on the frame the plan asked for rather than one either side.

**`setsar=1` on every clip.** ``concat`` also requires matching sample aspect
ratios, and anamorphic sources otherwise fail with an error that names neither
the clip nor the plan.

**Gaps become black, explicitly.** A plan may leave a track uncovered. Rather
than letting ffmpeg improvise, the compiler inserts a generated black segment,
so the output length always matches ``plan.duration_s``.

**One `concat` handles video and audio together.** Concatenating the two
separately would let them drift apart -- each stream would be assembled from its
own segment boundaries, and any rounding difference becomes creeping lip-sync
error. Interleaving the labels into a single ``concat`` makes the two share one
set of boundaries by construction.

**Every segment carries audio, even silent ones.** ``concat`` requires the same
stream count from every input, so a source with no audio track, and every
generated gap, gets silence from ``anullsrc``. Without it the filter fails with
an error that names no clip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Final

from vision_use.effects.animation import time_expression
from vision_use.effects.registry import EffectDef, EffectRegistry
from vision_use.effects.registry import registry as default_effects
from vision_use.errors import VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.plan import Clip, EditPlan, Effect, RenderProfile, Track, Transition

log = get_logger(__name__)

#: Bumped when compiled output changes for an identical plan, so cached renders
#: from an older compiler are not served.
COMPILER_VERSION: Final = "1"

#: Colour used to fill gaps and letterbox bars.
_FILL: Final = "black"

#: Sample rate every audio chain is resampled to. Sources disagree, and concat
#: requires agreement; 48 kHz is the video-production norm.
_AUDIO_RATE: Final = 48_000

#: Sample format and layout every audio chain is normalised to, for the same
#: reason as the rate.
_AUDIO_FORMAT: Final = (
    f"aformat=sample_fmts=fltp:sample_rates={_AUDIO_RATE}:channel_layouts=stereo"
)


@dataclass(frozen=True, slots=True)
class CompiledRender:
    """Everything needed to invoke ffmpeg for one render.

    Attributes:
        args: Full argument list, excluding the ffmpeg binary itself.
        width: Output width after profile scaling.
        height: Output height after profile scaling.
        duration_s: Expected output duration.
        segments: How many concat segments the graph contains, gaps included.

    """

    args: list[str]
    width: int
    height: int
    duration_s: float
    segments: int = field(default=0)


def _even(value: int) -> int:
    """Round ``value`` down to an even number.

    yuv420p subsamples chroma by two, so an odd dimension is not encodable.
    Proxy scaling routinely produces one.
    """
    return max(2, value - (value % 2))


def _overlay_tracks(plan: EditPlan) -> list[Track]:
    """Return the overlay tracks, in the order they should be composited.

    Later tracks sit on top, matching how every editor stacks layers.
    """
    return [
        track for track in plan.timeline.tracks if track.kind == "overlay" and not track.muted
    ]


def _video_track(plan: EditPlan) -> Track | None:
    """Return the track the video output is built from.

    The first video track. Multi-track compositing arrives with overlays; until
    then a second video track would be silently ignored, so the caller is told.
    """
    tracks = [track for track in plan.timeline.tracks if track.kind == "video"]
    if len(tracks) > 1:
        log.warning(
            "plan has multiple video tracks; only the first is rendered",
            extra=bind(tracks=len(tracks)),
        )
    return tracks[0] if tracks else None


@dataclass(frozen=True, slots=True)
class _Canvas:
    """Output geometry and cadence every segment is normalised to."""

    width: int
    height: int
    fps: str


def _transition_plan(clips: Sequence[Clip], effects: EffectRegistry) -> list[Transition | None]:
    """Return the transition joining each clip to the one before it.

    A boundary is described by the later clip's ``transition_in`` when it has
    one, and otherwise by the earlier clip's ``transition_out``. Allowing both
    ends to describe the same boundary would make a plan ambiguous, so the
    later clip wins and the earlier one is ignored.

    The first clip has no predecessor, so its entry is always ``None`` -- a
    fade from black at the very start is a ``fade`` effect, not a transition
    between two clips.
    """
    del effects  # reserved: custom transition types will resolve through here
    joins: list[Transition | None] = [None]
    joins.extend(
        clip.transition_in or previous.transition_out for previous, clip in pairwise(clips)
    )
    return joins


def _compile_effect(definition: EffectDef, effect: Effect, effects: EffectRegistry) -> str:
    """Return one effect's filter, animated or static.

    Time expressions are safe to emit here because `setpts=PTS-STARTPTS` has
    already run: `t` inside a clip's chain is measured from the clip's own
    start, which is the base keyframes use (PLAN.md D6).
    """
    params = effects.validate(effect)
    if not effect.animated:
        return definition.compile_filter(params)
    return definition.compile_animated_filter(
        params, {name: time_expression(frames) for name, frames in effect.animate.items()}
    )


def _clip_chain(
    clip: Clip,
    label_in: str,
    label_out: str,
    canvas: _Canvas,
    effects: EffectRegistry,
) -> str:
    """Return the filtergraph chain for one clip."""
    steps = [
        f"trim=start={clip.source_in_s}:end={clip.source_out_s}",
        "setpts=PTS-STARTPTS",
    ]

    for effect in clip.effects:
        definition = effects.get(effect.type)
        if definition.applies_to != "video":
            continue
        steps.append(_compile_effect(definition, effect, effects))

    steps.extend(
        [
            f"scale={canvas.width}:{canvas.height}:force_original_aspect_ratio=decrease",
            f"pad={canvas.width}:{canvas.height}:(ow-iw)/2:(oh-ih)/2:color={_FILL}",
            "setsar=1",
            f"fps={canvas.fps}",
        ]
    )
    return f"[{label_in}]{','.join(steps)}[{label_out}]"


def _audio_chain(
    clip: Clip,
    label_in: str,
    label_out: str,
    effects: EffectRegistry,
) -> str:
    """Return the filtergraph chain for one clip's audio."""
    steps = [
        f"atrim=start={clip.source_in_s}:end={clip.source_out_s}",
        "asetpts=PTS-STARTPTS",
    ]

    for effect in clip.effects:
        definition = effects.get(effect.type)
        if definition.applies_to != "audio":
            continue
        steps.append(definition.compile_filter(effects.validate(effect)))

    steps.extend([f"aresample={_AUDIO_RATE}", _AUDIO_FORMAT])
    return f"[{label_in}]{','.join(steps)}[{label_out}]"


@dataclass
class _GraphBuilder:
    """Accumulates ffmpeg inputs and filter chains for one plan.

    Extracted from :func:`compile_plan` so the stream-index bookkeeping lives in
    one object rather than a nest of closures sharing ``nonlocal`` state -- that
    bookkeeping is exactly where the input-numbering bug came from.
    """

    plan: EditPlan
    canvas: _Canvas
    effects: EffectRegistry
    inputs: list[str] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    segments: list[tuple[str, str]] = field(default_factory=list)
    _slots: dict[str, int] = field(default_factory=dict)
    # Counted explicitly rather than derived from len(inputs): a file input is
    # two arguments but a lavfi input is four, so any arithmetic on the argument
    # list misnumbers every stream after the first generated one.
    _slot_count: int = 0

    def _claim_slot(self) -> int:
        """Reserve and return the next ffmpeg input index."""
        slot = self._slot_count
        self._slot_count += 1
        return slot

    def add_source(self, source_id: str) -> int:
        """Return the input index for ``source_id``, adding the input if new."""
        if source_id not in self._slots:
            self._slots[source_id] = self._claim_slot()
            self.inputs.extend(["-i", str(self.plan.sources[source_id].path)])
        return self._slots[source_id]

    def add_silence(self, duration: float, label: str) -> None:
        """Append a generated silent audio segment of ``duration`` seconds."""
        slot = self._claim_slot()
        self.inputs.extend(
            ["-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate={_AUDIO_RATE}"]
        )
        self.chains.append(f"[{slot}:a]atrim=duration={duration},{_AUDIO_FORMAT}[{label}]")

    def add_gap(self, duration: float) -> None:
        """Append a generated black-and-silent segment of ``duration`` seconds.

        Raises:
            VisionUseError: If ``duration`` is not positive. A zero-duration
                ``color`` lavfi source is not a no-op: ffmpeg never emits EOF
                for it, so the whole render hangs forever rather than failing.
                Guarded here as well as at the caller, because a hang is too
                expensive a way to discover a caller stopped checking.

        """
        if duration <= 0:
            raise VisionUseError(
                f"Refusing to insert a {duration}s gap",
                remedy="this is a compiler bug, not a plan problem -- please report it",
            )
        slot = self._claim_slot()
        self.inputs.extend(
            [
                "-f",
                "lavfi",
                "-i",
                f"color=c={_FILL}:s={self.canvas.width}x{self.canvas.height}"
                f":r={self.canvas.fps}:d={duration}",
            ]
        )
        position = len(self.segments)
        video_label, audio_label = f"gv{position}", f"ga{position}"
        self.chains.append(f"[{slot}:v]setsar=1,fps={self.canvas.fps}[{video_label}]")
        self.add_silence(duration, audio_label)
        self.segments.append((video_label, audio_label))

    def add_clip(self, clip: Clip, index: int) -> None:
        """Append one clip's video and audio chains.

        Raises:
            VisionUseError: If one of the clip's effects cannot be compiled.

        """
        slot = self.add_source(clip.source_id)
        video_label, audio_label = f"v{index}", f"a{index}"
        try:
            self.chains.append(
                _clip_chain(clip, f"{slot}:v", video_label, self.canvas, self.effects)
            )
            if self.plan.sources[clip.source_id].has_audio:
                self.chains.append(_audio_chain(clip, f"{slot}:a", audio_label, self.effects))
            else:
                # concat demands the same stream count from every input, so a
                # silent source still has to contribute an audio leg.
                self.add_silence(round(clip.duration_s, 6), audio_label)
        except NotImplementedError as exc:
            raise VisionUseError(
                f"Clip {clip.id} uses an effect that cannot be rendered: {exc}",
                remedy="remove the effect, or implement compile_filter() for it",
            ) from exc
        self.segments.append((video_label, audio_label))

    def lay_out(self, clips: Sequence[Clip]) -> float:
        """Build every segment for ``clips``, filling gaps and joining them.

        Returns:
            The output duration. Transitions shorten it, since each one
            consumes its duration from the boundary it spans.

        """
        cursor = 0.0
        for index, clip in enumerate(clips):
            # Rounded, then checked against zero rather than checking
            # `timeline_start_s > cursor` and rounding the gap afterwards:
            # `duration_s` is an unquantised subtraction (`source_out_s -
            # source_in_s`, see `Clip`), so two clips meant to be exactly back
            # to back can leave `cursor` a few femtoseconds past
            # `timeline_start_s`. That is not a gap -- but `add_gap(0.0)` was
            # still being called, and a zero-duration `color=...:d=0.0` lavfi
            # source does not just render as nothing: ffmpeg never produces
            # EOF for it and the whole render hangs forever. Found on real
            # 40-cut media, where the render ran for eleven minutes and never
            # would have finished.
            gap = round(clip.timeline_start_s - cursor, 6)
            if gap > 0:
                self.add_gap(gap)
            self.add_clip(clip, index)
            cursor = clip.timeline_end_s

        overlaps = _transition_plan(clips, self.effects)
        if any(overlaps):
            return self._join_with_transitions(clips, overlaps, cursor)

        joined = "".join(f"[{video}][{audio}]" for video, audio in self.segments)
        self.chains.append(f"{joined}concat=n={len(self.segments)}:v=1:a=1[outv][outa]")
        return cursor

    def composite_overlays(self, tracks: Sequence[Track], label: str = "outv") -> str:
        """Composite each overlay clip onto the finished video, returning its label.

        Overlays are *not* concatenated. They sit on top of the assembled video
        for the span they occupy, which is what `enable=between(...)` expresses.
        Each is a separate `overlay` step, so an overlay track with three
        captions costs three filters rather than a re-assembly of the timeline.

        The overlay's own alpha does the blending, which is why a caption is
        rendered full-canvas with a transparent background: the position is
        already baked in and the composite is a plain origin overlay.
        """
        current = label
        step = 0
        for track in tracks:
            for clip in track.clips:
                if not clip.enabled:
                    continue
                slot = self.add_source(clip.source_id)
                scaled, merged = f"ov{step}", f"oc{step}"
                self.chains.append(
                    f"[{slot}:v]scale={self.canvas.width}:{self.canvas.height},"
                    f"setsar=1,format=rgba[{scaled}]"
                )
                start = clip.timeline_start_s
                end = clip.timeline_end_s
                self.chains.append(
                    f"[{current}][{scaled}]overlay=0:0:"
                    f"enable='between(t,{start},{end})'"
                    f":format=auto[{merged}]"
                )
                current = merged
                step += 1
        return current

    def _join_with_transitions(
        self,
        clips: Sequence[Clip],
        overlaps: Sequence[Transition | None],
        plain_duration: float,
    ) -> float:
        """Join segments with `xfade`/`acrossfade` instead of `concat`.

        `concat` butts segments together; a crossfade has to *overlap* them, so
        the two cannot be mixed. When any transition is present the whole track
        is joined pairwise instead, each step folding the next clip into the
        accumulated stream.

        `xfade` takes an absolute offset into the accumulated stream, and every
        preceding transition has already shortened that stream -- so the offset
        is tracked cumulatively rather than derived from the clip's timeline
        position.
        """
        video, audio = self.segments[0]
        elapsed = clips[0].duration_s
        total = plain_duration

        for index in range(1, len(self.segments)):
            next_video, next_audio = self.segments[index]
            transition = overlaps[index]
            merged_video, merged_audio = f"xv{index}", f"xa{index}"

            if transition is None:
                # No transition here: a two-input concat is the butt join.
                self.chains.append(
                    f"[{video}][{audio}][{next_video}][{next_audio}]"
                    f"concat=n=2:v=1:a=1[{merged_video}][{merged_audio}]"
                )
                elapsed += clips[index].duration_s
            else:
                offset = round(elapsed - transition.duration_s, 6)
                if offset < 0:
                    raise VisionUseError(
                        f"Transition into {clips[index].id} is longer than the "
                        f"footage before it ({transition.duration_s}s over "
                        f"{elapsed}s)",
                        remedy="shorten the transition, or lengthen the earlier clip",
                    )
                self.chains.append(
                    f"[{video}][{next_video}]xfade=transition={transition.type}"
                    f":duration={transition.duration_s}:offset={offset}[{merged_video}]"
                )
                self.chains.append(
                    f"[{audio}][{next_audio}]acrossfade=d={transition.duration_s}"
                    f"[{merged_audio}]"
                )
                elapsed += clips[index].duration_s - transition.duration_s
                total = round(total - transition.duration_s, 6)

            video, audio = merged_video, merged_audio

        self.chains.append(f"[{video}]null[outv]")
        self.chains.append(f"[{audio}]anull[outa]")
        return total


def compile_plan(
    plan: EditPlan,
    output: Path,
    *,
    profile: RenderProfile | None = None,
    effects: EffectRegistry | None = None,
    overwrite: bool = True,
) -> CompiledRender:
    """Compile ``plan`` into an ffmpeg argument list.

    Args:
        plan: The plan to render.
        output: Where the rendered file should be written.
        profile: Which render profile to use. Defaults to the plan's master.
        effects: Effect registry for validating and compiling effect stacks.
        overwrite: Pass ``-y``.

    Returns:
        The compiled invocation.

    Raises:
        VisionUseError: If the plan has no video track or no clips, or if an
            effect cannot be compiled.

    """
    registry = default_effects if effects is None else effects
    chosen = profile or plan.render.master

    track = _video_track(plan)
    if track is None or not track.clips:
        raise VisionUseError(
            "This plan has no video clips to render",
            remedy="add a clip to a video track before rendering",
        )

    enabled = [clip for clip in track.clips if clip.enabled]
    if not enabled:
        raise VisionUseError(
            "Every clip in this plan is disabled",
            remedy="enable at least one clip before rendering",
        )

    canvas = _Canvas(
        width=_even(int(plan.timeline.width * chosen.scale)),
        height=_even(int(plan.timeline.height * chosen.scale)),
        fps=str(plan.timeline.frame_rate),
    )

    builder = _GraphBuilder(plan=plan, canvas=canvas, effects=registry)
    duration = builder.lay_out(enabled)
    video_label = builder.composite_overlays(_overlay_tracks(plan))
    inputs, chains = builder.inputs, builder.chains
    segments = builder.segments

    args: list[str] = ["-hide_banner", "-loglevel", "error"]
    if overwrite:
        args.append("-y")
    args.extend(inputs)
    args.extend(
        [
            "-filter_complex",
            ";".join(chains),
            "-map",
            f"[{video_label}]",
            "-map",
            "[outa]",
            "-c:v",
            chosen.codec,
            "-crf",
            str(chosen.crf),
            "-preset",
            chosen.preset,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            plan.render.audio_codec,
            str(output),
        ]
    )

    log.debug(
        "compiled plan",
        extra=bind(
            clips=len(enabled),
            segments=len(segments),
            size=f"{canvas.width}x{canvas.height}",
            fps=canvas.fps,
        ),
    )
    return CompiledRender(
        args=args,
        width=canvas.width,
        height=canvas.height,
        duration_s=duration,
        segments=len(segments),
    )
