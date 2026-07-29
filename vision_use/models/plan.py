"""``EditPlan`` -- the document that describes what should happen to a video.

This is the correction to the largest gap in the original architecture, and the
reason the rest of the framework is shaped the way it is (``PLAN.md`` § D3).

A skill does not produce a video. It produces a *patch against this document*.
Pixels are rendered exactly once, at export. The consequences are worth stating
plainly, because every other design choice here follows from them:

* **Composable.** Three skills applied in sequence are three edits to one
  document, not three decode/encode round-trips. No generation loss, and the
  composition can be inspected before a single frame is rendered.
* **Reversible.** A plan is data. An edit can be undone by discarding a patch.
* **Reviewable.** A plan is small, ordered JSON -- diffable by a human and small
  enough to hand to a model, which is good at editing JSON and bad at driving
  encoders.
* **Validated before export.** Principle 7. Overlaps, dangling references, and
  out-of-range source ranges are caught at construction, not discovered halfway
  through a render.

Like ``VideoContext``, times are quantised and structural invariants are
enforced by the model rather than documented and hoped for.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from vision_use.models.media import FrameRate

Seconds = Annotated[float, Field(ge=0)]
PlanId = Annotated[str, Field(pattern=r"^(src|trk|clp)_\d+$")]

#: How far apart two clips may be on paper and still be treated as touching
#: exactly, with neither a gap nor an overlap between them. Timestamps in a
#: plan are quantised to `timebase.TIME_PRECISION` (six decimal places, i.e.
#: microseconds); this is three orders of magnitude below that, so it absorbs
#: float subtraction noise -- `duration_s` is a plain, unquantised
#: `source_out_s - source_in_s` -- without ever masking a real gap or overlap,
#: either of which would be at least one frame.
_TIME_NOISE_TOLERANCE_S: Final = 1e-9

#: Media a plan can lay onto a track.
TrackKind = Literal["video", "audio", "overlay"]


class Source(BaseModel):
    """A media file a plan draws from.

    ``sha256`` is what ties a plan to specific bytes: a plan that renders
    correctly today must render identically tomorrow, and a source whose digest
    has changed is a different file wearing the same name.
    """

    model_config = ConfigDict(frozen=True)

    id: PlanId
    path: Path
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    duration_s: Seconds
    has_audio: bool = Field(
        default=True,
        description=(
            "Whether the file carries an audio stream. Recorded in the plan "
            "rather than probed at compile time, so compiling stays free of "
            "media I/O and a plan remains self-describing. When false the "
            "compiler generates silence for this source's clips, which is what "
            "keeps concat's stream counts consistent."
        ),
    )


class Keyframe(BaseModel):
    """One value of an animated parameter, at one moment.

    ``t_s`` is **clip-local**: zero is where the clip starts, whatever it was
    trimmed to and wherever it sits on the timeline. That is what lets a skill
    retrim or move a clip without invalidating an animation someone else put on
    it (``PLAN.md`` § D6).
    """

    model_config = ConfigDict(frozen=True)

    t_s: Seconds
    value: float


class Effect(BaseModel):
    """One entry in a clip's ordered effect stack.

    ``type`` names a registered effect; ``params`` is validated against that
    effect's own schema by the registry (M2.2), not here -- the model must be
    able to hold effects it has never heard of, or a plan produced by a plugin
    could not round-trip through a core that lacks it.

    ``schema_hash`` records the shape ``params`` was written against and always
    participates in the render cache key, so an additive schema change that
    alters output cannot silently serve a stale artefact. See ``PLAN.md`` § D4:
    schemas are additive-only and the type *is* the version, so there is no
    version field here and no migration on load.
    """

    model_config = ConfigDict(frozen=True)

    type: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)
    schema_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description="Digest of the effect's param schema at authoring time.",
    )
    animate: dict[str, tuple[Keyframe, ...]] = Field(
        default_factory=dict,
        description=(
            "Parameters that vary over the clip, keyed by parameter name. "
            "Beside `params` rather than inside it so that no effect's schema "
            "changes shape -- see PLAN.md D6. `params` still holds the value "
            "that applies when nothing is animating, so a reader that ignores "
            "this field still produces a sensible static result."
        ),
    )

    @model_validator(mode="after")
    def _check_animation(self) -> Self:
        """Reject animation that could not be interpolated."""
        for name, frames in self.animate.items():
            if not frames:
                raise ValueError(
                    f"animated parameter {name!r} has no keyframes; remove it or give it one"
                )
            times = [frame.t_s for frame in frames]
            if times != sorted(times):
                raise ValueError(f"keyframes for {name!r} are out of order")
            if len(set(times)) != len(times):
                raise ValueError(
                    f"keyframes for {name!r} repeat a time, so the value there is ambiguous"
                )
            # Animating something `params` never mentioned means the static
            # fallback is undefined, and a renderer ignoring `animate` would
            # produce a different edit than one honouring it.
            if name not in self.params:
                raise ValueError(
                    f"animated parameter {name!r} is not in params; add its static value"
                )
            if not isinstance(self.params[name], (int, float)):
                # ValueError, not TypeError: Pydantic only reports the former as
                # a validation error, and this is a bad *plan*, not a bad call.
                raise ValueError(  # noqa: TRY004 - Pydantic requires ValueError
                    f"parameter {name!r} is not numeric, so it cannot be interpolated"
                )
        return self

    def value_at(self, name: str, t_s: float) -> float | None:
        """Return an animated parameter's value at clip-local ``t_s``.

        Linear between keyframes and held flat outside them: before the first
        and after the last, the nearest keyframe's value applies. "What happens
        outside the animation" needs an obvious answer, and holding is the one
        that never surprises.

        Returns:
            The interpolated value, or ``None`` if the parameter is not
            animated -- in which case ``params`` already holds its value.

        """
        frames = self.animate.get(name)
        if not frames:
            return None
        if t_s <= frames[0].t_s:
            return frames[0].value
        if t_s >= frames[-1].t_s:
            return frames[-1].value

        for earlier, later in pairwise(frames):
            if earlier.t_s <= t_s <= later.t_s:
                span = later.t_s - earlier.t_s
                if span <= 0:  # pragma: no cover - repeated times are rejected
                    return later.value
                share = (t_s - earlier.t_s) / span
                return earlier.value + (later.value - earlier.value) * share
        return frames[-1].value  # pragma: no cover - covered by the bounds above

    @property
    def animated(self) -> bool:
        """Whether any parameter varies over the clip."""
        return bool(self.animate)


class Transition(BaseModel):
    """A timed blend into or out of a clip."""

    model_config = ConfigDict(frozen=True)

    type: str = Field(min_length=1)
    duration_s: Seconds
    params: dict[str, JsonValue] = Field(default_factory=dict)


class Clip(BaseModel):
    """A span of a source, placed at a point on the timeline.

    The distinction that matters: ``source_in_s``/``source_out_s`` address the
    *source*, while ``timeline_start_s`` addresses the *output*. Keeping them
    separate is what lets a clip be moved without re-deciding what it shows, and
    trimmed without re-deciding where it sits.
    """

    model_config = ConfigDict(frozen=True)

    id: PlanId
    source_id: PlanId
    source_in_s: Seconds
    source_out_s: Seconds
    timeline_start_s: Seconds
    effects: tuple[Effect, ...] = ()
    transition_in: Transition | None = None
    transition_out: Transition | None = None
    enabled: bool = Field(
        default=True,
        description="Disabled clips stay in the plan but render nothing, so an "
        "edit can be auditioned without discarding it.",
    )

    @model_validator(mode="after")
    def _check_source_range(self) -> Self:
        """Reject a clip that ends before it starts or has no duration."""
        if self.source_out_s <= self.source_in_s:
            raise ValueError(
                f"clip {self.id} has an empty or reversed source range: "
                f"{self.source_in_s} -> {self.source_out_s}"
            )
        return self

    @property
    def duration_s(self) -> float:
        """How long this clip occupies the timeline."""
        return self.source_out_s - self.source_in_s

    @property
    def timeline_end_s(self) -> float:
        """Where this clip stops on the timeline."""
        return self.timeline_start_s + self.duration_s

    def overlaps(self, other: Clip) -> bool:
        """Whether this clip shares timeline space with ``other``.

        Tolerant of float noise far below a frame: ``duration_s`` is a plain
        subtraction of ``source_out_s - source_in_s``, and when a skill builds
        many clips end to end from independently quantised boundaries -- as
        silence removal does -- that subtraction can land a few femtoseconds on
        either side of where two clips are meant to touch exactly. Two clips
        that were built to be back to back must not be rejected as overlapping
        because of arithmetic noise thousands of times smaller than one
        microsecond, which is the framework's own timestamp resolution
        (``timebase.TIME_PRECISION``). Found on real 40-cut media, where several
        boundaries landed exactly on this noise.
        """
        return (
            self.timeline_start_s < other.timeline_end_s - _TIME_NOISE_TOLERANCE_S
            and other.timeline_start_s < self.timeline_end_s - _TIME_NOISE_TOLERANCE_S
        )


class Track(BaseModel):
    """An ordered lane of clips of one kind."""

    model_config = ConfigDict(frozen=True)

    id: PlanId
    kind: TrackKind
    clips: tuple[Clip, ...] = ()
    muted: bool = False

    @model_validator(mode="after")
    def _check_clips(self) -> Self:
        """Reject duplicate IDs, unordered clips, or overlapping placements.

        Two clips overlapping on one track is not a stylistic choice -- it has
        no defined rendering, since both would claim the same output frames.
        Layering is expressed with a second track.
        """
        seen: set[str] = set()
        for clip in self.clips:
            if clip.id in seen:
                raise ValueError(f"duplicate clip id {clip.id} in track {self.id}")
            seen.add(clip.id)

        for left, right in pairwise(self.clips):
            if right.timeline_start_s < left.timeline_start_s:
                raise ValueError(
                    f"clips in track {self.id} are not in timeline order: "
                    f"{right.id} starts before {left.id}"
                )
            if left.overlaps(right):
                raise ValueError(f"clips {left.id} and {right.id} overlap on track {self.id}")
        return self

    @property
    def duration_s(self) -> float:
        """Where the last clip on this track ends."""
        return max((clip.timeline_end_s for clip in self.clips), default=0.0)

    def gaps(self) -> list[tuple[float, float]]:
        """Return uncovered spans between clips.

        Gaps are legal -- they render as black or silence -- but a gap is more
        often an accident than an intent, so the validator surfaces them.

        Tolerant of the same float noise :meth:`Clip.overlaps` is: two clips
        meant to sit exactly back to back can leave `cursor` a few
        femtoseconds short of the next clip's start. That is not a gap -- and
        reporting it as one used to make the *render* compiler insert a
        zero-duration filler segment, which does not just render as nothing:
        ffmpeg never emits EOF for a zero-duration lavfi source, so the render
        hung forever. This is the other place that same false positive came
        from.
        """
        found: list[tuple[float, float]] = []
        cursor = 0.0
        for clip in self.clips:
            if clip.timeline_start_s > cursor + _TIME_NOISE_TOLERANCE_S:
                found.append((cursor, clip.timeline_start_s))
            cursor = max(cursor, clip.timeline_end_s)
        return found


class Timeline(BaseModel):
    """The output canvas and everything laid onto it."""

    model_config = ConfigDict(frozen=True)

    frame_rate: FrameRate
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    tracks: tuple[Track, ...] = ()

    @model_validator(mode="after")
    def _check_tracks(self) -> Self:
        """Reject duplicate track IDs."""
        seen: set[str] = set()
        for track in self.tracks:
            if track.id in seen:
                raise ValueError(f"duplicate track id {track.id}")
            seen.add(track.id)
        return self

    @property
    def duration_s(self) -> float:
        """Where the last clip on any track ends."""
        return max((track.duration_s for track in self.tracks), default=0.0)

    def clips(self) -> list[Clip]:
        """Every clip across every track, in track then timeline order."""
        return [clip for track in self.tracks for clip in track.clips]


class RenderProfile(BaseModel):
    """Encoder settings for one output size."""

    model_config = ConfigDict(frozen=True)

    scale: float = Field(default=1.0, gt=0, le=1.0, description="Fraction of canvas size.")
    codec: str = "libx264"
    crf: int = Field(default=20, ge=0, le=51)
    preset: str = "medium"


class RenderSettings(BaseModel):
    """How a plan becomes a file.

    Two profiles, not one. The agent loop renders at proxy scale, where an
    iteration costs seconds; export renders at master scale. The plan itself is
    resolution-independent, so the same document produces both -- which is what
    keeps the retry loop affordable.
    """

    model_config = ConfigDict(frozen=True)

    container: str = "mp4"
    master: RenderProfile = RenderProfile()
    proxy: RenderProfile = RenderProfile(scale=0.5, crf=28, preset="veryfast")
    audio_codec: str = "aac"
    loudnorm_lufs: float | None = Field(
        default=None,
        description="Target integrated loudness. None leaves levels untouched.",
    )


class EditPlan(BaseModel):
    """A complete, renderable description of an edit.

    Skills mutate this document and never touch pixels. The renderer is the only
    component that encodes, and it does so once, at export.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1"] = "1"
    sources: dict[str, Source] = Field(default_factory=dict)
    timeline: Timeline
    render: RenderSettings = RenderSettings()

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        """Reject clips naming unknown sources or ranges past a source's end.

        Caught here rather than at render time because a dangling reference
        discovered halfway through an export has already cost the user the
        render, and because a skill that produces one has a bug worth surfacing
        immediately.
        """
        for key, source in self.sources.items():
            if key != source.id:
                raise ValueError(
                    f"source keyed {key!r} declares id {source.id!r}; they must match"
                )

        for clip in self.timeline.clips():
            referenced = self.sources.get(clip.source_id)
            if referenced is None:
                raise ValueError(f"clip {clip.id} references unknown source {clip.source_id}")
            if referenced.duration_s and clip.source_out_s > referenced.duration_s:
                raise ValueError(
                    f"clip {clip.id} reads to {clip.source_out_s}s of "
                    f"{clip.source_id}, which is {referenced.duration_s}s long"
                )
        return self

    @property
    def duration_s(self) -> float:
        """Output duration."""
        return self.timeline.duration_s

    def clip_by_id(self, clip_id: str) -> Clip | None:
        """Return the clip with ``clip_id``, or ``None``."""
        return next((clip for clip in self.timeline.clips() if clip.id == clip_id), None)

    def track_by_id(self, track_id: str) -> Track | None:
        """Return the track with ``track_id``, or ``None``."""
        return next((track for track in self.timeline.tracks if track.id == track_id), None)

    def unused_sources(self) -> list[str]:
        """Return declared sources no clip draws from.

        Not an error: a plan under construction may declare a source before
        placing it. The validator reports these so an accident is visible.
        """
        used = {clip.source_id for clip in self.timeline.clips()}
        return sorted(key for key in self.sources if key not in used)
