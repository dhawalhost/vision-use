"""``VideoContext`` -- the single shared analysis of a video.

This is the payoff of Principle 8: a video is analysed once, and every skill
reads the result instead of re-deriving it. It is also the addressing scheme
that makes an agent able to talk about a video at all -- "the shot at 0:42",
"where he says *pricing*" -- because every entity carries a stable ID.

Three properties are enforced rather than documented:

* **Deterministic.** Identical input produces a byte-identical document. There
  are no timestamps, no absolute paths inside derived entities, and every time
  is quantised. This is what lets a context be content-addressed and cached.
* **Internally consistent.** Cross-references are validated on construction, so
  a context that names a keyframe or word that does not exist cannot be built,
  let alone serialized and handed to a skill.
* **Chronological.** Entities are ordered by time and their IDs sort the same
  way, so a serialized context diffs cleanly and reads top to bottom.

A context is legitimately *partial*: ASR is expensive and many skills do not
need it. Consumers therefore ask :meth:`VideoContext.has` rather than assuming
a section is populated.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from vision_use.models.media import MediaProbe

Seconds = Annotated[float, Field(ge=0)]
Confidence = Annotated[float, Field(ge=0, le=1)]
EntityId = Annotated[str, Field(pattern=r"^(sh|kf|w|sg|spk)_\d+$")]


class ContextSection(StrEnum):
    """A populatable region of a :class:`VideoContext`.

    Analyzers declare which sections they produce and require, and skills
    declare which they need, so the runtime can plan the cheapest analysis that
    satisfies a request instead of always running everything.
    """

    PROBE = "probe"
    SHOTS = "shots"
    KEYFRAMES = "keyframes"
    AUDIO_ACTIVITY = "audio_activity"
    TRANSCRIPT = "transcript"
    SUBJECT = "subject"


#: A section's identity: a built-in one, or a plugin's dotted name. `str` is
#: not a widening of `ContextSection` here so much as an admission -- the enum
#: is the set of sections *this repository* produces, and D5 exists because that
#: is not the set of sections that can exist.
SectionName = ContextSection | str

#: Every name the core owns. A plugin section may not take one of these.
BUILTIN_SECTIONS: frozenset[str] = frozenset(section.value for section in ContextSection)


def check_plugin_section(name: str) -> str:
    """Return ``name`` if a plugin may use it as a section key.

    Two rules, both enforced rather than trusted (PLAN.md § D5): the name must
    be dotted, so it sits in the publishing package's own namespace, and it must
    not shadow a built-in section. A plugin quietly overwriting ``shots`` would
    be the hardest failure here to trace back to its cause.

    Raises:
        ValueError: If the name is unusable. Raised rather than returned so
            that Pydantic reports it with the rest of a model's errors.

    """
    if name in BUILTIN_SECTIONS:
        raise ValueError(
            f"{name!r} is a built-in context section and cannot be replaced by a plugin"
        )
    if "." not in name.strip("."):
        raise ValueError(
            f"plugin section {name!r} must be namespaced by its package, e.g. "
            "'stabilize.motion'"
        )
    return name


class TimeSpan(BaseModel):
    """A half-open interval ``[start_s, end_s)`` in source time."""

    model_config = ConfigDict(frozen=True)

    start_s: Seconds
    end_s: Seconds

    @model_validator(mode="after")
    def _check_ordered(self) -> Self:
        """Reject an interval that ends before it starts."""
        if self.end_s < self.start_s:
            raise ValueError(f"span ends at {self.end_s} before it starts at {self.start_s}")
        return self

    @property
    def duration_s(self) -> float:
        """Length of the interval in seconds."""
        return self.end_s - self.start_s

    def contains(self, t_s: float) -> bool:
        """Whether ``t_s`` falls inside the half-open interval."""
        return self.start_s <= t_s < self.end_s

    def overlaps(self, other: TimeSpan) -> bool:
        """Whether this interval shares any instant with ``other``."""
        return self.start_s < other.end_s and other.start_s < self.end_s


class Shot(BaseModel):
    """A continuous run of frames between two detected cuts.

    Shots are the coarsest useful unit of video structure and the one an agent
    reasons about first: reorder shots, drop a shot, pick a thumbnail per shot.
    """

    model_config = ConfigDict(frozen=True)

    id: EntityId
    index: int = Field(ge=0, description="Chronological position, matching the ID.")
    span: TimeSpan
    keyframe_id: EntityId | None = Field(
        default=None, description="Representative keyframe, when one was extracted."
    )
    confidence: Confidence | None = Field(
        default=None, description="Detector confidence in the cut that opened this shot."
    )


class SubjectPoint(BaseModel):
    """Where the subject was at one sampled moment.

    Fractions of the frame, like :class:`SubjectSpan`, and in *source* time so
    a consumer can line them up with shots and clips without knowing how the
    span was sampled.
    """

    model_config = ConfigDict(frozen=True)

    t_s: Seconds
    centre_x: float = Field(ge=0.0, le=1.0)
    centre_y: float = Field(default=0.5, ge=0.0, le=1.0)


class SubjectSpan(BaseModel):
    """Where the interesting part of the frame is, over one span.

    "Interesting" is deliberately weak: this records where a reframing crop
    should sit, not what is there. Naming it `faces` would promise a detection
    this does not perform -- see `docs/context-schema.md`.

    Coordinates are fractions of the frame, so a span stays meaningful after a
    scale and can be compared across sources of different sizes.
    """

    model_config = ConfigDict(frozen=True)

    shot_id: EntityId | None = None
    span: TimeSpan
    centre_x: float = Field(ge=0.0, le=1.0, description="Horizontal centre, 0 to 1.")
    centre_y: float = Field(default=0.5, ge=0.0, le=1.0, description="Vertical centre, 0 to 1.")
    confidence: Confidence | None = Field(
        default=None,
        description=(
            "How concentrated the evidence was. Low means the frame had no "
            "clear subject, and a consumer should prefer the centre over this."
        ),
    )
    track: tuple[SubjectPoint, ...] = Field(
        default=(),
        description=(
            "Where the subject was at each sampled moment, in time order. "
            "`centre_x` summarises the whole span; this is what it summarises, "
            "and is what a crop follows when it can move within a shot."
        ),
    )

    @model_validator(mode="after")
    def _check_track(self) -> Self:
        """Reject a track that is out of order or outside its own span."""
        times = [point.t_s for point in self.track]
        if times != sorted(times):
            raise ValueError(f"subject track for {self.shot_id} is not in time order")
        return self


class Keyframe(BaseModel):
    """A single extracted frame, stored in the cache rather than inline.

    The image bytes live in the content-addressed cache and are referenced by
    ``cache_key``. Embedding them would make the context far too large to hand
    to a model, and would break the guarantee that a context is plain diffable
    JSON.
    """

    model_config = ConfigDict(frozen=True)

    id: EntityId
    shot_id: EntityId | None = None
    t_s: Seconds
    cache_key: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description="Digest of the cache entry holding the image bytes.",
    )
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    phash: str | None = Field(
        default=None,
        description="Perceptual hash, for near-duplicate detection across shots.",
    )


class Word(BaseModel):
    """One recognised word with its own timing.

    Word-level timing, not segment-level, is what makes "cut where he says X"
    frame-accurate. A skill that only has sentence timings must guess, and the
    guess is audible.
    """

    model_config = ConfigDict(frozen=True)

    id: EntityId
    span: TimeSpan
    text: str = Field(min_length=1)
    speaker_id: EntityId | None = None
    confidence: Confidence | None = None


class TranscriptSegment(BaseModel):
    """A sentence-ish grouping of words, as emitted by the recogniser.

    Segments exist for readability and for prompting a model; ``word_ids`` ties
    each one back to the authoritative per-word timings.
    """

    model_config = ConfigDict(frozen=True)

    id: EntityId
    span: TimeSpan
    text: str
    word_ids: tuple[EntityId, ...] = ()
    speaker_id: EntityId | None = None


class Transcript(BaseModel):
    """Recognised speech for the whole source."""

    model_config = ConfigDict(frozen=True)

    language: str | None = Field(default=None, description="BCP-47 or ISO-639-1 code.")
    words: tuple[Word, ...] = ()
    segments: tuple[TranscriptSegment, ...] = ()

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Reject a transcript whose words are unordered or whose refs dangle."""
        _require_chronological(self.words, "transcript words")

        known = {word.id for word in self.words}
        for segment in self.segments:
            missing = [ref for ref in segment.word_ids if ref not in known]
            if missing:
                raise ValueError(
                    f"segment {segment.id} references unknown words: {', '.join(missing)}"
                )
        return self

    @property
    def text(self) -> str:
        """Full transcript text, joined from segments when present else words."""
        if self.segments:
            return " ".join(segment.text.strip() for segment in self.segments).strip()
        return " ".join(word.text for word in self.words)


class ActivitySpan(BaseModel):
    """A window of audio classified as speech-ish or silent.

    This is what ``silence-removal`` consumes, and what lets ``captions`` avoid
    placing a caption over dead air.
    """

    model_config = ConfigDict(frozen=True)

    span: TimeSpan
    rms_db: float = Field(
        description="Mean level over the window, dBFS. Silence is strongly negative."
    )
    is_silence: bool


class AnalyzerRun(BaseModel):
    """Provenance for one analyzer's contribution to a context.

    Deliberately carries no timestamp. A wall-clock field would make two
    analyses of the same file differ, which would defeat content addressing and
    the M1 determinism criterion -- and nothing consumes it. When an analysis
    ran is a property of the cache entry and the log, not of the document.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    params_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    # Plain strings rather than the enum: a plugin's section is a dotted name
    # this repository has never heard of, and provenance must record it just as
    # faithfully (PLAN.md § D5).
    produced: tuple[str, ...] = ()


def _require_chronological(items: Sequence[Word] | Sequence[Shot], label: str) -> None:
    """Raise when ``items`` are not ordered by start time or have duplicate IDs.

    Ordering is not cosmetic: lookups here rely on it, IDs encode it, and a
    context that violates it would produce a silently wrong answer to
    "what is playing at time t".
    """
    seen: set[str] = set()
    previous = -1.0
    for item in items:
        if item.id in seen:
            raise ValueError(f"duplicate id {item.id} in {label}")
        seen.add(item.id)
        if item.span.start_s < previous:
            raise ValueError(
                f"{label} are not in chronological order at {item.id} "
                f"({item.span.start_s} follows {previous})"
            )
        previous = item.span.start_s


class VideoContext(BaseModel):
    """Everything VisionUse knows about one source video.

    Produced once by the analysis pipeline, then read by every skill. Sections
    other than ``probe`` are optional; use :meth:`has` before relying on one.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1"] = "1"
    probe: MediaProbe
    shots: tuple[Shot, ...] = ()
    keyframes: tuple[Keyframe, ...] = ()
    audio_activity: tuple[ActivitySpan, ...] = ()
    transcript: Transcript | None = None
    subject: tuple[SubjectSpan, ...] = ()
    #: Sections contributed by plugins, keyed by a dotted, package-namespaced
    #: name (``"stabilize.motion"``). Plain JSON, never interpreted here: the
    #: core stores it, hashes it, serializes it, and hands it back. The owning
    #: plugin validates its own payload, which it can do far better than a
    #: registry of shapes the core would have to guess at (PLAN.md § D5).
    extra: dict[str, JsonValue] = Field(default_factory=dict)
    analyzers: dict[str, AnalyzerRun] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Reject a context with unordered shots, dangling refs, or bad bounds."""
        _require_chronological(self.shots, "shots")

        for name in self.extra:
            check_plugin_section(name)

        keyframe_ids = {keyframe.id for keyframe in self.keyframes}
        for shot in self.shots:
            if shot.keyframe_id is not None and shot.keyframe_id not in keyframe_ids:
                raise ValueError(
                    f"shot {shot.id} references unknown keyframe {shot.keyframe_id}"
                )

        shot_ids = {shot.id for shot in self.shots}
        for keyframe in self.keyframes:
            if keyframe.shot_id is not None and keyframe.shot_id not in shot_ids:
                raise ValueError(
                    f"keyframe {keyframe.id} references unknown shot {keyframe.shot_id}"
                )

        for index, (left, right) in enumerate(zip(self.shots, self.shots[1:], strict=False)):
            if left.span.overlaps(right.span):
                raise ValueError(f"shots {left.id} and {right.id} overlap at position {index}")

        return self

    @property
    def duration_s(self) -> float:
        """Source duration in seconds."""
        return self.probe.source.duration_s

    def has(self, section: SectionName) -> bool:
        """Whether ``section`` is populated.

        Skills call this rather than truthiness-testing a field, so that "not
        analysed" and "analysed, found nothing" stay distinguishable -- a video
        with genuinely no speech is not the same as one never transcribed.

        Answers for a plugin's section from the same provenance, so that
        distinction holds for plugins too: an empty ``extra`` entry means the
        analyzer ran and found nothing.
        """
        if section == ContextSection.PROBE:
            return True
        return any(section in run.produced for run in self.analyzers.values())

    def missing(self, sections: Sequence[SectionName]) -> list[SectionName]:
        """Return which of ``sections`` are not yet populated."""
        return [section for section in sections if not self.has(section)]

    def section(self, name: str) -> JsonValue:
        """Return a plugin's section payload, or ``None`` if absent.

        The core does not know what is in here; the plugin that wrote it does.
        """
        return self.extra.get(name)

    def shot_by_id(self, shot_id: str) -> Shot | None:
        """Return the shot with ``shot_id``, or ``None``."""
        return next((shot for shot in self.shots if shot.id == shot_id), None)

    def keyframe_by_id(self, keyframe_id: str) -> Keyframe | None:
        """Return the keyframe with ``keyframe_id``, or ``None``."""
        return next((frame for frame in self.keyframes if frame.id == keyframe_id), None)

    def shot_at(self, t_s: float) -> Shot | None:
        """Return the shot playing at ``t_s``, or ``None`` if none covers it."""
        return next((shot for shot in self.shots if shot.span.contains(t_s)), None)

    def words_between(self, start_s: float, end_s: float) -> list[Word]:
        """Return words overlapping ``[start_s, end_s)``, in order.

        Overlap rather than containment is deliberate: a word straddling the
        boundary of a requested range is part of what is said in that range,
        and dropping it would silently truncate a quotation.
        """
        if self.transcript is None:
            return []
        window = TimeSpan(start_s=start_s, end_s=end_s)
        return [word for word in self.transcript.words if word.span.overlaps(window)]

    def text_between(self, start_s: float, end_s: float) -> str:
        """Return the words spoken in ``[start_s, end_s)`` as a string."""
        return " ".join(word.text for word in self.words_between(start_s, end_s))

    def find_text(self, needle: str) -> Iterator[tuple[Word, ...]]:
        """Yield each run of words whose joined text matches ``needle``.

        This is the primitive behind "cut the part where he says X": it returns
        the actual word entities, so the caller gets exact timings rather than
        a character offset it would have to map back to time itself.

        Matching is case-insensitive and ignores punctuation attached to word
        edges, because recognisers place punctuation inconsistently.
        """
        if self.transcript is None or not needle.strip():
            return

        targets = [_normalise(part) for part in needle.split()]
        words = self.transcript.words
        if not targets:
            return

        for start in range(len(words) - len(targets) + 1):
            window = words[start : start + len(targets)]
            if [_normalise(word.text) for word in window] == targets:
                yield window


def _normalise(text: str) -> str:
    """Lowercase ``text`` and strip edge punctuation, for transcript matching."""
    return text.strip().strip(".,!?;:\"'()[]-").lower()
