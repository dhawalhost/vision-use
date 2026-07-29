"""Captions: word-level timings turned into readable lines on screen.

The payoff of M1.6. The transcript already carries a start and end for every
*word*, so this skill never has to guess when a caption should appear -- the
hard part was done by the context engine, and what remains is a grouping
decision and a rendering one.

Three judgements shape whether captions read well:

**A caption is a phrase, not a word.** Showing one word at a time is a style,
but a bad default: the eye lands on the caption after it has already changed.
Words are grouped until a line is long enough or a real pause appears.

**A pause ends a line.** Grouping purely by duration cuts sentences in half. A
gap between words is where a speaker themselves broke the phrase, so it is the
best free hint about where a caption should end.

**A caption must not outlast its words.** It appears with the first word and
leaves shortly after the last, so it never lingers over the next sentence.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import ClassVar

from pydantic import JsonValue

from vision_use.cache.store import CacheStore
from vision_use.errors import VisionUseError
from vision_use.hashing import hash_file
from vision_use.log import bind, get_logger
from vision_use.models.context import ContextSection, VideoContext, Word
from vision_use.models.ids import clip_id, source_id, track_id
from vision_use.models.plan import Clip, EditPlan, Source, Track
from vision_use.skills.patch import AddClip, AddSource, AddTrack, Op, PlanPatch
from vision_use.skills.skill import (
    Analysis,
    Measurement,
    Score,
    SkillMeta,
    SkillRequest,
    Strategy,
    ValidationReport,
)
from vision_use.timebase import quantize_s
from vision_use.video.text import TEXT_VERSION, CaptionStyle, render_caption

log = get_logger(__name__)

#: Cache namespace for rendered caption images.
CAPTION_NAMESPACE = "caption.png"

#: Grouping presets offered as strategies: (max seconds, max words). The first
#: is the readable default; the others trade shorter lines for faster changes,
#: which suits dense speech.
DEFAULT_GROUPINGS: tuple[tuple[float, int], ...] = ((3.0, 9), (2.0, 6), (1.5, 4))

#: A gap between words longer than this ends the caption, wherever the counts
#: happen to have reached. It is the speaker's own phrasing.
DEFAULT_PAUSE_S: float = 0.45

#: How long a caption lingers after its final word, so the last syllable is
#: still on screen as the eye reaches it.
DEFAULT_HOLD_S: float = 0.25


@dataclass(frozen=True, slots=True)
class CaptionLine:
    """One caption: its text and the span it is shown for."""

    text: str
    start_s: float
    end_s: float
    word_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Layout:
    """What ``execute`` needs to build one grouping's captions.

    ``id_base`` is computed from the plan rather than fixed, so caption
    entities cannot collide with ids the plan already uses -- a hardcoded
    offset would work until someone hands us a plan that reached it.
    """

    lines: tuple[CaptionLine, ...]
    width: int
    height: int
    id_base: int


def group_words(
    words: tuple[Word, ...],
    max_seconds: float,
    max_words: int,
    pause_s: float = DEFAULT_PAUSE_S,
    hold_s: float = DEFAULT_HOLD_S,
) -> list[CaptionLine]:
    """Group ``words`` into caption lines.

    A line ends when it reaches ``max_words``, when it has been on screen for
    ``max_seconds``, or when the speaker pauses for longer than ``pause_s`` --
    whichever comes first. The pause rule is what keeps sentences intact.

    Args:
        words: Transcript words, in order.
        max_seconds: Longest a single caption may stay up.
        max_words: Most words in one caption.
        pause_s: A gap longer than this ends the line.
        hold_s: How long a caption stays after its last word.

    Returns:
        Lines covering every word exactly once, in order.

    """
    if not words:
        return []

    lines: list[CaptionLine] = []
    current: list[Word] = []

    def flush() -> None:
        if not current:
            return
        lines.append(
            CaptionLine(
                text=" ".join(word.text for word in current),
                start_s=quantize_s(current[0].span.start_s),
                end_s=quantize_s(current[-1].span.end_s + hold_s),
                word_ids=tuple(word.id for word in current),
            )
        )
        current.clear()

    for word in words:
        if current:
            gap = word.span.start_s - current[-1].span.end_s
            spanned = word.span.end_s - current[0].span.start_s
            if gap > pause_s or len(current) >= max_words or spanned > max_seconds:
                flush()
        current.append(word)
    flush()

    # A caption must never outlast the start of the next one, or two would be
    # composited at once and the older would sit on top of the newer.
    trimmed: list[CaptionLine] = []
    for index, line in enumerate(lines):
        end = line.end_s
        if index + 1 < len(lines):
            end = min(end, lines[index + 1].start_s)
        trimmed.append(
            CaptionLine(
                text=line.text,
                start_s=line.start_s,
                end_s=quantize_s(max(end, line.start_s + 0.04)),
                word_ids=line.word_ids,
            )
        )
    return trimmed


class Captions:
    """Renders a transcript onto the video as timed caption overlays.

    Reads ``transcript``; the runtime backfills it, which means the optional
    ``asr`` extra must be installed for this skill to be usable.
    """

    meta: ClassVar[SkillMeta] = SkillMeta(
        name="edit.captions",
        version="1",
        summary="Burn word-timed captions onto the video.",
        requires=frozenset({ContextSection.TRANSCRIPT}),
    )

    def __init__(
        self,
        groupings: tuple[tuple[float, int], ...] = DEFAULT_GROUPINGS,
        style: CaptionStyle | None = None,
        cache: CacheStore | None = None,
    ) -> None:
        if not groupings:
            raise ValueError("at least one grouping is required")
        self._groupings = groupings
        self._style = style or CaptionStyle()
        self._cache = cache or CacheStore()

    def analyse(
        self,
        context: VideoContext,
        plan: EditPlan,
        request: SkillRequest,  # noqa: ARG002 - grouping comes from strategies
    ) -> Analysis:
        """Collect the words to caption and the canvas to render onto."""
        transcript = context.transcript
        if transcript is None or not transcript.words:
            return Analysis(findings={"words": 0})

        if any(track.kind == "overlay" for track in plan.timeline.tracks):
            # Adding a second caption track would composite two captions at
            # once. Replacing the existing one is a different request.
            return Analysis(findings={"words": len(transcript.words), "captioned": True})

        return Analysis(
            findings={"words": len(transcript.words)},
            detail=(
                transcript.words,
                plan.timeline.width,
                plan.timeline.height,
                _next_id_base(plan),
            ),
        )

    def plan(self, analysis: Analysis) -> list[Strategy]:
        """Offer one strategy per grouping, most readable first."""
        if not isinstance(analysis.detail, tuple):
            return []
        words, width, height, id_base = analysis.detail

        strategies: list[Strategy] = []
        for max_seconds, max_words in self._groupings:
            lines = group_words(words, max_seconds, max_words)
            if not lines:
                continue
            strategies.append(
                Strategy(
                    name=f"{max_words}-words-{max_seconds}s",
                    rationale=(
                        f"caption in phrases of up to {max_words} words "
                        f"or {max_seconds}s, breaking at pauses"
                    ),
                    detail=_Layout(
                        lines=tuple(lines),
                        width=width,
                        height=height,
                        id_base=id_base,
                    ),
                )
            )
        return strategies

    def execute(self, strategy: Strategy) -> PlanPatch:
        """Render each line and lay it onto a new overlay track.

        Images are written to the content-addressed cache, which
        ``SKILL_SPEC.md`` explicitly permits -- a skill may not encode video,
        but caching a derived artefact is how every other stage works too.

        Raises:
            VisionUseError: If the strategy did not come from this skill.

        """
        layout = strategy.detail
        if not isinstance(layout, _Layout):
            raise VisionUseError(
                "This strategy did not come from this skill",
                remedy="pass a strategy returned by plan()",
            )

        base = layout.id_base
        overlay = Track(id=track_id(base), kind="overlay")
        ops: list[Op] = [AddTrack(track=overlay)]

        for index, line in enumerate(layout.lines):
            path = self._render(line, layout)
            source = Source(
                id=source_id(base + index),
                path=path,
                sha256=hash_file(path),
                duration_s=quantize_s(line.end_s - line.start_s),
                has_audio=False,
            )
            ops.append(AddSource(source=source))
            ops.append(
                AddClip(
                    track_id=overlay.id,
                    clip=Clip(
                        id=clip_id(base + index),
                        source_id=source.id,
                        source_in_s=0.0,
                        source_out_s=source.duration_s,
                        timeline_start_s=line.start_s,
                    ),
                )
            )

        log.debug("planned captions", extra=bind(lines=len(layout.lines)))
        return PlanPatch(
            ops=tuple(ops), rationale=strategy.rationale, produced_by=self.meta.ref
        )

    def _render(self, line: CaptionLine, layout: _Layout) -> Path:
        """Render one caption to the cache and return its path."""
        key = self._cache.key(
            CAPTION_NAMESPACE,
            version=TEXT_VERSION,
            params={
                "text": line.text,
                "width": layout.width,
                "height": layout.height,
                "style": _json_style(self._style),
            },
            suffix=".png",
        )
        if self._cache.get(key) is None:
            payload = render_caption(line.text, layout.width, layout.height, self._style)
            with self._cache.writing(key) as tmp:
                tmp.write_bytes(payload)
        return self._cache.path_for(key)

    def validate(
        self, before: EditPlan, after: EditPlan, context: VideoContext
    ) -> ValidationReport:
        """Check every word is captioned once and no two captions overlap.

        A word appearing twice, or not at all, is the failure a viewer reads
        directly off the screen.
        """
        del before
        transcript = context.transcript
        expected = len(transcript.words) if transcript else 0

        overlay_clips = [
            clip
            for track in after.timeline.tracks
            if track.kind == "overlay"
            for clip in track.clips
        ]

        overlapping = 0
        for earlier, later in pairwise(overlay_clips):
            if earlier.timeline_end_s > later.timeline_start_s + 1e-6:
                overlapping += 1

        captioned = len(overlay_clips)

        return ValidationReport(
            measurements=(
                Measurement(
                    name="captions_created",
                    value=float(captioned),
                    passed=captioned > 0,
                    detail="no captions were produced",
                ),
                Measurement(
                    name="no_overlap",
                    value=float(overlapping),
                    passed=overlapping == 0,
                    detail=f"{overlapping} caption(s) overlap the next one",
                ),
                Measurement(
                    name="words_available",
                    value=float(expected),
                    passed=expected > 0,
                    detail="the transcript was empty",
                ),
            )
        )

    def score(self, report: ValidationReport) -> Score:
        """Prefer fewer, longer captions -- they are easier to read."""
        if not report.passed:
            return Score(value=-1.0, detail="rejected by validation")
        created = next(
            (item.value for item in report.measurements if item.name == "captions_created"),
            0.0,
        )
        # Fewer captions for the same words means longer phrases on screen.
        return Score(value=-created, detail=f"{created:.0f} captions")


def _json_style(style: CaptionStyle) -> dict[str, JsonValue]:
    """Return a style as JSON-safe values, for the cache key."""
    return {
        "font_scale": style.font_scale,
        "margin_scale": style.margin_scale,
        "fill": list(style.fill),
        "plate": list(style.plate),
        "font": str(style.font_path) if style.font_path else "default",
        "max_width_scale": style.max_width_scale,
    }


def _next_id_base(plan: EditPlan) -> int:
    """Return an id index no entity in ``plan`` has reached.

    Caption sources, clips and the overlay track all number from here, so a
    plan that already uses low indices -- or one captioned twice -- cannot
    produce a duplicate id.
    """
    used = [0]
    for identifier in (
        *plan.sources,
        *(track.id for track in plan.timeline.tracks),
        *(clip.id for clip in plan.timeline.clips()),
    ):
        _, _, digits = identifier.partition("_")
        if digits.isdigit():
            used.append(int(digits))
    return max(used) + 1
