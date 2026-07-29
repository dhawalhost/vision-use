"""Silence removal: the jump cut, done from the shared context.

The first skill that answers a request in the user's own words -- *"cut out the
dead air"* -- and the one every piece before it was built to support. It reads
``audio_activity`` from the context, decides which silences are worth cutting,
and returns a patch that replaces one clip with several.

Three judgements make the difference between a usable cut and an unusable one,
and none of them are about the algorithm:

**Not every silence is dead air.** A pause between sentences is speech. Only
silences longer than a threshold are cut, and the threshold is what the
strategies vary.

**Cuts need padding.** Slicing at the exact moment the waveform drops clips the
tail off the last word and the breath before the next. A little silence is kept
on each side, which is the difference between a cut that sounds deliberate and
one that sounds broken.

**Cuts land on frames.** A cut point derived from audio falls mid-frame, and the
encoder then resolves it arbitrarily. Every boundary is snapped through
``timebase`` first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from vision_use.errors import VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.context import ContextSection, VideoContext
from vision_use.models.ids import clip_id
from vision_use.models.plan import Clip, EditPlan
from vision_use.skills.patch import AddClip, Op, PlanPatch, RemoveClip
from vision_use.skills.skill import (
    Analysis,
    Measurement,
    Score,
    SkillMeta,
    SkillRequest,
    Strategy,
    ValidationReport,
)
from vision_use.timebase import quantize_s, snap_to_frame

log = get_logger(__name__)

#: Silence thresholds offered as successive strategies, most cautious first.
#: A shorter threshold cuts more and risks clipping natural pauses, so the
#: retry loop only reaches for it when the cautious one is rejected.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.8, 0.5, 0.35)

#: Seconds of silence kept either side of a cut. Without it, a cut lands the
#: instant the waveform drops, which clips the tail of the last word.
DEFAULT_PAD_S: float = 0.12

#: A kept span shorter than this is not worth its own clip -- it would be a
#: flash of a frame or two between cuts.
MIN_KEEP_S: float = 0.2

#: Fraction of the source's speech that must survive a cut. Some loss at the
#: padded boundaries is unavoidable; much more than this means the cut ate
#: words, which is the failure a viewer actually notices.
MIN_SPEECH_KEPT: float = 0.98


@dataclass(frozen=True, slots=True)
class _Span:
    """A stretch of source time to keep."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """How long this span lasts."""
        return self.end_s - self.start_s


@dataclass(frozen=True, slots=True)
class _Cut:
    """What ``execute`` needs to carry out one threshold."""

    threshold_s: float
    pad_s: float
    track_id: str
    clip: Clip
    keeps: tuple[_Span, ...]


def _silences(context: VideoContext, threshold_s: float) -> list[_Span]:
    """Return silences longer than ``threshold_s``, in source time."""
    return [
        _Span(span.span.start_s, span.span.end_s)
        for span in context.audio_activity
        if span.is_silence and span.span.duration_s > threshold_s
    ]


def _keeps(clip: Clip, silences: list[_Span], pad_s: float) -> tuple[_Span, ...]:
    """Return the spans of ``clip`` worth keeping once ``silences`` are cut.

    Padding shrinks each silence from both ends before it is removed, so the
    kept spans overlap the quiet moments either side of speech.
    """
    kept: list[_Span] = []
    cursor = clip.source_in_s

    for silence in silences:
        start = max(silence.start_s + pad_s, clip.source_in_s)
        end = min(silence.end_s - pad_s, clip.source_out_s)
        if end <= start:
            # Padding was wider than the silence, so there is nothing to remove.
            continue
        if end <= cursor:
            # Already inside a stretch an earlier silence consumed.
            continue
        # `start > cursor` is checked separately from the keep length: a silence
        # beginning exactly where the cursor sits -- leading dead air, which is
        # very common -- contributes no keep but must still advance the cursor.
        if start > cursor and start - cursor >= MIN_KEEP_S:
            kept.append(_Span(cursor, start))
        cursor = max(cursor, end)

    if clip.source_out_s - cursor >= MIN_KEEP_S:
        kept.append(_Span(cursor, clip.source_out_s))
    return tuple(kept)


class RemoveSilence:
    """Cuts long silences out of a clip, closing the gaps.

    Reads ``audio_activity``; the runtime backfills it if the caller has not
    already analysed the source.
    """

    meta: ClassVar[SkillMeta] = SkillMeta(
        name="edit.remove_silence",
        version="1",
        summary="Cut long silences and close the gaps.",
        requires=frozenset({ContextSection.AUDIO_ACTIVITY}),
    )

    def __init__(
        self,
        thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
        pad_s: float = DEFAULT_PAD_S,
    ) -> None:
        if not thresholds:
            raise ValueError("at least one silence threshold is required")
        if pad_s < 0:
            raise ValueError(f"pad_s must not be negative, got {pad_s}")
        self._thresholds = thresholds
        self._pad_s = pad_s

    def analyse(
        self,
        context: VideoContext,
        plan: EditPlan,
        request: SkillRequest,  # noqa: ARG002 - the threshold comes from strategies
    ) -> Analysis:
        """Find the clip to cut and how much silence the context reports."""
        located = [
            (track.id, clip)
            for track in plan.timeline.tracks
            for clip in track.clips
            if clip.enabled
        ]
        if len(located) != 1:
            # Cutting an already-cut timeline needs a rule for what happens to
            # the existing boundaries, and there is no good default. Better to
            # do nothing than to guess.
            return Analysis(findings={"clips": len(located)})

        silent_s = sum(
            span.span.duration_s for span in context.audio_activity if span.is_silence
        )
        track_id, clip = located[0]
        return Analysis(
            findings={
                "silent_s": round(silent_s, 3),
                "spans": len(context.audio_activity),
            },
            detail=(track_id, clip, context),
        )

    def plan(self, analysis: Analysis) -> list[Strategy]:
        """Offer one strategy per threshold, most cautious first."""
        if not isinstance(analysis.detail, tuple):
            return []
        track_id, clip, context = analysis.detail

        strategies: list[Strategy] = []
        for threshold in self._thresholds:
            keeps = snap_keeps(
                _keeps(clip, _silences(context, threshold), self._pad_s), context
            )
            if not keeps:
                continue
            if len(keeps) == 1 and keeps[0].duration_s >= clip.duration_s:
                # Nothing was actually removed; offering it would waste an
                # attempt from the retry budget.
                continue
            strategies.append(
                Strategy(
                    name=f"over-{threshold}s",
                    rationale=(
                        f"cut silences longer than {threshold}s, "
                        f"keeping {self._pad_s}s either side"
                    ),
                    detail=_Cut(
                        threshold_s=threshold,
                        pad_s=self._pad_s,
                        track_id=track_id,
                        clip=clip,
                        keeps=keeps,
                    ),
                )
            )
        return strategies

    def execute(self, strategy: Strategy) -> PlanPatch:
        """Replace the clip with one per kept span, laid end to end.

        Raises:
            VisionUseError: If the strategy did not come from this skill.

        """
        cut = strategy.detail
        if not isinstance(cut, _Cut):
            raise VisionUseError(
                "This strategy did not come from this skill",
                remedy="pass a strategy returned by plan()",
            )

        ops: list[Op] = [RemoveClip(clip_id=cut.clip.id)]
        cursor = cut.clip.timeline_start_s

        for index, span in enumerate(cut.keeps):
            # Quantized once, then reused for both the clip's own bounds and the
            # cursor advance. Quantizing `span.duration_s` separately looked
            # equivalent but is not: `Clip.timeline_end_s` is defined as
            # `source_out_s - source_in_s` on the *stored* (independently
            # quantized) bounds, and that can differ from
            # `quantize_s(end - start)` by float noise on the order of 1e-15 --
            # enough to trip the strict overlap check between adjacent clips.
            # Found on real media: a 40-cut source produced clips that were
            # supposed to be exactly back-to-back and were rejected as
            # overlapping by femtoseconds.
            source_in_s = quantize_s(span.start_s)
            source_out_s = quantize_s(span.end_s)
            duration = source_out_s - source_in_s
            ops.append(
                AddClip(
                    track_id=cut.track_id,
                    clip=cut.clip.model_copy(
                        update={
                            "id": clip_id(index),
                            "source_in_s": source_in_s,
                            "source_out_s": source_out_s,
                            "timeline_start_s": quantize_s(cursor),
                        }
                    ),
                )
            )
            cursor += duration

        log.debug(
            "planned silence cut",
            extra=bind(threshold_s=cut.threshold_s, kept=len(cut.keeps)),
        )
        return PlanPatch(
            ops=tuple(ops), rationale=strategy.rationale, produced_by=self.meta.ref
        )

    def validate(
        self, before: EditPlan, after: EditPlan, context: VideoContext
    ) -> ValidationReport:
        """Check the cut shortened the timeline without swallowing speech.

        The measurement that matters is ``speech_kept``: cutting a word in half
        is the failure users notice, and it is checkable from the activity map
        alone -- every second of non-silent audio in the source must still be
        covered by some clip.
        """
        kept = [clip for clip in after.timeline.clips() if clip.enabled]
        covered = [(clip.source_in_s, clip.source_out_s) for clip in kept]

        speech_total = 0.0
        speech_kept = 0.0
        for span in context.audio_activity:
            if span.is_silence:
                continue
            speech_total += span.span.duration_s
            speech_kept += sum(
                max(0.0, min(end, span.span.end_s) - max(start, span.span.start_s))
                for start, end in covered
            )

        ratio = speech_kept / speech_total if speech_total else 1.0
        shortened = after.duration_s < before.duration_s

        return ValidationReport(
            measurements=(
                Measurement(
                    name="speech_kept",
                    value=round(ratio, 4),
                    # A little loss is inevitable at the padded boundaries; a
                    # lot means the cut ate words.
                    passed=ratio >= MIN_SPEECH_KEPT,
                    detail=f"only {ratio:.1%} of speech survived the cut",
                ),
                Measurement(
                    name="shortened",
                    value=round(before.duration_s - after.duration_s, 3),
                    passed=shortened,
                    detail="the cut made no difference",
                ),
                Measurement(
                    name="not_empty",
                    value=float(len(kept)),
                    passed=bool(kept),
                    detail="the render would be empty",
                ),
            )
        )

    def score(self, report: ValidationReport) -> Score:
        """Prefer the attempt that removed the most while keeping the speech."""
        if not report.passed:
            return Score(value=-1.0, detail="rejected by validation")
        removed = next(
            (item.value for item in report.measurements if item.name == "shortened"), 0.0
        )
        return Score(value=removed, detail=f"removed {removed:.2f}s")


def snap_keeps(keeps: tuple[_Span, ...], context: VideoContext) -> tuple[_Span, ...]:
    """Snap every span boundary to a frame.

    A boundary derived from audio falls mid-frame, and the encoder then decides
    for itself which frame to use. Snapping makes the cut land where the plan
    says it does.
    """
    rate = context.probe.video.frame_rate
    return tuple(
        _Span(snap_to_frame(span.start_s, rate), snap_to_frame(span.end_s, rate))
        for span in keeps
    )
