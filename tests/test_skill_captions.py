"""Tests for the captions skill.

Grouping is where captions are won or lost: a line that changes before the eye
reaches it, or one that outlasts the sentence it belongs to, is unreadable no
matter how well it is rendered.
"""

from __future__ import annotations

import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from vision_use.cache.store import CacheStore
from vision_use.models.context import (
    AnalyzerRun,
    ContextSection,
    TimeSpan,
    Transcript,
    VideoContext,
    Word,
)
from vision_use.models.ids import clip_id, source_id, track_id, word_id
from vision_use.models.media import FrameRate, MediaProbe, SourceInfo, VideoStreamInfo
from vision_use.models.plan import Clip, EditPlan, Source, Timeline, Track
from vision_use.planning import straight_through
from vision_use.render import render
from vision_use.skills import SkillOutcome, SkillRequest, apply_patch, run_skill
from vision_use.skills.captions import (
    Captions,
    _next_id_base,
    group_words,
)
from vision_use.video.frames import extract_frames

from tests.conftest import FFMPEG, requires_ffmpeg

PAL = FrameRate(num=25, den=1)
DIGEST = "a" * 64
PRESENT = Path(__file__)


def _words(*spec: tuple[str, float, float]) -> tuple[Word, ...]:
    return tuple(
        Word(id=word_id(index), span=TimeSpan(start_s=start, end_s=end), text=text)
        for index, (text, start, end) in enumerate(spec)
    )


def _context(words: tuple[Word, ...], duration_s: float = 30.0) -> VideoContext:
    return VideoContext(
        probe=MediaProbe(
            source=SourceInfo(
                path=PRESENT,
                sha256=DIGEST,
                size_bytes=1,
                container="mp4",
                duration_s=duration_s,
            ),
            video=VideoStreamInfo(index=0, codec="h264", width=320, height=240, frame_rate=PAL),
        ),
        transcript=Transcript(language="en", words=words),
        analyzers={
            "transcript.whisper": AnalyzerRun(
                name="transcript.whisper",
                version="1",
                params_hash=DIGEST,
                produced=(ContextSection.TRANSCRIPT,),
            )
        },
    )


def _plan(duration_s: float = 30.0) -> EditPlan:
    source = Source(id=source_id(0), path=PRESENT, sha256=DIGEST, duration_s=duration_s)
    return EditPlan(
        sources={source.id: source},
        timeline=Timeline(
            frame_rate=PAL,
            width=320,
            height=240,
            tracks=(
                Track(
                    id=track_id(0),
                    kind="video",
                    clips=(
                        Clip(
                            id=clip_id(0),
                            source_id=source.id,
                            source_in_s=0.0,
                            source_out_s=duration_s,
                            timeline_start_s=0.0,
                        ),
                    ),
                ),
            ),
        ),
    )


def _run(context: VideoContext, plan: EditPlan, tmp_path: Path) -> SkillOutcome:
    store = CacheStore(tmp_path / "cache")
    return run_skill(
        Captions(cache=store),
        PRESENT,
        plan,
        SkillRequest(prompt="add captions"),
        context=context,
        cache=store,
    )


# --- grouping --------------------------------------------------------------


def test_no_words_makes_no_captions() -> None:
    assert group_words((), 3.0, 9) == []


def test_words_group_into_a_phrase() -> None:
    """One word at a time changes before the eye lands on it."""
    words = _words(("our", 0.0, 0.3), ("pricing", 0.3, 0.7), ("is", 0.7, 0.9))

    lines = group_words(words, 3.0, 9)

    assert len(lines) == 1
    assert lines[0].text == "our pricing is"


def test_a_pause_ends_the_line() -> None:
    """A gap is where the speaker themselves broke the phrase."""
    words = _words(("one", 0.0, 0.3), ("two", 0.3, 0.6), ("three", 2.0, 2.3))

    lines = group_words(words, 10.0, 99, pause_s=0.45)

    assert len(lines) == 2
    assert lines[1].text == "three"


def test_the_word_limit_ends_the_line() -> None:
    words = _words(*[(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(10)])

    lines = group_words(words, 100.0, 4)

    assert all(len(line.word_ids) <= 4 for line in lines)


def test_the_duration_limit_ends_the_line() -> None:
    words = _words(*[(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(10)])

    lines = group_words(words, 1.5, 99)

    assert len(lines) > 1


def test_every_word_appears_exactly_once() -> None:
    """A word shown twice, or not at all, is read straight off the screen."""
    words = _words(*[(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(12)])

    lines = group_words(words, 1.5, 4)

    seen = [wid for line in lines for wid in line.word_ids]
    assert seen == [word.id for word in words]


def test_a_caption_holds_past_its_last_word() -> None:
    """The last syllable should still be on screen as the eye reaches it."""
    words = _words(("hello", 0.0, 0.5))

    (line,) = group_words(words, 3.0, 9, hold_s=0.25)

    assert line.end_s > 0.5


def test_captions_never_overlap() -> None:
    """Two at once would composite, and the older would sit on the newer."""
    words = _words(*[(f"w{i}", i * 0.3, i * 0.3 + 0.2) for i in range(12)])

    lines = group_words(words, 0.5, 2, hold_s=1.0)

    for earlier, later in pairwise(lines):
        assert earlier.end_s <= later.start_s + 1e-6


def test_a_caption_always_has_positive_duration() -> None:
    """Trimming against the next line must not collapse one to nothing."""
    words = _words(*[(f"w{i}", i * 0.05, i * 0.05 + 0.04) for i in range(8)])

    lines = group_words(words, 0.1, 1, hold_s=2.0)

    assert all(line.end_s > line.start_s for line in lines)


# --- id allocation ---------------------------------------------------------


def test_id_base_clears_everything_the_plan_uses() -> None:
    """A fixed offset works until a plan reaches it."""
    plan = _plan()

    assert _next_id_base(plan) > 0


def test_id_base_clears_high_numbered_entities() -> None:
    source = Source(id="src_0900", path=PRESENT, sha256=DIGEST, duration_s=10.0)
    plan = EditPlan(
        sources={source.id: source},
        timeline=Timeline(frame_rate=PAL, width=320, height=240),
    )

    assert _next_id_base(plan) > 900


# --- lifecycle -------------------------------------------------------------


def test_no_transcript_means_nothing_to_do(tmp_path: Path) -> None:
    context = _context(())

    outcome = _run(context, _plan(), tmp_path)

    assert outcome.accepted is None
    assert outcome.attempts == ()


def test_captions_are_added_as_an_overlay_track(tmp_path: Path) -> None:
    words = _words(("hello", 0.5, 1.0), ("world", 1.0, 1.5))

    outcome = _run(_context(words), _plan(), tmp_path)

    assert outcome.accepted is not None
    overlays = [t for t in outcome.plan.timeline.tracks if t.kind == "overlay"]
    assert len(overlays) == 1
    assert overlays[0].clips


def test_the_video_track_is_untouched(tmp_path: Path) -> None:
    """Captions sit on top; they must not disturb the cut."""
    plan = _plan()
    words = _words(("hello", 0.5, 1.0))

    outcome = _run(_context(words), plan, tmp_path)

    assert outcome.accepted is not None
    before = [t for t in plan.timeline.tracks if t.kind == "video"]
    after = [t for t in outcome.plan.timeline.tracks if t.kind == "video"]
    assert before == after


def test_each_caption_declares_its_own_source(tmp_path: Path) -> None:
    """A rendered caption is media the plan did not start with."""
    words = _words(*[(f"w{i}", i * 2.0, i * 2.0 + 1.0) for i in range(3)])

    outcome = _run(_context(words), _plan(), tmp_path)

    assert outcome.accepted is not None
    assert len(outcome.plan.sources) > len(_plan().sources)


def test_caption_images_land_in_the_cache(tmp_path: Path) -> None:
    words = _words(("hello", 0.5, 1.0))

    outcome = _run(_context(words), _plan(), tmp_path)

    assert outcome.accepted is not None
    assert list((tmp_path / "cache" / "caption.png").rglob("*.png"))


def test_captions_line_up_with_their_words(tmp_path: Path) -> None:
    words = _words(("hello", 1.25, 1.75))

    outcome = _run(_context(words), _plan(), tmp_path)

    assert outcome.accepted is not None
    overlay = next(t for t in outcome.plan.timeline.tracks if t.kind == "overlay")
    assert overlay.clips[0].timeline_start_s == pytest.approx(1.25, abs=0.01)


def test_the_readable_grouping_is_tried_first(tmp_path: Path) -> None:
    words = _words(*[(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(12)])

    outcome = _run(_context(words), _plan(), tmp_path)

    assert outcome.accepted is not None
    assert outcome.accepted.strategy.name == "9-words-3.0s"


def test_an_already_captioned_plan_is_left_alone(tmp_path: Path) -> None:
    """A second caption track would composite two captions at once."""
    words = _words(("hello", 0.5, 1.0))
    once = _run(_context(words), _plan(), tmp_path)

    assert once.accepted is not None
    twice = _run(_context(words), once.plan, tmp_path)

    assert twice.accepted is None


def test_the_edit_is_reversible(tmp_path: Path) -> None:
    plan = _plan()
    words = _words(("hello", 0.5, 1.0), ("again", 3.0, 3.5))

    outcome = _run(_context(words), plan, tmp_path)

    assert outcome.accepted is not None
    restored = apply_patch(outcome.plan, outcome.accepted.inverse).plan
    assert restored.model_dump_json() == plan.model_dump_json()


def test_the_patch_explains_itself(tmp_path: Path) -> None:
    words = _words(("hello", 0.5, 1.0))

    outcome = _run(_context(words), _plan(), tmp_path)

    assert outcome.accepted is not None
    assert "caption in phrases" in outcome.accepted.patch.rationale


def test_rejects_empty_groupings() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Captions(groupings=())


# --- rendering -------------------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_a_captioned_plan_renders(tmp_path: Path) -> None:
    """The whole point: the captions have to reach the output frames."""

    assert FFMPEG is not None

    clip = tmp_path / "src.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=320x240:r=25:d=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )

    base = straight_through(clip)
    words = _words(("hello", 0.5, 1.0), ("world", 1.0, 1.5))
    store = CacheStore(tmp_path / "cache")
    outcome = run_skill(
        Captions(cache=store), clip, base, context=_context(words, 3.0), cache=store
    )

    assert outcome.accepted is not None
    captioned = tmp_path / "captioned.mp4"
    plain = tmp_path / "plain.mp4"
    render(outcome.plan, captioned, cache=store)
    render(base, plain, cache=store)

    def band(path: Path) -> float:
        (_, frame), *_ = list(extract_frames(path, [1.0]))
        pixels = np.asarray(frame.to_ndarray(format="rgb24"))
        low = pixels[int(pixels.shape[0] * 0.82) : int(pixels.shape[0] * 0.95)]
        return float(low.mean())

    # Compared against the same render without captions rather than an absolute
    # threshold: the plate covers only the centre of the band, so the average
    # across the full width is nowhere near the plate's own darkness.
    assert band(captioned) < band(plain) - 10
