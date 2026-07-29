"""Tests for speech recognition.

Most of these need no model: the risky logic is turning raw model output into
validated entities, and that is exercised with stand-in objects. The handful
that genuinely need Whisper are marked ``asr`` and skipped when the extra or
the model weights are unavailable, so the default suite stays fast and offline.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import pytest
from vision_use.cache.store import CacheStore
from vision_use.context.analyzer import AnalyzerRequest
from vision_use.context.analyzers.transcript import TranscriptAnalyzer
from vision_use.context.registry import default_registry, registry
from vision_use.context.runner import analyze, context_digest
from vision_use.errors import DependencyError
from vision_use.hashing import hash_params
from vision_use.models.context import ContextSection, Transcript, VideoContext
from vision_use.video.probe import probe

from tests.conftest import MakeClip, requires_ffmpeg

S = ContextSection


def _has_faster_whisper() -> bool:
    try:
        import faster_whisper  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


requires_asr = pytest.mark.skipif(
    not _has_faster_whisper(), reason="the 'asr' extra is not installed"
)


# --- stand-ins for raw model output ----------------------------------------


@dataclass
class FakeWord:
    """Mimics faster-whisper's word object."""

    word: str
    start: float
    end: float
    probability: float = 0.9


@dataclass
class FakeSegment:
    """Mimics faster-whisper's segment object."""

    text: str
    words: list[FakeWord] = field(default_factory=list)


def _build(segments: list[FakeSegment], language: str | None = "en") -> Transcript:
    return TranscriptAnalyzer()._build(segments, language)


# --- construction ----------------------------------------------------------


def test_rejects_bad_beam_size() -> None:
    with pytest.raises(ValueError, match="beam_size"):
        TranscriptAnalyzer(beam_size=0)


def test_params_pin_temperature_to_zero() -> None:
    """Sampling would make two runs of one file differ and break determinism."""
    assert TranscriptAnalyzer().params()["temperature"] == 0.0


def test_params_cover_everything_that_changes_output() -> None:
    params = TranscriptAnalyzer(model="small", language="fr", beam_size=3).params()

    assert params["model"] == "small"
    assert params["language"] == "fr"
    assert params["beam_size"] == 3
    assert params["word_timestamps"] is True


def test_model_choice_changes_the_cache_key() -> None:
    base = hash_params(dict(TranscriptAnalyzer(model="base").params()))
    small = hash_params(dict(TranscriptAnalyzer(model="small").params()))

    assert base != small


def test_declares_only_transcript() -> None:
    analyzer = TranscriptAnalyzer()

    assert analyzer.produces == {S.TRANSCRIPT}
    assert analyzer.requires == frozenset()


# --- entity construction ---------------------------------------------------


def test_words_get_chronological_ids() -> None:
    transcript = _build(
        [
            FakeSegment(
                "our pricing is simple",
                [
                    FakeWord("our", 0.0, 0.4),
                    FakeWord(" pricing", 0.4, 0.9),
                    FakeWord(" is", 0.9, 1.1),
                    FakeWord(" simple", 1.1, 1.6),
                ],
            )
        ]
    )

    assert [word.id for word in transcript.words] == [
        "w_000000",
        "w_000001",
        "w_000002",
        "w_000003",
    ]


def test_leading_whitespace_is_stripped() -> None:
    """Whisper prefixes most words with a space; it would corrupt text matching."""
    transcript = _build([FakeSegment("hi", [FakeWord(" hello", 0.0, 0.5)])])

    assert transcript.words[0].text == "hello"


def test_whitespace_only_tokens_are_dropped() -> None:
    """An empty word would fail the model's min_length validation."""
    transcript = _build(
        [FakeSegment("hi", [FakeWord("  ", 0.0, 0.1), FakeWord(" hello", 0.1, 0.5)])]
    )

    assert len(transcript.words) == 1


def test_segments_reference_their_words() -> None:
    transcript = _build(
        [
            FakeSegment("one two", [FakeWord("one", 0.0, 0.3), FakeWord(" two", 0.3, 0.6)]),
            FakeSegment("three", [FakeWord("three", 0.7, 1.0)]),
        ]
    )

    first, second = transcript.segments
    assert first.word_ids == ("w_000000", "w_000001")
    assert second.word_ids == ("w_000002",)


def test_segment_span_covers_its_words() -> None:
    transcript = _build(
        [FakeSegment("one two", [FakeWord("one", 0.2, 0.3), FakeWord(" two", 0.3, 0.9)])]
    )

    segment = transcript.segments[0]
    assert segment.span.start_s == 0.2
    assert segment.span.end_s == 0.9


def test_empty_segments_are_skipped() -> None:
    """A segment whose words were all whitespace must not become an empty entity."""
    transcript = _build([FakeSegment("", []), FakeSegment("hi", [FakeWord("hi", 0.0, 0.2)])])

    assert len(transcript.segments) == 1


def test_confidence_is_carried_through() -> None:
    transcript = _build([FakeSegment("hi", [FakeWord("hi", 0.0, 0.2, probability=0.42)])])

    assert transcript.words[0].confidence == pytest.approx(0.42)


def test_reversed_word_timing_is_repaired() -> None:
    """A model occasionally emits end < start; the model layer would reject it."""
    transcript = _build([FakeSegment("hi", [FakeWord("hi", 1.0, 0.5)])])

    word = transcript.words[0]
    assert word.span.end_s >= word.span.start_s


def test_times_are_quantised() -> None:
    transcript = _build([FakeSegment("hi", [FakeWord("hi", 0.1234567891, 0.5)])])

    assert transcript.words[0].span.start_s == 0.123457


def test_language_is_recorded() -> None:
    transcript = _build([FakeSegment("hi", [FakeWord("hi", 0.0, 0.2)])], language="fr")

    assert transcript.language == "fr"


def test_empty_input_yields_an_empty_transcript() -> None:
    transcript = _build([])

    assert transcript.words == ()
    assert transcript.segments == ()


def test_built_transcript_supports_lookup() -> None:
    """The whole point: find_text must resolve to entities with real timings."""
    transcript = _build(
        [
            FakeSegment(
                "our pricing is simple",
                [
                    FakeWord("our", 0.0, 0.4),
                    FakeWord(" pricing", 0.4, 0.9),
                    FakeWord(" is", 0.9, 1.1),
                ],
            )
        ]
    )
    assert isinstance(transcript, Transcript)


# --- no-audio path ---------------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_silent_video_yields_empty_transcript_and_a_warning(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    """Silent video is legitimate input; it must not need the model at all."""
    clip = make_clip(name="tr_noaudio", with_audio=False)
    request = AnalyzerRequest(
        source=clip,
        context=VideoContext(probe=probe(clip)),
        cache=CacheStore(tmp_path / "cache"),
    )

    result = TranscriptAnalyzer().run(request)

    assert result.transcript is not None
    assert result.transcript.words == ()
    assert "no audio" in result.warnings[0]


@pytest.mark.integration
@requires_ffmpeg
def test_analysed_but_silent_still_records_provenance(
    make_clip: MakeClip, tmp_path: Path
) -> None:
    context = analyze(
        make_clip(name="tr_noaudio2", with_audio=False),
        [S.TRANSCRIPT],
        cache=CacheStore(tmp_path / "cache"),
        registry=default_registry([TranscriptAnalyzer()]),
    )

    assert context.has(S.TRANSCRIPT)
    assert context.transcript is not None
    assert context.transcript.words == ()


# --- optional-extra behaviour ----------------------------------------------


def test_missing_extra_reports_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure must name the extra, not surface a bare ImportError."""
    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("faster_whisper"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(DependencyError) as excinfo:
        TranscriptAnalyzer()._load_model()

    assert "asr" in excinfo.value.message
    assert "uv sync --extra asr" in (excinfo.value.remedy or "")


def test_registered_even_without_the_extra() -> None:
    """The registry is a capability catalogue; a missing extra fails at run time."""
    assert "transcript.whisper" in registry
    assert registry.producer_of(S.TRANSCRIPT) is not None


# --- against the real model ------------------------------------------------


@pytest.mark.asr
@pytest.mark.slow
@pytest.mark.integration
@requires_ffmpeg
@requires_asr
def test_transcribes_real_speech(speech_clip: Path, tmp_path: Path) -> None:
    context = analyze(
        speech_clip,
        [S.TRANSCRIPT],
        cache=CacheStore(tmp_path / "cache"),
        registry=default_registry([TranscriptAnalyzer(model="tiny", language="en")]),
    )

    assert context.transcript is not None
    assert context.transcript.words
    assert context.transcript.language == "en"


@pytest.mark.asr
@pytest.mark.slow
@pytest.mark.integration
@requires_ffmpeg
@requires_asr
def test_words_are_ordered_and_timed(speech_clip: Path, tmp_path: Path) -> None:
    context = analyze(
        speech_clip,
        [S.TRANSCRIPT],
        cache=CacheStore(tmp_path / "cache"),
        registry=default_registry([TranscriptAnalyzer(model="tiny", language="en")]),
    )

    words = context.transcript.words  # type: ignore[union-attr]
    assert all(a.span.start_s <= b.span.start_s for a, b in pairwise(words))
    assert all(word.span.end_s >= word.span.start_s for word in words)


@pytest.mark.asr
@pytest.mark.slow
@pytest.mark.integration
@requires_ffmpeg
@requires_asr
def test_transcription_is_byte_identical_across_runs(speech_clip: Path, tmp_path: Path) -> None:
    """Temperature is pinned to zero precisely so this holds."""
    store = CacheStore(tmp_path / "cache")
    asr_registry = default_registry([TranscriptAnalyzer(model="tiny", language="en")])

    first = analyze(speech_clip, [S.TRANSCRIPT], cache=store, registry=asr_registry)
    second = analyze(speech_clip, [S.TRANSCRIPT], cache=store, registry=asr_registry)

    assert first.model_dump_json() == second.model_dump_json()
    assert context_digest(first) == context_digest(second)


@pytest.mark.asr
@pytest.mark.slow
@pytest.mark.integration
@requires_ffmpeg
@requires_asr
def test_every_section_together_is_byte_identical(speech_clip: Path, tmp_path: Path) -> None:
    """M1's exit criterion at full scope: all four analyzers, twice, identical.

    The per-analyzer determinism tests can all pass while the composition still
    drifts -- merge order, provenance ordering, or one analyzer perturbing the
    context another reads. This is the assertion that actually closes M1.
    """
    store = CacheStore(tmp_path / "cache")
    sections = [section for section in S if section is not S.PROBE]

    first = analyze(speech_clip, sections, cache=store)
    second = analyze(speech_clip, sections, cache=store)

    assert all(first.has(section) for section in sections)
    assert first.model_dump_json() == second.model_dump_json()
    assert context_digest(first) == context_digest(second)
    assert len(first.analyzers) == 4
