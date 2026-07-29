"""Speech recognition with word-level timestamps.

Word-level timing is the point. Segment-level timings force a skill to guess
where inside a sentence a word falls, and the guess is audible: a caption
appears late, a cut lands mid-syllable. With per-word spans,
``context.find_text("pricing")`` returns entities a caller can cut on directly.

**This analyzer is optional.** ``faster-whisper`` pulls in a substantial ML
runtime, and most of what VisionUse does needs no speech at all, so it lives
behind the ``asr`` extra (Principle 3: plugins over built-ins). The import is
lazy and its absence produces an actionable message rather than an ImportError
from somewhere deep in the call stack.

**Determinism is configured, not hoped for.** Whisper will sample when its
temperature is non-zero, so two runs of the same file would differ and the M1
byte-identical criterion would fail. Temperature is pinned to zero and the
fallback ladder disabled; everything that could change output appears in
``params()`` so a settings change invalidates the cache instead of silently
serving a transcript produced under different settings.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Final

from pydantic import JsonValue

from vision_use.context.analyzer import AnalyzerRequest, AnalyzerResult
from vision_use.errors import DependencyError, VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.context import (
    ContextSection,
    SectionName,
    TimeSpan,
    Transcript,
    TranscriptSegment,
    Word,
)
from vision_use.models.ids import segment_id, word_id
from vision_use.timebase import quantize_s

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

log = get_logger(__name__)

#: Default model. "base" is the smallest size whose word timings are reliable
#: enough to cut on; "tiny" drifts by tenths of a second, which is audible.
DEFAULT_MODEL: Final = "base"

#: Beam width. Fixed rather than tuned: it participates in the cache key, so
#: changing it re-transcribes everything.
DEFAULT_BEAM_SIZE: Final = 5

#: Compute type. int8 runs everywhere including CPU-only machines, which
#: ``CLAUDE.md`` requires; GPU users can pass "float16".
DEFAULT_COMPUTE_TYPE: Final = "int8"


class TranscriptAnalyzer:
    """Transcribes speech into word-level entities, filling ``TRANSCRIPT``.

    Args:
        model: Whisper model size or a local path.
        language: BCP-47/ISO-639-1 code, or ``None`` to auto-detect. Pinning it
            is both faster and more reliable when the language is known.
        beam_size: Decoder beam width.
        compute_type: CTranslate2 compute type.

    """

    name: str = "transcript.whisper"
    version: str = "1"
    produces: frozenset[SectionName] = frozenset({ContextSection.TRANSCRIPT})
    requires: frozenset[SectionName] = frozenset()

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        language: str | None = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ) -> None:
        if beam_size < 1:
            raise ValueError(f"beam_size must be at least 1, got {beam_size}")
        self._model = model
        self._language = language
        self._beam_size = beam_size
        self._compute_type = compute_type

    def params(self) -> Mapping[str, JsonValue]:
        """Return the settings that affect this analyzer's output.

        ``temperature`` is included even though it is not configurable: it is
        part of what produced the transcript, and recording it means a future
        version that exposes sampling cannot collide with entries written now.
        """
        return {
            "model": self._model,
            "language": self._language,
            "beam_size": self._beam_size,
            "compute_type": self._compute_type,
            "temperature": 0.0,
            "word_timestamps": True,
        }

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        """Transcribe ``request.source``.

        Returns:
            A result carrying a :class:`Transcript`. A source with no audio
            yields an empty transcript and a warning rather than an error --
            silent video is legitimate input, and the recorded provenance still
            separates "analysed, no speech" from "never transcribed".

        Raises:
            DependencyError: If the ``asr`` extra is not installed.
            VisionUseError: If transcription fails.

        """
        if not request.context.probe.has_audio:
            return AnalyzerResult(
                transcript=Transcript(),
                warnings=(f"{request.source.name} has no audio stream",),
            )

        segments, language = self._transcribe(request)
        transcript = self._build(segments, language)

        log.debug(
            "transcribed",
            extra=bind(
                source=request.source.name,
                language=transcript.language,
                words=len(transcript.words),
                segments=len(transcript.segments),
            ),
        )
        return AnalyzerResult(transcript=transcript)

    def _load_model(self) -> object:
        """Import and construct the Whisper model, lazily.

        Raises:
            DependencyError: If the ``asr`` extra is not installed.

        """
        try:
            # Imported here, not at module scope: the transcript analyzer is
            # registered unconditionally, so a top-level import would make the
            # whole framework unusable without the optional extra.
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError as exc:
            raise DependencyError(
                "Speech recognition requires the 'asr' extra, which is not installed",
                remedy="install it with `uv sync --extra asr` or `pip install vision-use[asr]`",
            ) from exc

        return WhisperModel(self._model, device="auto", compute_type=self._compute_type)

    def _transcribe(self, request: AnalyzerRequest) -> tuple[list[object], str | None]:
        """Run the model and return its raw segments plus the detected language."""
        model = self._load_model()

        try:
            segments, info = model.transcribe(  # type: ignore[attr-defined]
                str(request.source),
                language=self._language,
                beam_size=self._beam_size,
                word_timestamps=True,
                # Pinned for reproducibility. A non-zero temperature makes
                # Whisper sample, and the fallback ladder silently raises it on
                # a low-confidence segment -- either would break determinism.
                temperature=0.0,
                condition_on_previous_text=False,
            )
            materialised = list(segments)
        except Exception as exc:
            raise VisionUseError(
                f"Transcription failed on {request.source.name}",
                remedy="check the file decodes (`vision inspect`) and the model name is valid",
            ) from exc

        detected = getattr(info, "language", None)
        return materialised, detected if isinstance(detected, str) else None

    def _build(self, segments: Iterable[object], language: str | None) -> Transcript:
        """Turn raw model output into validated, ID-bearing entities.

        IDs are assigned in chronological order across the whole transcript, so
        they are stable for a given input and sort the way they read.
        """
        words: list[Word] = []
        built_segments: list[TranscriptSegment] = []

        for segment in segments:
            raw_words: Sequence[object] = getattr(segment, "words", None) or ()
            first_word_index = len(words)

            for raw in raw_words:
                text = str(getattr(raw, "word", "")).strip()
                if not text:
                    # Whisper occasionally emits a whitespace-only token; an
                    # empty word would fail the model's min_length validation.
                    continue
                start = quantize_s(float(getattr(raw, "start", 0.0)))
                end = quantize_s(float(getattr(raw, "end", start)))
                probability = getattr(raw, "probability", None)

                words.append(
                    Word(
                        id=word_id(len(words)),
                        span=TimeSpan(start_s=start, end_s=max(start, end)),
                        text=text,
                        confidence=(
                            round(float(probability), 4)
                            if isinstance(probability, (int, float))
                            else None
                        ),
                    )
                )

            segment_words = words[first_word_index:]
            if not segment_words:
                continue

            built_segments.append(
                TranscriptSegment(
                    id=segment_id(len(built_segments)),
                    span=TimeSpan(
                        start_s=segment_words[0].span.start_s,
                        end_s=segment_words[-1].span.end_s,
                    ),
                    text=str(getattr(segment, "text", "")).strip(),
                    word_ids=tuple(word.id for word in segment_words),
                )
            )

        return Transcript(
            language=language,
            words=tuple(words),
            segments=tuple(built_segments),
        )
