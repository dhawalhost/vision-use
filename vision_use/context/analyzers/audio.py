"""Audio activity detection: where the source is speaking and where it is silent.

This is what ``silence-removal`` consumes to build a jump-cut plan, and what lets
``captions`` avoid placing a caption over dead air. It is deliberately *level*
detection rather than voice-activity detection: a level threshold is
deterministic, needs no model, and runs on CPU in a fraction of real time. Real
VAD can be added later as a second analyzer producing the same section.

Two details matter more than the thresholding itself:

**Streaming.** Audio is decoded and reduced frame by frame. A one-hour stereo
source is several hundred megabytes as float samples; loading it whole to take a
mean would violate ``CLAUDE.md`` § Performance for no benefit.

**Hysteresis on duration, not level.** A raw per-window classification flickers
on every breath and plosive. Spans shorter than the configured minimum are
absorbed into their neighbour, so the output is spans a human would agree with
rather than a few thousand alternating windows.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import av
import numpy as np
from av.audio.resampler import AudioResampler
from pydantic import JsonValue

from vision_use.context.analyzer import AnalyzerRequest, AnalyzerResult
from vision_use.errors import VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.context import ActivitySpan, ContextSection, SectionName, TimeSpan
from vision_use.timebase import quantize_s

log = get_logger(__name__)

#: Analysis sample rate. Level detection needs no fidelity, and resampling every
#: source to one rate makes window boundaries identical regardless of input.
ANALYSIS_RATE: Final = 16_000

#: Window length. 20 ms is short enough to catch a gap between words and long
#: enough that a single plosive does not dominate the measurement.
DEFAULT_WINDOW_S: Final = 0.02

#: Level at or below which a window counts as silent, in dBFS. Typical room tone
#: on a decent recording sits near -50; speech peaks well above -20.
DEFAULT_SILENCE_DB: Final = -40.0

#: Shortest span of either kind that survives. Shorter spans are absorbed into
#: the neighbouring span, which is what stops the output flickering.
DEFAULT_MIN_SPAN_S: Final = 0.2

#: Reported level for a digitally silent window. ``log10(0)`` is undefined, and
#: an infinity would not survive a JSON round trip.
SILENCE_FLOOR_DB: Final = -100.0

#: Decimal places retained on reported levels. Summing floats is not guaranteed
#: bit-identical across platforms or SIMD widths, and M1 requires two analyses of
#: one file to be byte-identical -- including between the Linux and macOS CI
#: legs. Two decimal places is far finer than any threshold decision needs.
_DB_PRECISION: Final = 2


@dataclass(frozen=True, slots=True)
class _Window:
    """One analysed window: its bounds and its measured level.

    Deliberately carries no ``is_silence``: classification depends on the
    analyzer's threshold, so storing it here would bake one analyzer's
    configuration into a value shared by all of them.
    """

    start_s: float
    end_s: float
    rms_db: float


def _to_db(rms: float) -> float:
    """Convert a linear RMS in ``[0, 1]`` to dBFS, floored and rounded."""
    if rms <= 0.0:
        return SILENCE_FLOOR_DB
    db = 20.0 * math.log10(rms)
    return max(round(db, _DB_PRECISION), SILENCE_FLOOR_DB)


class AudioActivityAnalyzer:
    """Detects silent and non-silent spans, filling ``AUDIO_ACTIVITY``.

    Args:
        window_s: Analysis window length in seconds.
        silence_db: Level at or below which a window counts as silent, dBFS.
        min_span_s: Spans shorter than this are absorbed into their neighbour.

    """

    name: str = "audio.activity"
    version: str = "1"
    produces: frozenset[SectionName] = frozenset({ContextSection.AUDIO_ACTIVITY})
    requires: frozenset[SectionName] = frozenset()

    def __init__(
        self,
        window_s: float = DEFAULT_WINDOW_S,
        silence_db: float = DEFAULT_SILENCE_DB,
        min_span_s: float = DEFAULT_MIN_SPAN_S,
    ) -> None:
        if window_s <= 0:
            raise ValueError(f"window_s must be positive, got {window_s}")
        if min_span_s < 0:
            raise ValueError(f"min_span_s must not be negative, got {min_span_s}")
        self._window_s = window_s
        self._silence_db = silence_db
        self._min_span_s = min_span_s

    def params(self) -> Mapping[str, JsonValue]:
        """Return the settings that affect this analyzer's output."""
        return {
            "window_s": self._window_s,
            "silence_db": self._silence_db,
            "min_span_s": self._min_span_s,
            "rate": ANALYSIS_RATE,
        }

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        """Measure audio level across ``request.source`` and derive spans.

        Returns:
            A result whose ``audio_activity`` tiles the source. A source with no
            audio stream yields an empty tuple and a warning -- not an error,
            since silent video is legitimate input, and the recorded provenance
            still distinguishes "analysed, no audio" from "never analysed".

        """
        if not request.context.probe.has_audio:
            return AnalyzerResult(
                audio_activity=(),
                warnings=(f"{request.source.name} has no audio stream",),
            )

        duration_s = quantize_s(request.context.duration_s)
        windows = list(self._measure(request.source))
        spans = self._build_spans(windows, duration_s)

        silent = sum(1 for span in spans if span.is_silence)
        log.debug(
            "measured audio activity",
            extra=bind(
                source=request.source.name,
                windows=len(windows),
                spans=len(spans),
                silent_spans=silent,
            ),
        )
        return AnalyzerResult(audio_activity=spans)

    def _measure(self, source: Path) -> Iterator[_Window]:
        """Yield one :class:`_Window` per analysis window, streaming.

        Samples are resampled to mono at :data:`ANALYSIS_RATE` and accumulated
        into a rolling buffer, so peak memory is one window plus one decoded
        frame regardless of source length.
        """
        window_samples = max(1, round(self._window_s * ANALYSIS_RATE))
        buffer = np.zeros(0, dtype=np.float32)
        index = 0

        try:
            container = av.open(str(source))
        except av.FFmpegError as exc:
            raise VisionUseError(
                f"Could not open {source.name} to measure audio",
                remedy="check the file decodes (`vision inspect`)",
            ) from exc

        with container:
            if not container.streams.audio:  # pragma: no cover - guarded by caller
                return
            stream = container.streams.audio[0]
            resampler = AudioResampler(format="flt", layout="mono", rate=ANALYSIS_RATE)

            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    samples = resampled.to_ndarray().reshape(-1).astype(np.float32)
                    buffer = np.concatenate((buffer, samples))

                    while buffer.size >= window_samples:
                        chunk = buffer[:window_samples]
                        buffer = buffer[window_samples:]
                        yield self._window(index, chunk)
                        index += 1

        # The final partial window still describes real audio; dropping it would
        # leave a gap at the end of every source whose length is not an exact
        # multiple of the window.
        if buffer.size:
            yield self._window(index, buffer)

    def _window(self, index: int, samples: np.ndarray) -> _Window:
        """Build a window record from ``samples`` at position ``index``."""
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        start = quantize_s(index * self._window_s)
        return _Window(
            start_s=start,
            end_s=quantize_s(start + samples.size / ANALYSIS_RATE),
            rms_db=_to_db(rms),
        )

    def _build_spans(
        self, windows: Sequence[_Window], duration_s: float
    ) -> tuple[ActivitySpan, ...]:
        """Merge classified windows into contiguous spans covering the source.

        Runs shorter than ``min_span_s`` are absorbed into the preceding span,
        which removes per-breath flicker. The result tiles ``[0, duration_s)``
        with no gaps, so a consumer can walk it without bounds checks.
        """
        if not windows:
            return ()

        runs: list[tuple[bool, float, float, list[float]]] = []
        for window in windows:
            silent = window.rms_db <= self._silence_db
            if runs and runs[-1][0] == silent:
                kind, start, _, levels = runs[-1]
                levels.append(window.rms_db)
                runs[-1] = (kind, start, window.end_s, levels)
            else:
                runs.append((silent, window.start_s, window.end_s, [window.rms_db]))

        merged: list[tuple[bool, float, float, list[float]]] = []
        for run in runs:
            silent, start, end, levels = run
            too_short = end - start < self._min_span_s
            if too_short and merged:
                prev_kind, prev_start, _, prev_levels = merged[-1]
                merged[-1] = (prev_kind, prev_start, end, [*prev_levels, *levels])
            else:
                merged.append((silent, start, end, list(levels)))

        # A leading run shorter than the minimum has no predecessor to absorb it,
        # so it is merged forwards into its successor instead of being dropped.
        if len(merged) > 1 and merged[0][2] - merged[0][1] < self._min_span_s:
            head, nxt = merged[0], merged[1]
            merged[1] = (nxt[0], head[1], nxt[2], [*head[3], *nxt[3]])
            merged.pop(0)

        # Absorbing a runt leaves its neighbours adjacent and of the same kind --
        # e.g. speech, one-window plosive, speech becomes two touching speech
        # spans. Coalescing is not cosmetic: a consumer walking the spans would
        # otherwise see a boundary where nothing changes, and `silence-removal`
        # would plan a cut across it.
        coalesced: list[tuple[bool, float, float, list[float]]] = []
        for silent, start, end, levels in merged:
            if coalesced and coalesced[-1][0] == silent:
                prev_kind, prev_start, _, prev_levels = coalesced[-1]
                coalesced[-1] = (prev_kind, prev_start, end, [*prev_levels, *levels])
            else:
                coalesced.append((silent, start, end, levels))
        merged = coalesced

        spans: list[ActivitySpan] = []
        for position, (silent, start, end, levels) in enumerate(merged):
            final = position == len(merged) - 1
            spans.append(
                ActivitySpan(
                    span=TimeSpan(
                        start_s=0.0 if position == 0 else start,
                        end_s=max(duration_s, end) if final else end,
                    ),
                    rms_db=round(sum(levels) / len(levels), _DB_PRECISION),
                    is_silence=silent,
                )
            )
        return tuple(spans)
