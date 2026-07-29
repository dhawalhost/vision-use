"""Shot boundary detection.

Shots are the coarsest useful structure in a video and the first thing an agent
reasons about, so this is the analyzer everything else builds on: keyframes are
extracted per shot, reframing tracks a subject within a shot, and a jump cut
must never be placed across one.

Detection is delegated to PySceneDetect's ``ContentDetector``, which compares
consecutive frames in HSV space and reports a cut when the weighted difference
crosses a threshold. VisionUse orchestrates rather than reinvents (Principle 1),
and content-based cut detection is a solved, well-tested problem.

What this module adds on top is everything the framework needs and the detector
does not provide: quantised times, stable IDs, a minimum shot duration, and the
guarantee that the returned shots tile the source exactly with no gaps and no
overlaps -- which the context model then enforces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import JsonValue

from vision_use.context.analyzer import AnalyzerRequest, AnalyzerResult
from vision_use.errors import VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.context import ContextSection, SectionName, Shot, TimeSpan
from vision_use.models.ids import shot_id
from vision_use.timebase import quantize_s

log = get_logger(__name__)

#: Default detector sensitivity. PySceneDetect's own default; lower finds more
#: cuts. Exposed as a parameter because the right value is content-dependent --
#: a fast-cut trailer and a static interview need very different settings.
DEFAULT_THRESHOLD: Final = 27.0

#: Shots shorter than this are merged into their predecessor. Detectors emit
#: spurious one- or two-frame shots on flashes, camera pops, and hard lighting
#: changes; a shot nothing can be done with is noise in the context and noise
#: in an agent's prompt.
DEFAULT_MIN_SHOT_S: Final = 0.4


class ShotAnalyzer:
    """Detects shot boundaries and fills :attr:`ContextSection.SHOTS`.

    Args:
        threshold: Detector sensitivity; see :data:`DEFAULT_THRESHOLD`.
        min_shot_s: Shots shorter than this are merged backwards; see
            :data:`DEFAULT_MIN_SHOT_S`.

    """

    name: str = "shots.content"
    version: str = "1"
    produces: frozenset[SectionName] = frozenset({ContextSection.SHOTS})
    requires: frozenset[SectionName] = frozenset()

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        min_shot_s: float = DEFAULT_MIN_SHOT_S,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if min_shot_s < 0:
            raise ValueError(f"min_shot_s must not be negative, got {min_shot_s}")
        self._threshold = threshold
        self._min_shot_s = min_shot_s

    def params(self) -> Mapping[str, JsonValue]:
        """Return the settings that affect this analyzer's output."""
        return {
            "detector": "content",
            "threshold": self._threshold,
            "min_shot_s": self._min_shot_s,
        }

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        """Detect shot boundaries in ``request.source``.

        Returns:
            A result whose ``shots`` tile the whole source. A video with no
            detected cut yields exactly one shot spanning it -- never an empty
            list, which a consumer would have to special-case.

        Raises:
            VisionUseError: If the detector cannot read the source.

        """
        duration_s = quantize_s(request.context.duration_s)
        boundaries = self._detect(request, duration_s)
        shots = self._build_shots(boundaries, duration_s)

        log.debug(
            "detected shots",
            extra=bind(
                source=request.source.name,
                shots=len(shots),
                threshold=self._threshold,
            ),
        )
        return AnalyzerResult(shots=shots)

    def _detect(self, request: AnalyzerRequest, duration_s: float) -> list[float]:
        """Return detected cut times in seconds, ascending, excluding 0.

        Imported lazily: PySceneDetect pulls in OpenCV, which is slow to import
        and unnecessary for callers that never detect shots.
        """
        from scenedetect import ContentDetector, detect  # noqa: PLC0415

        try:
            scenes = detect(str(request.source), ContentDetector(threshold=self._threshold))
        except Exception as exc:
            raise VisionUseError(
                f"Shot detection failed on {request.source.name}",
                remedy="check the file decodes (`vision inspect`) and is not truncated",
            ) from exc

        cuts = [quantize_s(start.seconds) for start, _ in scenes]
        return [cut for cut in cuts if 0.0 < cut < duration_s]

    def _build_shots(self, boundaries: Sequence[float], duration_s: float) -> tuple[Shot, ...]:
        """Turn cut times into contiguous, ID-bearing, gap-free shots.

        Boundaries closer together than ``min_shot_s`` are dropped rather than
        producing a shot too short to use. The final shot always ends at the
        source duration, so the shots tile the timeline exactly.
        """
        edges: list[float] = [0.0]
        for cut in boundaries:
            if cut - edges[-1] >= self._min_shot_s:
                edges.append(cut)

        # A trailing shot shorter than the minimum is merged backwards by
        # dropping the boundary that created it, rather than emitting a runt.
        if len(edges) > 1 and duration_s - edges[-1] < self._min_shot_s:
            edges.pop()

        return tuple(
            Shot(
                id=shot_id(index),
                index=index,
                span=TimeSpan(start_s=start, end_s=end),
            )
            for index, (start, end) in enumerate(
                zip(edges, [*edges[1:], duration_s], strict=True)
            )
        )
