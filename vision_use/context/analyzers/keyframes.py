"""Keyframe extraction: one representative still per shot.

Keyframes are how a model *sees* the video. `VISION.md` says the AI never
manipulates pixels but does not say how it perceives them; this is half the
answer (the contact sheet in M1.8 is the other half). A shot list with timings
tells a model what the structure is; a still per shot tells it what is actually
on screen.

The image bytes live in the content-addressed cache and the context holds only
`Keyframe.cache_key`. Inlining base64 stills would push a context for a
ten-minute video past what fits in a model's context window, and would stop it
being plain diffable JSON.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from pydantic import JsonValue

from vision_use.cache.store import CacheStore
from vision_use.context.analyzer import AnalyzerRequest, AnalyzerResult
from vision_use.log import bind, get_logger
from vision_use.models.context import ContextSection, Keyframe, SectionName, Shot
from vision_use.models.ids import keyframe_id
from vision_use.timebase import quantize_s
from vision_use.video.frames import (
    DEFAULT_MAX_EDGE,
    DEFAULT_QUALITY,
    FRAMES_VERSION,
    QUALITY_MAX,
    QUALITY_MIN,
    encode_jpeg,
    extract_frames,
)

log = get_logger(__name__)

#: Where in each shot the representative frame is taken from, as a fraction of
#: the shot's length. The midpoint rather than the first frame: the first frame
#: of a shot is frequently mid-transition, motion-blurred, or still showing the
#: tail of a dissolve, none of which represent the shot.
DEFAULT_POSITION: Final = 0.5

#: Cache namespace for the encoded stills.
KEYFRAME_NAMESPACE: Final = "keyframe.jpeg"


class KeyframeAnalyzer:
    """Extracts one still per shot, filling ``KEYFRAMES``.

    Args:
        position: Fraction into each shot to sample, in ``[0, 1]``.
        max_edge: Longest edge of the stored still, in pixels.
        quality: JPEG quality, 1-100.

    """

    name: str = "keyframes.per_shot"
    version: str = "1"
    produces: frozenset[SectionName] = frozenset({ContextSection.KEYFRAMES})
    requires: frozenset[SectionName] = frozenset({ContextSection.SHOTS})

    def __init__(
        self,
        position: float = DEFAULT_POSITION,
        max_edge: int = DEFAULT_MAX_EDGE,
        quality: int = DEFAULT_QUALITY,
    ) -> None:
        if not 0.0 <= position <= 1.0:
            raise ValueError(f"position must be within [0, 1], got {position}")
        if max_edge <= 0:
            raise ValueError(f"max_edge must be positive, got {max_edge}")
        if not QUALITY_MIN <= quality <= QUALITY_MAX:
            raise ValueError(
                f"quality must be within [{QUALITY_MIN}, {QUALITY_MAX}], got {quality}"
            )
        self._position = position
        self._max_edge = max_edge
        self._quality = quality

    def params(self) -> Mapping[str, JsonValue]:
        """Return the settings that affect this analyzer's output."""
        return {
            "position": self._position,
            "max_edge": self._max_edge,
            "quality": self._quality,
            "encoder": f"mjpeg-{FRAMES_VERSION}",
        }

    def sample_time(self, shot: Shot) -> float:
        """Return the time sampled for ``shot``.

        Clamped a hair inside the shot's end so that a midpoint on a
        zero-length shot, or a ``position`` of exactly 1.0, does not sample the
        first frame of the *next* shot.
        """
        span = shot.span
        raw = span.start_s + span.duration_s * self._position
        return quantize_s(min(raw, max(span.start_s, span.end_s - 1e-3)))

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        """Extract and cache one still per shot.

        Returns:
            A result whose ``keyframes`` reference cached JPEG bytes. Shots
            whose frame could not be decoded are skipped with a warning rather
            than failing the analysis: one unreadable frame near a container's
            end should not cost the caller every other keyframe.

        """
        shots = request.context.shots
        if not shots:
            return AnalyzerResult(
                keyframes=(),
                warnings=("no shots to extract keyframes from",),
            )

        targets = [self.sample_time(shot) for shot in shots]
        by_time = dict(zip(targets, shots, strict=True))

        keyframes: list[Keyframe] = []
        warnings: list[str] = []

        for index, (target, frame) in enumerate(extract_frames(request.source, targets)):
            shot = by_time[target]
            payload, width, height = encode_jpeg(
                frame, quality=self._quality, max_edge=self._max_edge
            )
            key = self._store(request.cache, request, shot.id, payload)

            keyframes.append(
                Keyframe(
                    id=keyframe_id(index),
                    shot_id=shot.id,
                    t_s=target,
                    cache_key=key,
                    width=width,
                    height=height,
                )
            )

        missing = len(shots) - len(keyframes)
        if missing:
            warnings.append(f"{missing} shot(s) had no decodable frame")

        log.debug(
            "extracted keyframes",
            extra=bind(
                source=request.source.name,
                shots=len(shots),
                keyframes=len(keyframes),
            ),
        )
        return AnalyzerResult(keyframes=tuple(keyframes), warnings=tuple(warnings))

    def _store(
        self,
        cache: CacheStore,
        request: AnalyzerRequest,
        shot_id: str,
        payload: bytes,
    ) -> str:
        """Write ``payload`` to the cache and return its digest.

        The key includes the source digest and the shot ID, so two shots of the
        same video never collide and the same shot of the same video with the
        same settings always resolves to one entry.
        """
        key = cache.key(
            KEYFRAME_NAMESPACE,
            version=f"{FRAMES_VERSION}.{self.version}",
            inputs=[request.context.probe.source.sha256],
            params={"shot": shot_id, **dict(self.params())},
            suffix=".jpg",
        )
        if cache.get(key) is None:
            with cache.writing(key) as tmp:
                tmp.write_bytes(payload)
        return key.digest
