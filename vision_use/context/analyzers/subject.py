"""Locating the subject of each shot, for reframing.

`reframe` has always cropped to the centre, while `SKILL_SPEC.md` described it
as speaker-aware. This closes half that gap: it finds *where the action is* in
each shot, so a 16:9 to 9:16 crop can follow the subject instead of trusting
that whoever shot the video centred them.

**This is not face detection, and the section is not called `faces`.** OpenCV 5
no longer ships the Haar cascades, and a real detector means a heavy ML
dependency with a CPU path to justify (`CONTRIBUTING.md` § Dependencies). What
this does instead needs nothing new: per-column temporal energy -- how much each
vertical strip of the frame changes from one sampled frame to the next --
summed over the shot, and the centroid of that. A person talking, gesturing, or
moving is the thing that changes; a static background is not.

The honest limits: it finds motion, not people. A shot where the subject holds
still and the background moves will point at the background, and `confidence`
is how it says so -- low when the energy is spread evenly, which a consumer
should read as "no clear subject, prefer the centre".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from vision_use.context.analyzer import AnalyzerRequest, AnalyzerResult
from vision_use.log import bind, get_logger
from vision_use.models.context import (
    ContextSection,
    SectionName,
    Shot,
    SubjectPoint,
    SubjectSpan,
    TimeSpan,
)
from vision_use.timebase import TIME_PRECISION
from vision_use.video.transform import read_frames

log = get_logger(__name__)

#: Luma weights (ITU-R BT.601). Motion is a brightness phenomenon here; colour
#: triples the work for no extra signal.
_LUMA: Final = np.array([0.299, 0.587, 0.114], dtype=np.float32)

#: Frames sampled per second of shot. Two is enough to see where someone is
#: moving and cheap enough to run over a long source: this analyzer decodes, and
#: decoding is the expensive part of every analysis.
_SAMPLE_RATE_HZ: Final = 2.0

#: Frames beyond which a shot stops being sampled. A five-minute locked-off
#: interview does not need six hundred samples to say "the subject is on the
#: left"; the estimate stops improving long before that.
_MAX_SAMPLES: Final = 60

#: Below this concentration the energy is spread across the frame and there is
#: no subject to speak of. Reported as low confidence rather than suppressed:
#: "measured, found nothing" and "never measured" must stay distinguishable.
_FLAT_CONCENTRATION: Final = 1.2


def _columns(frame: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Return one brightness value per column of ``frame``."""
    grey = np.asarray(frame, dtype=np.float32) @ _LUMA
    return np.asarray(grey.sum(axis=0), dtype=np.float32)


def _rows(frame: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Return one brightness value per row of ``frame``."""
    grey = np.asarray(frame, dtype=np.float32) @ _LUMA
    return np.asarray(grey.sum(axis=1), dtype=np.float32)


def _centroid(energy: NDArray[np.float32]) -> tuple[float, float]:
    """Return the normalised centroid of ``energy`` and how concentrated it is.

    Concentration is the peak divided by the mean: 1.0 means perfectly flat --
    no subject -- and higher means the energy clusters somewhere.
    """
    total = float(energy.sum())
    if total <= 0.0 or energy.size == 0:
        return 0.5, 1.0

    weights = energy / total
    position = float((np.arange(energy.size) * weights).sum()) / max(1, energy.size - 1)
    concentration = float(weights.max() * energy.size)
    return min(1.0, max(0.0, position)), concentration


class SubjectAnalyzer:
    """Finds the centre of activity in each shot."""

    name = "subject.energy"
    version = "1"
    produces: frozenset[SectionName] = frozenset({ContextSection.SUBJECT})
    # Shots, not just probe: a subject that moves between shots must be found
    # per shot, and averaging across a cut would place the crop between two
    # positions where the subject never was.
    requires: frozenset[SectionName] = frozenset({ContextSection.SHOTS})

    def params(self) -> Mapping[str, JsonValue]:
        """Everything that changes the result, and nothing that does not."""
        return {
            "sample_rate_hz": _SAMPLE_RATE_HZ,
            "max_samples": _MAX_SAMPLES,
            "flat_concentration": _FLAT_CONCENTRATION,
        }

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        """Locate the subject of every shot."""
        shots = request.context.shots
        if not shots:
            return AnalyzerResult(warnings=("no shots to locate a subject in",))

        spans = tuple(self._locate(request, shot) for shot in shots)
        log.debug("located subjects", extra=bind(shots=len(spans)))
        return AnalyzerResult(subject=spans)

    def _locate(self, request: AnalyzerRequest, shot: Shot) -> SubjectSpan:
        """Return where the activity sits within one shot."""
        # Every Nth decoded frame, where N comes from the source's own rate.
        # Sampling by index rather than seeking per sample: a seek is
        # keyframe-accurate and costs more than decoding forwards across the
        # short spans a shot usually is.
        fps = request.context.probe.video.frame_rate.decimal
        stride = max(1, round(fps / _SAMPLE_RATE_HZ))
        wanted = min(_MAX_SAMPLES, max(2, int(shot.span.duration_s * _SAMPLE_RATE_HZ)))

        column_energy: NDArray[np.float32] | None = None
        row_energy: NDArray[np.float32] | None = None
        previous_columns: NDArray[np.float32] | None = None
        previous_rows: NDArray[np.float32] | None = None
        track: list[SubjectPoint] = []
        taken = 0

        for index, frame in enumerate(
            read_frames(request.source, shot.span.start_s, shot.span.end_s)
        ):
            if index % stride != 0:
                continue

            columns, rows = _columns(frame), _rows(frame)
            if previous_columns is not None and previous_rows is not None:
                delta_columns = np.abs(columns - previous_columns)
                delta_rows = np.abs(rows - previous_rows)
                column_energy = (
                    delta_columns if column_energy is None else column_energy + delta_columns
                )
                row_energy = delta_rows if row_energy is None else row_energy + delta_rows
                taken += 1

                # Each sample's own position, kept as well as accumulated: the
                # sum says where the subject was on average, the track says
                # where it was at each moment, and a crop that moves needs the
                # second. Sampled in source time so a consumer can line it up
                # with clips without knowing the stride.
                sample_x, _ = _centroid(delta_columns)
                sample_y, _ = _centroid(delta_rows)
                track.append(
                    SubjectPoint(
                        t_s=round(shot.span.start_s + index / fps, TIME_PRECISION),
                        centre_x=round(sample_x, 4),
                        centre_y=round(sample_y, 4),
                    )
                )
            previous_columns, previous_rows = columns, rows

            if taken >= wanted:
                break

        if column_energy is None or row_energy is None:
            # One frame, or none: nothing moved because nothing was compared.
            return SubjectSpan(
                shot_id=shot.id,
                span=TimeSpan(start_s=shot.span.start_s, end_s=shot.span.end_s),
                centre_x=0.5,
                centre_y=0.5,
                confidence=0.0,
            )

        centre_x, concentration_x = _centroid(column_energy)
        centre_y, concentration_y = _centroid(row_energy)
        confidence = _confidence(min(concentration_x, concentration_y))

        return SubjectSpan(
            shot_id=shot.id,
            span=TimeSpan(start_s=shot.span.start_s, end_s=shot.span.end_s),
            centre_x=round(centre_x, 4),
            centre_y=round(centre_y, 4),
            confidence=round(confidence, 4),
            track=_smoothed(tuple(track)),
        )


#: Samples averaged together when smoothing the track. A single frame
#: difference is noisy -- a flicker or a compression artefact moves the centroid
#: a long way -- and a crop that jitters reads far worse than one that lags.
_SMOOTHING_WINDOW: Final = 3


def _smoothed(track: tuple[SubjectPoint, ...]) -> tuple[SubjectPoint, ...]:
    """Return ``track`` with a moving average over its positions.

    Centred rather than trailing, so the smoothed position does not lag the
    subject by half the window -- this runs offline over a whole shot, so there
    is no reason to accept the lag a causal filter would impose.
    """
    if len(track) <= _SMOOTHING_WINDOW:
        return track

    half = _SMOOTHING_WINDOW // 2
    smoothed = []
    for index, point in enumerate(track):
        window = track[max(0, index - half) : index + half + 1]
        smoothed.append(
            SubjectPoint(
                t_s=point.t_s,
                centre_x=round(sum(item.centre_x for item in window) / len(window), 4),
                centre_y=round(sum(item.centre_y for item in window) / len(window), 4),
            )
        )
    return tuple(smoothed)


def _confidence(concentration: float) -> float:
    """Map energy concentration onto 0..1.

    Flat energy is no evidence at all, so it maps to zero rather than to a small
    number that a consumer might mistake for a weak signal.
    """
    if concentration <= _FLAT_CONCENTRATION:
        return 0.0
    # Saturating rather than linear: past a certain peakiness the subject is
    # located about as well as this method can locate it, and reporting 0.99 vs
    # 0.999 would imply a precision that is not there.
    return float(min(1.0, (concentration - _FLAT_CONCENTRATION) / 8.0))
