"""Media probing via PyAV.

Probing reads container and stream headers without decoding frames, so it is
fast enough to run unconditionally at the head of every pipeline.

PyAV is used rather than shelling out to ``ffprobe`` because it returns frame
rates as exact rationals instead of decimal strings that must be re-parsed, and
because it avoids a subprocess per file. See ``PLAN.md`` § Dependencies.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Final

import av
from av.container import InputContainer
from av.video.stream import VideoStream

from vision_use.errors import ProbeError, UnsupportedMediaError
from vision_use.hashing import hash_file
from vision_use.log import bind, get_logger
from vision_use.models.media import (
    AudioStreamInfo,
    FrameRate,
    MediaProbe,
    SourceInfo,
    VideoStreamInfo,
)

log = get_logger(__name__)

#: Bumped whenever probe output changes for identical input, so that cached
#: probes derived from an older implementation are not served.
PROBE_VERSION: Final = "1"

#: PyAV reports container duration in units of this many per second.
_AV_TIME_BASE: Final = 1_000_000

#: Relative disagreement between average and nominal frame rate above which a
#: stream is treated as variable. Containers routinely round the nominal rate,
#: so an exact comparison would flag almost everything; 0.5% is comfortably
#: above that rounding noise and well below any real VFR swing.
_VFR_TOLERANCE: Final = 0.005


def _normalise_rotation(raw: float) -> int:
    """Map an arbitrary rotation in degrees onto one of 0, 90, 180, 270.

    Container rotation is conventionally expressed clockwise-negative by the
    display matrix and clockwise-positive by the legacy ``rotate`` tag, and may
    arrive as a float. Anything not near a right angle is discarded: VisionUse
    cannot losslessly express an arbitrary rotation as a display flag.
    """
    degrees = round(raw) % 360
    return degrees if degrees in (0, 90, 180, 270) else 0


def _extract_rotation(stream: VideoStream) -> int:
    """Return the display rotation recorded for ``stream``, in degrees.

    Two sources are consulted, newest first: the H.264/QuickTime display
    matrix, and the legacy ``rotate`` metadata tag written by older muxers.
    Phone footage is nearly always stored landscape with one of these set, so
    ignoring them would misreport the aspect ratio of most real input.
    """
    side_data = getattr(stream, "side_data", None)
    if side_data is not None:
        try:
            matrix = side_data.get("DISPLAYMATRIX")
        except (AttributeError, TypeError, KeyError):
            matrix = None
        if matrix is not None:
            rotation = getattr(matrix, "rotation", None)
            if isinstance(rotation, (int, float)):
                # The display matrix expresses counter-clockwise rotation;
                # VisionUse stores the clockwise angle to apply on display.
                return _normalise_rotation(-rotation)

    tag = stream.metadata.get("rotate")
    if tag is not None:
        try:
            return _normalise_rotation(float(tag))
        except ValueError:
            log.warning(
                "ignoring unparseable rotate tag",
                extra=bind(value=tag, stream=stream.index),
            )
    return 0


def _frame_rate(stream: VideoStream) -> tuple[FrameRate, bool]:
    """Return the stream's frame rate and whether it looks variable.

    ``average_rate`` is measured across the file and is what timeline maths
    must use. ``base_rate`` is the rate the container declares. A meaningful
    gap between them means frames are not evenly spaced, which makes
    frame-indexed addressing unsafe.
    """
    average: Fraction | None = stream.average_rate
    nominal: Fraction | None = getattr(stream, "base_rate", None)

    if average is None or average <= 0:
        if nominal is None or nominal <= 0:
            raise UnsupportedMediaError(
                "Video stream declares no usable frame rate",
                remedy="remux the file with ffmpeg to write a valid rate",
            )
        return FrameRate.from_fraction(nominal), True

    variable = False
    if nominal is not None and nominal > 0:
        variable = abs(float(average) - float(nominal)) / float(nominal) > _VFR_TOLERANCE

    return FrameRate.from_fraction(average), variable


def _frame_count(
    stream: VideoStream, duration_s: float, rate: FrameRate
) -> tuple[int | None, bool]:
    """Return the frame count and whether it had to be estimated.

    ``stream.frames`` is authoritative when the container records it. MPEG-TS
    and fragmented MP4 commonly do not, in which case the count is derived from
    duration -- useful for progress reporting, never for seeking.
    """
    declared = int(getattr(stream, "frames", 0) or 0)
    if declared > 0:
        return declared, False
    if duration_s > 0:
        return int(duration_s * rate.decimal), True
    return None, False


def _container_duration_s(container: InputContainer, stream: VideoStream) -> float:
    """Return the best available duration in seconds, preferring the container."""
    if container.duration is not None:
        return float(container.duration) / _AV_TIME_BASE
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    return 0.0


def probe(path: Path, *, compute_hash: bool = True) -> MediaProbe:
    """Read container and stream metadata for a media file.

    Args:
        path: File to probe.
        compute_hash: Whether to compute the source SHA-256. Hashing streams
            the whole file, so callers that only need dimensions -- a doctor
            check, a fast preflight -- can skip it. When skipped, the digest
            field is filled with zeroes and the probe must not be used as a
            cache input.

    Returns:
        A fully populated :class:`~vision_use.models.media.MediaProbe`.

    Raises:
        ProbeError: If the file does not exist or cannot be opened.
        UnsupportedMediaError: If the file has no video stream, or has one
            whose frame rate cannot be determined.

    """
    if not path.exists():
        raise ProbeError(
            f"No such media file: {path}",
            remedy="check the path, or pass an absolute path",
        )
    if not path.is_file():
        raise ProbeError(
            f"Not a file: {path}",
            remedy="pass a media file rather than a directory",
        )

    try:
        container = av.open(str(path))
    except av.FFmpegError as exc:
        raise ProbeError(
            f"Could not open {path.name} as media",
            remedy="confirm the file is not truncated and its format is supported",
        ) from exc

    with container:
        if not container.streams.video:
            raise UnsupportedMediaError(
                f"{path.name} contains no video stream",
                remedy="pass a video file; audio-only input is not yet supported",
            )

        stream = container.streams.video[0]
        duration_s = _container_duration_s(container, stream)
        rate, variable = _frame_rate(stream)
        count, estimated = _frame_count(stream, duration_s, rate)

        video = VideoStreamInfo(
            index=stream.index,
            codec=stream.codec_context.name,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            frame_rate=rate,
            pix_fmt=getattr(stream.codec_context, "pix_fmt", None),
            bit_rate=stream.bit_rate or None,
            rotation=_normalise_rotation(_extract_rotation(stream)),
            frame_count=count,
            frame_count_estimated=estimated,
            variable_frame_rate=variable,
        )

        audio: AudioStreamInfo | None = None
        if container.streams.audio:
            track = container.streams.audio[0]
            audio = AudioStreamInfo(
                index=track.index,
                codec=track.codec_context.name,
                sample_rate=track.codec_context.sample_rate,
                channels=track.codec_context.channels,
                layout=getattr(track.layout, "name", None),
                bit_rate=track.bit_rate or None,
            )

        other = len(container.streams) - 1 - (1 if audio else 0)
        container_name = container.format.name

    source = SourceInfo(
        path=path,
        sha256=hash_file(path) if compute_hash else "0" * 64,
        size_bytes=path.stat().st_size,
        container=container_name,
        duration_s=duration_s,
    )

    log.debug(
        "probed media",
        extra=bind(
            path=str(path),
            container=container_name,
            codec=video.codec,
            size=f"{video.width}x{video.height}",
            fps=str(rate),
            duration_s=round(duration_s, 3),
            has_audio=audio is not None,
        ),
    )

    return MediaProbe(source=source, video=video, audio=audio, other_stream_count=max(other, 0))
