"""Frame extraction and still-image encoding.

Like the rest of :mod:`vision_use.video`, this is the only layer permitted to
touch PyAV. Callers ask for a frame at a time and get back JPEG bytes; nothing
above here knows about pixel formats, seeking, or codecs.

**Encoding uses PyAV rather than Pillow or OpenCV.** Both would work, but PyAV is
already a declared dependency and owns every other media path in the framework;
adding an imaging library to write a thumbnail would be a dependency bought for
one function. OpenCV in particular is only present transitively, via
PySceneDetect, and depending on a transitive dependency directly is how a build
breaks when an upstream drops it.

**Seeking is per-frame, not a full decode pass.** Extracting one frame per shot
from a one-hour source by decoding the whole file would cost minutes; seeking to
the nearest keyframe before each target and decoding forward costs milliseconds.
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Final, cast

import av
from av.video.frame import VideoFrame
from av.video.stream import VideoStream

from vision_use.errors import ProbeError, UnsupportedMediaError
from vision_use.log import bind, get_logger

log = get_logger(__name__)

#: Bumped when encoded output changes for identical input, so that cached
#: thumbnails from an older implementation are not served.
FRAMES_VERSION: Final = "1"

#: Longest edge of an extracted still, in pixels. Keyframes exist to be looked
#: at by a model and shown in a contact sheet; full resolution would bloat the
#: cache for no gain in either use.
DEFAULT_MAX_EDGE: Final = 640

#: JPEG quality on the familiar 1-100 scale.
DEFAULT_QUALITY: Final = 85

#: Bounds of the familiar quality scale, exposed so callers can validate against
#: the same numbers this module enforces.
QUALITY_MIN: Final = 1
QUALITY_MAX: Final = 100

#: JPEG's own quantiser scale, which runs backwards from the quality scale:
#: 2 is best, 31 is worst. Quality is mapped onto this range.
_Q_BEST: Final = 2
_Q_WORST: Final = 31


def _quantiser(quality: int) -> int:
    """Map a 1-100 quality onto libavcodec's inverted 2-31 quantiser scale."""
    clamped = min(QUALITY_MAX, max(QUALITY_MIN, quality))
    span = _Q_WORST - _Q_BEST
    return _Q_BEST + round(span * (QUALITY_MAX - clamped) / (QUALITY_MAX - QUALITY_MIN))


def scaled_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
    """Return ``(width, height)`` scaled so the longest edge is at most ``max_edge``.

    Aspect ratio is preserved and images are never enlarged. Both dimensions are
    forced to at least 1, since a very wide, very short source can otherwise
    scale to zero height and fail to encode.
    """
    if max_edge <= 0:
        raise ValueError(f"max_edge must be positive, got {max_edge}")
    scale = min(1.0, max_edge / max(width, height))
    return max(1, int(width * scale)), max(1, int(height * scale))


def encode_jpeg(
    frame: VideoFrame,
    *,
    quality: int = DEFAULT_QUALITY,
    max_edge: int = DEFAULT_MAX_EDGE,
) -> tuple[bytes, int, int]:
    """Encode ``frame`` as JPEG.

    Args:
        frame: Frame to encode.
        quality: 1-100; higher is better and larger.
        max_edge: Longest output edge in pixels.

    Returns:
        ``(jpeg_bytes, width, height)`` for the encoded image.

    Raises:
        UnsupportedMediaError: If the frame cannot be encoded.

    """
    width, height = scaled_size(frame.width, frame.height, max_edge)
    quantiser = _quantiser(quality)

    buffer = io.BytesIO()
    try:
        # yuvj420p is JPEG's full-range flavour; using yuv420p here produces a
        # visibly washed-out image because the decoder assumes limited range.
        converted = frame.reformat(width=width, height=height, format="yuvj420p")
        with av.open(buffer, "w", format="mjpeg") as container:
            # Quality is set through encoder options rather than by assigning
            # codec_context.qmin/qmax: the options path is typed, and pinning
            # both bounds to one value is what makes output byte-reproducible.
            # add_stream is typed as returning any stream kind; asking for
            # "mjpeg" always yields a video stream.
            stream = cast(
                "VideoStream",
                container.add_stream(
                    "mjpeg",
                    rate=1,
                    options={"qmin": str(quantiser), "qmax": str(quantiser)},
                ),
            )
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuvj420p"
            for packet in stream.encode(converted):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except av.FFmpegError as exc:
        raise UnsupportedMediaError(
            f"Could not encode a {width}x{height} frame as JPEG",
            remedy="check the source pixel format is convertible (`vision inspect`)",
        ) from exc

    return buffer.getvalue(), width, height


def extract_frames(
    source: Path, times_s: Sequence[float]
) -> Iterator[tuple[float, VideoFrame]]:
    """Yield ``(requested_time, frame)`` for each time in ``times_s``.

    Each target is reached by seeking to the nearest preceding keyframe and
    decoding forward, so cost scales with the number of targets rather than the
    length of the source.

    A target past the last frame yields the final frame rather than nothing: a
    shot boundary derived from container duration can land marginally beyond the
    last decodable frame, and returning nothing there would leave a shot with no
    keyframe for a rounding error.

    Args:
        source: Media file to read.
        times_s: Target times in seconds. Need not be sorted.

    Yields:
        The requested time paired with the frame displayed at it.

    Raises:
        ProbeError: If the file cannot be opened.
        UnsupportedMediaError: If it has no video stream.

    """
    if not times_s:
        return

    try:
        container = av.open(str(source))
    except av.FFmpegError as exc:
        raise ProbeError(
            f"Could not open {source.name} to extract frames",
            remedy="check the file decodes (`vision inspect`)",
        ) from exc

    with container:
        if not container.streams.video:
            raise UnsupportedMediaError(
                f"{source.name} contains no video stream",
                remedy="pass a video file",
            )
        stream = container.streams.video[0]
        time_base = stream.time_base or Fraction(1, 1000)

        for target in times_s:
            frame = _frame_at(container, stream, time_base, target)
            if frame is None:
                log.warning(
                    "no frame decodable at target time",
                    extra=bind(source=source.name, t_s=target),
                )
                continue
            yield target, frame


def _frame_at(
    container: av.container.InputContainer,
    stream: VideoStream,
    time_base: Fraction,
    target_s: float,
) -> VideoFrame | None:
    """Return the frame displayed at ``target_s``, or the last frame if past the end."""
    offset = max(0, int(target_s / time_base))
    try:
        container.seek(offset, stream=stream, backward=True)
    except av.FFmpegError:
        # Some containers refuse a seek near their end; decoding from wherever
        # the demuxer currently sits still finds the frame, just more slowly.
        log.debug("seek failed, decoding forward", extra=bind(t_s=target_s))

    last: VideoFrame | None = None
    for frame in container.decode(stream):
        last = frame
        if frame.time is not None and frame.time >= target_s:
            return frame
    return last
