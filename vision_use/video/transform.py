"""Frame-by-frame transformation of a source span into a new file.

The escape hatch for effects a filtergraph cannot express. Stabilisation needs
motion estimated across the whole span before the first frame can be warped;
object removal needs a segmentation model; upscaling needs a network. None of
those fit in a filter chain, and all of them are ordinary Python once frames are
numpy arrays.

Everything here is deterministic given the same input and callable, because the
result is content-addressed and cached: an intermediate that varied between runs
would be served for the wrong input.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import Final, cast

import av
import numpy as np
from av.container import OutputContainer
from av.video.frame import VideoFrame
from av.video.stream import VideoStream
from numpy.typing import NDArray

from vision_use.errors import ProbeError, UnsupportedMediaError
from vision_use.log import bind, get_logger
from vision_use.models.media import FrameRate

log = get_logger(__name__)

#: Bumped when transform output changes for identical input, so cached
#: intermediates from an older implementation are not served.
TRANSFORM_VERSION: Final = "1"

#: Codec used for intermediates. Lossless so that a baked effect is not taxed
#: with generation loss before the final encode even begins.
_INTERMEDIATE_CODEC: Final = "ffv1"

#: Pixel format for intermediates: packed RGB, not yuv420p.
#:
#: ffv1 is lossless in whatever format it is given, but *converting* RGB frames
#: to yuv420p subsamples chroma 2x2 first, which visibly shifts saturated edges.
#: A baked effect would then lose colour detail before the real encode had even
#: started. bgr0 keeps full chroma, so a round trip through an intermediate is
#: exact. (ffv1 accepts gbrp only at 9 bits and above, hence bgr0 rather than
#: planar RGB.)
_INTERMEDIATE_PIX_FMT: Final = "bgr0"

#: Container for intermediates. Matroska takes ffv1 without complaint; mp4 does
#: not.
INTERMEDIATE_SUFFIX: Final = ".mkv"

#: Container format named explicitly when encoding. The cache writes through a
#: temporary file whose name ends in ".partial", so nothing can be inferred from
#: the extension -- and an inferred format would be a silent dependency on a
#: filename anyway.
_INTERMEDIATE_FORMAT: Final = "matroska"

#: Signature of a frame transformation: RGB array in, RGB array out.
FrameOp = Callable[[NDArray[np.uint8]], NDArray[np.uint8]]


def read_frames(
    source: Path,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> Iterator[NDArray[np.uint8]]:
    """Yield RGB arrays for the frames of ``source`` within a time range.

    Frames are decoded and yielded one at a time; nothing accumulates, so peak
    memory is one frame regardless of how long the span is.

    Args:
        source: Media file to read.
        start_s: First presentation time to include.
        end_s: Stop at this time. ``None`` reads to the end.

    Yields:
        ``(height, width, 3)`` uint8 arrays.

    Raises:
        ProbeError: If the file cannot be opened.
        UnsupportedMediaError: If it has no video stream.

    """
    try:
        container = av.open(str(source))
    except av.FFmpegError as exc:
        raise ProbeError(
            f"Could not open {source.name} to transform it",
            remedy="check the file decodes (`vision inspect`)",
        ) from exc

    with container:
        if not container.streams.video:
            raise UnsupportedMediaError(
                f"{source.name} contains no video stream",
                remedy="pass a video file",
            )
        stream = container.streams.video[0]
        if start_s > 0 and stream.time_base is not None:
            container.seek(int(start_s / stream.time_base), stream=stream, backward=True)

        for frame in container.decode(stream):
            # PyAV's stubs type `time` as non-optional, so a checker calls this
            # branch dead. It is not: a frame with no timestamp comes back from
            # damaged or truncated streams, and comparing None would raise.
            timestamp: float | None = frame.time
            if timestamp is None:
                continue
            if timestamp < start_s:
                continue
            if end_s is not None and timestamp >= end_s:
                break
            yield np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)


def write_frames(
    frames: Iterator[NDArray[np.uint8]],
    output: Path,
    rate: FrameRate,
) -> int:
    """Encode ``frames`` into ``output`` losslessly.

    Args:
        frames: RGB arrays, all the same shape.
        output: Destination file. Should carry :data:`INTERMEDIATE_SUFFIX`.
        rate: Frame rate to stamp on the output.

    Returns:
        How many frames were written.

    Raises:
        UnsupportedMediaError: If nothing could be encoded.

    """
    output.parent.mkdir(parents=True, exist_ok=True)
    container: OutputContainer | None = None
    stream: VideoStream | None = None
    written = 0

    try:
        for array in frames:
            if container is None:
                container = av.open(str(output), "w", format=_INTERMEDIATE_FORMAT)
                # add_stream is typed as returning any stream kind; asking
                # for a video codec always yields a video stream.
                stream = cast(
                    "VideoStream",
                    container.add_stream(
                        _INTERMEDIATE_CODEC, rate=Fraction(rate.num, rate.den)
                    ),
                )
                stream.width = array.shape[1]
                stream.height = array.shape[0]
                stream.pix_fmt = _INTERMEDIATE_PIX_FMT

            assert stream is not None
            frame = VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            written += 1

        if container is None or stream is None:
            raise UnsupportedMediaError(
                "Nothing to encode: the transformation produced no frames",
                remedy="check the clip's source range covers decodable frames",
            )

        for packet in stream.encode():
            container.mux(packet)
    finally:
        if container is not None:
            container.close()

    log.debug("wrote intermediate", extra=bind(path=str(output), frames=written))
    return written


def transform_span(
    source: Path,
    output: Path,
    operation: FrameOp,
    rate: FrameRate,
    span: tuple[float, float | None] = (0.0, None),
    on_frame: Callable[[int], None] | None = None,
) -> int:
    """Apply ``operation`` to every frame of a span and write the result.

    Args:
        source: Media file to read.
        output: Where to write the transformed span.
        operation: Called once per frame with an RGB array; returns one.
        rate: Frame rate for the output.
        span: ``(start_s, end_s)``. ``end_s`` of ``None`` reads to the end.
        on_frame: Called with the running frame count. Kept as a bare int
            callback rather than a `Progress` so this module stays unaware of
            what the frames are for.

    Returns:
        How many frames were written.

    """
    start_s, end_s = span

    def transformed() -> Iterator[NDArray[np.uint8]]:
        """Apply the operation, counting frames as they pass."""
        for index, frame in enumerate(read_frames(source, start_s, end_s), start=1):
            yield operation(frame)
            if on_frame is not None:
                on_frame(index)

    return write_frames(transformed(), output, rate)
