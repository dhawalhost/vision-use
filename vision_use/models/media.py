"""Media description models: the output of probing, the input to everything else.

These models are the foundation of ``VideoContext`` (milestone M1) and of the
renderer's timebase handling (M2), so exactness matters more than convenience
here. In particular frame rate is stored as an explicit rational rather than a
float: NTSC rates such as 30000/1001 are not representable in binary floating
point, and accumulating a rounded frame duration across a long timeline drifts
by whole frames. Cuts must land on the frame the user asked for.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class FrameRate(BaseModel):
    """An exact frame rate, stored as a rational number.

    Attributes:
        num: Numerator, e.g. ``30000``.
        den: Denominator, e.g. ``1001``.

    """

    model_config = ConfigDict(frozen=True)

    num: PositiveInt
    den: PositiveInt

    @classmethod
    def from_fraction(cls, value: Fraction) -> Self:
        """Build a :class:`FrameRate` from a :class:`fractions.Fraction`."""
        return cls(num=value.numerator, den=value.denominator)

    @property
    def fraction(self) -> Fraction:
        """Return this rate as an exact :class:`fractions.Fraction`."""
        return Fraction(self.num, self.den)

    @property
    def decimal(self) -> float:
        """Return this rate as a float. For display and heuristics only, never timing."""
        return self.num / self.den

    @property
    def frame_duration_s(self) -> Fraction:
        """Exact duration of one frame in seconds."""
        return Fraction(self.den, self.num)

    def __str__(self) -> str:
        """Return ``"30000/1001"`` for non-integer rates, ``"25"`` for integer ones."""
        return str(self.num) if self.den == 1 else f"{self.num}/{self.den}"


class SourceInfo(BaseModel):
    """Identity of a media file on disk.

    ``sha256`` is what makes analysis and render results content-addressable:
    two files with the same digest are interchangeable, and a file whose digest
    has changed invalidates every derived artefact.
    """

    model_config = ConfigDict(frozen=True)

    path: Path
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    container: str = Field(description="Container format short name, e.g. 'mov,mp4,m4a'.")
    duration_s: NonNegativeFloat = Field(
        description="Container duration. May disagree slightly with stream duration."
    )


class VideoStreamInfo(BaseModel):
    """Properties of the video stream VisionUse will operate on."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    codec: str
    width: PositiveInt
    height: PositiveInt
    frame_rate: FrameRate = Field(description="Average rate; see 'variable_frame_rate'.")
    pix_fmt: str | None = None
    bit_rate: int | None = Field(default=None, ge=0)
    rotation: Literal[0, 90, 180, 270] = Field(
        default=0,
        description=(
            "Display rotation from the container. Phone footage is commonly stored "
            "landscape with a 90 degree rotation flag, so consumers must apply this "
            "before reasoning about aspect ratio."
        ),
    )
    frame_count: int | None = Field(
        default=None,
        ge=0,
        description="Exact count when the container records one, else None.",
    )
    frame_count_estimated: bool = Field(
        default=False,
        description="True when 'frame_count' was derived from duration and rate.",
    )
    variable_frame_rate: bool = Field(
        default=False,
        description=(
            "True when average and nominal rates disagree, which is typical of "
            "screen recordings and some phone cameras. Frame-indexed operations "
            "are unreliable on such sources and must address time instead."
        ),
    )

    @property
    def display_width(self) -> int:
        """Width after applying :attr:`rotation`."""
        return self.height if self.rotation in (90, 270) else self.width

    @property
    def display_height(self) -> int:
        """Height after applying :attr:`rotation`."""
        return self.width if self.rotation in (90, 270) else self.height

    @property
    def display_aspect_ratio(self) -> float:
        """Width divided by height, after rotation."""
        return self.display_width / self.display_height

    @property
    def is_portrait(self) -> bool:
        """Whether the displayed frame is taller than it is wide."""
        return self.display_height > self.display_width


class AudioStreamInfo(BaseModel):
    """Properties of the audio stream, when the container has one."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    codec: str
    sample_rate: PositiveInt
    channels: PositiveInt
    layout: str | None = None
    bit_rate: int | None = Field(default=None, ge=0)


class MediaProbe(BaseModel):
    """Everything learned from opening a file without decoding its content.

    Probing is the cheapest analysis VisionUse performs and the precondition for
    every other one, so it is kept separate from the richer ``VideoContext``
    that follows in M1.
    """

    model_config = ConfigDict(frozen=True)

    source: SourceInfo
    video: VideoStreamInfo
    audio: AudioStreamInfo | None = None
    other_stream_count: int = Field(
        default=0,
        ge=0,
        description="Streams VisionUse ignores, e.g. subtitles, timecode, data.",
    )

    @model_validator(mode="after")
    def _check_stream_indices_differ(self) -> Self:
        """Reject a probe whose audio and video claim the same stream index."""
        if self.audio is not None and self.audio.index == self.video.index:
            raise ValueError(
                f"video and audio streams share index {self.video.index}; "
                "the probe is inconsistent"
            )
        return self

    @property
    def has_audio(self) -> bool:
        """Whether an audio stream is present.

        Skills that consume speech (captions, silence removal) check this
        before declaring their context requirements satisfied.
        """
        return self.audio is not None
