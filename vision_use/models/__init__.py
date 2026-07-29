"""Serializable data models shared across the framework.

Models are the contract between the runtime, skills, and plugins. They are
Pydantic models so that a plugin author gets validation for free and so that
every document VisionUse produces round-trips through JSON unchanged.
"""

from vision_use.models.context import (
    ActivitySpan,
    AnalyzerRun,
    ContextSection,
    Keyframe,
    Shot,
    TimeSpan,
    Transcript,
    TranscriptSegment,
    VideoContext,
    Word,
)
from vision_use.models.media import (
    AudioStreamInfo,
    FrameRate,
    MediaProbe,
    SourceInfo,
    VideoStreamInfo,
)
from vision_use.models.plan import (
    Clip,
    EditPlan,
    Effect,
    RenderProfile,
    RenderSettings,
    Source,
    Timeline,
    Track,
    Transition,
)

__all__ = [
    "ActivitySpan",
    "AnalyzerRun",
    "AudioStreamInfo",
    "Clip",
    "ContextSection",
    "EditPlan",
    "Effect",
    "FrameRate",
    "Keyframe",
    "MediaProbe",
    "RenderProfile",
    "RenderSettings",
    "Shot",
    "Source",
    "SourceInfo",
    "TimeSpan",
    "Timeline",
    "Track",
    "Transcript",
    "TranscriptSegment",
    "Transition",
    "VideoContext",
    "VideoStreamInfo",
    "Word",
]
