"""Low-level media access: probing, decoding, and frame extraction.

This package is the only place in VisionUse that talks to PyAV or invokes
ffmpeg. Everything above it works with the models in :mod:`vision_use.models`,
which keeps the media backend replaceable and keeps skills free of codec
concerns.
"""

from vision_use.video.frames import encode_jpeg, extract_frames, scaled_size
from vision_use.video.probe import probe

__all__ = ["encode_jpeg", "extract_frames", "probe", "scaled_size"]
