"""Exception hierarchy for VisionUse.

Every error raised by the framework derives from :class:`VisionUseError` so
that embedding applications can catch VisionUse failures without catching
unrelated exceptions.

Per ``CLAUDE.md``: errors are never silently ignored, are always wrapped with
context, and always carry an actionable message. To that end every exception
here accepts an optional ``remedy`` describing what the caller should do.
"""

from __future__ import annotations


class VisionUseError(Exception):
    """Base class for every error raised by VisionUse.

    Args:
        message: What went wrong, in the imperative past tense.
        remedy: Optional concrete action the caller can take to fix it.

    """

    def __init__(self, message: str, *, remedy: str | None = None) -> None:
        self.message = message
        self.remedy = remedy
        super().__init__(f"{message} ({remedy})" if remedy else message)


class ConfigurationError(VisionUseError):
    """Raised when VisionUse is misconfigured, e.g. an unwritable cache root."""


class DependencyError(VisionUseError):
    """Raised when a required external tool or library is missing or too old.

    Used by ``vision doctor`` to report an unusable environment rather than
    letting a downstream call fail with an obscure ``FileNotFoundError``.
    """


class MediaError(VisionUseError):
    """Base class for failures that originate in the media itself."""


class ProbeError(MediaError):
    """Raised when a media file cannot be opened or its metadata cannot be read."""


class UnsupportedMediaError(MediaError):
    """Raised when a file opens cleanly but lacks something VisionUse requires.

    For example, a container with no video stream, or a variable frame rate
    that the caller explicitly requested be constant.
    """


class CacheError(VisionUseError):
    """Raised when the content-addressed cache cannot be read or written."""
