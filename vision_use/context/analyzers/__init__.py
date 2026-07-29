"""Built-in analyzers.

Importing this package registers every built-in analyzer on the default
registry. Registration is explicit and happens here rather than by scanning the
filesystem, so that what is available is readable in one place and load order
is deterministic.
"""

from vision_use.context.analyzers.audio import AudioActivityAnalyzer
from vision_use.context.analyzers.keyframes import KeyframeAnalyzer
from vision_use.context.analyzers.shots import ShotAnalyzer
from vision_use.context.analyzers.subject import SubjectAnalyzer
from vision_use.context.analyzers.transcript import TranscriptAnalyzer
from vision_use.context.registry import registry

__all__ = [
    "AudioActivityAnalyzer",
    "KeyframeAnalyzer",
    "ShotAnalyzer",
    "SubjectAnalyzer",
    "TranscriptAnalyzer",
    "register_builtins",
]


def register_builtins() -> None:
    """Register the built-in analyzers on the default registry.

    Idempotent: re-registration replaces the existing entry rather than
    raising, so importing this module twice is harmless.
    """
    registry.register(ShotAnalyzer(), replace=True)
    registry.register(AudioActivityAnalyzer(), replace=True)
    registry.register(KeyframeAnalyzer(), replace=True)
    registry.register(SubjectAnalyzer(), replace=True)
    # Registered even when the 'asr' extra is absent: the registry is a
    # capability catalogue, and a missing extra should produce an actionable
    # error at run time, not a section that silently has no producer.
    registry.register(TranscriptAnalyzer(), replace=True)


register_builtins()
