"""VisionUse: an open source AI video intelligence and editing framework.

VisionUse orchestrates existing computer vision and media libraries behind a
uniform skill interface so that AI agents can analyse, edit, and enhance video
without manipulating pixels directly.

See ``PLAN.md`` for the implementation roadmap and ``VISION.md`` for the
project constitution.
"""

from vision_use.errors import (
    CacheError,
    ConfigurationError,
    DependencyError,
    ProbeError,
    UnsupportedMediaError,
    VisionUseError,
)

__all__ = [
    "CacheError",
    "ConfigurationError",
    "DependencyError",
    "ProbeError",
    "UnsupportedMediaError",
    "VisionUseError",
    "__version__",
]

__version__ = "0.1.0.dev0"
