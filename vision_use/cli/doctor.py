"""Environment diagnosis for ``vision doctor``.

The framework depends on native media libraries whose absence otherwise
surfaces deep inside a pipeline as an obscure failure. Doctor moves that
discovery to the front, names the missing piece, and says how to install it.

Checks are pure functions returning a :class:`Check`; rendering lives in
:mod:`vision_use.cli.main`. Keeping them separate makes the same checks usable
from the MCP server and from tests without capturing terminal output.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

from vision_use.cache.store import CACHE_ROOT_ENV, default_cache_root

#: Minimum interpreter required. Below this, syntax used across the package
#: (PEP 695 generics, `Self`) is unavailable.
MIN_PYTHON: Final = (3, 12)

#: Minimum ffmpeg major version. Filters relied on by the M2 renderer are not
#: present or behave differently before this.
MIN_FFMPEG_MAJOR: Final = 6

#: How long to wait for a version-probing subprocess before giving up.
_SUBPROCESS_TIMEOUT_S: Final = 10

#: Minimum token count in a parseable ffmpeg banner: "ffmpeg version 7.1 ...".
_BANNER_MIN_TOKENS: Final = 3


class Status(Enum):
    """Outcome of a single environment check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class Check:
    """The result of one environment check.

    Attributes:
        name: Short label, e.g. ``"ffmpeg"``.
        status: Whether the check passed, is degraded, or blocks use.
        detail: What was found, e.g. a version string or a path.
        remedy: Present when the check did not pass; how to fix it.

    """

    name: str
    status: Status
    detail: str
    remedy: str | None = None


def check_python() -> Check:
    """Verify the running interpreter meets :data:`MIN_PYTHON`."""
    found = sys.version_info[:3]
    version = ".".join(str(part) for part in found)
    if found[:2] < MIN_PYTHON:
        required = ".".join(str(part) for part in MIN_PYTHON)
        return Check(
            name="python",
            status=Status.FAIL,
            detail=f"{version} ({platform.python_implementation()})",
            remedy=f"VisionUse requires Python {required}+; run `uv sync` to provision it",
        )
    return Check("python", Status.OK, f"{version} ({platform.python_implementation()})")


def _binary_version(binary: str) -> str | None:
    """Return the first line of ``<binary> -version``, or ``None`` if unavailable."""
    if shutil.which(binary) is None:
        return None
    try:
        completed = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.splitlines()[0] if completed.stdout else None


def _parse_ffmpeg_major(banner: str) -> int | None:
    """Extract the major version from an ffmpeg banner line.

    Distribution builds append suffixes (``7.1-static``, ``6.0.1-3ubuntu``), so
    only the leading integer is trusted.
    """
    parts = banner.split()
    if len(parts) < _BANNER_MIN_TOKENS or parts[1] != "version":
        return None
    head = parts[2].split(".", maxsplit=1)[0]
    return int(head) if head.isdigit() else None


def _check_ffmpeg_binary(binary: str) -> Check:
    """Verify one ffmpeg-family binary is installed and recent enough."""
    banner = _binary_version(binary)
    if banner is None:
        return Check(
            name=binary,
            status=Status.FAIL,
            detail="not found on PATH",
            remedy="install ffmpeg (`brew install ffmpeg`, `apt install ffmpeg`)",
        )
    major = _parse_ffmpeg_major(banner)
    if major is None:
        return Check(binary, Status.WARN, banner, "could not parse version; assuming usable")
    if major < MIN_FFMPEG_MAJOR:
        return Check(
            name=binary,
            status=Status.FAIL,
            detail=f"version {major}",
            remedy=f"VisionUse requires ffmpeg {MIN_FFMPEG_MAJOR}+; upgrade it",
        )
    return Check(binary, Status.OK, f"version {major}")


def check_ffmpeg() -> Check:
    """Verify the ``ffmpeg`` binary used by the renderer."""
    return _check_ffmpeg_binary("ffmpeg")


def check_ffprobe() -> Check:
    """Verify the ``ffprobe`` binary used for cross-checking container metadata."""
    return _check_ffmpeg_binary("ffprobe")


def check_pyav() -> Check:
    """Verify PyAV imports and report the libavformat build it links against."""
    try:
        # Imported lazily and locally: whether this import succeeds is the
        # check itself, so it must not run at module import time.
        import av  # noqa: PLC0415
    except ImportError as exc:
        return Check(
            name="pyav",
            status=Status.FAIL,
            detail=f"import failed: {exc}",
            remedy="run `uv sync` to install project dependencies",
        )

    libav = getattr(av, "library_versions", {}).get("libavformat")
    linked = ".".join(str(part) for part in libav) if libav else "unknown"
    return Check("pyav", Status.OK, f"{av.__version__} (libavformat {linked})")


def check_cache_root() -> Check:
    """Verify the cache root exists or can be created, and is writable."""
    root = default_cache_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        handle, probe_path = tempfile.mkstemp(dir=root, suffix=".doctor")
    except OSError as exc:
        return Check(
            name="cache",
            status=Status.FAIL,
            detail=f"{root}: {exc.strerror or exc}",
            remedy=f"create the directory, or set {CACHE_ROOT_ENV} to a writable path",
        )
    os.close(handle)
    Path(probe_path).unlink(missing_ok=True)
    return Check("cache", Status.OK, f"writable at {root}")


#: Every check doctor runs, in report order.
CHECKS: Final[Sequence[Callable[[], Check]]] = (
    check_python,
    check_pyav,
    check_ffmpeg,
    check_ffprobe,
    check_cache_root,
)


def run_checks() -> list[Check]:
    """Run every environment check and return the results in report order."""
    return [check() for check in CHECKS]


def worst_status(checks: Sequence[Check]) -> Status:
    """Return the most severe status across ``checks``."""
    if any(check.status is Status.FAIL for check in checks):
        return Status.FAIL
    if any(check.status is Status.WARN for check in checks):
        return Status.WARN
    return Status.OK
