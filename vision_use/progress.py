"""Reporting progress on work measured in minutes.

Baking a ten-minute clip takes twenty seconds at 640x360 and minutes at 1080p.
Silence for that long is indistinguishable from a hang, and the reasonable
response to a hang is Ctrl-C -- which throws away the work.

**A callback, not a progress bar.** The library must not know whether it is
being driven by a terminal, an MCP server, or a test. The CLI draws a bar, an
agent polls a number, and a test asserts on the sequence; none of that belongs
in the renderer.

**Reporting is best-effort and never fatal.** A callback that raises must not
destroy a bake that was otherwise about to succeed -- the work is expensive and
the reporting is not. Failures are logged once, then that callback is dropped
for the rest of the run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from vision_use.log import bind, get_logger

log = get_logger(__name__)


class Stage(StrEnum):
    """Which phase of the work a report describes.

    Named rather than free text so a caller can label a bar, or ignore a stage
    it does not care about, without parsing anything.
    """

    ANALYZE = "analyze"
    BAKE = "bake"
    RENDER = "render"


@dataclass(frozen=True, slots=True)
class Progress:
    """One report about work in flight.

    Attributes:
        stage: Which phase this describes.
        done: Units finished so far -- frames, clips, or analyzers.
        total: Units expected, or ``None`` when it cannot be known in advance.
            A variable-frame-rate source has no reliable frame count, and
            claiming one would make a bar that jumps or stalls near the end.
        detail: Short human-readable context, e.g. the clip being baked.

    """

    stage: Stage
    done: int = 0
    total: int | None = None
    detail: str = ""

    @property
    def fraction(self) -> float | None:
        """Completed share, or ``None`` when the total is unknown."""
        if not self.total:
            return None
        return min(1.0, self.done / self.total)


#: What a caller passes to be told how the work is going.
ProgressCallback = Callable[[Progress], None]


class Reporter:
    """Delivers progress to a callback, tolerating a callback that fails.

    Constructed per operation. Holding the "has it failed" flag here rather than
    globally means one bad callback disables itself and nothing else.
    """

    __slots__ = ("_callback", "_failed")

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self._callback = callback
        self._failed = False

    def __call__(
        self, stage: Stage, done: int, total: int | None = None, detail: str = ""
    ) -> None:
        """Report one update. Never raises."""
        if self._callback is None or self._failed:
            return
        try:
            self._callback(Progress(stage=stage, done=done, total=total, detail=detail))
        except Exception as exc:  # noqa: BLE001 - caller's code, any failure
            # Dropped rather than retried: a callback that raised once will
            # almost certainly raise every frame, and the log would be the
            # slowest part of the render.
            self._failed = True
            log.warning(
                "progress callback failed and was disabled",
                extra=bind(stage=str(stage), detail=str(exc)),
            )

    @property
    def active(self) -> bool:
        """Whether reports are still being delivered."""
        return self._callback is not None and not self._failed
