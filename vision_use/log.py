"""Structured logging for VisionUse.

``CLAUDE.md`` forbids ``print()`` and requires structured logging, so every
diagnostic in the framework goes through :func:`get_logger`. Structured fields
are passed via the standard ``extra=`` mechanism and are rendered as key/value
pairs by the human formatter or as JSON keys by the machine formatter.

Configuration is applied exactly once, by the process entry point (the CLI, or
an embedding application). Library code never configures logging -- it only
calls :func:`get_logger`.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from typing import Final, Literal

LogFormat = Literal["human", "json"]

_LOGGER_ROOT: Final = "vision_use"

# Attributes present on every LogRecord. Anything outside this set was supplied
# by the caller via extra= and is therefore a structured field worth emitting.
_STANDARD_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


def _extra_fields(record: logging.LogRecord) -> dict[str, object]:
    """Return the caller-supplied structured fields attached to ``record``."""
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD_ATTRS and not key.startswith("_")
    }


class HumanFormatter(logging.Formatter):
    """Single-line human-readable formatter with trailing ``key=value`` fields."""

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        """Format ``record``, appending any structured fields as ``key=value``."""
        base = super().format(record)
        fields = _extra_fields(record)
        if not fields:
            return base
        rendered = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
        return f"{base} | {rendered}"


class JsonFormatter(logging.Formatter):
    """Newline-delimited JSON formatter for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        """Format ``record`` as a single JSON object."""
        payload: dict[str, object] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **_extra_fields(record),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    *,
    level: int | str = logging.INFO,
    fmt: LogFormat = "human",
) -> None:
    """Configure VisionUse logging. Call once, from the process entry point.

    Handlers are attached to the ``vision_use`` logger rather than the root
    logger so that embedding applications keep control of their own logging.

    Args:
        level: Minimum level to emit.
        fmt: ``"human"`` for console use, ``"json"`` for log aggregation.

    """
    logger = logging.getLogger(_LOGGER_ROOT)
    logger.setLevel(level)
    logger.propagate = False

    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else HumanFormatter())
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return the VisionUse logger for ``name``.

    Args:
        name: Usually ``__name__``. A ``vision_use.`` prefix is added when the
            caller lives outside the package, keeping all framework output
            under a single logger tree.

    """
    if name == _LOGGER_ROOT or name.startswith(f"{_LOGGER_ROOT}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_ROOT}.{name}")


def bind(**fields: object) -> Mapping[str, object]:
    """Return ``fields`` as a mapping suitable for a logging ``extra=`` argument.

    Exists purely so call sites read as ``log.info("msg", extra=bind(k=v))``
    instead of building dict literals inline.
    """
    return fields
