"""The analyzer contract.

An analyzer takes a partially populated :class:`VideoContext` and returns the
sections it knows how to fill. It never mutates the context it is given --
contexts are frozen, and the pipeline is responsible for merging results, which
is what keeps provenance honest and makes a failed analyzer a no-op rather than
a partial write.

Every analyzer must be **deterministic**: identical input and parameters must
yield identical output. Anything that samples randomly seeds explicitly, and
anything that reports a wall-clock time keeps it out of its result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from vision_use.cache.store import CacheStore
from vision_use.hashing import hash_params
from vision_use.models.context import (
    ActivitySpan,
    AnalyzerRun,
    Keyframe,
    SectionName,
    Shot,
    SubjectSpan,
    Transcript,
    VideoContext,
)


@dataclass(frozen=True, slots=True)
class AnalyzerRequest:
    """Everything an analyzer needs to do its work.

    Attributes:
        source: Path to the media file being analysed.
        context: The context as populated so far. An analyzer reads the
            sections it declared in ``requires`` and must not assume any other
            section is present.
        cache: Store for derived artefacts such as extracted frames.

    """

    source: Path
    context: VideoContext
    cache: CacheStore


class AnalyzerResult(BaseModel):
    """What one analyzer contributed.

    Only the fields corresponding to the analyzer's declared ``produces`` are
    meaningful; the pipeline ignores anything else, so an analyzer cannot
    quietly overwrite a section it did not claim.

    A Pydantic model rather than a dataclass specifically so that results
    serialize losslessly. That is what lets the runner cache *any* analyzer's
    output generically -- otherwise every analyzer would grow its own bespoke
    caching, which is the duplication Principle 8 exists to prevent.
    """

    model_config = ConfigDict(frozen=True)

    shots: tuple[Shot, ...] = ()
    keyframes: tuple[Keyframe, ...] = ()
    audio_activity: tuple[ActivitySpan, ...] = ()
    transcript: Transcript | None = None
    subject: tuple[SubjectSpan, ...] = ()
    #: Plugin-contributed sections, keyed by dotted name. Only the keys the
    #: analyzer declared in `produces` are merged, so an analyzer cannot
    #: quietly fill a section it never claimed (PLAN.md § D5).
    extra: dict[str, JsonValue] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@runtime_checkable
class Analyzer(Protocol):
    """Fills one or more sections of a :class:`VideoContext`.

    Implementations are plain classes with class-level metadata:

    ``name``
        Dotted identifier, also the cache namespace, e.g. ``"analyze.shots"``.
    ``version``
        Bumped whenever output changes for identical input. Stale cache entries
        and stale contexts are detected by comparing this.
    ``produces``
        Sections this analyzer fills. A built-in :class:`ContextSection`, or a
        plugin's own dotted name (``"stabilize.motion"``), whose payload is
        returned in :attr:`AnalyzerResult.extra`.
    ``requires``
        Sections that must already be populated before it runs.
    """

    # Declared as instance attributes rather than ClassVar so that an
    # analyzer may compute them per instance -- a configurable analyzer that
    # produces different sections depending on how it was constructed is
    # legitimate. A class-level attribute satisfies this just as well.
    name: str
    version: str
    # `ContextSection` is a StrEnum, so a built-in member is already a string
    # and every existing declaration keeps working unchanged.
    produces: frozenset[SectionName]
    requires: frozenset[SectionName]

    def params(self) -> Mapping[str, JsonValue]:
        """Return this analyzer's configuration.

        Feeds both the cache key and the recorded provenance, so it must
        include every setting that can change the output and nothing that
        cannot -- an incidental value here causes needless cache misses.
        """
        ...

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        """Analyse ``request.source`` and return the sections produced.

        Raises:
            VisionUseError: On any failure the caller could act on.

        """
        ...


def provenance(analyzer: Analyzer) -> AnalyzerRun:
    """Build the :class:`AnalyzerRun` record for ``analyzer``.

    Sections are sorted so that the record -- and therefore the serialized
    context -- is byte-stable regardless of set iteration order.
    """
    return AnalyzerRun(
        name=analyzer.name,
        version=analyzer.version,
        params_hash=hash_params(dict(analyzer.params())),
        # `str(section)` rather than `section.value`: a plugin's section is
        # already a plain string and has no `.value`. Sorting keeps the record
        # -- and so the serialized context -- byte-stable across set orderings.
        produced=tuple(sorted(str(section) for section in analyzer.produces)),
    )
