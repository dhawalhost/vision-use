"""Running analyzers and merging their results into a context.

This is where the M1 exit criterion is met. Two guarantees live here:

**Caching is generic.** An analyzer's whole :class:`AnalyzerResult` is cached
as JSON, keyed on the analyzer's identity and parameters plus the source
digest. No analyzer implements caching itself, so none can get it subtly wrong,
and a re-run of an unchanged file does no decoding at all.

**Merging is authoritative.** Only sections an analyzer declared in
``produces`` are taken from its result. An analyzer that computes something it
did not claim cannot silently overwrite another's work, and provenance always
matches what actually landed in the context.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from vision_use.cache.store import CacheKey, CacheStore
from vision_use.context.analyzer import (
    Analyzer,
    AnalyzerRequest,
    AnalyzerResult,
    provenance,
)
from vision_use.context.registry import AnalyzerRegistry
from vision_use.context.registry import registry as default_registry
from vision_use.errors import VisionUseError
from vision_use.hashing import hash_params
from vision_use.log import bind, get_logger
from vision_use.models.context import (
    BUILTIN_SECTIONS,
    ContextSection,
    SectionName,
    VideoContext,
    check_plugin_section,
)
from vision_use.video.probe import probe

log = get_logger(__name__)

#: Bumped when the cached representation of a result changes shape. Guards
#: against deserialising an entry written by an incompatible older build.
RESULT_CACHE_VERSION = "1"


def _cache_key(store: CacheStore, analyzer: Analyzer, source_digest: str) -> CacheKey:
    """Derive the cache key for one analyzer run against one source."""
    return store.key(
        f"analyze.{analyzer.name}",
        version=f"{RESULT_CACHE_VERSION}.{analyzer.version}",
        inputs=[source_digest],
        params=dict(analyzer.params()),
        suffix=".json",
    )


def run_analyzer(
    analyzer: Analyzer,
    request: AnalyzerRequest,
    *,
    use_cache: bool = True,
) -> AnalyzerResult:
    """Run ``analyzer``, returning a cached result when one exists.

    Args:
        analyzer: The analyzer to run.
        request: Source, context so far, and cache store.
        use_cache: Set ``False`` to force recomputation, for debugging a
            suspected stale entry. The fresh result is still written back.

    Returns:
        The analyzer's contribution.

    Raises:
        VisionUseError: Whatever the analyzer raises, unwrapped -- the
            analyzer is closer to the failure and writes the better message.

    """
    digest = request.context.probe.source.sha256
    key = _cache_key(request.cache, analyzer, digest)

    if use_cache:
        cached = request.cache.get_json(key)
        if cached is not None:
            log.debug("analyzer cache hit", extra=bind(analyzer=analyzer.name))
            return AnalyzerResult.model_validate(cached)

    result = analyzer.run(request)

    for warning in result.warnings:
        log.warning("analyzer warning", extra=bind(analyzer=analyzer.name, detail=warning))

    request.cache.put_json(key, result.model_dump(mode="json"))
    return result


def merge(context: VideoContext, analyzer: Analyzer, result: AnalyzerResult) -> VideoContext:
    """Return ``context`` with ``result``'s declared sections applied.

    Sections the analyzer did not declare are ignored. Provenance is recorded
    under the analyzer's name, so a later run of the same analyzer replaces its
    own entry rather than accumulating duplicates.
    """
    updates: dict[str, object] = {}

    if ContextSection.SHOTS in analyzer.produces:
        updates["shots"] = result.shots
    if ContextSection.KEYFRAMES in analyzer.produces:
        updates["keyframes"] = result.keyframes
    if ContextSection.AUDIO_ACTIVITY in analyzer.produces:
        updates["audio_activity"] = result.audio_activity
    if ContextSection.TRANSCRIPT in analyzer.produces:
        updates["transcript"] = result.transcript
    if ContextSection.SUBJECT in analyzer.produces:
        updates["subject"] = result.subject

    # Plugin sections, merged by name. Only what the analyzer declared is taken
    # -- `extra` carrying a key it never claimed is ignored the same way a
    # built-in section would be. `check_plugin_section` runs here as well as in
    # the model so that the error names the analyzer that caused it.
    plugin_sections = {
        name: result.extra[name]
        for name in analyzer.produces
        if name not in BUILTIN_SECTIONS and name in result.extra
    }
    if plugin_sections:
        for name in plugin_sections:
            try:
                check_plugin_section(name)
            except ValueError as exc:
                raise VisionUseError(
                    f"Analyzer {analyzer.name!r} produced an unusable section: {exc}",
                    remedy="namespace the section by your package, e.g. 'stabilize.motion'",
                ) from exc
        updates["extra"] = {**context.extra, **plugin_sections}

    analyzers = {**context.analyzers, analyzer.name: provenance(analyzer)}
    return context.model_copy(update={**updates, "analyzers": analyzers})


def analyze(
    source: Path,
    sections: Iterable[SectionName] = (),
    *,
    cache: CacheStore | None = None,
    registry: AnalyzerRegistry | None = None,
    use_cache: bool = True,
) -> VideoContext:
    """Build a :class:`VideoContext` for ``source``, populating ``sections``.

    The probe always runs; it is cheap and every analyzer needs it. Beyond
    that, only the analyzers required to satisfy ``sections`` -- and their
    transitive prerequisites -- are executed.

    Args:
        source: Media file to analyse.
        sections: Sections the caller needs. Empty means probe only.
        cache: Store for results and derived artefacts. Defaults to the
            standard cache root.
        registry: Analyzer registry. Defaults to the process-wide one.
        use_cache: Set ``False`` to recompute everything.

    Returns:
        A context populated with the requested sections.

    Raises:
        VisionUseError: If the source cannot be probed, or a requested section
            has no registered producer.

    """
    store = cache or CacheStore()
    resolver = registry if registry is not None else default_registry

    context = VideoContext(probe=probe(source))
    plan: Sequence[Analyzer] = resolver.resolve(sections)

    log.info(
        "analyzing",
        extra=bind(
            source=source.name,
            duration_s=round(context.duration_s, 3),
            analyzers=[analyzer.name for analyzer in plan],
        ),
    )

    for analyzer in plan:
        request = AnalyzerRequest(source=source, context=context, cache=store)
        context = merge(context, analyzer, run_analyzer(analyzer, request, use_cache=use_cache))

    return context


def context_digest(context: VideoContext) -> str:
    """Return a stable digest of ``context``.

    Two analyses of the same file with the same settings must produce the same
    digest -- that is M1's exit criterion, and this is how it is asserted.
    """
    return hash_params({"context": context.model_dump(mode="json")})
