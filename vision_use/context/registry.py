"""Analyzer registration and execution ordering.

The registry answers one question: *given the sections a caller wants, which
analyzers must run and in what order?* It resolves that from the analyzers'
declared ``requires`` and ``produces`` rather than from a hardcoded list, so
adding an analyzer -- including one from a plugin -- needs no change here.

The registry holds no video state. It is a lookup table plus a topological
sort, which keeps it testable without media and keeps the runtime free of the
global mutable state ``CLAUDE.md`` forbids: the module-level :data:`registry`
is a convenience default, and every consumer accepts an explicit instance.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from vision_use.context.analyzer import Analyzer
from vision_use.errors import ConfigurationError, VisionUseError
from vision_use.log import bind, get_logger
from vision_use.models.context import BUILTIN_SECTIONS, ContextSection, SectionName

log = get_logger(__name__)


class AnalyzerRegistry:
    """A collection of analyzers, resolvable into an execution order."""

    def __init__(self) -> None:
        self._by_name: dict[str, Analyzer] = {}

    def register(self, analyzer: Analyzer, *, replace: bool = False) -> None:
        """Add ``analyzer`` to the registry.

        Args:
            analyzer: The analyzer to add.
            replace: Permit replacing an existing registration of the same
                name. Off by default so that two plugins claiming one name is
                a loud error rather than a load-order-dependent surprise.

        Raises:
            ConfigurationError: If the analyzer produces nothing, if its
                requirements and products overlap, or if the name collides.

        """
        if not analyzer.produces:
            raise ConfigurationError(
                f"Analyzer {analyzer.name!r} declares no produced sections",
                remedy="set the 'produces' class variable",
            )

        overlap = analyzer.produces & analyzer.requires
        if overlap:
            names = ", ".join(sorted(str(section) for section in overlap))
            raise ConfigurationError(
                f"Analyzer {analyzer.name!r} both requires and produces: {names}",
                remedy="an analyzer cannot depend on its own output; split it in two",
            )

        if analyzer.name in self._by_name and not replace:
            raise ConfigurationError(
                f"An analyzer named {analyzer.name!r} is already registered",
                remedy="rename one of them, or pass replace=True to override",
            )

        self._by_name[analyzer.name] = analyzer
        log.debug(
            "registered analyzer",
            extra=bind(
                name=analyzer.name,
                version=analyzer.version,
                produces=sorted(str(section) for section in analyzer.produces),
            ),
        )

    def unregister(self, name: str) -> None:
        """Remove the analyzer called ``name``, if present."""
        self._by_name.pop(name, None)

    def get(self, name: str) -> Analyzer:
        """Return the analyzer called ``name``.

        Raises:
            ConfigurationError: If no such analyzer is registered.

        """
        try:
            return self._by_name[name]
        except KeyError:
            known = ", ".join(sorted(self._by_name)) or "none"
            raise ConfigurationError(
                f"No analyzer named {name!r}",
                remedy=f"registered analyzers: {known}",
            ) from None

    def all(self) -> list[Analyzer]:
        """Return every registered analyzer, ordered by name."""
        return [self._by_name[name] for name in sorted(self._by_name)]

    def producer_of(self, section: SectionName) -> Analyzer | None:
        """Return the analyzer that fills ``section``, or ``None``.

        When several analyzers claim the same section, the one whose name sorts
        first wins. That is arbitrary but deterministic; selecting between real
        alternatives is the AI decision engine's job, not the registry's.
        """
        candidates = [analyzer for analyzer in self.all() if section in analyzer.produces]
        return candidates[0] if candidates else None

    def resolve(self, wanted: Iterable[SectionName]) -> list[Analyzer]:
        """Return the analyzers needed for ``wanted``, in execution order.

        Prerequisites are pulled in transitively, each analyzer appears once,
        and the result is stable across runs.

        Args:
            wanted: Sections the caller needs populated. ``PROBE`` is ignored:
                it is always present before any analyzer runs.

        Returns:
            Analyzers ordered so that each one's requirements are satisfied by
            those before it.

        Raises:
            ConfigurationError: If a section has no producer, or if the
                declarations contain a dependency cycle.

        """
        targets = [
            section for section in dict.fromkeys(wanted) if section != ContextSection.PROBE
        ]

        ordered: list[Analyzer] = []
        placed: set[str] = set()
        visiting: list[str] = []

        def visit(section: SectionName) -> None:
            if section == ContextSection.PROBE:
                return

            analyzer = self.producer_of(section)
            if analyzer is None:
                raise ConfigurationError(
                    f"No registered analyzer produces {str(section)!r}",
                    remedy=(
                        "register one, or drop that section from the request"
                        if "." not in str(section)
                        # A dotted name is almost always a plugin's section, and
                        # the usual cause is that the plugin is not installed.
                        else f"install the plugin that provides {str(section)!r}, "
                        "or drop that section from the request"
                    ),
                )

            if analyzer.name in placed:
                return
            if analyzer.name in visiting:
                cycle = " -> ".join([*visiting, analyzer.name])
                raise ConfigurationError(
                    f"Analyzer dependency cycle: {cycle}",
                    remedy="break the cycle by removing a declared requirement",
                )

            visiting.append(analyzer.name)
            for prerequisite in sorted(analyzer.requires, key=str):
                visit(prerequisite)
            visiting.pop()

            placed.add(analyzer.name)
            ordered.append(analyzer)

        for section in targets:
            visit(section)

        return ordered

    def sections(self) -> set[SectionName]:
        """Return every section any registered analyzer can produce."""
        return {section for analyzer in self._by_name.values() for section in analyzer.produces}

    def __len__(self) -> int:
        """Return how many analyzers are registered."""
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        """Whether an analyzer with this name is registered."""
        return isinstance(name, str) and name in self._by_name


def default_registry(analyzers: Sequence[Analyzer] = ()) -> AnalyzerRegistry:
    """Return a fresh registry pre-populated with ``analyzers``.

    Tests use this to build an isolated registry rather than mutating the
    module-level default.
    """
    instance = AnalyzerRegistry()
    for analyzer in analyzers:
        instance.register(analyzer)
    return instance


#: Process-wide default registry. Built-in analyzers register themselves here
#: on import; consumers that need isolation construct their own.
registry = AnalyzerRegistry()


def resolve_section_names(
    names: Iterable[str], *, analyzers: AnalyzerRegistry | None = None
) -> list[SectionName]:
    """Resolve section names given by a user into sections.

    A built-in section is named by its own value; a plugin's section by the
    dotted name it publishes (``"stabilize.motion"``). Both are looked up here
    rather than against the enum alone, because a plugin's section is exactly as
    real to a caller as a built-in one (PLAN.md § D5).

    Args:
        names: What the caller typed.
        analyzers: Registry used to recognise plugin sections. Defaults to the
            process-wide one.

    Returns:
        The resolved sections, in the order given.

    Raises:
        VisionUseError: If a name is neither a built-in section nor one any
            registered analyzer produces. The message lists what is available,
            since a typo is the likeliest cause -- and if a plugin is simply not
            installed, seeing its section missing from that list is the fastest
            way to notice.

    """
    resolver = registry if analyzers is None else analyzers
    builtin = {section.value: section for section in ContextSection}
    plugin = {str(section) for section in resolver.sections() if str(section) not in builtin}

    resolved: list[SectionName] = []
    for name in names:
        if name in builtin:
            resolved.append(builtin[name])
        elif name in plugin:
            resolved.append(name)
        else:
            raise VisionUseError(
                f"Unknown context section {name!r}",
                remedy=f"choose from: {', '.join(sorted(set(builtin) | plugin))}",
            )
    return resolved


def available_sections(analyzers: AnalyzerRegistry | None = None) -> list[SectionName]:
    """Return every section that can be populated, built-in and plugin alike.

    ``probe`` is excluded: it is always present before any analyzer runs, so
    asking for it is a no-op.
    """
    resolver = registry if analyzers is None else analyzers
    builtin = [section for section in ContextSection if section is not ContextSection.PROBE]
    plugin = sorted(
        str(section) for section in resolver.sections() if str(section) not in BUILTIN_SECTIONS
    )
    return [*builtin, *plugin]
