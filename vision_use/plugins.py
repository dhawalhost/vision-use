"""Discovering skills, analyzers and effects published by other packages.

Principle 3 says plugins over built-ins, which only means anything if a
third-party package can add a capability *without the core changing*. Entry
points are how: a package declares what it provides, VisionUse finds it at
import, and nothing in this repository needs to know the plugin exists.

**A broken plugin must not take the framework down.** One package with a typo in
its module path would otherwise make `vision --help` fail, which is a terrible
trade for an optional extension. Every failure is caught, recorded against the
plugin's name, and reported by ``vision plugins`` -- so the framework still
works and the user still finds out.

A plugin declares itself like this:

```toml
[project.entry-points."vision_use.skills"]
stabilize = "vision_stabilize:StabilizeSkill"
```

The object may be the class itself or a zero-argument factory; both are common
in the wild and telling a plugin author which one to use buys nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Final

from vision_use.context.registry import AnalyzerRegistry
from vision_use.context.registry import registry as default_analyzers
from vision_use.effects.registry import EffectRegistry
from vision_use.effects.registry import registry as default_effects
from vision_use.log import bind, get_logger
from vision_use.skills.registry import SkillRegistry
from vision_use.skills.registry import registry as default_skills

log = get_logger(__name__)

#: Entry-point group for skills a plugin publishes.
SKILL_GROUP: Final = "vision_use.skills"

#: Entry-point group for analyzers, which fill context sections.
ANALYZER_GROUP: Final = "vision_use.analyzers"

#: Entry-point group for effects a clip can carry.
EFFECT_GROUP: Final = "vision_use.effects"

#: Every group VisionUse looks in.
GROUPS: Final[tuple[str, ...]] = (SKILL_GROUP, ANALYZER_GROUP, EFFECT_GROUP)


@dataclass(frozen=True, slots=True)
class LoadedPlugin:
    """One capability a plugin successfully contributed."""

    group: str
    name: str
    module: str


@dataclass(frozen=True, slots=True)
class FailedPlugin:
    """One capability that could not be loaded, and why.

    Kept rather than raised: a plugin that fails to import is the plugin's
    problem, and the rest of the framework should stay usable.
    """

    group: str
    name: str
    reason: str


@dataclass(frozen=True, slots=True)
class PluginReport:
    """What discovery found."""

    loaded: tuple[LoadedPlugin, ...] = field(default=())
    failed: tuple[FailedPlugin, ...] = field(default=())

    @property
    def ok(self) -> bool:
        """Whether every discovered plugin loaded."""
        return not self.failed


def _discover(group: str) -> Iterator[EntryPoint]:
    """Yield the entry points declared in ``group``.

    Wrapped because a corrupt distribution in site-packages can make the
    metadata scan itself raise, and that would take down a framework the broken
    package has nothing to do with.
    """
    try:
        yield from entry_points(group=group)
    except Exception as exc:  # noqa: BLE001 - importlib.metadata raises broadly
        log.warning(
            "could not scan for plugins",
            extra=bind(group=group, detail=str(exc)),
        )


def _instantiate(loaded: object) -> object:
    """Return an instance from an entry point's object.

    A plugin may publish the class or a factory. Calling a class gives an
    instance; calling a factory gives whatever it makes. Either is fine, and
    requiring one over the other would only produce a support question.
    """
    return loaded() if callable(loaded) else loaded


def load_plugins(
    *,
    skills: SkillRegistry | None = None,
    analyzers: AnalyzerRegistry | None = None,
    effects: EffectRegistry | None = None,
) -> PluginReport:
    """Find and register every published plugin.

    Args:
        skills: Registry to add skills to. Defaults to the process-wide one.
        analyzers: Registry to add analyzers to.
        effects: Registry to add effects to.

    Returns:
        What loaded and what did not. Failures are recorded, never raised.

    """
    # Each registry's `register` takes its own protocol; they have no common
    # supertype, and a plugin's object is not typed until it is loaded anyway.
    # `x if x is not None else default`, never `x or default`: an empty registry
    # is falsy, so `or` would quietly register a caller's plugins into the
    # process-wide one -- which is exactly what a caller passing their own
    # registry is trying to avoid.
    targets: dict[str, Callable[[Any], None]] = {
        SKILL_GROUP: (default_skills if skills is None else skills).register,
        ANALYZER_GROUP: (default_analyzers if analyzers is None else analyzers).register,
        EFFECT_GROUP: (default_effects if effects is None else effects).register,
    }

    loaded: list[LoadedPlugin] = []
    failed: list[FailedPlugin] = []

    for group, register in targets.items():
        for entry in _discover(group):
            try:
                register(_instantiate(entry.load()))
            except Exception as exc:  # noqa: BLE001 - third-party code, any failure
                failed.append(
                    FailedPlugin(
                        group=group,
                        name=entry.name,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
                log.warning(
                    "plugin failed to load",
                    extra=bind(group=group, plugin=entry.name, detail=str(exc)),
                )
                continue

            loaded.append(LoadedPlugin(group=group, name=entry.name, module=entry.value))
            log.debug("loaded plugin", extra=bind(group=group, plugin=entry.name))

    if loaded or failed:
        log.info(
            "plugin discovery complete",
            extra=bind(loaded=len(loaded), failed=len(failed)),
        )
    return PluginReport(loaded=tuple(loaded), failed=tuple(failed))


#: What the last discovery found, so ``vision plugins`` can report it without
#: re-importing every plugin -- a second import would re-run module-level code.
_last_report: PluginReport | None = None


def ensure_loaded() -> PluginReport:
    """Load plugins once per process and return what was found."""
    global _last_report  # noqa: PLW0603 - one-shot process-wide discovery
    if _last_report is None:
        _last_report = load_plugins()
    return _last_report
