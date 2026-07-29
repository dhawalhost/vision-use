"""Skill registration and lookup.

Mirrors :mod:`vision_use.context.registry` deliberately: a plugin author who has
written an analyzer should recognise this immediately, and a skill from a plugin
is indistinguishable from a built-in one.

The registry holds no video state -- it is a lookup table, which keeps it
testable without media and keeps the runtime free of the global mutable state
``CLAUDE.md`` forbids. The module-level :data:`registry` is a convenience
default; every consumer accepts an explicit instance.
"""

from __future__ import annotations

from collections.abc import Sequence

from vision_use.errors import ConfigurationError
from vision_use.log import bind, get_logger
from vision_use.models.context import SectionName
from vision_use.skills.skill import Skill

log = get_logger(__name__)


class SkillRegistry:
    """A collection of skills, addressable by name."""

    def __init__(self) -> None:
        self._by_name: dict[str, Skill] = {}

    def register(self, skill: Skill, *, replace: bool = False) -> None:
        """Add ``skill`` to the registry.

        Raises:
            ConfigurationError: If the skill declares no name, or if the name
                is already taken. Two plugins claiming one name must fail
                loudly rather than depend on import order.

        """
        meta = skill.meta
        if not meta.name:
            raise ConfigurationError(
                "A skill must declare a non-empty name",
                remedy="set SkillMeta.name",
            )
        if meta.name in self._by_name and not replace:
            raise ConfigurationError(
                f"A skill named {meta.name!r} is already registered",
                remedy="rename one of them, or pass replace=True to override",
            )

        self._by_name[meta.name] = skill
        log.debug(
            "registered skill",
            extra=bind(
                name=meta.name,
                version=meta.version,
                requires=sorted(str(section) for section in meta.requires),
            ),
        )

    def unregister(self, name: str) -> None:
        """Remove the skill called ``name``, if present."""
        self._by_name.pop(name, None)

    def get(self, name: str) -> Skill:
        """Return the skill called ``name``.

        Raises:
            ConfigurationError: If no such skill is registered.

        """
        try:
            return self._by_name[name]
        except KeyError:
            known = ", ".join(sorted(self._by_name)) or "none"
            raise ConfigurationError(
                f"No skill named {name!r}",
                remedy=f"registered skills: {known}",
            ) from None

    def all(self) -> list[Skill]:
        """Return every registered skill, ordered by name."""
        return [self._by_name[name] for name in sorted(self._by_name)]

    def requiring(self, section: SectionName) -> list[Skill]:
        """Return the skills that read ``section``.

        Useful for answering "what becomes possible if I transcribe this?"
        """
        return [skill for skill in self.all() if section in skill.meta.requires]

    def __len__(self) -> int:
        """Return how many skills are registered."""
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        """Whether a skill with this name is registered."""
        return isinstance(name, str) and name in self._by_name


def default_registry(skills: Sequence[Skill] = ()) -> SkillRegistry:
    """Return a fresh registry pre-populated with ``skills``.

    Tests use this to build an isolated registry rather than mutating the
    module-level default.
    """
    instance = SkillRegistry()
    for skill in skills:
        instance.register(skill)
    return instance


#: Process-wide default registry. Built-in skills register themselves here on
#: import; consumers needing isolation construct their own.
registry = SkillRegistry()
