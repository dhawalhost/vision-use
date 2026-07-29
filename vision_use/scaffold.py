"""Generating a working plugin package.

Documentation tells an author what a plugin looks like; a scaffold gets them a
package that already installs, already registers, and already passes a test.
The gap between those two is where most would-be plugin authors give up, so
this exists to close it: ``vision plugin new my-plugin`` produces something
runnable in one command.

**Templates are strings here, not files, and there is no cookiecutter
dependency.** The substitution needed is four names; a templating engine and a
package-data story would both be more machinery than the job has (Principle 5).
Placeholders are ``__LIKE_THIS__`` rather than ``{{ this }}`` because the
templates are Python and TOML full of braces.

The generated skill is a real edit, not a stub. An author's first act is to
change what it does, and starting from something that works beats starting from
something that raises ``NotImplementedError``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vision_use.errors import VisionUseError
from vision_use.log import bind, get_logger
from vision_use.plugins import SKILL_GROUP

log = get_logger(__name__)

#: A distribution name PyPI would accept: letters, digits, and single
#: separators between them.
_NAME_PATTERN: Final = re.compile(r"^[a-z][a-z0-9]*([-_][a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class PluginNames:
    """The four names a template needs, derived from one.

    Attributes:
        dist: Distribution name, as it would appear on PyPI (``my-plugin``).
        module: Importable package name (``my_plugin``).
        cls: Class name for the generated skill (``MyPluginSkill``).
        skill: Registry key the skill claims (``my_plugin.trim_tail``).

    """

    dist: str
    module: str
    cls: str
    skill: str

    @classmethod
    def parse(cls, name: str) -> PluginNames:
        """Derive every name from a distribution name.

        Raises:
            VisionUseError: If the name would not make a valid package.

        """
        cleaned = name.strip().lower()
        if not _NAME_PATTERN.match(cleaned):
            raise VisionUseError(
                f"{name!r} is not a usable package name",
                remedy="use lowercase letters, digits, and single hyphens, e.g. my-plugin",
            )
        module = cleaned.replace("-", "_")
        return cls(
            dist=cleaned.replace("_", "-"),
            module=module,
            cls="".join(part.title() for part in module.split("_")) + "Skill",
            skill=f"{module}.trim_tail",
        )


_PYPROJECT = """\
[project]
name = "__DIST__"
version = "0.1.0"
description = "A VisionUse plugin."
requires-python = ">=3.12"
dependencies = ["vision-use"]

# This block is the whole plugin mechanism. VisionUse finds the skill through
# it; nothing in VisionUse changes to accommodate this package.
[project.entry-points."__SKILL_GROUP__"]
trim-tail = "__MODULE__:__CLASS__"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/__MODULE__"]

[dependency-groups]
dev = ["pytest>=8.3.0"]
"""

_INIT = '''\
"""__DIST__ -- a VisionUse plugin."""

from __MODULE__.skill import __CLASS__

__all__ = ["__CLASS__"]
'''

_SKILL = '''\
"""The skill this plugin publishes.

Trims dead air off the end of a timeline: the last clip is retrimmed, nothing
is removed, and nothing is encoded. Replace it with whatever you came here to
build -- it is a working starting point, not a stub.

Four rules keep a skill composable with skills it has never heard of:

1. Return a `PlanPatch`, never a rendered file.
2. Never import another skill; compose through the `EditPlan`.
3. Declare what context you need in `meta.requires`, rather than decoding.
4. Be idempotent for the same context, plan, and parameters -- the retry loop
   re-runs skills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from vision_use.errors import VisionUseError
from vision_use.models.context import ContextSection, VideoContext
from vision_use.models.plan import EditPlan
from vision_use.skills.patch import PlanPatch, SetClipRange
from vision_use.skills.skill import (
    Analysis,
    Measurement,
    Score,
    SkillMeta,
    SkillRequest,
    Strategy,
    ValidationReport,
)


@dataclass(frozen=True, slots=True)
class _Trim:
    """What `execute` needs to carry out one strategy."""

    clip_id: str
    source_out_s: float


class __CLASS__:
    """Trims a fixed amount off the end of the last clip."""

    meta: ClassVar[SkillMeta] = SkillMeta(
        name="__SKILL__",
        version="1",
        summary="Trim dead air off the end of the timeline.",
        # The runtime guarantees these sections are populated before `analyse`
        # runs. Ask for what you read, and nothing more: every section costs.
        requires=frozenset({ContextSection.PROBE}),
    )

    #: Offered as successive strategies, boldest first. Several is what gives
    #: the retry loop somewhere to go when validation rejects the first.
    AMOUNTS_S: ClassVar[tuple[float, ...]] = (2.0, 1.0, 0.5)

    #: A clip shorter than this is not worth trimming further.
    MIN_REMAINING_S: ClassVar[float] = 0.5

    def analyse(
        self, context: VideoContext, plan: EditPlan, request: SkillRequest
    ) -> Analysis:
        """Find the last clip. Read-only: decide nothing yet."""
        del context, request
        clips = plan.timeline.clips()
        if not clips:
            return Analysis(findings={"clips": 0})
        last = max(clips, key=lambda clip: clip.timeline_start_s)
        return Analysis(findings={"clips": len(clips), "last": last.id}, detail=last)

    def plan(self, analysis: Analysis) -> list[Strategy]:
        """Offer one strategy per amount, boldest first."""
        last = analysis.detail
        if last is None:
            return []
        return [
            Strategy(
                name=f"trim-{amount}s",
                rationale=f"trim {amount}s off the end",
                detail=_Trim(clip_id=last.id, source_out_s=last.source_out_s - amount),
            )
            for amount in self.AMOUNTS_S
            if last.duration_s - amount >= self.MIN_REMAINING_S
        ]

    def execute(self, strategy: Strategy) -> PlanPatch:
        """Turn one strategy into an edit.

        Raises:
            VisionUseError: If handed a strategy this skill did not produce.

        """
        trim = strategy.detail
        if not isinstance(trim, _Trim):
            raise VisionUseError(
                "This strategy did not come from this skill",
                remedy="pass a strategy returned by plan()",
            )
        return PlanPatch(
            ops=(SetClipRange(clip_id=trim.clip_id, source_out_s=trim.source_out_s),),
            rationale=strategy.rationale,
            produced_by=self.meta.ref,
        )

    def validate(
        self, before: EditPlan, after: EditPlan, context: VideoContext
    ) -> ValidationReport:
        """Judge the edit. A skill that cannot say what would make its own
        output wrong is not finished.
        """
        del context
        return ValidationReport(
            measurements=(
                Measurement(
                    name="shortened",
                    value=before.duration_s - after.duration_s,
                    passed=after.duration_s < before.duration_s,
                    detail="the trim had no effect",
                ),
                Measurement(
                    name="survives",
                    value=after.duration_s,
                    passed=after.duration_s > 0,
                    detail="the render would be empty",
                ),
            )
        )

    def score(self, report: ValidationReport) -> Score:
        """Prefer the boldest trim that still passed."""
        if not report.passed:
            return Score(value=-1.0)
        shortened = next(
            (item.value for item in report.measurements if item.name == "shortened"),
            0.0,
        )
        return Score(value=shortened)
'''

_TEST = '''\
"""Tests for __DIST__.

The first one is the one that matters: it proves the *plugin* works, not just
the class. A skill that is correct but never discovered helps nobody.
"""

from __future__ import annotations

from __MODULE__ import __CLASS__
from vision_use.plugins import load_plugins
from vision_use.skills.registry import SkillRegistry
from vision_use.skills.skill import Analysis, Skill


def test_the_plugin_is_discovered() -> None:
    """Fails until this package is installed -- run `uv sync` first."""
    registry = SkillRegistry()
    load_plugins(skills=registry)

    assert "__SKILL__" in registry


def test_the_skill_satisfies_the_contract() -> None:
    assert isinstance(__CLASS__(), Skill)


def test_no_clips_means_no_strategies() -> None:
    """Having nothing to do is a legitimate outcome, not an error."""
    assert __CLASS__().plan(Analysis(findings={"clips": 0})) == []
'''

_README = """\
# __DIST__

A [VisionUse](https://github.com/dhawalhost/vision-use) plugin.

## Install

```bash
uv sync
```

## Verify VisionUse can see it

```bash
uv run vision plugins
```

```bash
uv run vision skills
```

`__SKILL__` should appear in both. If it does not, the entry-point block in
`pyproject.toml` is the first place to look — and remember that a plugin is only
discovered once its package is *installed*, not merely present on disk.

## Use it

```bash
uv run vision plan new input.mp4 -o plan.json
uv run vision apply plan.json --skill __SKILL__
uv run vision render plan.json -o out.mp4 --proxy
```

## Test

```bash
uv run pytest
```
"""

#: Relative path to template, in the order they are written.
_FILES: Final[tuple[tuple[str, str], ...]] = (
    ("pyproject.toml", _PYPROJECT),
    ("README.md", _README),
    ("src/__MODULE__/__init__.py", _INIT),
    ("src/__MODULE__/skill.py", _SKILL),
    ("tests/test___MODULE__.py", _TEST),
)


def _fill(text: str, names: PluginNames) -> str:
    """Substitute every placeholder in ``text``."""
    return (
        text.replace("__DIST__", names.dist)
        .replace("__MODULE__", names.module)
        .replace("__CLASS__", names.cls)
        .replace("__SKILL__", names.skill)
        .replace("__SKILL_GROUP__", SKILL_GROUP)
    )


def scaffold_plugin(name: str, parent: Path) -> tuple[PluginNames, tuple[Path, ...]]:
    """Write a working plugin package under ``parent``.

    Args:
        name: Distribution name, e.g. ``my-plugin``.
        parent: Directory to create the package directory in.

    Returns:
        The derived names and every file written, in order.

    Raises:
        VisionUseError: If the name is unusable, or the target already exists.
            Refusing beats overwriting an author's work.

    """
    names = PluginNames.parse(name)
    root = parent / names.dist
    if root.exists():
        raise VisionUseError(
            f"{root} already exists",
            remedy="choose another name, or remove the existing directory",
        )

    written: list[Path] = []
    for relative, template in _FILES:
        target = root / _fill(relative, names)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_fill(template, names), encoding="utf-8")
        written.append(target)

    log.info("scaffolded plugin", extra=bind(plugin=names.dist, files=len(written)))
    return names, tuple(written)
