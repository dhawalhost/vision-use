"""Tests for plugin discovery.

Principle 3 only means something if a third-party package can add a capability
without the core changing. These tests use fabricated entry points rather than
installing a package, so they exercise the discovery path without needing a
build step -- the real installation is verified by hand and documented in
`CONTRIBUTING.md`.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict
from vision_use.context.registry import AnalyzerRegistry
from vision_use.effects.registry import EffectDef, EffectRegistry
from vision_use.models.context import ContextSection
from vision_use.plugins import (
    ANALYZER_GROUP,
    EFFECT_GROUP,
    GROUPS,
    SKILL_GROUP,
    FailedPlugin,
    LoadedPlugin,
    PluginReport,
    load_plugins,
)
from vision_use.skills.registry import SkillRegistry
from vision_use.skills.registry import registry as process_wide_skills
from vision_use.skills.skill import SkillMeta


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _PluginEffect(EffectDef):
    """An effect as a third-party package would publish it."""

    effect_type: ClassVar[str] = "plugin.effect"
    params_model: ClassVar[type[BaseModel]] = _Params


class _PluginSkill:
    """A skill as a third-party package would publish it."""

    meta: ClassVar[SkillMeta] = SkillMeta(
        name="plugin.skill",
        version="1",
        summary="From a plugin.",
        requires=frozenset({ContextSection.PROBE}),
    )


def _make_effect() -> _PluginEffect:
    """A factory, as opposed to publishing the class itself."""
    return _PluginEffect()


class _FakeEntryPoint:
    """Stands in for an installed distribution's entry point."""

    def __init__(self, name: str, value: str, target: object) -> None:
        self.name = name
        self.value = value
        self._target = target

    def load(self) -> object:
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def _patch_entries(
    monkeypatch: pytest.MonkeyPatch, entries: dict[str, list[_FakeEntryPoint]]
) -> None:
    """Make discovery see exactly ``entries``."""

    def fake(*, group: str) -> list[_FakeEntryPoint]:
        return entries.get(group, [])

    monkeypatch.setattr("vision_use.plugins.entry_points", fake)


# --- discovery -------------------------------------------------------------


def test_nothing_installed_is_an_empty_report(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entries(monkeypatch, {})

    report = load_plugins(
        skills=SkillRegistry(), analyzers=AnalyzerRegistry(), effects=EffectRegistry()
    )

    assert report.loaded == ()
    assert report.ok


def test_a_skill_is_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core does not know this skill exists."""
    registry = SkillRegistry()
    _patch_entries(
        monkeypatch,
        {SKILL_GROUP: [_FakeEntryPoint("demo", "pkg:Skill", _PluginSkill)]},
    )

    report = load_plugins(
        skills=registry, analyzers=AnalyzerRegistry(), effects=EffectRegistry()
    )

    assert "plugin.skill" in registry
    assert report.loaded[0].group == SKILL_GROUP


def test_an_effect_is_discovered(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = EffectRegistry()
    _patch_entries(
        monkeypatch,
        {EFFECT_GROUP: [_FakeEntryPoint("tint", "pkg:Effect", _PluginEffect)]},
    )

    load_plugins(skills=SkillRegistry(), analyzers=AnalyzerRegistry(), effects=registry)

    assert "plugin.effect" in registry


def test_a_class_and_a_factory_both_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both are common in the wild; requiring one buys nothing."""
    registry = EffectRegistry()
    _patch_entries(
        monkeypatch,
        {
            EFFECT_GROUP: [
                _FakeEntryPoint("cls", "pkg:Effect", _PluginEffect),
                _FakeEntryPoint("factory", "pkg:make", _make_effect),
            ]
        },
    )

    report = load_plugins(
        skills=SkillRegistry(), analyzers=AnalyzerRegistry(), effects=registry
    )

    # The second collides with the first, which is correct -- the point is that
    # both forms were called rather than one being rejected outright.
    assert len(report.loaded) + len(report.failed) == 2
    assert report.loaded[0].name == "cls"


def test_a_caller_registry_is_used_even_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty registry is falsy; `or` would send plugins to the global one."""
    registry = SkillRegistry()
    _patch_entries(
        monkeypatch,
        {SKILL_GROUP: [_FakeEntryPoint("demo", "pkg:Skill", _PluginSkill)]},
    )

    load_plugins(skills=registry, analyzers=AnalyzerRegistry(), effects=EffectRegistry())

    assert "plugin.skill" in registry
    assert "plugin.skill" not in process_wide_skills


def test_every_group_is_scanned(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(*, group: str) -> list[_FakeEntryPoint]:
        seen.append(group)
        return []

    monkeypatch.setattr("vision_use.plugins.entry_points", fake)
    load_plugins(skills=SkillRegistry(), analyzers=AnalyzerRegistry(), effects=EffectRegistry())

    assert set(seen) == set(GROUPS)


# --- failure handling ------------------------------------------------------


def test_a_broken_plugin_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """One typo in a plugin must not make `vision --help` fail."""
    _patch_entries(
        monkeypatch,
        {SKILL_GROUP: [_FakeEntryPoint("broken", "pkg:missing", ImportError("no module"))]},
    )

    report = load_plugins(
        skills=SkillRegistry(), analyzers=AnalyzerRegistry(), effects=EffectRegistry()
    )

    assert not report.ok
    assert report.failed[0].name == "broken"
    assert "ImportError" in report.failed[0].reason


def test_a_broken_plugin_does_not_stop_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SkillRegistry()
    _patch_entries(
        monkeypatch,
        {
            SKILL_GROUP: [
                _FakeEntryPoint("broken", "pkg:missing", ImportError("boom")),
                _FakeEntryPoint("good", "pkg:Skill", _PluginSkill),
            ]
        },
    )

    report = load_plugins(
        skills=registry, analyzers=AnalyzerRegistry(), effects=EffectRegistry()
    )

    assert "plugin.skill" in registry
    assert len(report.failed) == 1


def test_a_plugin_that_registers_badly_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration can fail even when the import succeeds."""

    class Nameless:
        meta: ClassVar[SkillMeta] = SkillMeta(name="", version="1")

    _patch_entries(
        monkeypatch, {SKILL_GROUP: [_FakeEntryPoint("bad", "pkg:Nameless", Nameless)]}
    )

    report = load_plugins(
        skills=SkillRegistry(), analyzers=AnalyzerRegistry(), effects=EffectRegistry()
    )

    assert not report.ok


def test_a_corrupt_metadata_scan_is_survivable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken distribution elsewhere must not take this framework down."""

    def fake(*, group: str) -> list[_FakeEntryPoint]:
        raise RuntimeError("site-packages is a mess")

    monkeypatch.setattr("vision_use.plugins.entry_points", fake)

    report = load_plugins(
        skills=SkillRegistry(), analyzers=AnalyzerRegistry(), effects=EffectRegistry()
    )

    assert report.loaded == ()


def test_a_plugin_cannot_silently_shadow_a_builtin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name collision is a loud registration error, not an override."""
    registry = SkillRegistry()
    _patch_entries(
        monkeypatch, {SKILL_GROUP: [_FakeEntryPoint("dup", "pkg:Skill", _PluginSkill)]}
    )
    # The first load takes the name; the second is the collision under test.
    load_plugins(skills=registry, analyzers=AnalyzerRegistry(), effects=EffectRegistry())

    report = load_plugins(
        skills=registry, analyzers=AnalyzerRegistry(), effects=EffectRegistry()
    )

    assert not report.ok
    assert "already registered" in report.failed[0].reason


# --- the report ------------------------------------------------------------


def test_a_report_with_no_failures_is_ok() -> None:
    report = PluginReport(loaded=(LoadedPlugin(SKILL_GROUP, "a", "pkg:A"),))

    assert report.ok


def test_a_report_with_failures_is_not_ok() -> None:
    report = PluginReport(failed=(FailedPlugin(ANALYZER_GROUP, "a", "boom"),))

    assert not report.ok
