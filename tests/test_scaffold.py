"""Tests for the plugin scaffold.

A template that drifts from the framework is worse than no template: it teaches
an author the wrong shape and fails at install time, far from the cause. So
these tests do not check that files exist -- they check that what is generated
still *matches this version of VisionUse*: the entry-point group is the one
discovery scans, the generated skill satisfies the protocol, and the code is
importable rather than merely present.

The full loop -- scaffold, install, discover, apply, render -- was verified by
hand and is recorded in `TASKS.md`; it needs a build step and a network, which
does not belong in a unit test.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner
from vision_use.cli.main import app
from vision_use.errors import VisionUseError
from vision_use.plugins import SKILL_GROUP
from vision_use.scaffold import PluginNames, scaffold_plugin
from vision_use.skills.skill import Analysis, Skill

runner = CliRunner()


# --- names -----------------------------------------------------------------


def test_names_are_derived_from_one() -> None:
    names = PluginNames.parse("my-plugin")

    assert names.dist == "my-plugin"
    assert names.module == "my_plugin"
    assert names.cls == "MyPluginSkill"
    assert names.skill == "my_plugin.trim_tail"


def test_underscores_and_hyphens_are_interchangeable() -> None:
    """PyPI normalises them; an author should not have to care which they typed."""
    assert PluginNames.parse("my_plugin") == PluginNames.parse("my-plugin")


def test_case_and_whitespace_are_normalised() -> None:
    """PyPI normalises these too; refusing them would be pedantry."""
    assert PluginNames.parse("  My-Plugin ") == PluginNames.parse("my-plugin")


@pytest.mark.parametrize(
    "name", ["", "9lives", "has space", "trailing-", "double--dash", "dot.ted"]
)
def test_an_unusable_name_is_refused_with_a_remedy(name: str) -> None:
    """Failing here beats failing at `pip install` with a packaging error."""
    with pytest.raises(VisionUseError) as excinfo:
        PluginNames.parse(name)

    assert excinfo.value.remedy


# --- what is written -------------------------------------------------------


def test_scaffolding_writes_a_package(tmp_path: Path) -> None:
    names, written = scaffold_plugin("demo-plugin", tmp_path)

    assert (tmp_path / names.dist / "pyproject.toml") in written
    assert (tmp_path / names.dist / "src" / "demo_plugin" / "skill.py").exists()
    assert (tmp_path / names.dist / "tests" / "test_demo_plugin.py").exists()


def test_no_placeholder_survives(tmp_path: Path) -> None:
    """A leftover placeholder is a syntax error in someone else's package."""
    _, written = scaffold_plugin("demo-plugin", tmp_path)

    for path in written:
        assert "__DIST__" not in path.read_text()
        assert "__MODULE__" not in path.read_text()
        assert "__CLASS__" not in path.read_text()


def test_an_existing_directory_is_never_overwritten(tmp_path: Path) -> None:
    scaffold_plugin("demo-plugin", tmp_path)

    with pytest.raises(VisionUseError, match="already exists"):
        scaffold_plugin("demo-plugin", tmp_path)


# --- does it match this VisionUse? -----------------------------------------


def _pyproject(tmp_path: Path) -> dict[str, Any]:
    _, written = scaffold_plugin("demo-plugin", tmp_path)
    target = next(path for path in written if path.name == "pyproject.toml")
    data: dict[str, Any] = tomllib.loads(target.read_text())
    return data


def test_the_entry_point_group_is_the_one_discovery_scans(tmp_path: Path) -> None:
    """The template's group and `plugins.SKILL_GROUP` must not drift apart."""
    groups = _pyproject(tmp_path)["project"]["entry-points"]

    assert SKILL_GROUP in groups
    assert groups[SKILL_GROUP]["trim-tail"] == "demo_plugin:DemoPluginSkill"


def test_the_generated_pyproject_is_valid_toml(tmp_path: Path) -> None:
    project = _pyproject(tmp_path)["project"]

    assert project["name"] == "demo-plugin"
    assert "vision-use" in project["dependencies"]


def test_the_generated_skill_satisfies_the_protocol(tmp_path: Path) -> None:
    """Imported and instantiated, not merely grepped for method names."""
    skill = _load_skill(tmp_path)

    assert isinstance(skill, Skill)
    assert skill.meta.name == "demo_plugin.trim_tail"


def test_the_generated_skill_does_nothing_when_there_is_nothing_to_do(
    tmp_path: Path,
) -> None:
    """Returning no strategies is a legitimate outcome, and the template shows it."""
    skill = _load_skill(tmp_path)

    assert skill.plan(Analysis(findings={"clips": 0})) == []


def _load_skill(tmp_path: Path) -> Any:
    """Import the generated skill module and return an instance."""
    _, written = scaffold_plugin("demo-plugin", tmp_path)
    module_path = next(path for path in written if path.name == "skill.py")

    spec = importlib.util.spec_from_file_location("demo_plugin_skill", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_plugin_skill"] = module
    spec.loader.exec_module(module)
    return module.DemoPluginSkill()


# --- the command -----------------------------------------------------------


def test_the_command_scaffolds_and_says_what_is_next(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plugin", "new", "demo-plugin", "--into", str(tmp_path)])

    assert result.exit_code == 0
    assert "uv sync" in result.stdout
    assert (tmp_path / "demo-plugin" / "pyproject.toml").exists()


def test_the_command_refuses_a_bad_name(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plugin", "new", "Bad Name", "--into", str(tmp_path)])

    assert result.exit_code != 0
