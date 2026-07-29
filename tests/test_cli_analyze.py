"""Tests for `vision analyze` and `vision sections`."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import JsonValue
from typer.testing import CliRunner
from vision_use.cli.main import DEFAULT_SECTIONS, _parse_sections, app
from vision_use.context.analyzer import AnalyzerRequest, AnalyzerResult
from vision_use.context.registry import registry as default_registry
from vision_use.errors import VisionUseError
from vision_use.models.context import ContextSection, SectionName

from tests.conftest import MakeClip, requires_ffmpeg

runner = CliRunner()
S = ContextSection


# --- section parsing -------------------------------------------------------


def test_default_sections_exclude_transcript() -> None:
    """Transcript needs an optional extra and downloads weights; opt in explicitly."""
    assert S.TRANSCRIPT not in DEFAULT_SECTIONS
    assert S.SHOTS in DEFAULT_SECTIONS


def test_no_selection_uses_defaults() -> None:
    assert _parse_sections(None, everything=False) == list(DEFAULT_SECTIONS)


def test_all_covers_every_producible_section() -> None:
    parsed = _parse_sections(None, everything=True)

    assert S.TRANSCRIPT in parsed
    assert S.PROBE not in parsed


def test_explicit_selection_is_parsed() -> None:
    assert _parse_sections("shots,transcript", everything=False) == [S.SHOTS, S.TRANSCRIPT]


def test_whitespace_and_empties_tolerated() -> None:
    assert _parse_sections(" shots , , keyframes ", everything=False) == [
        S.SHOTS,
        S.KEYFRAMES,
    ]


def test_all_overrides_explicit_selection() -> None:
    assert len(_parse_sections("shots", everything=True)) > 1


def test_unknown_section_lists_the_valid_ones() -> None:
    """A typo here is the likeliest mistake, so the error must be self-correcting."""
    with pytest.raises(VisionUseError) as excinfo:
        _parse_sections("bogus", everything=False)

    assert "bogus" in excinfo.value.message
    assert "shots" in (excinfo.value.remedy or "")


def test_empty_selection_yields_nothing() -> None:
    assert _parse_sections("", everything=False) == []


# --- sections command ------------------------------------------------------


def test_sections_lists_every_section() -> None:
    result = runner.invoke(app, ["sections"])

    assert result.exit_code == 0
    for section in ContextSection:
        assert section.value in result.stdout


def test_sections_names_producers_and_dependencies() -> None:
    result = runner.invoke(app, ["sections"])

    assert "shots.content" in result.stdout
    assert "keyframes.per_shot" in result.stdout


# --- analyze ---------------------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_analyze_reports_shots(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="cli_a", seconds=2.0)

    result = runner.invoke(app, ["analyze", str(clip), "--with", "shots"])

    assert result.exit_code == 0
    assert "shots" in result.stdout


@pytest.mark.integration
@requires_ffmpeg
def test_analyze_json_is_parseable(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Logging goes to stderr precisely so --json output stays machine-readable."""
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="cli_b", seconds=2.0)

    result = runner.invoke(app, ["analyze", str(clip), "--with", "shots", "--json"])

    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1"
    assert payload["shots"]


@pytest.mark.integration
@requires_ffmpeg
def test_analyze_writes_output_file(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="cli_c", seconds=2.0)
    target = tmp_path / "context.json"

    result = runner.invoke(app, ["analyze", str(clip), "--with", "shots", "-o", str(target)])

    assert result.exit_code == 0
    assert json.loads(target.read_text())["shots"]


@pytest.mark.integration
@requires_ffmpeg
def test_analyze_output_and_json_agree(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="cli_d", seconds=2.0)
    target = tmp_path / "context.json"

    runner.invoke(app, ["analyze", str(clip), "--with", "shots", "-o", str(target)])
    streamed = runner.invoke(app, ["analyze", str(clip), "--with", "shots", "--json"])

    assert json.loads(target.read_text()) == json.loads(streamed.stdout)


@pytest.mark.integration
@requires_ffmpeg
def test_analyze_resolves_prerequisites(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Asking for keyframes must pull shots in without the user naming them."""
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="cli_e", seconds=2.0)

    result = runner.invoke(app, ["analyze", str(clip), "--with", "keyframes", "--json"])

    payload = json.loads(result.stdout)
    assert payload["shots"]
    assert payload["keyframes"]


@pytest.mark.integration
@requires_ffmpeg
def test_analyze_no_cache_still_succeeds(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="cli_f", seconds=2.0)

    result = runner.invoke(
        app, ["analyze", str(clip), "--with", "shots", "--no-cache", "--json"]
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["shots"]


@pytest.mark.integration
@requires_ffmpeg
def test_repeated_analysis_is_byte_identical(
    make_clip: MakeClip, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """M1's exit criterion, asserted through the user-facing surface."""
    monkeypatch.chdir(tmp_path)
    clip = make_clip(name="cli_g", seconds=2.0)

    first = runner.invoke(app, ["analyze", str(clip), "--with", "shots", "--json"])
    second = runner.invoke(app, ["analyze", str(clip), "--with", "shots", "--json"])

    assert json.loads(first.stdout) == json.loads(second.stdout)


def test_analyze_unknown_section_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["analyze", str(tmp_path / "x.mp4"), "--with", "bogus"])

    assert result.exit_code == 1


def test_analyze_missing_file_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["analyze", str(tmp_path / "absent.mp4")])

    assert result.exit_code == 1


# --- plugin sections (PLAN.md D5) ------------------------------------------


class _PluginAnalyzer:
    """An analyzer publishing a section the CLI has never heard of."""

    name = "demo.motion"
    version = "1"
    produces: frozenset[SectionName] = frozenset({"demo.motion"})
    requires: frozenset[SectionName] = frozenset({ContextSection.PROBE})

    def params(self) -> dict[str, JsonValue]:
        return {}

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        del request
        return AnalyzerResult(extra={"demo.motion": {"mean_px": 4.5}})


@pytest.fixture
def with_plugin_analyzer() -> Iterator[None]:
    """Register a plugin analyzer on the process-wide registry, then remove it."""
    analyzer = _PluginAnalyzer()
    default_registry.register(analyzer)
    try:
        yield
    finally:
        default_registry.unregister(analyzer.name)


def test_a_plugin_section_can_be_requested(with_plugin_analyzer: None) -> None:
    """`--with stabilize.motion` must work exactly like `--with shots`."""
    del with_plugin_analyzer

    assert _parse_sections("demo.motion", everything=False) == ["demo.motion"]


def test_all_includes_plugin_sections(with_plugin_analyzer: None) -> None:
    """ "Everything" that silently excluded plugins would be a lie."""
    del with_plugin_analyzer

    assert "demo.motion" in _parse_sections(None, everything=True)


def test_an_unknown_section_lists_plugin_sections_too(with_plugin_analyzer: None) -> None:
    """Seeing a plugin's section absent from this list is how you spot it is uninstalled."""
    del with_plugin_analyzer

    with pytest.raises(VisionUseError) as excinfo:
        _parse_sections("bogus", everything=False)

    assert "demo.motion" in (excinfo.value.remedy or "")


def test_probe_is_not_offered_as_something_to_analyse() -> None:
    """It is always present, so requesting it is a no-op."""
    assert ContextSection.PROBE not in _parse_sections(None, everything=True)
