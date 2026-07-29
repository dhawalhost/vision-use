"""Tests for the command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner
from vision_use import __version__
from vision_use.cache.store import CACHE_ROOT_ENV, CacheStore, parse_size
from vision_use.cli.doctor import Check, Status, worst_status
from vision_use.cli.main import _progress_bar, app
from vision_use.errors import VisionUseError
from vision_use.progress import Progress as ProgressUpdate
from vision_use.progress import Stage

from tests.conftest import MakeClip, requires_ffmpeg

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "doctor" in result.stdout


def test_doctor_json_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--json"])

    payload = json.loads(result.stdout)
    assert payload["status"] in {"ok", "warn", "fail"}
    assert {check["name"] for check in payload["checks"]} >= {"python", "pyav", "ffmpeg"}


def test_doctor_exit_code_matches_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CI gates on this exit code, so it must track the worst check."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--json"])

    expected = 1 if json.loads(result.stdout)["status"] == "fail" else 0
    assert result.exit_code == expected


def test_doctor_failure_reports_remedy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--json"])

    for check in json.loads(result.stdout)["checks"]:
        if check["status"] != "ok":
            assert check["remedy"], f"{check['name']} failed without an actionable remedy"


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([Status.OK, Status.OK], Status.OK),
        ([Status.OK, Status.WARN], Status.WARN),
        ([Status.WARN, Status.FAIL], Status.FAIL),
        ([], Status.OK),
    ],
)
def test_worst_status(statuses: list[Status], expected: Status) -> None:
    checks = [Check(f"c{i}", status, "") for i, status in enumerate(statuses)]

    assert worst_status(checks) is expected


def test_inspect_missing_file_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["inspect", str(tmp_path / "absent.mp4")])

    assert result.exit_code == 1


@requires_ffmpeg
@pytest.mark.integration
def test_inspect_renders_probe(make_clip: MakeClip) -> None:
    result = runner.invoke(app, ["inspect", str(make_clip())])

    assert result.exit_code == 0
    assert "h264" in result.stdout


@requires_ffmpeg
@pytest.mark.integration
def test_inspect_json_is_parseable(make_clip: MakeClip) -> None:
    result = runner.invoke(app, ["inspect", str(make_clip()), "--json"])

    payload = json.loads(result.stdout)
    assert payload["video"]["codec"] == "h264"


def test_cache_path_reports_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["cache", "path"])

    assert result.exit_code == 0
    assert ".vision" in result.stdout


def test_cache_clear_on_empty_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["cache", "clear"])

    assert result.exit_code == 0
    assert "0" in result.stdout


# --- cache accounting (M7) -------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("512", 512),
        ("1KB", 1024),
        ("20GB", 20 * 1024**3),
        ("1.5gb", int(1.5 * 1024**3)),
        ("2 MB", 2 * 1024**2),
        ("0", 0),
    ],
)
def test_sizes_are_parsed(raw: str, expected: int) -> None:
    assert parse_size(raw) == expected


@pytest.mark.parametrize("raw", ["", "lots", "20 gigs", "-5MB", "GB"])
def test_an_unreadable_size_is_refused(raw: str) -> None:
    """A misread budget silently deletes the wrong amount, so guessing is not an option."""
    with pytest.raises(VisionUseError):
        parse_size(raw)


def test_cache_status_reports_an_empty_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "cache"))

    result = runner.invoke(app, ["cache", "status"])

    assert result.exit_code == 0
    assert "empty" in result.stdout.lower()


def test_cache_status_lists_producers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "cache"))
    store = CacheStore(tmp_path / "cache")
    key = store.key("bake", version="1", inputs=["a"], params={}, suffix=".bin")
    with store.writing(key) as tmp:
        tmp.write_bytes(b"x" * 2048)

    result = runner.invoke(app, ["cache", "status", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["bytes"] == 2048


def test_cache_prune_reports_what_it_freed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "cache"))
    store = CacheStore(tmp_path / "cache")
    key = store.key("bake", version="1", inputs=["a"], params={}, suffix=".bin")
    with store.writing(key) as tmp:
        tmp.write_bytes(b"x" * 4096)

    result = runner.invoke(app, ["cache", "prune", "--budget", "0"])

    assert result.exit_code == 0
    assert "Removed 1 file" in result.stdout


def test_cache_prune_refuses_a_bad_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "cache"))

    result = runner.invoke(app, ["cache", "prune", "--budget", "loads"])

    assert result.exit_code == 1


# --- progress reporting (M7) -----------------------------------------------


def test_the_progress_bar_yields_a_working_callback() -> None:
    """The bar is drawn by Rich; what matters here is that updates reach it."""
    with _progress_bar("baking") as report:
        report(ProgressUpdate(stage=Stage.BAKE, done=5, total=10, detail="cl_0000"))
        report(ProgressUpdate(stage=Stage.BAKE, done=10, total=10, detail="cl_0000"))


def test_the_progress_bar_tolerates_an_unknown_total() -> None:
    """A total is not always knowable, and a bar must not require one."""
    with _progress_bar("baking") as report:
        report(ProgressUpdate(stage=Stage.BAKE, done=3, total=None))
