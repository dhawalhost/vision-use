"""Tests for the content-addressed cache store."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest
from vision_use.cache.store import (
    CACHE_BUDGET_ENV,
    CACHE_ROOT_ENV,
    DEFAULT_CACHE_BUDGET_BYTES,
    CacheStore,
    default_cache_budget,
    default_cache_root,
)
from vision_use.errors import CacheError, VisionUseError


@pytest.fixture
def store(tmp_path: Path) -> CacheStore:
    return CacheStore(tmp_path / "cache")


def test_key_is_stable_across_calls(store: CacheStore) -> None:
    first = store.key("analyze.shots", version="1", inputs=["abc"], params={"t": 0.4})
    second = store.key("analyze.shots", version="1", inputs=["abc"], params={"t": 0.4})

    assert first == second


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "2"),
        ("inputs", ["different"]),
        ("params", {"t": 0.5}),
    ],
)
def test_key_changes_when_any_component_changes(
    store: CacheStore, field: str, value: object
) -> None:
    """Every component of the identity must participate in the digest."""
    base = {"version": "1", "inputs": ["abc"], "params": {"t": 0.4}}
    baseline = store.key("analyze.shots", **base)  # type: ignore[arg-type]
    changed = store.key("analyze.shots", **{**base, field: value})  # type: ignore[arg-type]

    assert baseline.digest != changed.digest


def test_key_changes_with_namespace(store: CacheStore) -> None:
    a = store.key("analyze.shots", version="1")
    b = store.key("analyze.faces", version="1")

    assert a.digest != b.digest


def test_miss_returns_none(store: CacheStore) -> None:
    assert store.get(store.key("ns", version="1")) is None


def test_write_then_read(store: CacheStore) -> None:
    key = store.key("ns", version="1", suffix=".txt")

    with store.writing(key) as tmp:
        tmp.write_text("payload", encoding="utf-8")

    hit = store.get(key)
    assert hit is not None
    assert hit.read_text(encoding="utf-8") == "payload"


def test_failed_write_leaves_no_entry(store: CacheStore) -> None:
    """A crash mid-write must not leave a partial file that later reads trust."""
    key = store.key("ns", version="1", suffix=".txt")

    with pytest.raises(RuntimeError), store.writing(key) as tmp:
        tmp.write_text("half", encoding="utf-8")
        raise RuntimeError("interrupted")

    assert store.get(key) is None
    assert not any(store.root.rglob("*.partial"))


def test_json_round_trip(store: CacheStore) -> None:
    key = store.key("ns", version="1", suffix=".json")
    store.put_json(key, {"shots": [1, 2, 3]})

    assert store.get_json(key) == {"shots": [1, 2, 3]}


def test_corrupt_json_raises(store: CacheStore) -> None:
    key = store.key("ns", version="1", suffix=".json")
    with store.writing(key) as tmp:
        tmp.write_text("{not json", encoding="utf-8")

    with pytest.raises(CacheError):
        store.get_json(key)


def test_put_file_copies_by_default(store: CacheStore, tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"data")

    cached = store.put_file(store.key("ns", version="1", suffix=".bin"), source)

    assert cached.read_bytes() == b"data"
    assert source.exists()


def test_put_file_can_move(store: CacheStore, tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"data")

    store.put_file(store.key("ns", version="1", suffix=".bin"), source, move=True)

    assert not source.exists()


def test_disabled_store_never_hits(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache", enabled=False)
    key = store.key("ns", version="1", suffix=".txt")

    with store.writing(key) as tmp:
        tmp.write_text("payload", encoding="utf-8")

    assert store.get(key) is None


def test_clear_scoped_to_namespace(store: CacheStore) -> None:
    with store.writing(store.key("keep", version="1", suffix=".txt")) as tmp:
        tmp.write_text("a", encoding="utf-8")
    drop_key = store.key("drop", version="1", suffix=".txt")
    with store.writing(drop_key) as tmp:
        tmp.write_text("b", encoding="utf-8")

    assert store.clear("drop") == 1
    assert store.get(drop_key) is None
    assert store.get(store.key("keep", version="1", suffix=".txt")) is not None


def test_clear_missing_namespace_is_zero(store: CacheStore) -> None:
    assert store.clear("never-written") == 0


def test_default_root_respects_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "elsewhere"))

    assert default_cache_root() == tmp_path / "elsewhere"


# --- accounting and eviction (M7) ------------------------------------------


def _write(store: CacheStore, namespace: str, name: str, size: int, atime: float) -> Path:
    """Put a file of a given size and access time into the cache."""
    key = store.key(namespace, version="1", inputs=[name], params={}, suffix=".bin")
    with store.writing(key) as tmp:
        tmp.write_bytes(b"x" * size)
    path = store.path_for(key)
    os.utime(path, (atime, atime))
    return path


def test_usage_reports_nothing_for_an_empty_cache(tmp_path: Path) -> None:
    usage = CacheStore(tmp_path / "cache").usage()

    assert usage.files == 0
    assert usage.bytes == 0


def test_usage_breaks_down_by_producer(tmp_path: Path) -> None:
    """Which producer is filling the disk is the first thing worth knowing."""
    store = CacheStore(tmp_path / "cache")
    _write(store, "bake", "a", 4096, time.time())
    _write(store, "analyze.shots", "b", 128, time.time())

    usage = store.usage()

    assert usage.files == 2
    assert {item.namespace for item in usage.namespaces} == {"bake", "analyze.shots"}
    assert next(i.bytes for i in usage.namespaces if i.namespace == "bake") == 4096


def test_pruning_under_budget_removes_nothing(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    _write(store, "bake", "a", 1000, time.time())

    result = store.prune(10_000)

    assert result.removed == 0
    assert store.usage().files == 1


def test_pruning_evicts_until_it_fits(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    now = time.time()
    for index in range(4):
        _write(store, "bake", f"e{index}", 1000, now - index * 7200)

    result = store.prune(2500)

    assert store.usage().bytes <= 2500
    assert result.freed_bytes >= 1500


def test_the_oldest_era_is_evicted_first(tmp_path: Path) -> None:
    """Yesterday's artefact goes before today's."""
    store = CacheStore(tmp_path / "cache")
    now = time.time()
    old = _write(store, "bake", "old", 1000, now - 86_400)
    fresh = _write(store, "bake", "fresh", 1000, now)

    store.prune(1000)

    assert not old.exists()
    assert fresh.exists()


def test_one_big_artefact_goes_before_many_small_ones(tmp_path: Path) -> None:
    """The case that made strict per-file LRU wrong.

    Pruning a real 485 MB cache deleted 123 analysis entries worth 1.2 MB before
    reaching the single 484 MB bake that was actually the problem -- they had
    been touched a few seconds earlier. Losing one big artefact beats losing a
    hundred small ones that each cost seconds to rebuild.
    """
    store = CacheStore(tmp_path / "cache")
    now = time.time()
    small = [_write(store, "keyframe", f"k{i}", 100, now - 60) for i in range(20)]
    big = _write(store, "bake", "huge", 100_000, now)

    result = store.prune(5000)

    assert not big.exists()
    assert all(path.exists() for path in small)
    assert result.removed == 1


def test_pruning_to_zero_empties_the_cache(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    _write(store, "bake", "a", 1000, time.time())

    store.prune(0)

    assert store.usage().files == 0


def test_pruning_removes_empty_directories(tmp_path: Path) -> None:
    """An empty tree makes `du` and a human's `ls` both misleading."""
    store = CacheStore(tmp_path / "cache")
    _write(store, "bake", "a", 1000, time.time())

    store.prune(0)

    assert not any(path.is_dir() for path in store.root.rglob("*"))


def test_a_negative_budget_is_refused(tmp_path: Path) -> None:
    with pytest.raises(VisionUseError, match="cannot be negative"):
        CacheStore(tmp_path / "cache").prune(-1)


# --- automatic budget (M8) -------------------------------------------------


def test_the_default_budget_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A framework whose normal operation fills a disk is not usable unattended."""
    monkeypatch.delenv(CACHE_BUDGET_ENV, raising=False)

    assert default_cache_budget() == DEFAULT_CACHE_BUDGET_BYTES


def test_the_budget_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CACHE_BUDGET_ENV, "500MB")

    assert default_cache_budget() == 500 * 1024**2


@pytest.mark.parametrize("raw", ["none", "off", "unbounded", "0", ""])
def test_the_budget_can_be_switched_off(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Someone with a big disk and expensive bakes may want to keep everything."""
    monkeypatch.setenv(CACHE_BUDGET_ENV, raw)

    assert default_cache_budget() is None


def test_an_unreadable_budget_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misread ceiling deletes the wrong amount, so guessing is not an option."""
    monkeypatch.setenv(CACHE_BUDGET_ENV, "quite a lot")

    with pytest.raises(VisionUseError):
        default_cache_budget()


def test_enforcing_does_nothing_under_budget(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    _write(store, "bake", "a", 1000, time.time())

    assert store.enforce_budget(10_000) is None
    assert store.usage().files == 1


def test_enforcing_evicts_when_over(tmp_path: Path) -> None:
    store = CacheStore(tmp_path / "cache")
    now = time.time()
    _write(store, "bake", "old", 4000, now - 86_400)
    _write(store, "bake", "new", 1000, now)

    result = store.enforce_budget(2000)

    assert result is not None
    assert result.removed == 1
    assert store.usage().bytes <= 2000


def test_an_unbounded_budget_evicts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CACHE_BUDGET_ENV, "none")
    store = CacheStore(tmp_path / "cache")
    _write(store, "bake", "a", 10_000, time.time())

    assert store.enforce_budget() is None
    assert store.usage().files == 1


def test_protected_entries_survive_eviction(tmp_path: Path) -> None:
    """The newest, largest entry is exactly what this policy evicts first."""
    store = CacheStore(tmp_path / "cache")
    now = time.time()
    in_use = _write(store, "bake", "in_use", 10_000, now)
    spare = _write(store, "bake", "spare", 1000, now - 86_400)

    store.prune(500, protect=[in_use])

    assert in_use.exists()
    assert not spare.exists()


def test_an_unreachable_budget_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Deleting what the caller is using would be worse than missing the ceiling."""
    store = CacheStore(tmp_path / "cache")
    in_use = _write(store, "bake", "in_use", 10_000, time.time())

    with caplog.at_level(logging.WARNING):
        result = store.prune(500, protect=[in_use])

    assert in_use.exists()
    assert result.remaining_bytes > 500
    assert "smaller than the artefacts in use" in caplog.text


def test_eviction_does_not_depend_on_the_wall_clock(tmp_path: Path) -> None:
    """Two files written seconds apart must be the same era, whatever time it is.

    With eras measured on absolute clock hours, a prune that happened to run
    just after the hour put them in different buckets and evicted the small
    ones first -- so the same cache pruned differently depending on when.
    """
    # A moment just past an hour boundary, which is where the old rule broke.
    boundary = (int(time.time() // 3600) + 1) * 3600
    store = CacheStore(tmp_path / "cache")
    small = [_write(store, "keyframe", f"k{i}", 100, boundary - 60) for i in range(20)]
    big = _write(store, "bake", "huge", 100_000, boundary + 1)

    store.prune(5000)

    assert not big.exists()
    assert all(path.exists() for path in small)
