"""Content-addressed cache store.

A cache entry is identified by everything that could change its bytes:

* the *namespace* -- which producer made it (``"analyze.shots"``, ``"render.proxy"``)
* the *version* of that producer, so a bug fix invalidates old entries
* the *inputs* -- content hashes of the files consumed
* the *params* -- the producer's configuration

Any change to any of those yields a different key, and identical keys are
guaranteed to describe identical work. Entries are therefore immutable: a key
is written once and never updated.

Writes are atomic. Producing an entry writes to a temporary file in the same
directory and renames it into place, so a crashed or cancelled run can never
leave a half-written file that a later run mistakes for a valid hit.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from pydantic import JsonValue

from vision_use.errors import CacheError, VisionUseError
from vision_use.hashing import hash_params, short
from vision_use.log import bind, get_logger

log = get_logger(__name__)

#: Directory created inside the working directory when no root is configured.
DEFAULT_CACHE_DIRNAME: Final = ".vision"

#: Width of an eviction "era", in seconds. Entries accessed within the same era
#: are treated as equally recent, so size decides between them. An hour is long
#: enough to cover one working session and short enough that yesterday's work is
#: still clearly older than today's.
_ERA_SECONDS: Final = 3600.0

#: Environment variable overriding the cache location, for CI and for users who
#: keep large intermediates off their project volume.
CACHE_ROOT_ENV: Final = "VISION_CACHE_ROOT"

#: Fanout directory width. Two hex characters gives 256 subdirectories, which
#: keeps directory listings fast on filesystems that degrade with large dirs.
_FANOUT: Final = 2


def default_cache_root() -> Path:
    """Return the cache root: ``$VISION_CACHE_ROOT`` or ``./.vision/cache``."""
    override = os.environ.get(CACHE_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    return Path.cwd() / DEFAULT_CACHE_DIRNAME / "cache"


#: One binary step. Sizes here are powers of 1024, matching what `du` reports.
_BINARY_STEP = 1024

#: Suffixes accepted by `--budget`, as powers of 1024.
SIZE_UNITS: dict[str, int] = {
    "b": 1,
    "k": _BINARY_STEP,
    "kb": _BINARY_STEP,
    "m": _BINARY_STEP**2,
    "mb": _BINARY_STEP**2,
    "g": _BINARY_STEP**3,
    "gb": _BINARY_STEP**3,
    "t": _BINARY_STEP**4,
    "tb": _BINARY_STEP**4,
}


def parse_size(raw: str) -> int:
    """Parse a human size such as ``20GB`` into bytes.

    Raises:
        VisionUseError: If the value cannot be read as a size. Rejecting beats
            guessing: a misread budget silently deletes the wrong amount.

    """
    text = raw.strip().lower().replace(" ", "")
    for suffix in sorted(SIZE_UNITS, key=len, reverse=True):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            multiplier = SIZE_UNITS[suffix]
            break
    else:
        multiplier = 1

    try:
        value = float(text)
    except ValueError as exc:
        raise VisionUseError(
            f"Could not read {raw!r} as a size",
            remedy="use a number with an optional unit, e.g. 500MB or 20GB",
        ) from exc
    if value < 0:
        raise VisionUseError(
            f"A cache budget cannot be negative (got {raw!r})",
            remedy="pass a size, or 0 to empty the cache",
        )
    return int(value * multiplier)


#: Environment variable holding the cache ceiling, e.g. `VISION_CACHE_BUDGET=5GB`.
#: `none` or `0` means unbounded.
CACHE_BUDGET_ENV: Final = "VISION_CACHE_BUDGET"

#: Ceiling applied when nothing says otherwise.
#:
#: A default rather than unbounded because the cache is *rebuildable by
#: construction* -- content-addressed, derived entirely from inputs it records
#: -- and a framework whose normal operation fills a disk is not usable
#: unattended. Twenty gigabytes holds several long bakes, which is what the
#: iteration loop needs, and is small enough to sit inside a container volume.
DEFAULT_CACHE_BUDGET_BYTES: Final = 20 * 1024**3


def default_cache_budget() -> int | None:
    """Return the configured cache ceiling, or ``None`` for unbounded.

    Raises:
        VisionUseError: If the environment variable cannot be read as a size.
            Refusing beats guessing: a misread ceiling deletes the wrong amount.

    """
    raw = os.environ.get(CACHE_BUDGET_ENV)
    if raw is None:
        return DEFAULT_CACHE_BUDGET_BYTES
    if raw.strip().lower() in {"none", "off", "unbounded", ""}:
        return None
    budget = parse_size(raw)
    return None if budget == 0 else budget


@dataclass(frozen=True, slots=True)
class NamespaceUsage:
    """How much space one producer's entries occupy."""

    namespace: str
    files: int = 0
    bytes: int = 0


@dataclass(frozen=True, slots=True)
class CacheUsage:
    """What the cache currently holds, broken down by producer."""

    namespaces: tuple[NamespaceUsage, ...] = ()

    @property
    def files(self) -> int:
        """Total number of cached files."""
        return sum(item.files for item in self.namespaces)

    @property
    def bytes(self) -> int:
        """Total bytes on disk."""
        return sum(item.bytes for item in self.namespaces)


@dataclass(frozen=True, slots=True)
class PruneResult:
    """What an eviction pass removed."""

    removed: int = 0
    freed_bytes: int = 0
    remaining_bytes: int = 0


@dataclass(frozen=True, slots=True)
class CacheKey:
    """An immutable cache identity.

    Attributes:
        namespace: Dotted name of the producer, e.g. ``"analyze.shots"``.
        digest: Hex SHA-256 over namespace, version, inputs, and params.
        suffix: File extension of the cached artefact, including the dot.

    """

    namespace: str
    digest: str
    suffix: str = ""

    @property
    def relpath(self) -> Path:
        """Path of this entry relative to the cache root."""
        return Path(self.namespace) / self.digest[:_FANOUT] / f"{self.digest}{self.suffix}"

    def __str__(self) -> str:
        """Return a compact form for logs: ``namespace:abcdef01``."""
        return f"{self.namespace}:{short(self.digest, length=8)}"


class CacheStore:
    """Immutable, atomically written, content-addressed artefact store.

    Args:
        root: Directory holding the cache. Created on first write.
        enabled: When ``False`` every lookup misses and writes still happen,
            which is the behaviour wanted for ``--no-cache`` debugging runs.

    """

    def __init__(self, root: Path | None = None, *, enabled: bool = True) -> None:
        self._root = (root or default_cache_root()).expanduser()
        self._enabled = enabled

    @property
    def root(self) -> Path:
        """Directory holding the cache."""
        return self._root

    @property
    def enabled(self) -> bool:
        """Whether lookups are permitted to hit."""
        return self._enabled

    def key(
        self,
        namespace: str,
        *,
        version: str,
        inputs: Sequence[str] = (),
        params: Mapping[str, JsonValue] | None = None,
        suffix: str = "",
    ) -> CacheKey:
        """Derive a :class:`CacheKey`.

        Args:
            namespace: Dotted producer name.
            version: Producer version. Bump it whenever output bytes change
                for the same inputs, otherwise stale entries are served.
            inputs: Content hashes of consumed files, from
                :func:`vision_use.hashing.hash_file`. Order is preserved and
                significant, since it usually encodes argument position.
            params: Producer configuration.
            suffix: Artefact file extension, including the dot.

        """
        digest = hash_params(
            {
                "namespace": namespace,
                "version": version,
                "inputs": list(inputs),
                "params": dict(params or {}),
            }
        )
        return CacheKey(namespace=namespace, digest=digest, suffix=suffix)

    def path_for(self, key: CacheKey) -> Path:
        """Return the absolute path an entry occupies, whether or not it exists."""
        return self._root / key.relpath

    def get(self, key: CacheKey) -> Path | None:
        """Return the path of a cached artefact, or ``None`` on a miss."""
        if not self._enabled:
            return None
        path = self.path_for(key)
        if path.exists():
            log.debug("cache hit", extra=bind(key=str(key)))
            return path
        log.debug("cache miss", extra=bind(key=str(key)))
        return None

    @contextmanager
    def writing(self, key: CacheKey) -> Iterator[Path]:
        """Yield a temporary path to write, then commit it to ``key`` atomically.

        The caller writes the artefact to the yielded path. On clean exit the
        file is renamed into the cache; if the body raises, the temporary file
        is removed and the cache is left untouched.

        If another process committed the same key concurrently, the existing
        entry wins -- entries are immutable, so both are byte-identical by
        construction.

        Raises:
            CacheError: If the cache directory cannot be created or written.

        """
        final = self.path_for(key)
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
            handle, tmp_name = tempfile.mkstemp(dir=final.parent, suffix=".partial")
            os.close(handle)
        except OSError as exc:
            raise CacheError(
                f"Could not prepare cache directory {final.parent}",
                remedy=f"check permissions or set {CACHE_ROOT_ENV} to a writable path",
            ) from exc

        tmp = Path(tmp_name)
        try:
            yield tmp
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

        try:
            if final.exists():
                tmp.unlink(missing_ok=True)
            else:
                tmp.replace(final)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise CacheError(
                f"Could not commit cache entry {key}",
                remedy=f"check free space and permissions under {self._root}",
            ) from exc
        else:
            log.debug("cache write", extra=bind(key=str(key), path=str(final)))

    def put_file(self, key: CacheKey, source: Path, *, move: bool = False) -> Path:
        """Store an existing file under ``key`` and return its cached path.

        Args:
            key: Identity to store under.
            source: File to ingest.
            move: Move rather than copy. Use for large intermediates the caller
                has just produced and does not otherwise need.

        """
        with self.writing(key) as tmp:
            if move:
                shutil.move(str(source), tmp)
            else:
                shutil.copyfile(source, tmp)
        return self.path_for(key)

    def get_json(self, key: CacheKey) -> JsonValue | None:
        """Return the decoded JSON artefact for ``key``, or ``None`` on a miss.

        Raises:
            CacheError: If the entry exists but is not valid JSON, which means
                the cache is corrupt rather than merely cold.

        """
        path = self.get(key)
        if path is None:
            return None
        try:
            return cast("JsonValue", json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheError(
                f"Cache entry {key} is unreadable or corrupt",
                remedy=f"delete {path} and re-run",
            ) from exc

    def put_json(self, key: CacheKey, value: JsonValue) -> Path:
        """Serialise ``value`` as JSON under ``key`` and return its cached path."""
        with self.writing(key) as tmp:
            tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        return self.path_for(key)

    def clear(self, namespace: str | None = None) -> int:
        """Delete cached entries and return how many files were removed.

        Args:
            namespace: Limit the purge to one producer. ``None`` clears all.

        """
        target = self._root / namespace if namespace else self._root
        if not target.exists():
            return 0
        removed = sum(1 for path in target.rglob("*") if path.is_file())
        shutil.rmtree(target)
        log.info("cache cleared", extra=bind(namespace=namespace or "*", files=removed))
        return removed

    def usage(self) -> CacheUsage:
        """Return how much space the cache is using, by namespace.

        Walks the tree rather than keeping a running total: an index would be
        another thing to keep consistent with the filesystem, and the filesystem
        is the authority here. A cache with a few thousand entries walks in
        milliseconds.
        """
        namespaces: dict[str, NamespaceUsage] = {}
        for path in sorted(self._root.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            name = path.relative_to(self._root).parts[0]
            current = namespaces.get(name) or NamespaceUsage(namespace=name)
            namespaces[name] = NamespaceUsage(
                namespace=name,
                files=current.files + 1,
                bytes=current.bytes + stat.st_size,
            )
        return CacheUsage(namespaces=tuple(namespaces[name] for name in sorted(namespaces)))

    def prune(self, budget_bytes: int, protect: Collection[Path] = ()) -> PruneResult:
        """Evict least-recently-used entries until the cache fits ``budget_bytes``.

        **Least recently used, coarsely, with ties broken by size.** Access
        time decides which *era* an entry belongs to; within an era the largest
        goes first. Strict per-file LRU is wrong here because artefacts differ
        in size by three orders of magnitude: pruning a 485 MB cache to 100 MB
        deleted 123 analysis entries worth 1.2 MB before it reached the one
        484 MB bake that was actually the problem, because they had been touched
        a few seconds earlier. Losing one big artefact beats losing a hundred
        small ones that cost seconds each to rebuild.

        Args:
            budget_bytes: Ceiling to bring the cache under. Zero empties it.
            protect: Entries that must survive regardless. An operation in
                flight passes the artefacts it is about to read: the newest and
                largest entry is exactly what this policy evicts first, so
                without this a bake would delete the intermediate its own
                render was about to open.

        Returns:
            What was removed, so a caller can report it rather than guess.

        Raises:
            VisionUseError: If the budget is negative, which is a caller bug
                rather than a condition to handle.

        """
        if budget_bytes < 0:
            raise VisionUseError(
                f"A cache budget cannot be negative (got {budget_bytes})",
                remedy="pass a byte count, or 0 to empty the cache",
            )

        keep = {path.resolve() for path in protect}
        files = [path for path in self._root.rglob("*") if path.is_file()]
        newest = max((path.stat().st_atime for path in files), default=0.0)
        entries = sorted(
            (
                (path.stat().st_atime, path.stat().st_size, path)
                for path in files
                if path.resolve() not in keep
            ),
            # Oldest era first, then largest within it. Bucketing keeps entries
            # touched in the same working session comparable, so size decides
            # between them rather than a few seconds of accident.
            #
            # Eras are measured back from the newest entry, not from absolute
            # clock hours: with absolute boundaries, two files written seconds
            # apart land in different eras whenever the clock happens to cross
            # one, which made eviction depend on what time the prune ran.
            # Negated: measured back from the newest entry, a *larger* bucket
            # means older, so descending age puts the oldest era first.
            key=lambda item: (-int((newest - item[0]) // _ERA_SECONDS), -item[1]),
        )
        # Protected bytes still count towards the ceiling -- they are on the
        # disk either way -- but cannot be reclaimed, so the budget may be
        # unreachable. Saying so beats deleting what the caller is using.
        protected_bytes = sum(path.stat().st_size for path in keep if path.is_file())
        total = sum(size for _, size, _ in entries) + protected_bytes
        if protected_bytes > budget_bytes:
            log.warning(
                "cache budget is smaller than the artefacts in use",
                extra=bind(protected_bytes=protected_bytes, budget_bytes=budget_bytes),
            )

        removed = 0
        freed = 0
        for _, size, path in entries:
            if total <= budget_bytes:
                break
            try:
                path.unlink()
            except OSError as exc:  # pragma: no cover - racing another process
                log.warning("could not evict", extra=bind(path=str(path), detail=str(exc)))
                continue
            total -= size
            freed += size
            removed += 1

        self._remove_empty_directories()
        log.info(
            "cache pruned",
            extra=bind(removed=removed, freed_bytes=freed, remaining_bytes=total),
        )
        return PruneResult(removed=removed, freed_bytes=freed, remaining_bytes=total)

    def enforce_budget(
        self, budget_bytes: int | None = None, protect: Collection[Path] = ()
    ) -> PruneResult | None:
        """Prune if the cache has grown past its ceiling.

        Called after operations that write large artefacts, so that a long
        unattended session stays bounded without anyone remembering to run
        `vision cache prune`. Cheap when the cache is comfortably under: one
        walk, no deletions.

        Args:
            budget_bytes: Ceiling to apply. Defaults to the configured one;
                ``None`` there means unbounded, and nothing is done.
            protect: Artefacts the caller is about to read.

        Returns:
            What was evicted, or ``None`` if the cache was already within
            budget or no budget applies.

        """
        limit = default_cache_budget() if budget_bytes is None else budget_bytes
        if limit is None:
            return None

        used = self.usage().bytes
        if used <= limit:
            return None

        log.info(
            "cache over budget, evicting",
            extra=bind(used_bytes=used, budget_bytes=limit),
        )
        return self.prune(limit, protect=protect)

    def _remove_empty_directories(self) -> None:
        """Delete directories left behind by eviction.

        Cosmetic but worth doing: an empty tree makes `du` and a human's `ls`
        both misleading about what the cache still holds.
        """
        for path in sorted(self._root.rglob("*"), key=lambda item: -len(item.parts)):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
