"""Deterministic hashing primitives.

Content addressing underpins both the cache and ``VideoContext`` provenance, so
these helpers must be stable across processes, platforms, and releases. Two
rules make that true:

* Files are hashed by streaming, never by reading them into memory -- video
  files routinely exceed available RAM (``CLAUDE.md`` § Performance).
* Structured values are canonicalised to sorted, separator-normalised JSON
  before hashing, so ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` agree.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from vision_use.errors import VisionUseError

#: Chunk size for streamed file hashing. 1 MiB balances syscall overhead
#: against resident memory for the many-gigabyte files VisionUse handles.
_CHUNK_BYTES: Final = 1024 * 1024

#: Number of hex characters retained in short digests. 160 bits of a SHA-256
#: digest; collision risk is negligible and the result stays readable in paths
#: and log lines.
SHORT_DIGEST_LEN: Final = 40


def hash_file(path: Path, *, chunk_bytes: int = _CHUNK_BYTES) -> str:
    """Return the hex SHA-256 digest of the file at ``path``.

    The file is read in fixed-size chunks so that memory use is constant
    regardless of file size.

    Args:
        path: File to hash.
        chunk_bytes: Read granularity in bytes.

    Returns:
        Lowercase hex digest, 64 characters.

    Raises:
        VisionUseError: If the file cannot be read.

    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_bytes):
                digest.update(chunk)
    except OSError as exc:
        raise VisionUseError(
            f"Could not read {path} to compute its hash",
            remedy="check that the path exists and is readable",
        ) from exc
    return digest.hexdigest()


def canonical_json(value: JsonValue) -> str:
    """Return a canonical JSON encoding of ``value`` suitable for hashing.

    Keys are sorted, whitespace is stripped, and non-ASCII characters are
    preserved so that logically equal structures always encode identically.

    Raises:
        VisionUseError: If ``value`` contains something JSON cannot represent.
            Hashing an unserialisable value would silently produce unstable
            keys, so this is an error rather than a fallback to ``repr``.

    """
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except TypeError as exc:
        raise VisionUseError(
            f"Value of type {type(value).__name__} cannot be hashed deterministically",
            remedy="convert it to a JSON-compatible type before hashing",
        ) from exc


def hash_params(params: Mapping[str, JsonValue]) -> str:
    """Return the hex SHA-256 digest of a parameter mapping.

    Used for the ``params_hash`` recorded against every analyzer run and for
    cache key derivation.
    """
    return hashlib.sha256(canonical_json(dict(params)).encode("utf-8")).hexdigest()


def short(digest: str, *, length: int = SHORT_DIGEST_LEN) -> str:
    """Truncate ``digest`` for use in filenames and log lines."""
    return digest[:length]
