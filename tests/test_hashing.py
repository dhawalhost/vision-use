"""Tests for deterministic hashing primitives."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from vision_use.errors import VisionUseError
from vision_use.hashing import canonical_json, hash_file, hash_params, short


def test_hash_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"vision-use" * 1000
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)

    assert hash_file(target) == hashlib.sha256(payload).hexdigest()


def test_hash_file_is_chunk_size_independent(tmp_path: Path) -> None:
    """Chunking is an implementation detail and must not change the digest."""
    target = tmp_path / "blob.bin"
    target.write_bytes(b"x" * 5000)

    assert hash_file(target, chunk_bytes=7) == hash_file(target, chunk_bytes=4096)


def test_hash_file_reports_missing_path(tmp_path: Path) -> None:
    with pytest.raises(VisionUseError) as excinfo:
        hash_file(tmp_path / "absent.bin")

    assert excinfo.value.remedy is not None


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_canonical_json_rejects_unserialisable() -> None:
    with pytest.raises(VisionUseError):
        # Deliberately unserialisable; the type error here is the point --
        # callers are meant to be stopped statically, and at runtime too.
        canonical_json({"bad": object()})  # type: ignore[dict-item]


def test_hash_params_distinguishes_values() -> None:
    assert hash_params({"strength": 1}) != hash_params({"strength": 2})


def test_hash_params_ignores_key_order() -> None:
    assert hash_params({"a": 1, "b": 2}) == hash_params({"b": 2, "a": 1})


def test_hash_params_distinguishes_types() -> None:
    """1 and "1" must not collide; a stringly-typed param would poison the cache."""
    assert hash_params({"n": 1}) != hash_params({"n": "1"})


def test_short_truncates() -> None:
    digest = "a" * 64
    assert short(digest, length=8) == "a" * 8
