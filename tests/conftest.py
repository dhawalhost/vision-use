"""Shared pytest fixtures.

Media fixtures are *generated* by ffmpeg rather than checked in. Binary test
assets bloat a repository permanently and can never be regenerated with
different properties; synthesising them keeps the repo text-only and lets a
test ask for exactly the frame rate, rotation, or duration it needs.

Generated clips are cached for the session, so the ffmpeg cost is paid once.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

FFMPEG = shutil.which("ffmpeg")

requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")


def _run_ffmpeg(args: Sequence[str]) -> None:
    """Invoke ffmpeg, raising with its stderr attached when it fails."""
    assert FFMPEG is not None
    completed = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped directory holding generated media fixtures."""
    return tmp_path_factory.mktemp("media")


MakeClip = Callable[..., Path]


@pytest.fixture(scope="session")
def make_clip(media_dir: Path) -> MakeClip:
    """Return a factory that synthesises small test clips.

    The factory is memoised on its arguments, so repeated requests for the same
    clip within a session reuse one file.
    """
    cache: dict[tuple[object, ...], Path] = {}

    def factory(
        *,
        name: str = "clip",
        seconds: float = 2.0,
        width: int = 320,
        height: int = 240,
        fps: str = "25",
        with_audio: bool = True,
        container: str = "mp4",
    ) -> Path:
        signature = (name, seconds, width, height, fps, with_audio, container)
        if signature in cache:
            return cache[signature]

        path = media_dir / f"{name}-{width}x{height}-{fps.replace('/', '_')}.{container}"
        args: list[str] = [
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate={fps}:duration={seconds}",
        ]
        if with_audio:
            args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if with_audio:
            args += ["-c:a", "aac", "-shortest"]
        args.append(str(path))

        _run_ffmpeg(args)
        cache[signature] = path
        return path

    return factory


@pytest.fixture(scope="session")
def speech_clip(media_dir: Path) -> Path:
    """A short clip containing real spoken words.

    Synthesised with the platform's text-to-speech so the suite still ships no
    binaries. Skips where no TTS is available rather than falling back to a
    tone, since a transcription test against a tone asserts nothing.
    """
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")

    path = media_dir / "speech.mp4"
    if path.exists():
        return path

    if shutil.which("say") is None:
        pytest.skip("no text-to-speech available to synthesise speech")

    voice = media_dir / "speech.aiff"
    spoken = subprocess.run(
        ["say", "-o", str(voice), "Our pricing is simple. Fifty dollars per month."],
        capture_output=True,
        text=True,
        check=False,
    )
    if spoken.returncode != 0 or not voice.exists():
        pytest.skip("text-to-speech failed")

    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=25:duration=5",
            "-i",
            str(voice),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    )
    return path
