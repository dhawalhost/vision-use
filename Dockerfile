# VisionUse runtime image.
#
# Two stages so that build tooling and the uv cache never reach the final image:
# the builder resolves and installs dependencies, the runtime carries only the
# interpreter, ffmpeg, and the resulting virtualenv.
#
# ffmpeg is a hard runtime requirement, not a convenience. The renderer shells
# out to it, and `vision doctor` fails without it.

# --- builder ---------------------------------------------------------------

FROM python:3.12-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies are installed from the lockfile alone, before the source is
# copied, so that editing code does not invalidate the dependency layer.
# --frozen fails rather than silently re-resolving: the image must match the
# lockfile that CI tested.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY vision_use/ ./vision_use/
COPY README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- runtime ---------------------------------------------------------------

FROM python:3.12-slim-trixie AS runtime

# Trixie rather than bookworm: bookworm ships ffmpeg 5.1, and `vision doctor`
# requires 6+ for filters the M2 renderer depends on. On bookworm the image
# builds fine and then reports itself unusable, which is worse than not building.
#
# ffmpeg brings the codecs; libgl1 and libglib2.0-0 are OpenCV's runtime
# dependencies, which the slim base does not carry. Without them `import cv2`
# fails at shot detection with an error that does not mention OpenCV.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged by default; media processing has no reason to run as root.
RUN useradd --create-home --uid 1000 vision

WORKDIR /app

COPY --from=builder --chown=vision:vision /app/.venv /app/.venv
COPY --from=builder --chown=vision:vision /app/vision_use /app/vision_use

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Keep cache artefacts on a writable path that is easy to mount as a volume; the
# default (./.vision/cache) would land in the working copy, which is not writable
# by the unprivileged user.
ENV VISION_CACHE_ROOT=/var/cache/vision

RUN mkdir -p /var/cache/vision && chown vision:vision /var/cache/vision

USER vision

# Media is mounted here; nothing is baked into the image.
VOLUME ["/media", "/var/cache/vision"]

# Fails the container when the environment cannot actually run VisionUse,
# rather than waiting for the first request to discover it.
HEALTHCHECK --interval=30s --timeout=15s --start-period=5s --retries=3 \
    CMD ["vision", "doctor"]

ENTRYPOINT ["vision"]
CMD ["--help"]
