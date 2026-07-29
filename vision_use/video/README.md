# `vision_use.video`

Low-level media access. **This is the only package permitted to import PyAV or
invoke ffmpeg.**

Everything above works with `vision_use.models`, which keeps the media backend
replaceable and keeps skills free of codec concerns.

## Contents

| Module | Responsibility |
|---|---|
| `probe.py` | Container and stream metadata, without decoding frames |

Decode and frame extraction land with keyframes (M1.4); the renderer lands in M2.

## Why PyAV and not `ffprobe`

PyAV returns frame rates as exact rationals instead of decimal strings that must
be re-parsed, and avoids a subprocess per file. The `ffmpeg` binary is still used
for *rendering*, where filtergraph maturity is unmatched.

## What probing gets right that a naive implementation does not

**Rotation.** Most phone footage is stored landscape with a 90-degree display
flag. Ignoring it misreports the aspect ratio of most real input, so `probe()`
reads both the H.264/QuickTime display matrix and the legacy `rotate` tag, and
`VideoStreamInfo` exposes `display_width` / `display_height` / `is_portrait`.

**Variable frame rate.** Screen recordings and some phone cameras have unevenly
spaced frames, which makes frame-indexed addressing unreliable. When average and
nominal rates disagree beyond a tolerance, `variable_frame_rate` is set and
consumers must address time rather than frame indices.

**Estimated frame counts.** MPEG-TS and fragmented MP4 often record no frame
count. It is then derived from duration, and `frame_count_estimated` says so —
useful for progress reporting, never for seeking.

## Rules

- Stream; never load a whole video into memory.
- Never re-encode to inspect. Probing is header-only and must stay that way.
- GPU paths may be added; a CPU path must always exist.
