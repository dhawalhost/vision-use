# `vision_use.models`

The contracts between runtime, skills, and plugins. Pydantic models, so plugin
authors get validation for free and every document round-trips through JSON
unchanged.

This package depends on nothing else in the framework. Everything else may depend
on it.

## Modules

| Module | Contents |
|---|---|
| `media.py` | `FrameRate`, `SourceInfo`, `VideoStreamInfo`, `AudioStreamInfo`, `MediaProbe` |
| `context.py` | `VideoContext` and its entities — `Shot`, `Keyframe`, `Word`, `Transcript`, `ActivitySpan`, `TimeSpan`, `AnalyzerRun` |
| `ids.py` | Stable entity ID generation |

## Three design decisions

**Frame rate is a rational, not a float.** NTSC rates such as 30000/1001 have no
exact binary float form. Accumulating a rounded frame duration across a long
timeline drifts by whole frames, so cuts land on the wrong frame. `FrameRate`
stores `num`/`den` and exposes `.fraction` for exact arithmetic; `.decimal` exists
for display and heuristics only, never for timing.

**IDs encode chronological order.** `sh_0003`, `w_000412`. Zero-padded so
lexicographic order matches time order, which makes a serialized context readable
and diffable. They are how an agent addresses part of a video — "cut from
`w_000412`" — so they must be stable across runs, skills, and reloads.

**Consistency is validated, not documented.** A `VideoContext` naming a keyframe
or word that does not exist cannot be constructed, let alone serialized and handed
to a skill. Shots must be chronological and non-overlapping. Failures surface at
construction, where the stack trace still points at the cause.

## Determinism

Every document here must serialise identically on every run — that is what makes
it content-addressable. Two consequences:

- No timestamps in derived entities. `AnalyzerRun` deliberately has no
  `produced_at`; when an analysis ran is a property of the cache entry and the
  log, not of the document.
- Every stored time passes through `timebase.quantize_s()` first.
