# `vision_use`

The framework package. Structure follows one rule: **each package has a single
responsibility, and dependencies point one way only.**

## Packages

| Package | Responsibility |
|---|---|
| `models/` | Serializable contracts — `VideoContext`, media description, IDs. Depends on nothing else in the framework. |
| `cache/` | Content-addressed artefact store. Immutable entries, atomic writes. |
| `context/` | The context engine: analyzers, registry, runner. Fills a `VideoContext`. |
| `video/` | The only place that touches PyAV or ffmpeg. Probing today, decode and frame extraction next. |
| `cli/` | Thin presentation layer over the library. No media logic. |

## Top-level modules

| Module | Responsibility |
|---|---|
| `errors.py` | Exception hierarchy. Every error carries an actionable `remedy`. |
| `log.py` | Structured logging, human and JSON formatters. Named `log`, not `logging`, to avoid shadowing the stdlib. |
| `hashing.py` | Deterministic hashing — streamed file digests, canonical JSON. |
| `timebase.py` | Time quantisation and exact rational frame mapping. |

## Dependency rules

Enforced by review and by `CLAUDE.md` § Project Structure:

- `models/` depends on nothing above it. Everything may depend on `models/`.
- Only `video/` may import `av` or invoke `ffmpeg`. Everything above works with
  models, which keeps the media backend replaceable.
- The runtime must not import skills; it resolves them through a registry.
- Skills must not import each other; they compose through the `EditPlan`.

## Two invariants worth knowing before reading the code

**Determinism.** Identical input produces byte-identical output. That is what
makes results content-addressable and cacheable, and it is why `AnalyzerRun`
carries no timestamp and every stored time is quantised.

**Frame rate is rational, never float.** 30000/1001 has no exact binary float
form; a rounded frame duration accumulated across a long timeline drifts by whole
frames. See `timebase.py`.
