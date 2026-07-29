# Architecture

VisionUse is a pipeline around two documents. Understand those two documents and
the rest of the system follows.

---

## The two documents

**`VideoContext`** — what the video *is*. Produced once by the analyse pipeline,
read by every skill, never mutated. Shots, keyframes, a word-level transcript, an
audio activity map, all carrying stable IDs. This is Principle 8 made concrete:
one analysis, shared context, no duplicated work.

**`EditPlan`** — what should *happen* to the video. A declarative timeline:
sources, tracks, clips with source in/out points, an ordered effect stack, render
settings. Skills mutate this, never pixels.

Everything else — the runtime, the skill manager, the validators — exists to move
information between these two.

---

## Layers

```
User prompt
    │
    ▼
AI Agent  ──── reads VideoContext, writes PlanPatch. Never touches frames.
    │
    ▼
Runtime   ──── owns config, cache, logging, retry budget. Depends on no skill.
    │
    ▼
Pipeline  ──── ordered steps: probe → analyse → context → skill → validate → export
    │
    ▼
Skill Manager ── registry, dependency resolution, context requirement checking
    │
    ▼
Skill     ──── analyse / plan / execute / validate / score. Returns a PlanPatch.
    │
    ▼
Renderer  ──── the ONLY component that encodes. EditPlan → filtergraph → file.
    │
    ▼
Output
```

### Dependency rules

These are enforced, not aspirational (`CLAUDE.md` § Project Structure):

- Runtime must not import from `skills`. It resolves them through the registry.
- Skills must not import each other. They compose through the `EditPlan`.
- Plugins communicate only through `SkillMeta` and `PlanPatch`.
- Only `vision_use.video` may import PyAV or invoke ffmpeg. Everything above it
  works with models, so the media backend stays replaceable.

---

## Why skills return patches and not video

The obvious design has each skill take a video and return a video. It fails on
four counts:

| Problem | Consequence |
|---|---|
| Re-encoding per skill | Generation loss compounds; three skills, three decodes |
| No shared state | Skill 2 cannot see what skill 1 decided, only its output |
| Irreversible | No undo, no review, no diff, no explanation |
| Validation too late | Principle 7 wants validation *before* export, not after each step |

So `execute()` returns a `PlanPatch` — a list of typed operations against the
`EditPlan`. Pixels are produced exactly once, by the renderer, at export.

An agent is good at editing a small JSON document and bad at driving encoders.
This design plays to that.

---

## Effects that ffmpeg cannot express

Stabilisation, object removal, and upscaling need real pixel work. They stay
inside the same model as a **baked effect**: the compiler renders that segment to
a cached intermediate keyed by
`sha256(effect_type + version + params + input_hash)` and splices the result.

No special-case path. No bypassing the pipeline. A baked effect is still an entry
in a clip's effect stack; only its compilation differs.

---

## Effect versioning

Effect parameter schemas are **additive-only**: they may gain optional fields with
defaults, nothing else. Any breaking change is a new effect type — `crop` becomes
`crop.v2` — and the old type stays registered, marked deprecated rather than
deleted.

The type *is* the version. There is no version field to resolve and no migration
code: a stored plan naming `crop` loads against `crop`, whose schema can by
construction still read it, and the plan is never rewritten on load. `Effect`
carries a `schema_hash` that always participates in the cache key, so an additive
change that alters output cannot serve a stale artefact.

This shape is chosen because Principle 3 means most effects will be third-party.
A scheme whose correctness depends on plugin authors writing migrations correctly
fails silently and corrupts stored plans; additive-only is mechanically checkable.
Enforcement — a golden schema snapshot that fails CI on non-additive diffs,
`deprecated_since`/`superseded_by` metadata, and an opt-in `migrate()` hook —
lands with the effect registry in M2. See `PLAN.md` § D4.

---

## Proxy and master

The same `EditPlan` renders at two scales. The agent loop iterates against a
480p, fast-preset proxy; export uses master settings. The plan itself is
resolution-independent, so nothing is re-decided between the two.

This is the difference between an agent iteration costing seconds and costing
minutes.

---

## Caching

Content-addressed, immutable, atomically written. A key is derived from:

```
namespace + producer version + input content hashes + params
```

Identical keys describe identical work by construction, so entries are written
once and never updated. Writes go to a temporary file in the destination
directory and are renamed into place, so an interrupted run cannot leave a
partial file that a later run trusts.

Without this, a validation retry is indistinguishable from a cold run and
framework cost is unbounded.

---

## How the agent perceives video

`VISION.md` says the AI never manipulates pixels. It does not say how the AI
*sees*. The bridge is explicit and read-only, entirely derived from
`VideoContext`:

- shot list with timings (text)
- keyframe contact sheets, one per shot, resolution-capped (image)
- word-level transcript with timestamps and speakers (text)
- silence and loudness map (text)

The agent reasons over these, emits a `PlanPatch`, and reviews a proxy render.

---

## Current implementation status

Milestone M0 is landing. What exists today:

| Module | Responsibility |
|---|---|
| `vision_use.errors` | Exception hierarchy; every error carries a `remedy` |
| `vision_use.log` | Structured logging; named `log` not `logging` to avoid shadowing stdlib |
| `vision_use.hashing` | Streamed file hashing, canonical JSON, param digests |
| `vision_use.cache` | Content-addressed store with atomic writes |
| `vision_use.models.media` | `FrameRate`, `SourceInfo`, `VideoStreamInfo`, `AudioStreamInfo`, `MediaProbe` |
| `vision_use.video.probe` | PyAV-backed probing; the only media-backend code so far |
| `vision_use.cli` | `vision doctor`, `vision inspect`, `vision cache`, `vision version` |

`VideoContext` arrives in M1, `EditPlan` and the renderer in M2, the skill SDK in
M3. See `PLAN.md`.