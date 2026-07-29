# VisionUse — Implementation Plan

Status: proposed
Supersedes: the ordering in `VISION.md` § Development Roadmap and § Initial Skill.
Does not supersede: `VISION.md` vision/mission/principles, `PRINCIPLES.md`, `CLAUDE.md`.

---

## Decisions

**D1. Context Engine and Timeline ship before any enhancement skill.**
`VISION.md` names `stabilize` as the only initial skill. That is deferred. Stabilization
is the least agent-dependent capability in the catalogue (it needs no shared context, no
LLM, and no timeline), so building it first would exercise none of the framework's load-
bearing abstractions, and any skill written before the Timeline exists would hardcode the
wrong contract. `stabilize` returns in M5 as the first *external plugin*, where it proves
the plugin boundary instead of blocking it.

**D2. VisionUse builds its own full Context Engine.**
`video-use` remains an optional future integration as listed in `VISION.md`, not a
dependency. Overlap is accepted deliberately.

**D3. Skills never write pixels. Skills return patches against a declarative EditPlan.**
This is the correction to the largest gap in the current architecture: as written, each
skill's `execute()` produces a video file, so composing three skills means three decode/
encode round-trips — generation loss, no reversibility, and no way to validate a
composition before export. That violates Principle 7 (validation before export) and
Principle 9 (composable). Pixels are produced exactly once, by the renderer, at export.

**D4. Effect param schemas are additive-only. The type is the version.**

An effect's parameter schema may only ever *gain* optional fields with defaults. Any
breaking change — removing a field, renaming one, changing a type, tightening a
constraint — is a **new effect type**: `crop` becomes `crop.v2`. The old type stays
registered and is marked deprecated; it is not deleted.

The consequence is that there is no version field to resolve, and no migration code.
An old plan naming `crop` loads against `crop`, whose schema is by construction still
able to read it. A plan is never rewritten on load, so it stays byte-comparable with
the plan that produced a given render.

`Effect` carries a `schema_hash` that always participates in the cache key, so an
additive change that alters rendered output cannot silently serve a stale artefact.

*Rationale:* by Principle 3 most effects will be third-party. Any scheme whose
correctness depends on plugin authors writing migration functions correctly will fail
in the field — the failure is silent, and it corrupts stored plans. Additive-only is
mechanically checkable; "did the author write a correct migration" is not.

*Enforcement lands with the effect registry in M2:*

- a golden schema snapshot at `tests/schemas.golden.json`; CI fails on any
  non-additive diff, which turns the rule into a build error rather than a convention
- registry metadata `deprecated_since` and `superseded_by`, so a deprecated type warns
  at load and points at its replacement
- an opt-in `migrate()` hook for authors who genuinely want to move users forward —
  opt-in, never required, and never load-bearing for correctness

---

### D5 — A plugin may add a context section, and the core never reads it

`ContextSection` stays an enum of the sections *this repository* produces. A
plugin's section is an ordinary dotted string in the same namespace as its
package (`stabilize.motion`), carried in `VideoContext.extra` as plain JSON.

**Why strings rather than a registry of section models.** A registry would let
the core validate a plugin's payload — and would also mean the core has an
opinion about a shape it cannot possibly know. The owning plugin already has a
Pydantic model for its own data; asking it to register that model with the core
buys validation the plugin can do better, at the cost of a lifecycle problem
(what validates a stored context whose plugin is no longer installed?). Plain
JSON has an answer: it still loads, and the skill that needs it says so.

**Rules, enforced not trusted:**

- A plugin section name must contain a dot and must not shadow a built-in
  section. Both are rejected loudly at merge time, because a plugin silently
  overwriting `shots` is the failure that would be hardest to trace.
- `requires` and `produces` widen from `frozenset[ContextSection]` to accept
  plain strings. `ContextSection` is a `StrEnum`, so a built-in member *is* a
  string and existing declarations keep working unchanged.
- `has()` and `missing()` answer for plugin sections from the same provenance
  they already use, so "analysed, found nothing" stays distinguishable from
  "never analysed" — for plugins too.
- The core never interprets `extra`. It stores it, hashes it into the cache key,
  serializes it, and hands it back.

**What this does not do.** It does not let a plugin add a *typed* section that
the core validates, and it does not give plugin data the ID stability rules that
`shots` and `words` have. A plugin that wants stable IDs mints them itself.

### D6 — Animation is a sidecar on the effect, not a change to any param schema

An effect parameter is a scalar today: `crop` takes `x`, and that `x` holds for
the clip's whole span. That is why `reframe` can follow a subject between shots
but not *within* one, and it is the first thing the document format genuinely
cannot express.

The tempting fix — letting any parameter be "a number *or* a keyframe track" —
breaks D4. Every effect's JSON schema would change shape, which is exactly the
non-additive change the golden snapshot exists to reject, and every plugin's
`params_model` would have to learn a union it never asked for.

**So animation lives beside `params`, not inside it:**

```json
{
  "type": "crop",
  "params": {"x": 219, "y": 0, "width": 202, "height": 360},
  "animate": {
    "x": [{"t_s": 0.0, "value": 219}, {"t_s": 2.5, "value": 438}]
  }
}
```

- `params` keeps the value that holds when nothing is animating, so **every
  existing plan stays valid and every existing effect schema is untouched.**
  A renderer that ignored `animate` entirely would still produce a sensible,
  static result — which is the property that makes this safe to add.
- `animate` maps a parameter name to keyframes. Interpolation is linear between
  them and held flat outside, because a spline needs tangents nobody has asked
  for and "what happens before the first keyframe" must have an obvious answer.
- Times are in **clip-local seconds**, not source or timeline time. A clip that
  is retrimmed or moved keeps its animation, which is what makes an animated
  effect survive the ordinary editing a skill does around it.
- Only numeric parameters animate. A keyframed enum or string is a different
  feature (a step function), and pretending otherwise would produce silently
  meaningless interpolation.

**Rendering is the effect's own business.** `EffectDef` gains an optional
`compile_animated_filter`; an effect that does not implement it is *baked*
instead, which already exists for effects a filtergraph cannot express. So
animation works for every effect on day one, and effects that can express it
cheaply in ffmpeg — `crop` can, via time expressions — opt into the fast path.

---

## Core Data Model

Two documents. Everything else is derived.

### `VideoContext` — produced once, read by every skill

Satisfies Principle 8 ("one video analysis, shared context, no duplicated work").
Immutable, serializable, cacheable, addressable.

```python
class VideoContext(BaseModel):
    schema_version: Literal["1"]
    source: SourceInfo  # path, sha256, container, duration_s
    video: VideoStreamInfo  # width, height, fps (Fraction), codec,
    # pix_fmt, rotation, frame_count, bit_rate
    audio: AudioStreamInfo | None  # sample_rate, channels, codec
    shots: list[Shot]  # id, start_s, end_s, keyframe_id, confidence
    keyframes: list[Keyframe]  # id, t_s, path, shot_id, phash
    transcript: Transcript | None  # language, words[], segments[]
    audio_activity: list[ActivitySpan]  # start_s, end_s, rms_db, is_silence
    analyzers: dict[str, AnalyzerRun]  # name, version, params_hash, produced_at
```

Rules:

- **Stable IDs.** `sh_0001`, `kf_0042`, `w_000317`. Assigned in deterministic
  chronological order. They are the addressing scheme an agent uses to say
  "cut where he says X" — without them every skill re-analyses the video.
- **Provenance.** `analyzers` records which analyzer version and parameter hash
  produced each section, so a stale context is detectable rather than silently wrong.
- **Partial is legal.** `transcript=None` when ASR was skipped. Skills declare which
  context sections they require via `SkillMeta.requires`; the runtime backfills or
  fails with an actionable message, never a `KeyError`.

### `EditPlan` — the editable document

```python
class EditPlan(BaseModel):
    schema_version: Literal["1"]
    sources: dict[str, Source]  # id -> path + sha256
    timeline: Timeline  # fps, resolution, tracks[]
    render: RenderSettings  # codec, crf, preset, container, loudnorm target


class Track(BaseModel):
    id: str
    kind: Literal["video", "audio", "overlay"]
    clips: list[Clip]


class Clip(BaseModel):
    id: str
    source_id: str
    source_in_s: float
    source_out_s: float
    timeline_start_s: float
    effects: list[Effect]  # ordered stack
    transition_in: Transition | None
    transition_out: Transition | None


class Effect(BaseModel):
    type: str  # registry key
    params: dict[str, Any]  # validated against the effect's own schema
```

Properties that matter: JSON, diffable, reversible, human-reviewable, and small enough
to fit in an LLM context window. An agent edits this document; it never touches frames.

### `PlanPatch` — what a skill actually returns

Typed operations, not a replacement plan:

```python
Op = AddClip | RemoveClip | SplitClip | SetClipRange | PushEffect | SetRenderSetting | ...


class PlanPatch(BaseModel):
    ops: list[Op]
    rationale: str  # surfaced to the user and to logs
    produced_by: SkillRef
```

Typed ops give free validation, a real audit trail, and cheap undo. A skill that returns
a whole new `EditPlan` cannot be reviewed or reverted.

### Effects that ffmpeg cannot express

Some future skills (stabilize, object-removal, upscale) need real pixel work. They stay
in the same model as a **baked effect**: the compiler renders that segment to a cached
intermediate, keyed by `sha256(effect_type + version + params + input_hash)`, and splices
the result. No special-case path, no bypass of the pipeline.

---

## Skill Interface

Keeps the `VISION.md` lifecycle. Only `execute()`'s return type changes.

```python
class Skill(Protocol):
    meta: SkillMeta  # name, version, requires, produces, deps, schema

    def analyse(self, ctx: VideoContext, plan: EditPlan, req: SkillRequest) -> Analysis: ...
    def plan(self, analysis: Analysis) -> list[Strategy]: ...
    def execute(self, strategy: Strategy) -> PlanPatch: ...
    def validate(
        self, before: EditPlan, after: EditPlan, ctx: VideoContext
    ) -> ValidationReport: ...
    def score(self, report: ValidationReport) -> Score: ...
```

Retry stays as specified in `VISION.md`, with one addition it currently lacks: a
**budget**. `RetryPolicy(max_attempts, max_wall_seconds, max_baked_frames)`. An
auto-retry loop over video without a cap is a runaway-compute bug, not a feature.

---

## Renderer

One compiler: `EditPlan -> ffmpeg filtergraph -> file`. Single place where encoding
settings live.

**Proxy/master split.** The same plan renders at proxy scale (480p, fast preset) for the
agent loop and at master scale for export. Standard NLE practice, and the difference
between an agent iteration costing seconds and costing minutes. `RenderSettings` carries
both profiles; the plan itself is resolution-independent.

**Cache.** Content-addressed, `.vision/cache/<hash>`. Keyed on
`(input sha256, op, params, tool version)`. Mandatory — the validate/retry loop
re-renders constantly.

---

## How the AI perceives the video

`VISION.md` states the AI never manipulates pixels but never says how it sees. Explicit
bridge, all read-only, all from `VideoContext`:

- shot list + timings (text)
- keyframe contact sheets, one per shot, at capped resolution (image)
- word-level transcript with timestamps and speakers (text)
- silence/loudness map (text)

The agent reasons over these, emits a `PlanPatch`, and reviews a proxy render.

---

## Milestones

Each milestone has a binary exit criterion. No milestone is done without tests.

**M0 — Skeleton.**
uv, ruff, strict mypy, pytest, GitHub Actions, structured logging, `vision doctor`,
content-addressed cache, PyAV/ffprobe wrapper.
*Exit:* `vision doctor` reports a green environment in CI on Linux and macOS.

**M1 — Context Engine.**
`vision analyze in.mp4 -o context.json`. Probe, shot detection, keyframe extraction,
word-level ASR, silence map, provenance, stable IDs.
*Exit:* re-running `analyze` on the same file is byte-identical and cache-hits.

**M2 — Timeline + Renderer.**
`EditPlan` model, filtergraph compiler, `vision render plan.json [--proxy]`,
baked-effect path, `vision plan new in.mp4`.
*Exit:* a hand-written plan with cuts, an overlay, and a transition renders correctly at
both proxy and master, and a straight-through plan is frame-accurate against the source.

**M3 — Skill SDK + agent loop.**
`Skill` protocol, registry, `PlanPatch` application with validation and rollback,
retry budget, `vision apply`.
*Exit:* a no-op reference skill round-trips through the full lifecycle with a rejected
patch cleanly rolled back.

**M4 — First three skills.**
Chosen to exercise the context engine and be useful on day one:
- `silence-removal` / jump-cut — pure context consumer, deterministic, no ML
- `reframe` (16:9 → 9:16, subject-aware) — highest real demand
- `captions` — word-level styled, straight off M1's ASR
*Exit:* `vision apply` on a real talking-head clip produces a publishable short.

**M5 — MCP server + Plugin SDK, then `stabilize` as the first plugin.**
MCP surface is **generic, not per-skill**: `analyze`, `query_context`, `list_skills`,
`apply_skill`, `render`. The `VISION.md` sketch (`stabilize_video`, `score_video`, …)
grows one tool per skill and reaches 40 tools; a registry behind five tools does not.
*Exit:* `stabilize` installs from a separate repo and runs with zero core changes.

**M6 — An open context.**
The gap `stabilize` found: a plugin can publish an analyzer but cannot publish a
*section* for it to fill, because `ContextSection` is a closed enum and
`AnalyzerResult` has fixed fields. So a plugin's skill cannot gather the
evidence it needs to judge its own work, which is Principle 7 asking for
something the framework does not permit. Resolved as **D5** below.
- Plugin-namespaced sections, opaque to the core
- `stabilize` publishes a motion analyzer and validates on measured shake
- Section-aware `query_context`, so an agent can read a plugin's section
*Exit:* `vision-stabilize` reports how much shake it removed, using a section
this repository has never heard of.

**M7 — Real footage.**
Every fixture until now has been seconds long. Measured on 10 minutes of 640x360
with 120 cuts, the streaming design holds up and the storage does not:

| | time | peak RSS | artefact |
|---|---|---|---|
| `analyze` (shots, keyframes, audio) | 4.8s | 148 MB | 43 KB context |
| `render --proxy` | 4.6s | 74 MB | — |
| `bake` (one frame op) | 20.3s | 112 MB | **508 MB intermediate** |

Memory is flat, which is the part that was designed for. The intermediate is
not: baking is lossless by design (`PLAN.md` § Effects that ffmpeg cannot
express), so a 40-minute 1080p source produces tens of gigabytes, and nothing
in the cache ever removes it. A framework that silently fills a disk is not
usable on real footage, however elegant it is on a 3-second clip.
- Cache accounting, a size budget, and least-recently-used eviction
- Progress reporting for operations measured in minutes
- Cancellation that leaves no partial artefact behind
*Exit:* a bake that would exceed the cache budget evicts rather than fills the
disk, and a long render reports progress and survives being interrupted.

**M9 — Animation.**
Effect parameters are static per clip, so a crop cannot follow a subject who
walks across frame: `reframe` picks one compromise window per shot. Resolved as
**D6** above — keyframes as a sidecar on the effect, leaving every existing
param schema and stored plan untouched.
- `animate` on `Effect`, with linear interpolation and held ends
- Validation: numeric params only, times inside the clip, ordered keyframes
- Rendering via time expressions where an effect can, baking where it cannot
- `reframe` follows the subject *within* a shot
*Exit:* a subject who crosses the frame stays in a vertical crop for the whole
shot, measured rather than eyeballed.

**Cross-cutting, starting at M1:** golden-fixture eval harness with small checked-in
clips. "Every skill validates its own output" is the central claim of the project; without
regression fixtures it is unverifiable.

---

## Dependencies

`CLAUDE.md` requires justifying each.

| Dependency | Why | Alternative rejected |
|---|---|---|
| PyAV | In-process demux/decode/probe with exact `Fraction` fps | parsing `ffprobe` JSON via subprocess |
| ffmpeg (CLI) | Render only; filtergraph maturity is unmatched | pure-PyAV encode — far more code, worse coverage |
| PySceneDetect | Mature, well-tested shot boundary detection | hand-rolled histogram diffing |
| faster-whisper | Word-level timestamps, CTranslate2, CPU path exists | openai-whisper (slower, no word timings by default) |
| pydantic | Mandated by `VISION.md`; schema validation for effect params | dataclasses — no runtime validation |
| typer + rich | Mandated by `VISION.md` | argparse |
| numpy | Transitively required; frame math | — |

GPU stays optional throughout. Every dependency above has a working CPU path, per
`CLAUDE.md` § Performance.

---

## Open items

- ~~`docs/` needs `context-schema.md` and `plan-schema.md` once M1/M2 land.~~
  **Resolved.** Both written.
- ~~A plugin cannot contribute a context section, so a plugin's skill cannot
  judge its own output.~~ **Resolved as D5**, landing in M6.
- ~~`ARCHITECTURE.md` and `SKILL_SPEC.md` are arrow diagrams with no interfaces.~~
  **Resolved.** Both rewritten against the contracts above.
- ~~Effect registry versioning policy is unresolved.~~ **Resolved as D4:**
  additive-only schemas, the type is the version, enforcement lands with the registry
  in M2.
