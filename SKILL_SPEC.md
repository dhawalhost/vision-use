# Skill Specification

A skill is the unit of capability in VisionUse. Everything the framework can do
to a video is a skill, and adding one requires no change to the runtime.

---

## Contract

```python
class Skill(Protocol):
    meta: SkillMeta

    def analyse(self, ctx: VideoContext, plan: EditPlan, req: SkillRequest) -> Analysis: ...
    def plan(self, analysis: Analysis) -> list[Strategy]: ...
    def execute(self, strategy: Strategy) -> PlanPatch: ...
    def validate(
        self, before: EditPlan, after: EditPlan, ctx: VideoContext
    ) -> ValidationReport: ...
    def score(self, report: ValidationReport) -> Score: ...
```

The lifecycle is the one in `VISION.md`. The single change: `execute()` returns a
**patch**, not a video.

---

## Lifecycle, stage by stage

### `analyse`

Read-only. Consumes the shared `VideoContext` and the current `EditPlan`; decides
what this skill is looking at.

**Must not re-analyse the video.** If a skill needs something the context does not
carry, that is a gap in the context engine, not a licence to run its own decoder.
Declare the need in `meta.requires` and the runtime backfills it once, for
everyone.

### `plan`

Turns the analysis into one or more *candidate* `Strategy` objects, ordered by
expected quality. Multiple strategies are what makes the retry loop meaningful:
when validation rejects the first, the runtime tries the next rather than
re-running the same computation and hoping.

### `execute`

Applies one strategy and returns a `PlanPatch`. Pure with respect to media: it
reads context, it writes operations. It does not encode, and it does not write
files outside the cache.

### `validate`

Compares the plan before and against after. Returns a `ValidationReport` of named
measurements, each with a pass/fail threshold. This is the skill judging its own
work, per Principle 7.

### `score`

Collapses the report into a single comparable `Score`. The runtime uses it to
pick a winner across strategies, so scores must be comparable *within* a skill.
They need not be comparable across skills.

---

## Metadata

```python
class SkillMeta(BaseModel):
    name: str  # registry key, e.g. "silence-removal"
    version: str  # semver; bump when output changes
    summary: str  # one line, shown to the agent
    requires: frozenset[ContextSection]  # SHOTS, TRANSCRIPT, AUDIO_ACTIVITY, ...
    produces: frozenset[OpKind]  # which plan operations it may emit
    dependencies: tuple[str, ...]  # external packages, for `vision doctor`
    supported_formats: tuple[str, ...]
    config_schema: type[BaseModel]  # validated skill parameters
    gpu: GpuPolicy  # NEVER | OPTIONAL | REQUIRED
```

There is no `FACES` section. `SUBJECT` records *where the frame changes*, which
is a usable stand-in for where a speaker is and is honest about being one:
OpenCV 5 no longer ships the Haar cascades, and a real detector is a heavy ML
dependency that has to earn its place (`CONTRIBUTING.md` § Dependencies). A
plugin may add `faces` when someone wants to pay that cost — D5 exists so it
needs no change here.

`requires` is the load-bearing field. The runtime reads it to decide which
analysers to run, and to fail early with an actionable message — *"captions needs
a transcript; this file has no audio stream"* — rather than with a `KeyError` deep
in the skill.

`produces` lets the runtime reject a patch that exceeds what a skill declared,
which matters once third-party plugins are installable.

---

## `PlanPatch`

```python
Op = AddClip | RemoveClip | SplitClip | SetClipRange | PushEffect | SetRenderSetting | ...


class PlanPatch(BaseModel):
    ops: list[Op]
    rationale: str  # surfaced to the user and to logs
    produced_by: SkillRef
```

Typed operations, not a replacement plan. This buys three things a wholesale
plan swap cannot:

- **Validation** — each op validates against the effect registry's schema
- **Audit** — the log says what changed and why, in the skill's own words
- **Undo** — every op has an inverse, so a rejected patch rolls back exactly

A skill that returns a whole new `EditPlan` cannot be reviewed or reverted, so
the runtime rejects it.

---

## Retry and budget

```python
class RetryPolicy(BaseModel):
    max_attempts: int
    max_wall_seconds: float
    max_baked_frames: int
```

`VISION.md` specifies retry → alternative algorithm → alternative parameters →
return the highest-scoring result. That is kept, with a budget added.

An unbounded auto-retry loop over video is a runaway-compute bug, not a feature.
The runtime stops at whichever limit binds first and returns the best result so
far, labelled as budget-limited rather than as validated.

---

## Composability

Every skill must compose with every other skill. In practice that means:

1. **Address time, not frame indices.** Variable-frame-rate sources make frame
   indices unreliable; `MediaProbe.variable_frame_rate` flags them.
2. **Never assume you run first.** Read the current `EditPlan`, not the source
   duration.
3. **Never assume you run last.** Do not flatten, do not bake unless the effect
   genuinely cannot be expressed as a filter.
4. **Reference context by ID.** `sh_0004`, `w_000317` survive an upstream edit;
   raw timestamps do not.

---

## Writing a skill

1. Define `SkillMeta`, most importantly `requires` and `config_schema`.
2. Implement the five lifecycle methods.
3. Register via the entry point group `vision_use.skills`. No core change.
4. Ship golden-fixture tests: a small generated clip, an expected patch, an
   expected score band.

A skill that cannot state what would make its own output wrong is not finished.
`validate` is not optional.

---

## Reference skills

The first three, chosen to exercise the context engine and be useful immediately
(M4 in `PLAN.md`):

| Skill | Requires | Emits | Why it is first |
|---|---|---|---|
| `silence-removal` | `AUDIO_ACTIVITY` | `SplitClip`, `RemoveClip` | Pure context consumer, deterministic, no ML |
| `reframe` | `SHOTS`, `SUBJECT` | `PushEffect` (crop path) | Highest real demand, 16:9 → 9:16 |
| `captions` | `TRANSCRIPT` | `AddClip` (overlay track) | Straight off M1's ASR |

`stabilize` follows in M5 as the first *external* plugin, where it proves the
plugin boundary instead of blocking it.