# Plugin SDK

How to add a capability to VisionUse from your own package, with **no change to
this repository**.

That constraint is the whole point (Principle 3). If adding a skill required a
pull request here, the ecosystem would grow only as fast as this repo's review
queue, and every capability would need to justify itself to a maintainer who has
never seen your footage.

---

## Start from something that works

```bash
uv run vision plugin new my-plugin
```

```bash
cd my-plugin && uv sync
```

```bash
uv run vision plugins
```

That is a complete, installable package: a working skill, an entry-point
declaration, a README, and three tests — one of which asserts the skill is
*discoverable*, which is the part a document cannot prove.

The generated skill trims dead air off the end of a timeline. It is a real edit,
not a stub, so your first act is to change what it does rather than to make it
run at all.

---

## The three extension points

| You want to add | Group | It must satisfy |
|---|---|---|
| A capability an agent can invoke | `vision_use.skills` | `Skill` protocol |
| A new kind of understanding | `vision_use.analyzers` | `Analyzer` protocol |
| A new visual or audio transform | `vision_use.effects` | `EffectDef` |

Declare them in your `pyproject.toml`:

```toml
[project.entry-points."vision_use.skills"]
stabilize = "vision_stabilize:StabilizeSkill"

[project.entry-points."vision_use.effects"]
deshake = "vision_stabilize:DeshakeEffect"

[project.entry-points."vision_use.analyzers"]
motion = "vision_stabilize:MotionAnalyzer"
```

The published object may be the class or a zero-argument factory. Both are
common in the wild, and requiring one would only produce a support question.

The entry-point *name* on the left is how the plugin identifies itself to
packaging; the capability's real identity is the `name` in its own metadata
(`stabilize.deshake`), which is what `vision skills` and `apply_skill` use.

---

## Writing a skill

A skill has five stages. The pipeline runs them in order and can reject a result
at the fourth, which is what makes the third worth doing more than once.

```python
class Skill(Protocol):
    meta: ClassVar[SkillMeta]

    def analyse(
        self, context: VideoContext, plan: EditPlan, request: SkillRequest
    ) -> Analysis: ...
    def plan(self, analysis: Analysis) -> list[Strategy]: ...
    def execute(self, strategy: Strategy) -> PlanPatch: ...
    def validate(
        self, before: EditPlan, after: EditPlan, context: VideoContext
    ) -> ValidationReport: ...
    def score(self, report: ValidationReport) -> Score: ...
```

Nothing is registered by inheritance. A class with these five methods and a
`meta` attribute *is* a skill — your plugin's skill is indistinguishable from a
built-in one, which is the test of whether the boundary is real.

### `analyse` — read, decide nothing

Read-only, and it must not re-analyse the video. If the context lacks something
you need, that is a gap in the context engine, not a licence to run your own
decoder: declare it in `meta.requires` and the runtime backfills it once, for
every skill (Principle 8).

### `plan` — offer several strategies

Return candidates in order of expected quality. This is what gives the retry
loop somewhere to go: re-running the same computation and hoping is not a retry.

Returning `[]` means the skill has nothing to do. That is a legitimate outcome,
not an error, and the pipeline leaves the plan untouched.

### `execute` — return a patch, never a video

```python
return PlanPatch(
    ops=(SetClipRange(clip_id=clip.id, source_out_s=4.0),),
    rationale="trim 2.0s off the end",
    produced_by=self.meta.ref,
)
```

Typed operations, not a replacement plan. That buys validation against the
effect registry, an audit trail in your own words, and an exact undo. A skill
that returns a whole new `EditPlan` cannot be reviewed or reverted, so the
runtime rejects it.

`rationale` is surfaced to the user and to the agent. Write it as a sentence a
person would accept as an explanation.

### `validate` — judge your own work

Compare before against after and return named `Measurement`s with pass/fail
verdicts. A bare number cannot be judged by a runtime that does not know your
skill; a verdict can, and that is what lets the pipeline reject a strategy and
try the next one without understanding what it measured.

**A skill that cannot state what would make its own output wrong is not
finished.** `validate` is not optional (Principle 7).

### `score` — pick a winner

Collapse the report into one number. Comparable *within* your skill, so the
pipeline can rank your own strategies. It is never compared across skills.

---

## The four composability rules

Your skill must compose with skills it has never heard of. In practice:

1. **Address time, not frame indices.** Variable-frame-rate sources make frame
   indices unreliable; `MediaProbe.variable_frame_rate` flags them.
2. **Never assume you run first.** Read the current `EditPlan`, not the source
   duration.
3. **Never assume you run last.** Do not flatten, and do not bake unless the
   effect genuinely cannot be expressed as a filter.
4. **Reference context by ID.** `sh_0004` and `w_000317` survive an upstream
   edit; raw timestamps do not.

And one more that the retry loop depends on: **be idempotent** for the same
context, plan, and parameters. A skill whose second run differs makes retry
unsound.

---

## Writing an effect

An effect is a parameter schema plus a way to apply it — as an ffmpeg filter
where possible, as a frame operation where not.

```python
class DeshakeEffect(EffectDef):
    effect_type: ClassVar[str] = "stabilize.deshake"
    params_model: ClassVar[type[BaseModel]] = DeshakeParams

    def compile_filter(self, params: BaseModel) -> str: ...
```

**Bump `version` when your effect's output changes.** A baked effect's artefact
is cached on the source, the range, the parameters — and `EffectDef.version`,
which is the only part that can see your code:

```python
class DeshakeEffect(EffectDef):
    effect_type: ClassVar[str] = "stabilize.deshake"
    version: ClassVar[str] = "2"  # bumped: the correction changed
```

Without it, fixing a bug in `frame_op` leaves every existing cache serving the
artefact the broken version produced. This was found the hard way while building
`vision-stabilize`, whose sign fix appeared to do nothing.

**Animated parameters.** Set `animatable = True` and implement one of two
methods, depending on how your effect renders:

- A **filter** effect implements `compile_animated_filter(params, animate)`,
  where `animate` arrives as ffmpeg time expressions already built from the
  plan's keyframes. `t` is clip-local.
- A **baked** effect implements `frame_op_animated(params)`, returning a
  callable that receives each frame *and that frame's interpolated parameters*.
  You never touch keyframes or timestamps.

Declaring `animatable` is opt-in on purpose: an effect that quietly ignored
animation would render a plan that says the crop moves as one that does not, so
validation refuses animation on any effect that has not claimed it. Baking is
not a free pass.

**Effect schemas are additive-only.** Parameters may gain optional fields with
defaults; anything else is a new effect type (`deshake` → `deshake.v2`), with
the old one left registered and marked `deprecated_since` / `superseded_by`.

This exists so that stored plans keep rendering, and so no plan's fate depends
on a plugin author having written a migration correctly. See `PLAN.md` § D4.

---

## Writing an analyzer, and adding your own context section

An analyzer fills part of the shared context. **It may fill a section VisionUse
has never heard of** — that is what lets your skill judge its own work rather
than checking that its own ops landed.

```python
SECTION = "stabilize.motion"  # dotted, namespaced by your package


class MotionAnalyzer:
    name = "stabilize.motion"
    version = "1"
    produces = frozenset({SECTION})
    requires = frozenset({ContextSection.PROBE})

    def params(self) -> dict[str, object]:
        return {"downscale": 4}

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        ...
        return AnalyzerResult(extra={SECTION: summary.model_dump()})
```

Your skill then declares it like any built-in section, and the runtime backfills
it by running your analyzer:

```python
meta = SkillMeta(
    name="stabilize.deshake",
    version="2",
    requires=frozenset({ContextSection.PROBE, SECTION}),
)
```

An agent can reach it too: `analyze` names the sections a context carries, and
`query_context(section="stabilize.motion")` returns the payload — the MCP server
does this without knowing your section exists. From the CLI, `vision analyze
--with stabilize.motion` and `vision sections` treat it like any built-in.

Read it back with `context.section(SECTION)`, and validate it with your own
model — **the core never interprets your payload.** It stores it, caches it,
serializes it, and hands it back (`PLAN.md` § D5).

Two rules are enforced, not trusted:

- **The name must be dotted and namespaced by your package.** A bare `motion`
  would collide with the next plugin that wants it.
- **You cannot shadow a built-in section.** A plugin quietly overwriting `shots`
  would be the hardest failure in this system to trace.

And one thing to keep honest: `context.has(SECTION)` answers from provenance, so
an *empty* payload still counts as analysed. "Measured, found nothing" and
"never measured" stay different answers, and your skill should treat them
differently.

---

## Reporting progress from a long effect

A baked effect on real footage runs for minutes. `bake_plan` and `render` take
an `on_progress` callback, and your `frame_op` needs to do nothing to
participate — the framework counts the frames it pulls through your operation:

```python
render(plan, output, on_progress=lambda update: print(update.fraction))
```

Two rules the framework holds up:

- **Progress reaches an agent too.** The MCP server forwards these updates as
  protocol-level progress notifications, rate-limited, so a plugin's slow effect
  is visible to whoever is driving — with no work on your part beyond the frames
  the framework already counts.
- **Reporting is never fatal.** A callback that raises is logged once and then
  dropped for the rest of the run. The work is expensive; the reporting is not.
- **Your baked artefacts count against the cache budget.** The cache prunes
  itself after a bake (20 GB by default, `VISION_CACHE_BUDGET` to change it),
  so a plugin that writes large intermediates does not need its own cleanup —
  and what the current render is reading is never evicted.
- **`total` may be `None`.** Draw a spinner rather than a bar in that case;
  `Progress.fraction` returns `None` rather than inventing a denominator.

---

## What the framework guarantees you

**A broken plugin never takes the framework down.** If your module fails to
import, the failure is caught, recorded against your entry-point name, and shown
by `vision plugins`. One typo in your module path must not make `vision --help`
fail for everyone.

```bash
uv run vision plugins
```

**A name collision is loud.** You cannot silently shadow a built-in skill or
effect; registration fails and says so. An installation's behaviour should not
depend on what happens to be installed.

**Discovery is explicit.** Plugins load at the entry points — the CLI callback
and the MCP server's construction — not at package import. Importing
`vision_use` as a library runs no third-party code.

---

## Testing your plugin

The generated package includes the test that matters most:

```python
def test_the_plugin_is_discovered() -> None:
    registry = SkillRegistry()
    load_plugins(skills=registry)

    assert "my_plugin.trim_tail" in registry
```

A skill that is correct but never discovered helps nobody. Note that this passes
only once your package is *installed* — being present on disk is not enough.

Beyond that, test what your skill measures rather than what it emits: an op list
is an implementation detail, and a test asserting it will break on every
refactor while catching nothing.

### Give your skill a scored baseline

`vision_use.evals` is a library, not a private test helper, so your plugin gets
the same regression guarantee the built-in skills have — with the baseline
living in *your* package:

```python
from vision_use.evals import compare, evaluate, load_baseline, save_baseline

results = [evaluate("stabilize/handheld", StabilizeSkill(), source, plan, cache=store)]
problems = [p for r in results for p in compare(r, load_baseline(BASELINE).get(r.case, {}))]
```

This catches the change no unit test can: a threshold nudged, a constant
"improved". Those pass every unit test, because every unit test was written to
allow them. `vision-stabilize` does exactly this — changing its motion
downscale from 4 to 8 moves the recorded shake by 8.1% and fails the baseline.

Do not pass a pre-built context if your skill declares a section: let the
runtime backfill it by running your analyzer, so the baseline guards the
measurement as well as the edit.

---

## Reference

- [SKILL_SPEC.md](../SKILL_SPEC.md) — the full skill contract
- [docs/context-schema.md](context-schema.md) — what `analyse` can read
- [docs/plan-schema.md](plan-schema.md) — what a patch may change
- [PRINCIPLES.md](../PRINCIPLES.md) — the ten the design answers to
- [CONTRIBUTING.md](../CONTRIBUTING.md) — for changes to VisionUse itself
