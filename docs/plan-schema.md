# `EditPlan` schema

What should *happen* to a video. A declarative timeline that skills patch and
the renderer compiles — never a video file, until export.

This is the decision the rest of the framework is shaped around (`PLAN.md` § D3).
A skill returns a patch against this document; pixels are produced exactly once.
The consequences: edits compose without generation loss, they are reversible,
they are reviewable before a frame is rendered, and a model gets to work on JSON
— which it is good at — instead of driving an encoder, which it is not.

Schema version: `1`. Defined in
[`vision_use/models/plan.py`](../vision_use/models/plan.py).

---

## Shape

```jsonc
{
  "schema_version": "1",
  "sources":  { "src_0000": { ... } },   // files the plan draws from
  "timeline": { "frame_rate": {...}, "width": 1920, "height": 1080,
                "tracks": [ ... ] },
  "render":   { "master": {...}, "proxy": {...} }
}
```

---

## Animated parameters

An effect parameter can vary over the clip. Keyframes live **beside** `params`,
never inside it:

```json
{
  "type": "crop",
  "params": {"x": 219, "y": 0, "width": 202, "height": 360},
  "animate": {
    "x": [{"t_s": 0.0, "value": 219}, {"t_s": 2.5, "value": 438}]
  }
}
```

Why a sidecar rather than "a number *or* a track" per parameter: that would
change the shape of every effect's schema, which is precisely the non-additive
change D4 exists to reject, and would force every plugin's `params_model` to
learn a union it never asked for. See `PLAN.md` § D6.

| Rule | Why |
|---|---|
| `params` still holds the static value | A reader ignoring `animate` produces a sensible result, and every stored plan stays valid |
| `t_s` is **clip-local** | A retrimmed or moved clip keeps its animation |
| Linear between keyframes, held outside | "What happens before the first keyframe" needs an obvious answer |
| Numeric parameters only | Interpolating a string or an enum is silently meaningless |
| Keyframes ordered, no repeated times | A repeated time makes the value there ambiguous |

**How it renders.** A filter effect declaring `animatable` compiles its
keyframes into an ffmpeg time expression, so a moving `crop` costs nothing
beyond the pass the render already makes. A baked effect declaring it receives
its interpolated parameters per frame. An effect that declares neither is a
validation **error** — rendering it statically while the plan says otherwise
would be silently wrong.

Interpolated values are validated against the effect's own schema at each frame,
so an animation that sweeps a parameter out of range fails with the same message
an authored value would, rather than producing a corrupt intermediate.

`crop` animates `x` and `y` only. Animating `width` or `height` would change the
frame size mid-clip, which no encoder accepts, so it is refused rather than
ignored.


## `sources`

```jsonc
"src_0000": {
  "id": "src_0000",
  "path": "/absolute/path/talk.mp4",
  "sha256": "<64 hex>",
  "duration_s": 612.4,
  "has_audio": true
}
```

The map key must equal the entry's `id`; a mismatch makes lookups silently miss,
so it is rejected on load.

**`sha256` ties the plan to specific bytes.** A plan that renders correctly today
must render identically tomorrow, and a file whose digest changed is a different
file wearing the same name.

**`path` is absolute.** Plans get moved and passed around; a relative path stops
resolving.

**`has_audio` is recorded, not probed.** Compiling stays free of media I/O and the
plan stays self-describing. It also has to be *right*: the compiler builds an
audio chain from it, and a plan claiming audio a file does not have fails in
ffmpeg with an error naming no clip.

---

## `timeline`

```jsonc
{
  "frame_rate": { "num": 30000, "den": 1001 },
  "width": 1920,
  "height": 1080,
  "tracks": [ { "id": "trk_0000", "kind": "video", "muted": false, "clips": [...] } ]
}
```

`kind` is `video`, `audio`, or `overlay`. Today the compiler renders the first
video track; a second one is warned about rather than silently ignored.

The frame rate is an exact rational, never a float — see
[context-schema.md](context-schema.md) § Time for why.

### `clips`

```jsonc
{
  "id": "clp_0003",
  "source_id": "src_0000",
  "source_in_s": 41.2,       // addresses the SOURCE
  "source_out_s": 47.9,
  "timeline_start_s": 12.0,  // addresses the OUTPUT
  "enabled": true,
  "effects": [ ... ],
  "transition_in": null,
  "transition_out": null
}
```

**Source range and timeline position are separate axes.** That is what lets a
clip be moved without re-deciding what it shows, and trimmed without re-deciding
where it sits. A clip's duration is `source_out_s - source_in_s`; its timeline
end follows from its start.

**Two clips may not overlap on one track.** Both would claim the same output
frames, which has no defined rendering. Layering is expressed with a second
track, and overlap is a construction error rather than a style choice.

**Gaps are legal.** They render as black and silence. `vision validate` reports
them because they are more often an accident than an intent.

**`enabled: false` keeps a clip in the plan but renders nothing**, so an edit can
be auditioned without being discarded.

---

## `effects`

An ordered stack, applied in order.

```jsonc
{ "type": "crop", "params": { "x": 0, "y": 0, "width": 1280, "height": 720 },
  "schema_hash": "<64 hex>" }
```

`params` is **not** validated by the plan model — only by the registry, on use. A
core that lacks a plugin's effect must still be able to load and re-save a plan
that uses it, or plans would stop round-tripping the moment a plugin is missing.

### Two kinds

| Kind | How it renders |
|---|---|
| **filter** | Compiled to a filtergraph fragment inside the clip's chain |
| **baked** (`bakes = True`) | Pre-rendered to a cached intermediate before the graph runs |

Baked effects exist for work a filtergraph cannot express — stabilisation needs
motion estimated across the whole span, object removal needs a model. A baked
effect is still an ordinary entry in the stack; only its compilation differs, and
it is removed from the clip once applied so the compiler does not try to filter
what is already rendered in.

Built in today: `crop`, `scale`, `fade` (filter), `volume` (audio filter),
`invert` (baked reference implementation).

### `schema_hash` and the additive-only rule

Effect parameter schemas may only ever *gain* optional fields with defaults. Any
breaking change is a **new effect type** — `crop` becomes `crop.v2` — with the
old one kept and marked deprecated.

The type *is* the version, so there is no version field to resolve and no
migration code: a stored plan naming `crop` loads against `crop`, whose schema can
by construction still read it. `schema_hash` records the shape the params were
written against and participates in the render cache key, so an additive change
that alters output cannot serve a stale artefact.

This is enforced in CI by a golden schema snapshot, not left to discipline. See
`PLAN.md` § D4 and `CONTRIBUTING.md`.

---

## `transitions`

```jsonc
"transition_in": { "type": "fade", "duration_s": 0.5, "params": {} }
```

A boundary is described by the **later** clip's `transition_in`, falling back to
the earlier clip's `transition_out`. Letting both ends describe one boundary
would be ambiguous, so the later clip wins.

**A transition shortens the output.** A plan cannot express overlap, so a
transition consumes its duration from the boundary it spans: two 1-second clips
with a 0.4s fade render 1.6 seconds. `CompiledRender.duration_s` reports the real
result rather than the plan's pre-transition length.

The first clip's `transition_in` is ignored — there is nothing before it. A fade
up from black at the start is a `fade` *effect*.

Any transition on a track switches that whole track from `concat` to a pairwise
`xfade` chain; boundaries without one become two-input concats inside it.

---

## `render`

```jsonc
{
  "container": "mp4",
  "master": { "scale": 1.0, "codec": "libx264", "crf": 20, "preset": "medium" },
  "proxy":  { "scale": 0.5, "codec": "libx264", "crf": 28, "preset": "veryfast" },
  "audio_codec": "aac",
  "loudnorm_lufs": null
}
```

**Two profiles, not one.** The plan is resolution-independent, so the same
document renders at proxy scale for the agent loop — where an iteration costs
seconds — and at master scale for export. That split is what keeps the retry loop
affordable.

Scaled dimensions are forced even: yuv420p subsamples chroma by two, and proxy
scaling routinely produces an odd number.

---

## Validation

Two layers, deliberately separated.

**Structural — rejected on load.** Overlapping or unordered clips, duplicate IDs,
dangling source references, source ranges past a source's end, mismatched source
keys. These are contradictions; a plan carrying one cannot be constructed.

**Advisory — reported by `vision validate`.** Gaps, unused sources, missing files,
unknown or misconfigured effects, all-disabled clips. Each is legal and
occasionally deliberate, so they are findings rather than exceptions. `render`
runs them first and refuses on errors unless `--skip-validation` is passed —
failing in a second beats failing after ffmpeg has spent minutes.

---

## Working with plans

```bash
vision plan new input.mp4 -o plan.json   # the identity edit
vision plan show plan.json
vision validate plan.json
vision render plan.json -o out.mp4 --proxy
```

```python
from vision_use.planning import straight_through, validate_plan
from vision_use.render import render

plan = straight_through(Path("input.mp4"))
assert validate_plan(plan).ok
render(plan, Path("out.mp4"), proxy=True)
```

`straight_through` produces the identity edit: one clip covering the whole file,
canvas and rate from the source. It renders back unchanged, and it is the
document every skill starts patching.

---

## Extending

Adding a field to `EditPlan` follows the same additive-only rule as effects:
optional with a default, or bump `schema_version`. A plan written last month must
keep loading.

New effects need a `params_model`, a `compile_filter()` (or `frame_op()` if they
bake), and a regenerated golden snapshot so the new type is guarded from then on.
See `CONTRIBUTING.md`.
