# `VideoContext` schema

The single shared analysis of a video. Produced once, read by every skill, never
mutated. This is Principle 8 made concrete — and it is also the *addressing
scheme*: it is what lets an agent say "the shot at 0:42" or "cut where he says
pricing" and have those phrases resolve to exact times.

Schema version: `1`. Defined in
[`vision_use/models/context.py`](../vision_use/models/context.py).

---

## Three guarantees

These are enforced in code, not documented aspirations. Each has tests.

**Deterministic.** Identical input produces byte-identical output. No
timestamps, no absolute paths inside derived entities, every time quantised.
This is what makes a context content-addressable and cacheable, and it is M1's
exit criterion.

**Internally consistent.** Cross-references are validated on construction. A
context naming a keyframe or word that does not exist cannot be built, let alone
serialized and handed to a skill. Shots must be chronological and
non-overlapping.

**Chronological.** Entities are ordered by time and their IDs sort the same way,
so a serialized context diffs cleanly and reads top to bottom.

---

## `subject` — where the crop should sit

Records where each shot's activity is concentrated, as fractions of the frame,
so `reframe` can follow the subject rather than assuming they were centred.

```json
{
  "shot_id": "sh_0000",
  "span": {"start_s": 0.0, "end_s": 4.0},
  "centre_x": 0.802,
  "centre_y": 0.497,
  "confidence": 1.0
}
```

**It is not face detection, and the section is deliberately not called
`faces`.** It sums per-column and per-row *temporal energy* — how much each
strip of the frame changes between sampled frames — and takes the centroid. A
person talking or gesturing is what changes; a static background is not.

The limits are real and `confidence` is how they are reported:

| Situation | What you get |
|---|---|
| One subject moving, static background | The subject's position, high confidence |
| Nothing moves | `0.5, 0.5` with confidence **0.0** — no evidence, not weak evidence |
| The whole frame changes | Near centre, low confidence |
| Subject still, background moving | The background. This method cannot tell them apart |

Treat low confidence as "prefer the centre". A consumer that ignores it will
happily crop to a passing car.


## Top level

```jsonc
{
  "schema_version": "1",
  "probe":          { ... },   // always present
  "shots":          [ ... ],   // optional section
  "keyframes":      [ ... ],   // optional section
  "audio_activity": [ ... ],   // optional section
  "transcript":     { ... },   // optional section, or null
  "analyzers":      { ... }    // provenance
}
```

A context is legitimately **partial**. ASR is expensive and most skills do not
need it, so sections are populated on request. Never test a section for
emptiness to decide whether it ran:

```python
if context.has(ContextSection.TRANSCRIPT):
    ...
```

`has()` reads provenance, not content. A video with genuinely no speech is not
the same as one that was never transcribed, and code that conflates them will
silently do the wrong thing on silent footage.

---

## Sections

### `probe` — always present

Container and stream facts, read without decoding. See
[`models/media.py`](../vision_use/models/media.py).

| Field | Meaning |
|---|---|
| `source.path` | Where the file was when analysed |
| `source.sha256` | Content digest; the identity everything else is keyed on |
| `source.duration_s`, `source.container`, `source.size_bytes` | Container facts |
| `video.width` / `height` | **Stored** dimensions, before rotation |
| `video.rotation` | Display rotation: 0, 90, 180, 270 |
| `video.frame_rate` | `{num, den}` — an exact rational, never a float |
| `video.variable_frame_rate` | True when frame spacing is uneven |
| `video.frame_count`, `frame_count_estimated` | Exact when the container records it |
| `audio` | Sample rate, channels, codec — or `null` |

Two traps worth naming:

- **Always apply rotation before reasoning about aspect ratio.** Most phone
  footage is stored landscape with a 90-degree flag. Use `display_width`,
  `display_height`, `is_portrait` rather than the raw fields.
- **Never use frame indices when `variable_frame_rate` is true.** Address time
  instead; frame-indexed operations are unreliable on such sources.

### `shots` — cut boundaries

```jsonc
{ "id": "sh_0003", "index": 3, "span": {"start_s": 12.4, "end_s": 18.9},
  "keyframe_id": "kf_0003", "confidence": null }
```

Shots **tile the source exactly**: no gaps, no overlaps, first starts at 0, last
ends at the duration. A video with no detected cut yields exactly one shot
covering it — never an empty list, so consumers need no special case.

### `keyframes` — one still per shot

```jsonc
{ "id": "kf_0003", "shot_id": "sh_0003", "t_s": 15.6,
  "cache_key": "<64 hex>", "width": 640, "height": 360, "phash": null }
```

Image bytes live in the content-addressed cache; the context holds only
`cache_key`. Inlining stills would push a ten-minute video's context past what
fits in a model's context window and stop it being plain diffable JSON.

Sampled at the shot **midpoint** by default — a shot's first frame is frequently
mid-transition or motion-blurred.

`phash` is reserved and currently always `null`; it lands when near-duplicate
detection has a consumer.

### `audio_activity` — where speech is and is not

```jsonc
{ "span": {"start_s": 4.0, "end_s": 6.2}, "rms_db": -52.4, "is_silence": true }
```

Spans tile the source. Levels are dBFS, rounded to 2 decimal places — float
summation is not bit-identical across platforms or SIMD widths, and determinism
must hold between the Linux and macOS CI legs.

Adjacent spans never share a kind: a span shorter than the minimum is absorbed
into a neighbour, then same-kind neighbours are coalesced. Without that, a
consumer would see a boundary where nothing changes.

This is level detection, not voice-activity detection. Real VAD can arrive later
as a second analyzer producing the same section.

### `transcript` — speech with word-level timing

```jsonc
{
  "language": "en",
  "words": [
    { "id": "w_000412", "span": {"start_s": 41.2, "end_s": 41.6},
      "text": "pricing", "speaker_id": null, "confidence": 0.98 }
  ],
  "segments": [
    { "id": "sg_0007", "span": {...}, "text": "Our pricing is simple.",
      "word_ids": ["w_000411", "w_000412", ...], "speaker_id": null }
  ]
}
```

**Words are authoritative; segments are for reading.** Segment-level timing
forces a skill to guess where inside a sentence a word falls, and the guess is
audible — a caption arrives late, a cut lands mid-syllable.

Requires the optional `asr` extra (`uv sync --extra asr`).

### `analyzers` — provenance

```jsonc
{ "shots.content": { "name": "shots.content", "version": "1",
                     "params_hash": "<64 hex>", "produced": ["shots"] } }
```

Records which analyzer version and parameter hash produced each section, so a
stale context is detectable rather than silently wrong.

**There is deliberately no timestamp.** A wall-clock field would make two
analyses of the same file differ, defeating content addressing and the
byte-identical criterion. When an analysis ran is a property of the cache entry
and the log, not of the document.

---

## Identifiers

| Prefix | Entity | Example |
|---|---|---|
| `sh_` | Shot | `sh_0003` |
| `kf_` | Keyframe | `kf_0003` |
| `w_` | Word | `w_000412` |
| `sg_` | Transcript segment | `sg_0007` |
| `spk_` | Speaker | `spk_0001` |

IDs depend only on an entity's position in chronological order — never on
iteration order or a random suffix. They are zero-padded so lexicographic order
matches chronological order, which keeps a serialized context readable and
diffable. Words get six digits because an hour of speech is roughly ten thousand
of them.

---

## Time

Every time is seconds as a float, quantised to 6 decimal places
([`timebase.py`](../vision_use/timebase.py)). Spans are **half-open**,
`[start_s, end_s)`, so adjacent entities tile without overlapping.

Converting a time to a frame index uses the exact rational frame rate and
carries a half-quantum tolerance. Both matter:

- A float rate drifts by a whole frame over about half an hour at 30000/1001.
- Quantising a frame boundary to microseconds can land just below it, and a
  plain floor would then name the *previous* frame.

Use `timebase.snap_to_frame()` when an audio-derived time — a word boundary, a
silence edge — has to become a video cut point.

---

## Worked example: cut where he says "pricing"

```python
from vision_use.context import analyze
from vision_use.models.context import ContextSection
from vision_use.timebase import snap_to_frame

context = analyze(Path("talk.mp4"), [ContextSection.TRANSCRIPT])

for match in context.find_text("our pricing is simple"):
    start = snap_to_frame(match[0].span.start_s, context.probe.video.frame_rate)
    end = snap_to_frame(match[-1].span.end_s, context.probe.video.frame_rate)
    print(f"{match[0].id}..{match[-1].id}  {start:.3f} -> {end:.3f}")
```

`find_text` is case-insensitive and ignores punctuation at word edges, because
recognisers place punctuation inconsistently. It returns the actual `Word`
entities rather than character offsets, so the caller gets exact timings without
mapping anything back itself.

Other lookups: `shot_at(t)`, `shot_by_id()`, `keyframe_by_id()`,
`words_between(a, b)`, `text_between(a, b)`.

`words_between` uses **overlap, not containment** — a word straddling a
requested boundary is part of what is said in that range, and dropping it would
silently truncate a quotation.

---

## Producing a context

```bash
vision analyze input.mp4 --with shots,keyframes,transcript --json
```

```bash
vision sections   # which producer fills each section, and what it requires
```

```python
context = analyze(path, [ContextSection.KEYFRAMES])
```

Requesting a section pulls in its prerequisites transitively — asking for
`keyframes` runs shot detection first, without the caller knowing the dependency
graph exists.

---

## Extending

A new section means a new analyzer (see
[`context/analyzer.py`](../vision_use/context/analyzer.py)) plus a field on
`VideoContext`. Requirements:

1. Declare `produces` and `requires`; the registry resolves the order.
2. Everything affecting output must appear in `params()` — it feeds both the
   cache key and the recorded provenance.
3. Be deterministic. Seed anything stochastic, pin any temperature, round
   anything whose last bits vary by platform.
4. Quantise every time through `timebase.quantize_s()`.
5. Never mutate the context you are given; return an `AnalyzerResult`.

Adding a field to `VideoContext` is a schema change. Follow the same
additive-only rule as effects (`PLAN.md` § D4): new optional fields with
defaults are fine, anything else bumps `schema_version`.
