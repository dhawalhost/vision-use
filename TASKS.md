# VisionUse — Task Breakdown

Milestones and exit criteria are defined in [PLAN.md](PLAN.md). This file is the
running ledger: it records what is done, what is next, and what was decided
along the way. It is the first file to read when picking the work back up.

Legend: `[x]` done · `[~]` in progress · `[ ]` not started

---

## Current state

**Milestone:** M0–M10 closed out.

**Last verified:** `ruff check` clean, `ruff format --check` clean, `mypy`
strict clean (89 files), `pytest` 845 passed + 4 ASR deselected — with
`vision-stabilize` installed, which is itself the check that a third-party
plugin does not disturb the host suite. The plugin's own 40 tests pass.

**Verified by running it, not by asserting it.** Each of these was driven by
hand against real media before its milestone was called done:

| What | Evidence |
|---|---|
| Render round trip (M2) | A source round-trips to 72 identical frames, audio intact; `--proxy` is smaller; a plan naming a missing file is refused before ffmpeg starts |
| Crossfade (M2) | A 2.0s plan renders 1.6s with a 0.4s fade, and the mid-transition frame is `[127, 0, 127]` — the exact blend of red and blue |
| Baking (M2) | Red → cyan through a frame op; a warm bake is ~700x faster |
| Skills (M4) | Silence removal 6s → 4.28s keeping 99.68% of speech; captions visible in the rendered frame; reframe 640x360 → 202x360, a true centre crop |
| MCP (M5.1) | A whole edit driven through the tools an agent would call: analyze → list_skills → plan → apply_skill → render |
| Plugin discovery (M5.2) | A package built and installed from outside the repository was found by `vision plugins`, `vision skills`, and MCP `list_skills` |
| Scaffold (M5.3) | `vision plugin new` → installed → applied → rendered 6s to 4.00s, with its own generated discovery test passing |
| `stabilize` (M5.4) | Synthetic handheld shake: 4.73px → 1.93px mean frame-to-frame motion, a 59% reduction, at a 3% crop |
| Plugin sections (M6/D5) | `vision-stabilize` publishes `stabilize.motion`; VisionUse backfills a section it cannot interpret, and the skill scores 10.37 = 1.0 + 9.37px measured shake. Steady footage: "found nothing to do" |
| Plugin surface (M6) | `vision analyze --with stabilize.motion` and `vision sections` treat a plugin's section like a built-in; MCP `query_context(section=...)` returns a payload the server cannot interpret |
| Plugin evals (M6) | `vision-stabilize` has its own scored baseline. Changing its motion downscale from 4 to 8 moved the recorded shake 8.1% and failed the baseline; reverted |
| Long media (M7) | 10 minutes, 120 cuts: analyze 4.8s / 148 MB, proxy render 4.6s / 74 MB, bake 20.3s / 112 MB. Memory flat throughout — but the bake left a **508 MB** intermediate |
| Progress + cancellation (M7) | A bake reports frames with a total a bar can fill; an interrupted bake leaves the cache with **zero** files, proven by removing the cleanup and watching the test fail |
| Following reframe (M9) | A subject crossing the frame stays visible in **100/100** output frames, against **50/100** for the best static crop — M9's exit criterion |
| Animated bake (M9) | A baked effect dimming over two seconds produces frames that get monotonically darker, and its first frame matches the source within 4/255 |
| Animated crop (M9) | A box crossing a 640px frame stays visible in **100/100** output frames with an animated crop, against **32/100** with a static one |
| MCP progress (M8) | A render reports advancing progress to a recording client and keeps an async heartbeat ticking throughout — both proven by breaking them and watching the tests fail |
| Subject-aware reframe (M8) | On a clip whose subject sits at x=470–550 of 640, the old centre crop showed **0** subject pixels; the subject-aware crop shows **9600**, the whole box, inside a 202x360 vertical output |
| Subject location (M8) | A white box moving at x=460 of 640 is located at **0.802** (true 0.781), confidence 1.0. A still frame gives 0.5 with confidence **0.0**; full-frame motion gives near-centre, low confidence |
| Cache budget (M8) | A render under a 5 MB budget with a 5.5 MB intermediate warns, keeps the artefact, and succeeds — verified on real media, which is how the eviction-eats-its-own-input bug was found |
| Cache eviction (M7) | Pruning a real 485 MB cache to 100 MB removes the one 484 MB bake and keeps all 123 analysis entries |
| Docker (M5.5) | `doctor` green inside the image (ffmpeg 7), then a full analyze → plan → apply → render against mounted media |
| End-to-end run (M10) | A real 2-minute, 40-cut, mixed-audio source driven through analyze → apply (silence removal, then reframe) → render, producing a valid 202x360, 81.84s, 2046-frame vertical file with audio intact |

**Start here:** M10 was end-to-end validation on real multi-shot media, and it
found three bugs that no unit test had -- because each needed dozens of clips
laid end to end from independently quantised boundaries to appear, and nothing
before this exercised that. All three fixed:

1. **`edit.remove_silence` rejected its own output on a 40-cut source.**
   `Clip.duration_s` is a plain `source_out_s - source_in_s`; two independently
   quantised endpoints can differ from the "true" duration by float noise on
   the order of 1e-15, which was enough to trip the strict `<` overlap check
   between clips meant to sit exactly back to back.
2. **The same noise, in the other direction, made `validate_plan` report
   phantom zero-width gaps** ("gap from 56.720s to 56.720s") between clips that
   were in fact touching.
3. **Compiling one of those phantom gaps hung the renderer forever.** The
   render compiler filled every reported gap with an ffmpeg
   `color=...:d=0.0` lavfi source, and a zero-duration lavfi source never
   reaches EOF -- confirmed directly against ffmpeg outside the framework. A
   render that should have taken two seconds ran for **over eleven minutes**
   and would never have finished. This is the most severe bug found so far:
   silent, and its symptom (a render that never returns) looks identical to
   the render simply being slow, not broken.

Fixed with one shared tolerance (`_TIME_NOISE_TOLERANCE_S`, three orders of
magnitude below the timebase's own quantisation) applied everywhere clips are
compared for overlap or gaps, plus a defensive guard in the compiler itself so
a non-positive gap duration can never reach ffmpeg regardless of how it was
computed. M11 is unplanned; candidates carried over from before this run:
1. **`captions` could animate.** Word-level timings are already in the context,
   so a caption that emphasises the current word is now expressible.
2. **The subject track samples twice a second.** Fine for a walking subject,
   coarse for a fast pan. Raising it costs decode time linearly, so it wants a
   measurement before a change.
3. **`apply_skill` and `analyze` are still synchronous on the MCP loop.** Only
   `render` was moved, because only `render` is measured in minutes.

---

## M0 — Skeleton ✅ closed out

Exit criterion: `vision doctor` reports a green environment in CI on Linux and macOS.

- [x] `pyproject.toml` with uv, hatchling, entry point `vision`
- [x] ruff configured (docstrings, annotations, no-print, no-blind-except enforced)
- [x] mypy strict
- [x] pytest with `integration` / `slow` markers
- [x] GitHub Actions: lint+types job, test matrix (Linux/macOS × py3.12/3.13)
- [x] pre-commit mirroring CI
- [x] `.gitignore` excluding media and `.vision/`
- [x] Exception hierarchy with actionable `remedy` on every error
- [x] Structured logging (`vision_use/log.py`, human + JSON formatters)
- [x] Deterministic hashing (`hash_file`, `canonical_json`, `hash_params`)
- [x] Content-addressed cache with atomic writes and immutable entries
- [x] Media models: `FrameRate` (rational), `SourceInfo`, `VideoStreamInfo`,
      `AudioStreamInfo`, `MediaProbe`
- [x] PyAV probe with rotation, VFR detection, estimated frame counts
- [x] CLI: `version`, `doctor`, `inspect`, `cache path`, `cache clear`
- [x] README
- [x] `CONTRIBUTING.md` — setup, the three gates, plugin authoring, dependency
      justification, PR template
- [x] Per-package READMEs (`vision_use/`, `cache/`, `models/`, `video/`, `cli/`)
- [x] `ARCHITECTURE.md` and `SKILL_SPEC.md` rewritten against real contracts
- [x] `PRINCIPLES.md` expanded with a why and a falsifier per principle
- [x] Docker image — multi-stage, ffmpeg + OpenCV runtime libs, non-root,
      `vision doctor` as healthcheck
- [ ] semantic-release — deferred until the first tagged release, which is the
      first point it has anything to do

---

## M1 — Context Engine ✅ closed out

Exit criterion: re-running `analyze` on the same file is byte-identical and
cache-hits. **Met**, verified across all four sections together.

- [x] **M1.1** `VideoContext` model: schema version, stable IDs, `analyzers`
      provenance, partial-context rules, cross-reference validation, and the
      `find_text` / `words_between` / `shot_at` lookups skills address video with
- [x] **M1.1a** `vision_use/timebase.py` — time quantisation and exact
      rational frame mapping (not in the original plan; extracted once it became
      clear determinism and frame accuracy are the same problem)
- [x] **M1.2** Analyzer protocol + registry with transitive dependency
      resolution, cycle detection, and stable ordering
- [x] **M1.3** Shot detection (PySceneDetect) → `shots[]`, with quantised
      times, stable IDs, minimum-shot merging, and gap-free tiling
- [x] **M1.3a** `context/runner.py` — generic analyzer caching, authoritative
      merging, and the public `analyze()` entry point (pulled forward from M1.7;
      caching had to be generic before a second analyzer existed, or every
      analyzer would have grown its own)
- [x] **M1.4** Keyframe extraction → `keyframes[]`, one representative still per
      shot at the shot midpoint, JPEG-encoded via PyAV into the cache and
      referenced by `cache_key`. `phash` deferred: it has no consumer until
      near-duplicate detection exists, and an unused field invites drift
- [x] **M1.5** Audio activity map: streaming RMS in dBFS per 20 ms window,
      merged silence/speech spans with duration hysteresis, gap-free tiling
- [x] **M1.6** ASR with word-level timestamps (faster-whisper) → `transcript`,
      behind the optional `asr` extra with a lazy import and an actionable
      error when it is missing
- [x] **M1.7** `vision analyze` CLI: `--with`/`--all` section selection,
      `--no-cache`, `--json`, `-o`, plus `vision sections` listing which
      producer fills each section and what it depends on
- [x] **M1.8** Contact-sheet builder — numbered tiles burned into one image
      plus a manifest mapping tile numbers back to shots and timestamps, and a
      `vision contact-sheet` command
- [x] **M1.9** Determinism tests: per-analyzer, plus a full-scope test running
      all four sections twice and asserting byte-identical output. The
      per-analyzer tests can all pass while the composition drifts — merge
      order, provenance ordering, one analyzer perturbing what another reads —
      so the composed assertion is the one that actually closes M1
- [x] **M1.10** `docs/context-schema.md` — sections, ID scheme, time and
      frame rules, provenance, a worked "cut where he says X" example, and
      what a new analyzer must guarantee

Settled in M1.1: keyframe image bytes live in the cache and `Keyframe` holds
only a `cache_key`. Inlining them would blow the context past what fits in a
model's context window and stop it being plain diffable JSON.

---

## M2 — Timeline + Renderer ✅ closed out

Exit criterion: a hand-written plan with cuts and a transition renders correctly
at proxy and master; a straight-through plan is frame-accurate. **Met** — an
overlay track is deferred to M4, where `captions` is its first real consumer.

- [x] **M2.1** `EditPlan`, `Timeline`, `Track`, `Clip`, `Effect`, `Transition`,
      with construction-time validation of overlaps, ordering, duplicate IDs,
      dangling source references and out-of-range source ranges; `RenderSettings`
      carrying both proxy and master profiles
- [x] **M2.2** Effect registry with per-effect Pydantic param schemas,
      validation, deprecation metadata, and `schema_hash` stamping
- [x] **M2.3** Plan validation — structural invariants enforced by the model;
      advisory report (gaps, unused sources, missing files, unknown or
      misconfigured effects, all-disabled clips) via `vision validate`, which
      exits non-zero on anything that would stop a render
- [x] **M2.4** Filtergraph compiler — video and audio: per-clip trim, timestamp
      rebase, ordered effect chains split by `applies_to`, canvas fit with
      letterboxing, per-clip rate/SAR normalisation, audio resampled and
      reformatted to one rate and layout, black and silence for gaps, silence
      for sources with no audio stream, and a single interleaved `concat`
      carrying both streams. Transitions join with `xfade`/`acrossfade`
- [x] **M2.5** `RenderSettings` with proxy and master profiles; `render(proxy=True)`
      renders the same plan at reduced scale
- [x] **M2.6** Baked-effect path — `EffectDef.bakes` + `frame_op()`, streamed
      frame transformation in `video/transform.py`, content-addressed
      intermediates, in-memory clip substitution, and `invert` as the reference
      baked effect. This is the road `stabilize` follows in M5
- [x] **M2.7** `vision plan new` (identity edit from a source), `vision plan
      show`, `vision validate`, `vision render` with `--proxy` and
      `--skip-validation`
- [x] **M2.8** Frame-accuracy tests — exact frame counts asserted for cuts,
      gaps, NTSC sources, audio presence, silent sources, baked effects and
      crossfades
- [x] **M2.9** `docs/plan-schema.md` — sources, tracks, clips, the
      source-range vs timeline-position split, filter vs baked effects,
      transitions and how they shorten output, render profiles, and the two
      layers of validation

Effect versioning is **resolved** as `PLAN.md` § D4: schemas are additive-only,
a breaking change is a new effect type (`crop` → `crop.v2`), and the type is the
version — so there is no version field to resolve and no migration code. M2.2
must ship the enforcement alongside the registry:

- [x] **M2.2a** Golden schema snapshot at `tests/schemas.golden.json`; CI fails
      on any non-additive diff. Verified by deliberately breaking a schema
- [x] **M2.2b** `deprecated_since` / `superseded_by` registry metadata; a
      deprecated effect still validates and logs its replacement
- [ ] **M2.2c** Opt-in `migrate()` hook — never required, never load-bearing

---

## M3 — Skill SDK + agent loop ✅ closed out

Exit criterion: a reference skill round-trips the full lifecycle; a rejected
patch rolls back cleanly. **Met** — and the reference skill performs a real edit
rather than being a no-op, so the wiring is proven by something with actual
behaviour.

- [x] **M3.1** `Skill` protocol and `SkillMeta` (`requires` / `produces`),
      plus the lifecycle types: `SkillRequest`, `Analysis`, `Strategy`,
      `Measurement`, `ValidationReport`, `Score`
- [x] **M3.2** Skill registry, mirroring the analyzer registry; `requiring()`
      answers "what becomes possible if I transcribe this?"
- [x] **M3.3** `PlanPatch` — nine typed ops discriminated by tag, atomic
      application, and exact inversion. Every op round-trips; a failed patch
      leaves the original plan untouched
- [x] **M3.4** Pipeline: analyse → plan → execute → validate → score → retry,
      returning the highest-scoring attempt when none passes
- [x] **M3.5** `RetryPolicy` — attempt count and wall-clock budget, both
      enforced between attempts
- [x] **M3.6** Context requirement resolution — `meta.requires` backfilled
      through the analyzer registry, a satisfied context reused untouched, and
      `resolvable()` reporting unproducible sections before any analysis is paid
      for
- [x] **M3.7** `vision apply` with `--prompt`, `--dry-run`, `--max-attempts`
      and `-o`, reporting each strategy's verdict and the accepted rationale;
      plus `vision skills` listing the registry and each skill's requirements
- [x] **M3.8** Reference skill (`edit.disable_runts`) — needs no media, no
      model and no new dependency, but performs a real edit, so it proves the
      wiring without being a stub

---

## M4 — First three skills ✅ closed out

Exit criterion: `vision apply` on a real talking-head clip produces a publishable
short. **Met in parts, honestly:** silence removal, captions and reframing each
work end to end on real media and compose through the plan. `reframe` is a static
crop rather than speaker-aware — see the decision below.

- [x] **M4.1** `silence-removal` — reads `audio_activity`, cuts silences past a
      threshold with padding, snaps every boundary to a frame, and validates
      that speech survived
- [~] **M4.2** `reframe` — aspect-ratio cropping with focus strategies
      (centre / left / right), canvas resize, and validation that the crop
      stays inside the frame. **Speaker-aware tracking is not implemented**:
      it needs subject detection in the context engine, and an honest static
      crop beats a guessed one
- [x] **M4.3** `captions` — text rasterising, overlay compositing, and the
      skill: word-level timings grouped into phrases (breaking at pauses),
      rendered per line into the cache, declared as sources, and laid onto a
      new overlay track
- [x] **M2.6a** Overlay track compositing — deferred from M2 until captions
      gave it a real consumer, as planned
- [x] **M4.4** Golden-fixture eval harness — `tests/evals.golden.json` records
      what each skill measured on a fixed fixture; CI reports any movement
      beyond 2%. Verified by nudging a padding constant and watching it fire

---

## M5 — MCP server, Plugin SDK, `stabilize`

Exit criterion: `stabilize` installs from a separate repo and runs with zero core changes.

- [x] **M5.1** MCP server — six generic tools (`analyze`, `query_context`,
      `list_skills`, `plan`, `apply_skill`, `render`), deliberately not one per
      skill. `plan` was added to the five in the plan: without it an agent has
      no way to obtain the document `apply_skill` edits
- [x] **M5.2** Plugin discovery via entry points — three groups
      (`vision_use.skills` / `.analyzers` / `.effects`), a `vision plugins`
      command reporting what loaded and what did not, and a broken plugin
      recorded rather than raised. Verified with a real external package
- [x] **M5.3** Plugin SDK docs + scaffold — `docs/plugin-sdk.md` and
      `vision plugin new`, which writes a package that installs, registers, and
      passes its own discovery test. No cookiecutter dependency
- [x] **M5.4** `stabilize` as the first external plugin — `vision-stabilize`
      lives at `../vision-stabilize`, publishes a skill and a baked effect, and
      needed no change to VisionUse to work. 59% measured shake reduction. It
      found two framework gaps; one is fixed, one is open (see below)
- [x] **M5.5** Docker image — built and verified for real, not just written:
      `vision doctor` green inside it (ffmpeg 7), and a full analyze → plan →
      apply → render round trip against mounted media. `.dockerignore` added;
      CI builds the image and runs that same round trip
- [x] **M5.6** Release automation — `release.yml`, tag-driven. Verifies the tag
      matches the packaged version, runs all three gates, publishes to PyPI via
      trusted publishing, pushes to ghcr.io, generates notes from the commits

---

## Decisions log

- **Stabilize deferred to M5.** `VISION.md` names it the initial skill; it needs no
  context, no timeline, and no LLM, so it would have exercised none of the
  framework's load-bearing abstractions. It returns as the first external plugin,
  where it proves the plugin boundary instead of blocking it.
- **Skills return `PlanPatch`, never video.** Pixels are produced once, at export.
- **VisionUse builds its own Context Engine.** `video-use` stays an optional
  future integration, not a dependency. Overlap accepted deliberately.
- **Frame rate is rational, never float.** 30000/1001 has no exact float form and
  accumulated rounding drifts by whole frames across a long timeline.
- **`vision_use/log.py`, not `logging.py`.** Avoids shadowing the stdlib module.
- **Test media is generated by ffmpeg, not committed.** Keeps the repo text-only
  and lets a test request the exact rate or rotation it needs.
- **`AnalyzerRun` carries no timestamp.** A wall-clock field would make two
  analyses of the same file differ, defeating content addressing and M1's
  byte-identical exit criterion. Nothing consumes it; when an analysis ran is a
  property of the cache entry and the log, not of the document.
- **Seconds-to-frame conversion carries a half-quantum tolerance.** Quantising a
  frame boundary to microseconds can land just below it, and a plain floor then
  names the previous frame. Caught by a round-trip test at 30000/1001; without
  the tolerance `frame_index_at(frame_start_s(n)) == n` fails for every NTSC rate.
- **Effect schemas are additive-only; the type is the version (D4).** Most
  effects will be third-party by Principle 3, and any scheme depending on plugin
  authors writing migrations correctly fails silently and corrupts stored plans.
  Additive-only is mechanically checkable in CI; "did the author write a correct
  migration" is not.
- **Audio activity is level detection, not VAD.** A level threshold is
  deterministic, needs no model, and runs on CPU far faster than real time. Real
  voice-activity detection can be added later as a second analyzer producing the
  same section — the registry picks one, and nothing else changes.
- **Reported levels are rounded to 2 dp.** Float summation is not guaranteed
  bit-identical across platforms or SIMD widths, and M1 requires two analyses of
  one file to be byte-identical — including between the Linux and macOS CI legs.
  Two decimals is far finer than any threshold decision needs.
- **Hysteresis is on duration, not level.** Raw per-window classification
  flickers on every breath and plosive. Spans shorter than `min_span_s` are
  absorbed into a neighbour, then adjacent same-kind spans are coalesced — a
  regression test covers the coalesce, since without it absorbing a runt left
  two touching speech spans and `silence-removal` would plan a cut across a
  boundary where nothing changes.
- **Stills are encoded with PyAV, not Pillow or OpenCV.** PyAV is already
  declared and owns every other media path; an imaging library bought for one
  function is a dependency for nothing. OpenCV specifically is only present
  transitively via PySceneDetect, and depending on a transitive dependency
  directly breaks the build when an upstream drops it.
- **JPEG quality is set through encoder `options`, not `codec_context.qmin`.**
  The options path is typed, and pinning qmin and qmax to one value is what
  makes the output byte-reproducible — which the cache depends on.
- **Keyframes sample the shot midpoint, not its first frame.** The first frame
  of a shot is frequently mid-transition, motion-blurred, or still showing the
  tail of a dissolve, none of which represent the shot to a model.
- **Frame extraction seeks per target rather than decoding the whole file.**
  One still per shot from a one-hour source costs milliseconds by seeking and
  minutes by full decode.
- **ASR is an optional extra, not a base dependency.** `faster-whisper` pulls
  in a ~300 MB ML runtime, and most of what VisionUse does needs no speech.
  Principle 3 says plugins over built-ins; the analyzer is still *registered*
  unconditionally, so the registry stays a full capability catalogue and a
  missing extra fails at run time with an install command rather than leaving
  `transcript` with no producer.
- **Whisper's temperature is pinned to zero and the fallback ladder disabled.**
  Whisper samples at non-zero temperature, and its fallback silently raises the
  temperature on a low-confidence segment — either would break M1's
  byte-identical criterion. A determinism test runs against the real model.
- **ASR tests are excluded from the default CI job.** They download weights from
  the HuggingFace hub, which makes a run non-hermetic and dependent on a third
  party's uptime; they get a dedicated job where a failure is unambiguous.
- **`vision analyze` excludes the transcript by default.** It needs the
  optional `asr` extra, downloads model weights, and is by far the slowest
  analyzer; the default should not surprise someone who wanted shot detection.
  `--with transcript` or `--all` opts in.
- **Diagnostics go to stderr, results to stdout.** That is what keeps `--json`
  parseable while logging is on, and there is a test asserting it.
- **`fps` and `setsar` are applied per clip, not once at the end.** `concat`
  requires its inputs to agree on frame rate and sample aspect ratio, and real
  sources routinely do not. Normalising per clip is what makes a cut land on
  the frame the plan asked for, and what stops anamorphic footage failing with
  an error that names neither the clip nor the plan.
- **Gaps are filled with an explicit generated black segment.** Leaving a track
  uncovered and hoping ffmpeg improvises produces an output whose length does
  not match `plan.duration_s`.
- **ffmpeg input indices are counted, not derived from the argument list.**
  A file input is two arguments and a lavfi input is four, so
  `len(args) // 2` misnumbers every stream after the first generated gap — and
  the symptom is a silently wrong source, not an error. There is a regression
  test.
- **MCP tools return summaries and paths, never whole documents.** A context
  for a ten-minute video does not belong in a model's context window; a test
  asserts the `analyze` summary stays under 600 bytes while the document it
  points at is larger.
- **`apply_skill` reports every attempt, including the rejected ones.** An agent
  needs the failures to decide what to try next; returning only the winner
  leaves it guessing.
- **The MCP extra is optional.** The library and CLI need nothing from the SDK,
  and the SDK is a substantial dependency tree (starlette, uvicorn, httpx).
- **The eval harness reports movement in either direction, not "worse".**
  Deciding which way is better for a given metric would need the harness to
  understand each skill, which it deliberately does not. A number that moved is
  a number a human should look at once.
- **A new measurement is not a regression.** Adding a validation check is how a
  skill gets stricter; requiring a baseline update for it would discourage the
  thing worth encouraging.
- **A corrupt baseline is an error, not an empty baseline.** Treating an
  unreadable file as "nothing recorded" would let every regression pass.
- **The caption eval uses a fixed transcript, not real ASR.** A Whisper update
  would move every caption measurement at once, and the baseline would be
  recording the model's behaviour rather than the skill's.
- **The patch vocabulary gained `SetCanvas`.** Reframing changes the shape of
  the result, not its content; without it a skill could crop the picture but not
  the frame it lands in, and the renderer would letterbox the crop straight back
  to the original aspect.
- **`reframe` ships a static crop, not speaker tracking.** Following a subject
  needs face detection — a real dependency and a real quality bar. Guessing at
  it would produce something worse than an honest centred crop, so tracking
  waits until the context engine can supply subject positions, which the skill
  will then read like any other section.
- **The patch vocabulary gained `AddSource`/`RemoveSource`.** Discovered while
  writing `captions`: without them a skill could only rearrange the media it was
  given, never introduce any — no caption image, title card or fetched b-roll.
  A gap in the vocabulary, not in the skill.
- **Caption entity ids are allocated above everything the plan already uses.**
  A fixed offset works right up until someone hands you a plan that reached it.
- **A pause ends a caption line.** Grouping purely by duration cuts sentences in
  half; a gap between words is where the speaker themselves broke the phrase.
- **A caption is trimmed so it never outlasts the next one.** Two composited at
  once would stack, with the older sitting on top of the newer.
- **Captions are rasterised in-process with Pillow, not by ffmpeg.** The
  development build has neither `drawtext` (libfreetype) nor `subtitles`
  (libass) — verified, not assumed — and distribution builds vary widely. A
  framework whose captions work on one machine and not the next is not usable.
  Pillow's bundled font means captions need no font files shipped or located.
- **A caption image is full-canvas and transparent.** Compositing is then a
  plain origin overlay with the position already baked in, which removes a
  whole class of alignment bug for a few kilobytes.
- **Overlays composite, they do not concatenate.** Each sits on top of the
  assembled video for its own span via `enable=between(...)`, so an overlay
  track costs one filter per clip rather than a re-assembly of the timeline.
- **A silence starting exactly where a clip starts must still advance the
  cursor.** Found by a test: leading dead air — very common in real footage —
  was silently never trimmed, because the guard that skips already-consumed
  silences also skipped this one.
- **Cuts are padded and frame-snapped.** Slicing the instant the waveform drops
  clips the tail off the last word; an audio-derived boundary falls mid-frame
  and the encoder then picks for itself.
- **`silence-removal` refuses an already-cut timeline.** Re-cutting needs a rule
  for what happens to the existing boundaries, and there is no good default —
  better to do nothing than to guess.
- **Each attempt starts from the original plan, not the previous attempt.** A
  rejected strategy cannot contaminate the next one, which is what makes trying
  several strategies meaningful rather than compounding.
- **When nothing passes validation, the best-scoring attempt is returned.**
  `VISION.md` asks for the highest-scoring result rather than an outright
  failure; `SkillOutcome.accepted` is how a caller that wanted strictness tells
  the difference.
- **The retry budget bounds attempts *and* wall time.** An unbounded auto-retry
  loop over video is a runaway-compute bug wearing a feature's clothes.
- **A patch that will not apply skips its strategy rather than failing the run.**
  It is a bug in the skill, and the next strategy may well be fine.
- **`model_copy(update=...)` skips validation, so every patch operation
  re-validates.** Found by a test: an operation could construct a plan the model
  would have rejected — overlapping clips, a range past the end of a source —
  and the failure surfaced much later, somewhere that could not say which
  operation caused it.
- **A patch is all-or-nothing.** If any operation fails the caller gets an
  exception and the original plan, never a half-edited one. That is what lets
  the retry loop try a strategy and cleanly abandon it.
- **The inverse unwinds in reverse order.** Undoing forwards would replay
  dependent operations in the wrong sequence.
- **`SetEffects` exists alongside `PushEffect`.** A push-only vocabulary cannot
  express reordering or clearing a stack, and it has no inverse.
- **A transition switches the whole track from `concat` to pairwise `xfade`.**
  `concat` butts segments together and a crossfade must overlap them, so the two
  cannot be mixed in one join. Boundaries without a transition become two-input
  concats inside the same pairwise chain.
- **Transitions shorten the output.** A plan cannot express overlap — the model
  forbids two clips sharing timeline space — so a transition consumes its
  duration from the boundary it spans. `CompiledRender.duration_s` reports the
  real result rather than the plan's pre-transition length.
- **`xfade` offsets are tracked cumulatively, not read from the timeline.**
  Each offset indexes the *accumulated* stream, which every preceding
  transition has already shortened; deriving it from a clip's timeline position
  drifts by the sum of the fades before it.
- **A boundary is described by the later clip's `transition_in`.** Letting both
  ends describe one boundary would make a plan ambiguous, so the later clip wins
  and the earlier clip's `transition_out` is ignored for that join.
- **Baked intermediates use ffv1 in `bgr0`, not yuv420p.** ffv1 is lossless in
  whatever format it is given, but *converting* RGB frames to yuv420p subsamples
  chroma 2x2 first, so a baked effect lost colour detail before the real encode
  began. Caught by a round-trip test asserting the pixels come back unchanged.
  (ffv1 accepts `gbrp` only at 9 bits and up, hence packed `bgr0`.)
- **Baking substitutes clips in memory, never on disk.** The document the user
  edits stays the document they wrote; the intermediate is an implementation
  detail of one render.
- **A baked effect is removed from the clip once applied.** Leaving it would
  make the compiler try to filter what has already been rendered in.
- **`render` validates before invoking ffmpeg.** An error found in a second
  beats the same error found after a render that was never going to be right;
  `--skip-validation` exists for when the caller knows better.
- **A plan records absolute source paths.** Plans get moved and passed around,
  and a relative path silently stops resolving.
- **One `concat` carries video and audio together.** Concatenating them
  separately lets each stream be assembled from its own segment boundaries, and
  any rounding difference becomes creeping lip-sync error. Interleaving the
  labels makes both share one set of boundaries by construction.
- **Every segment contributes an audio leg, even a silent one.** `concat`
  requires the same stream count from every input, so sources with no audio
  track and every generated gap get `anullsrc`. Without it the filter fails
  with an error naming no clip.
- **`Source.has_audio` is recorded in the plan, not probed at compile time.**
  Compiling stays free of media I/O and a plan stays self-describing.
- **`EffectDef.effect_type`, not `type`.** An attribute named `type` in a class
  body shadows the builtin, so the `ClassVar[type[BaseModel]]` annotation on the
  next line resolves to the string attribute instead. The plan model's field is
  still `type`; only the definition class differs.
- **Effect params use `extra="forbid"`.** A typo'd parameter name should be an
  error, not a silently ignored no-op that renders the wrong thing.
- **Two clips may not overlap on one track.** It has no defined rendering —
  both would claim the same output frames — so layering is expressed with a
  second track and overlap is a construction error, not a style choice.
- **Gaps and unused sources are reported, not rejected.** Both are legal: a gap
  renders as black or silence, and a plan under construction may declare a
  source before placing it. Both are usually accidents, so they surface in the
  validation report rather than failing construction.
- **A clip separates source range from timeline position.** That is what lets a
  clip be moved without re-deciding what it shows, and trimmed without
  re-deciding where it sits.
- **`Effect.params` is not validated by the model.** A core lacking a plugin's
  effect must still round-trip a plan that uses it; the registry validates
  params against the effect's own schema in M2.2.
- **Contact sheet tiles are numbered in the pixels, not just the manifest.**
  A model cannot reliably count grid positions, so an unlabelled sheet produces
  confident wrong answers about which shot it is describing.
- **The label font is a 3x5 bitmap table in the source.** Burning text would
  otherwise mean Pillow, or ffmpeg's drawtext and a libfreetype-enabled build.
  Only ten glyphs are needed, and a table cannot vary between platforms — which
  matters because the sheet is content-addressed.
- **Tiles are resized by nearest neighbour, not interpolation.** It needs no
  imaging library and is exactly reproducible; a resampler whose output varies
  with SIMD width or library version would break content addressing.
- **`Analyzer` protocol members are instance attributes, not `ClassVar`.** A
  configurable analyzer may compute `produces` per instance, and a class-level
  attribute satisfies the protocol either way.
- **`VideoContext.has()` reads provenance, not emptiness.** A video with
  genuinely no speech must stay distinguishable from one never transcribed.
- **`AnalyzerResult` is a Pydantic model, not a dataclass.** Results must
  serialize losslessly so the runner can cache *any* analyzer generically.
  A dataclass would have forced per-analyzer caching — the exact duplication
  Principle 8 exists to prevent.
- **Only `opencv-python` is declared, not `opencv-python-headless`.**
  `scenedetect` already requires the former, and both packages install the same
  `cv2` module, so declaring both installs OpenCV twice and uninstalling either
  deletes the shared files. Containers wanting headless should override the pin
  rather than add a second package.
- **OpenCV bundles its own libav, which warns against PyAV's on macOS.**
  Verified non-fatal: detection is correct and deterministic with both loaded.
  Worth revisiting if crashes appear — the fallback is feeding PySceneDetect's
  detectors frames decoded by PyAV instead of letting it open the file.
- **Analyzers are registered by importing `vision_use.context`.** Discovered by
  a regression: `analyze(path, [SHOTS])` failed with the default registry
  because built-ins were never imported. Explicit registration in
  `analyzers/__init__.py`, triggered from the package root.
- **Plugins load at the entry points, not at package import.** The first attempt
  called `ensure_loaded()` from `vision_use/skills/__init__.py`, which surfaced a
  circular-import hazard: `plugins.py` imports all three registries, so loading
  during that package's own import lets a plugin that imports VisionUse re-enter
  a half-initialised module. Discovery is now called explicitly from the CLI
  callback and `build_server()`.
- **A broken plugin is recorded, never raised.** One typo in a plugin's module
  path would otherwise make `vision --help` fail, which is a terrible trade for
  an optional extension. Failures are caught per entry point, attributed by
  name, and reported by `vision plugins`. The metadata *scan* is wrapped too — a
  corrupt distribution elsewhere in site-packages can make `entry_points()`
  itself raise.
- **A plugin name collision is a loud registration error, not an override.** A
  plugin that could silently shadow a built-in skill would make an installation's
  behaviour depend on what happens to be installed.
- **`x if x is None else default`, never `x or default`, for registries.** Found
  by a failing test: an empty registry is falsy, so `skills or default_skills`
  quietly registered a caller's plugins into the process-wide registry — exactly
  what a caller passing their own registry is trying to avoid.
- **The effect schema snapshot covers built-ins, not everything registered.**
  Found the same way: with a plugin installed, `test_snapshot_covers_every_
  registered_effect` failed on the plugin's effect. The D4 guard is about this
  repository's schemas; a plugin's schemas are the plugin's to guard.
- **The plugin scaffold is a command, not a cookiecutter.** `vision plugin new`
  substitutes four names into string templates. A templating engine plus a
  package-data story is more machinery than the job has, and a new dependency
  for what `str.replace` does is exactly what Principle 5 warns about.
- **The generated skill is a real edit, not a stub.** It trims the tail of a
  timeline: something an author changes rather than something they must first
  make run. A template that raises `NotImplementedError` teaches nothing about
  the five stages.
- **The scaffold's tests assert it still matches this VisionUse.** Not that
  files exist -- that the entry-point group is the one discovery scans, and that
  the generated skill imports and satisfies the protocol. A template that drifts
  from the framework is worse than no template: it fails at install time, far
  from the cause.
- **Four more `x or default` registry fallbacks fixed.** `planning.validate_plan`,
  `render.bake`, `render.compiler` and `skills.builtin.register_builtins` all
  silently fell back to the global registry when handed an empty one -- the same
  bug as in `plugins.py`, found by looking for the pattern rather than by a
  failure. `validate_plan(plan, effects=EffectRegistry())` would have validated
  against effects the caller deliberately excluded.
- **`vision-stabilize` is a real repository, not a fixture.** It lives outside
  the VisionUse checkout at `../vision-stabilize` and was built with
  `vision plugin new`, so the scaffold is proven to be the start of real work
  rather than a demo. VisionUse contains no reference to it.
- **`EffectDef.version` mixed into the bake cache key.** Found by fixing a sign
  error in the plugin's `frame_op` and watching the re-render produce identical
  output: source, range and parameters were unchanged, so the stale bake was
  served and a plugin had no way to invalidate it. `TRANSFORM_VERSION` covers
  the framework's own transform code; nothing covered the effect's. The guard
  was proven by removing the fix and confirming the test fails.
- **Phase correlation's sign convention is content displacement.** Getting it
  backwards does not crash and passes every structural check -- it just doubles
  the shake. Pinned by a test in the plugin, because no framework-level check
  could catch it.
- **`stabilize` validates structurally, and says so.** It checks the edit landed
  and the timeline is undisturbed, not whether the result is steadier. That is
  the honest consequence of the open finding above, recorded in the plugin's
  README rather than hidden behind a measurement that sounds like quality but
  is not.
- **A plugin cannot add a context section (open).** `ContextSection` is a closed
  enum and `AnalyzerResult` has fixed fields. The `vision_use.analyzers` group
  therefore only lets a plugin fill sections the core already knows about --
  which is not what Principle 3 promises, and it is what stops a plugin's skill
  from satisfying Principle 7.
- **The Docker image is verified by running it, not by building it.** `doctor`
  green inside the container proves ffmpeg and OpenCV's shared libraries are
  actually present -- a `libGL.so` missing from the slim base builds perfectly
  and fails on first use. CI runs a whole render inside the image for the same
  reason.
- **The CI image job makes its media directory world-writable.** The image runs
  as uid 1000 by design and a runner's workspace is owned by a different uid, so
  without it the job would fail on a permission error that says nothing about
  VisionUse. Caught by rehearsing the exact command sequence locally rather than
  by pushing and watching CI.
- **`.dockerignore` added.** The build context was sending `.venv` (454 MB) and
  `.git` to the daemon on every build; it is now 8 kB.
- **Releases are tag-driven, not commit-driven.** semantic-release's automatic
  mode makes every merge to main a potential publish, which is the wrong default
  for a framework other people pin. `git tag v0.2.0` is where the decision to
  release is recorded, and the workflow refuses a tag that disagrees with the
  packaged version -- otherwise `pip install vision-use==0.2.0` installs
  something that calls itself 0.1.0.
- **PyPI publishing uses trusted publishing.** No long-lived token in repository
  secrets.
- **D5: a plugin may add a context section, and the core never reads it.** The
  finding M5.4 left open. `VideoContext.extra` holds plugin payloads keyed by
  dotted, package-namespaced names; `ContextSection` stays the enum of sections
  *this repository* produces, and `SectionName = ContextSection | str` is what
  everything now accepts. A registry of section models was rejected: it would
  give the core an opinion about shapes it cannot know, and leaves no answer for
  a stored context whose plugin is no longer installed. Plain JSON has one.
- **`str(section)` everywhere, not `section.value`.** A plugin's section is a
  plain string with no `.value`. Caught by mypy across six call sites once the
  types widened, which is the argument for `strict` in one line.
- **Protocol attributes are invariant, so the alias had to be applied
  uniformly.** `frozenset[ContextSection]` does not satisfy
  `frozenset[ContextSection | str]` on a protocol member. Rather than fight it
  with read-only properties, every declaration now says `frozenset[SectionName]`
  -- 13 files, mechanical, and it reads better than the union did.
- **The measurement a skill validates with must be the one its work used.**
  `vision-stabilize`'s analyzer and its effect share `_shift_between`. A skill
  that judged itself with a different estimator than the one it corrected with
  would produce a confident number that means nothing.
- **A plugin skill can now decline to act, with evidence.** `stabilize` leaves
  steady footage alone because it *measured* that it was steady, rather than
  stabilising everything and calling it success. That is the behaviour D5 was
  for, and it is what Principle 7 was asking for all along.
- **The eval harness is a library, and a plugin uses it.** `vision-stabilize`
  keeps its own `evals.golden.json` and gets the same regression guarantee the
  built-in skills have, with nothing about it in this repository. Verified by
  changing the plugin's motion downscale from 4 to 8: the recorded shake moved
  8.1% and the baseline failed.
- **A plugin's eval must let the runtime backfill its section.** Handing
  `evaluate()` a pre-built context excludes the analyzer from the very thing
  meant to guard it. Discovered by doing it wrong: a hand-built context has no
  provenance, so `has()` was correctly False and the runtime tried to backfill
  from a registry with no plugins loaded.
- **`resolve_section_names` is shared by the CLI and MCP.** Both surfaces used
  to check the enum directly, so `--with stabilize.motion` failed with "choose
  from: audio_activity, keyframes, probe, shots, transcript" -- a message that
  is actively misleading once plugins exist. One resolver now answers for both,
  and lists plugin sections in its remedy: seeing a section *absent* from that
  list is the fastest way to notice a plugin is not installed.
- **`--all` includes plugin sections.** "Everything" that silently excluded
  plugins would be a lie.
- **`query_context` names sections but does not dump them.** A plugin's payload
  can be large, and the MCP rule is summaries and paths, never whole documents.
  Naming them is what lets an agent discover a section it has never seen; the
  payload arrives only when asked for.
- **Long media was measured before anything was designed.** 10 minutes with 120
  real cuts: analyze 4.8s at 148 MB, proxy render 4.6s at 74 MB, bake 20.3s at
  112 MB. Memory is flat, which is what the streaming design was for -- so the
  work went to storage, which is where the actual problem was.
- **The first two long fixtures measured nothing.** A 10-minute `testsrc` is one
  continuous scene (1 shot), and 6-degree hue steps between segments are too
  subtle to be cuts. Only alternating genuinely different sources produced the
  120 cuts worth profiling. A fixture that exercises nothing gives a green
  number that means nothing.
- **Cache eviction is era-bucketed LRU, largest-first within an era.** Strict
  per-file LRU is wrong when artefacts differ in size by three orders of
  magnitude: pruning the real 485 MB cache to 100 MB deleted 123 analysis
  entries worth 1.2 MB before reaching the single 484 MB bake that was the
  problem, because they had been touched seconds earlier. Access time now
  decides the *era* (one hour); size decides within it.
- **`vision cache status` breaks down by producer.** Which producer is filling
  the disk is the first thing worth knowing, and the answer is always `bake`.
- **`--budget` refuses what it cannot read.** A misread budget silently deletes
  the wrong amount, so `20 gigs` is an error rather than a guess.
- **Progress is a callback, not a bar.** The library must not know whether it is
  driven by a terminal, an MCP server, or a test. The CLI draws the bar, on
  stderr, transient, so `--json` stays machine-readable and piping to a file
  does not fill it with escape codes.
- **A failing progress callback is dropped, not retried.** It would raise on
  every frame and the logging would outweigh the render. One bad callback
  disables itself and nothing else -- the flag lives on the per-operation
  `Reporter`, not globally.
- **The frame total is an estimate, and says so.** A plan does not record
  whether its source is variable-rate -- that is a probe fact, and compiling
  stays free of media I/O -- so `Progress.fraction` clamps at 1.0 rather than
  reporting 103%. I nearly shipped a comment claiming a VFR check that the plan
  model cannot support.
- **Cancellation was tested by actually interrupting a bake.** `writing()`
  creates its temporary inside the cache directory and removes it on
  `BaseException`, so a KeyboardInterrupt mid-bake leaves zero files -- no
  entry and no `.partial`. Proven by deleting the cleanup and confirming the
  test fails, because an untested atomicity claim is just a comment.
- **The cache prunes itself, bounded by default.** 20 GB unless
  `VISION_CACHE_BUDGET` says otherwise (`none` disables it). A default rather
  than unbounded because the cache is rebuildable by construction, and a
  framework whose normal operation fills a disk is not usable unattended.
- **Eviction nearly ate the render's own input.** Enforcing the budget after a
  bake deleted the intermediate the returned plan pointed at: it is both the
  newest and the largest entry, and the policy takes the largest first within an
  era. `prune(protect=...)` now exempts what the caller is about to read, and
  `bake_plan` passes every source in the prepared plan. Found by writing the
  test that asked the obvious question, and confirmed load-bearing by removing
  the fix and watching it fail.
- **An unreachable budget warns rather than deleting what is in use.**
  Protected bytes still count towards the ceiling -- they are on the disk either
  way -- so a budget smaller than the artefacts in flight cannot be met. Saying
  so is better than breaking the render in progress. Verified on real media: a
  5 MB budget with a 5.5 MB intermediate warns, keeps it, and renders.
- **`parse_size` moved from the CLI into the cache module.** The automatic
  budget needed the same parser the `--budget` flag uses, and two parsers for
  one syntax is how they drift apart.
- **`SUBJECT`, not `FACES` -- the docs were corrected, not the claim inflated.**
  `SKILL_SPEC.md` has specified a `FACES` section and a "speaker-aware"
  `reframe` since M4, and neither existed. OpenCV 5 no longer ships the Haar
  cascades, so a real detector now means a heavy ML dependency that has to earn
  its place. The section records where the frame *changes*, which is a usable
  stand-in, and both documents now say that instead of promising detection. A
  plugin can add `faces` when someone wants to pay that cost -- D5 means it
  needs no change here.
- **Measured before designing, again.** A white box moving at x=460 of 640 is
  located at 0.802 against a true 0.781. The negative cases matter more: a still
  frame reports confidence **0.0** rather than a small number a consumer might
  mistake for weak evidence, and `testsrc` -- which changes everywhere -- reports
  near-centre with low confidence.
- **The subject is located per shot, not per file.** Averaging across a cut
  places the crop between two positions the subject never occupied.
- **`SubjectSpan` carries fractions, not pixels.** A span stays meaningful after
  a scale and is comparable across sources of different sizes.
- **`reframe` follows the subject, per clip.** `requires` gained `SUBJECT`, so
  the skill that was purely geometric now costs an analysis pass -- paid once,
  for everyone, which is what Principle 8 buys. Each clip's window centres on
  the subject of the shot its midpoint falls in.
- **Low confidence falls back to centre, and that is the point of confidence.**
  The threshold sits above what uniform motion scores (`testsrc` reports ~0.32),
  because cropping to the centroid of "everything is moving" is worse than not
  trying. A number nothing acts on is not worth reporting.
- **The subject strategy is offered first, the fixed ones behind it.** A source
  with no locatable subject reframes exactly as it did before -- proven by the
  eval baseline, which still records `vertical-centre` and passes byte-identical
  on a fixture with no subject to find.
- **Windows differ only in offset, so one canvas still fits.** Per-clip crop
  positions with a single timeline shape, and no special case in the compiler.
- **A crop still cannot move *within* a shot.** Effect params are static per
  clip; a drifting window would need keyframed parameters. Stated in the skill's
  own docstring rather than left for a user to discover.
- **MCP progress is a protocol notification, not a polling shape.** I expected
  to need a job registry and a `render_status` tool; the SDK already has
  `Context.report_progress`, so the library's existing callback maps straight
  onto it. Checking what the protocol supported before designing saved building
  the wrong thing.
- **`render` now runs off the event loop.** Baking decodes frames and ffmpeg is
  a subprocess; running either on the loop stalled the entire server, so an
  agent could not call another tool or cancel until the render finished. That
  was already true before progress -- the progress work is what surfaced it.
- **Progress is rate-limited to twice a second.** Baking reports every frame;
  forwarding all of them would spend more of the transport on progress than on
  the work.
- **`Context` is imported at runtime, not under TYPE_CHECKING.** The SDK reads a
  tool's annotations with `eval_str=True` to build its schema, so a
  typing-only name makes registration fail outright -- and it must come from
  `mcp.server.mcpserver.context`, which is the class the injector checks against.
  Both were found by the server refusing to start, not by reading the docs.
- **D6: animation is a sidecar, not a change to any param schema.** Letting a
  parameter be "a number or a track" would change the shape of every effect's
  JSON schema -- exactly the non-additive change D4's golden snapshot exists to
  reject -- and would force every plugin's `params_model` to learn a union it
  never asked for. `animate` sits beside `params`, which keeps the static value,
  so every stored plan stays valid and a reader that ignores animation still
  produces a sensible edit.
- **Keyframe times are clip-local.** A clip that is retrimmed or moved keeps its
  animation, which is what lets an animated effect survive the ordinary editing
  a skill does around it.
- **An animated effect is a validation *error* until the renderer honours it.**
  Rendering a plan that says the crop moves as one that does not is silently
  wrong, and silently wrong is the worst outcome available. Better to refuse.
- **Only numeric parameters animate.** Interpolating a string or an enum is
  meaningless, and allowing it would produce output nobody could explain.
- **Animation renders as an ffmpeg time expression, no bake.** `crop` takes an
  expression for `x` and `y`, so a moving crop costs nothing beyond the pass the
  render already makes. Measured: a box crossing the frame stays visible in
  100/100 output frames, against 32/100 with a static crop.
- **Clip-local time came free.** The compiler already emits
  `setpts=PTS-STARTPTS` before any effect, so `t` inside a clip's chain is
  measured from the clip's own start -- the same base D6 chose for keyframes.
  The two agree by construction rather than by conversion, which is luck worth
  writing down before someone "simplifies" the setpts away.
- **The expression and `value_at` are tested against each other.** The
  filtergraph path and the baked path must mean the same thing by an animation,
  or a clip renders differently depending on which path it took. The test
  evaluates the emitted expression rather than a Python restatement of it.
- **`crop` refuses to animate its own size.** A frame that changes size mid-clip
  is not encodable and the timeline's single canvas could not describe it, so it
  is refused rather than silently ignored.
- **Numbers are formatted fixed-point.** `1e-05` is valid Python and not valid
  in a filtergraph.
- **Baking is not a free pass for animation.** A baked effect animates only if
  it implements `frame_op_animated`; declaring `bakes` no longer implies it.
  The earlier rule would have let a baked effect ignore animation silently,
  which is the failure this whole design is arranged to prevent.
- **An animated baked effect receives its interpolated parameters, not the
  time.** It never touches keyframes or timestamps, so the interpolation is
  defined in exactly one place and the filter and bake paths cannot drift apart.
- **Interpolated values are validated at every frame.** An animation sweeping a
  crop's `x` negative is a bad plan; finding out at the frame it happens beats
  producing a corrupt intermediate. Pydantic validation costs microseconds
  against milliseconds of decode, so correctness was the easy call.
- **Interpolating an integer parameter rounds to an integer.** A crop offset of
  12.5 pixels is not a thing, and the params model would reject it.
- **`reframe` follows the subject within a shot.** The subject analyzer already
  sampled twice a second and threw everything but the centroid away; keeping
  those samples as a track was most of the work, and they become the crop's
  keyframes directly.
- **Summary confidence does not gate following.** A subject who crosses the
  frame has *low* confidence by construction -- there is no single good place
  for a fixed crop -- which is precisely when a moving crop is right. Gating on
  it would have disabled the feature exactly where it matters. The measurement
  showed this: the travelling-box fixture reports confidence 0.0.
- **A flat track degrades into a static crop by itself.** So no guard is needed
  for "there is no real subject": uniform motion produces a track of 0.5s, which
  animates to a constant, which is a centre crop.
- **A clip keeps only the keyframes it covers.** A clip trimmed out of a longer
  shot must not inherit keyframes for footage it never shows.
- **The eval harness could not see a strategy change.** `compare` checked
  acceptance and measurements but not *which* strategy was accepted, so
  `reframe` switching from a fixed crop to a following one left every number
  identical and the baseline green. Found by checking rather than trusting the
  pass. `compare` now reports it, and the baseline was regenerated deliberately.
- **Cache eras are measured back from the newest entry, not clock hours.** A
  suite run turned up an eviction failure that vanished on re-run: with absolute
  hour boundaries, two files written seconds apart landed in different eras
  whenever the prune happened to run just after the hour, so the same cache
  pruned differently depending on the time of day. Relative eras remove the
  class of bug. Pinned by a test that constructs a moment past a boundary, and
  confirmed by reverting the rule and watching it fail.
- **Which also caught a sign error I introduced fixing it.** Measured back from
  the newest entry, a *larger* bucket means older, so the ascending sort briefly
  evicted the newest first. Two existing eviction tests failed immediately,
  which is the argument for having written them.
- **M10 was end-to-end validation, and it found what unit tests structurally
  could not.** Every piece here was individually tested; nothing before this
  drove forty real clips through the whole pipeline at once. The bugs only
  exist at that scale -- 2, 4, and 6 clips all rendered in under 1.2s each with
  no error, and float noise never crossed the threshold that trips the checks
  until well past that.
- **`Clip.duration_s` is an unquantised subtraction, and it should probably stay
  that way.** Quantising it would only move the noise problem rather than
  remove it -- the model would then need to guarantee `start + duration == end`
  under quantisation too. A shared tolerance at the comparison points --
  overlap, gaps -- is the narrower, more honest fix: it says plainly "noise
  this small is not a real gap or overlap" rather than pretending the
  arithmetic is exact.
- **A zero-duration lavfi source is not a no-op, it is a hang.** Confirmed
  directly, outside the framework: `ffmpeg -f lavfi -i color=...:d=0.0 ...`
  never produces EOF and never terminates. Worth remembering anywhere else a
  duration might reach zero -- `add_silence` takes the same risk and was not
  exercised by this session, though nothing found a path to zero there.
- **The compiler refuses a non-positive gap itself, as well as its caller no
  longer asking for one.** Belt and suspenders deliberately: the caller-side
  fix stops the *known* path to the bug, the compiler-side guard stops any
  path, known or not, from reaching ffmpeg. Both are tested, and both were
  proven to catch the regression by reverting each independently.
- **Every regression test uses numbers taken verbatim from the failing run, not
  invented ones.** A hand-picked example for the overlap case did not
  reproduce the noise and passed even with the bug present; the fix was to
  extract the actual clip values from the real failure and use those. A float
  precision bug is exactly the class where "an example that looks like the
  real one" is not the same as reproducing it.
