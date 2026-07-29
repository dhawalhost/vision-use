# VisionUse

**Open source AI video intelligence and editing framework.**

VisionUse lets AI agents understand, analyse, edit, and enhance video. It is not
a video editor and not an FFmpeg wrapper — it is an orchestration framework that
exposes reusable *skills* to agents, in the same spirit as browser-use.

> Status: pre-alpha. M0–M5 are complete — analysis, timeline, rendering,
> skills, MCP, and the plugin boundary. Nothing is API-stable yet. See
> [PLAN.md](PLAN.md) for the roadmap and [VISION.md](VISION.md) for the project
> constitution.

---

## Why it is built this way

Two ideas carry the whole design.

**One analysis, shared context.** A video is analysed once into a `VideoContext`
— shots, keyframes, a word-level transcript, an audio activity map — with stable
IDs. Every skill reads that context instead of re-deriving it. That is how an
agent can say *"cut the part where he says X"* at all.

**Skills edit a document, not pixels.** A skill returns a patch against a
declarative `EditPlan`. Nothing is encoded until export, so skills compose
without generation loss, an edit is reversible and reviewable, and an agent gets
to work on JSON — which it is good at — rather than on encoders, which it is not.

---

## Install

Requires Python 3.12+ and ffmpeg 6+.

```bash
uv sync
```

Verify the environment:

```bash
uv run vision doctor
```

---

## Usage

M0 ships the foundations: environment diagnosis, media probing, and the
content-addressed cache.

```bash
uv run vision inspect input.mp4
```

```bash
uv run vision inspect input.mp4 --json
```

```bash
uv run vision cache path
```

Baked effects store *lossless* intermediates, so the cache grows fast on real
footage — ten minutes of 640x360 is around half a gigabyte. Check it, and keep
it under a budget:

```bash
uv run vision cache status
```

```bash
uv run vision cache prune --budget 20GB
```

The cache also prunes itself: it stays under **20 GB** by default, evicting
least-recently-used artefacts after a bake. Change or disable that with
`VISION_CACHE_BUDGET`:

```bash
VISION_CACHE_BUDGET=5GB uv run vision render plan.json -o out.mp4
```

```bash
VISION_CACHE_BUDGET=none uv run vision render plan.json -o out.mp4
```

Artefacts the current render is reading are never evicted. If the budget is
smaller than those, VisionUse says so and keeps them rather than breaking the
render it is in the middle of.

M1 adds analysis. Build a shared `VideoContext` — shots, keyframes, an audio
activity map, and optionally a word-level transcript:

```bash
uv run vision analyze input.mp4
```

```bash
uv run vision analyze input.mp4 --with shots,transcript --json
```

Speech recognition is an optional extra, since it pulls a sizeable ML runtime:

```bash
uv sync --extra asr
```

M2 adds editing. A plan is a declarative document — skills patch it, and pixels
are produced exactly once, at export:

```bash
uv run vision plan new input.mp4 -o plan.json
```

```bash
uv run vision validate plan.json
```

```bash
uv run vision render plan.json -o out.mp4 --proxy
```

Baking a long clip shows a progress bar on stderr, so `--json` output stays
clean and piping to a file does not fill it with escape codes.

M3 adds skills. A skill reads the shared context, proposes strategies, and
returns a patch — the first whose own validation passes is kept:

```bash
uv run vision skills
```

```bash
uv run vision apply plan.json --skill edit.remove_silence --prompt "cut the dead air"
```

`edit.reframe` follows the subject: it crops 16:9 to 9:16 around wherever the
activity is and **keeps following as the subject moves**, rather than assuming
whoever shot the video centred them. It falls back to a fixed crop when there is
no subject to track.

Nothing is rendered until you ask: `apply` edits the plan, `render` turns it
into a file.

M5 opens the framework up. A third-party package can add a skill, an analyzer or
an effect with no change to this repository — start from a working one:

```bash
uv run vision plugin new my-plugin
```

```bash
uv run vision plugins
```

See [docs/plugin-sdk.md](docs/plugin-sdk.md).

---

## For agents

VisionUse speaks MCP. Six generic tools — `analyze`, `query_context`,
`list_skills`, `plan`, `apply_skill`, `render` — rather than one per skill, so
the surface stays the same size as the ecosystem grows.

```bash
uv sync --extra mcp
```

```bash
uv run vision-mcp
```

Tools return summaries and file paths, never whole documents: a `VideoContext`
for a ten-minute video does not belong in a model's context window.

`render` reports progress over the protocol as it bakes, and runs off the event
loop — so a long render neither looks like a hang nor stalls the server.

---

## Docker

The image carries ffmpeg, so it needs nothing installed on the host:

```bash
docker run --rm -v "$PWD:/media" ghcr.io/dhawalhost/vision-use:latest doctor
```

```bash
docker run --rm -v "$PWD:/media" ghcr.io/dhawalhost/vision-use:latest plan new /media/input.mp4 -o /media/plan.json
```

Mount `/var/cache/vision` as a volume to keep the analysis cache between runs —
without it every container re-analyses from scratch.

---

## Development

```bash
uv run ruff check . && uv run ruff format --check .
```

```bash
uv run mypy
```

```bash
uv run pytest
```

Tests that need real media generate their clips with ffmpeg rather than checking
binaries into the repository, and skip automatically when ffmpeg is absent. To
run only the fast unit tests:

```bash
uv run pytest -m "not integration"
```

---

## Project documents

| File | What it is |
|---|---|
| [VISION.md](VISION.md) | Project constitution — vision, mission, non-goals |
| [PRINCIPLES.md](PRINCIPLES.md) | The ten principles the design answers to |
| [PLAN.md](PLAN.md) | Implementation plan, data model, milestones |
| [CLAUDE.md](CLAUDE.md) | Engineering standards for contributors and agents |
| [TASKS.md](TASKS.md) | Milestone task breakdown |
| [docs/context-schema.md](docs/context-schema.md) | `VideoContext` reference — sections, IDs, time rules |
| [docs/plan-schema.md](docs/plan-schema.md) | `EditPlan` reference — clips, effects, transitions, render profiles |
| [docs/plugin-sdk.md](docs/plugin-sdk.md) | Adding a skill, analyzer or effect from your own package |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, the three gates, plugin authoring |

---

## Licence

See [LICENSE](LICENSE).
