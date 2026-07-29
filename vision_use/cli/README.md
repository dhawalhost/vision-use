# `vision_use.cli`

The `vision` command. A thin presentation layer: parse arguments, configure
logging, call the library, render results.

**It contains no media logic.** Anything reachable from the terminal must be
equally reachable from the Python API and from the MCP server in M5. If a
behaviour only exists in the CLI, it is in the wrong place.

## Commands

| Command | Status |
|---|---|
| `vision version` | M0 |
| `vision doctor` | M0 — environment diagnosis; exits non-zero when anything blocks use |
| `vision inspect FILE` | M0 — container and stream properties, no decoding |
| `vision cache path` / `cache clear` | M0 |
| `vision analyze FILE` | M1 — build a `VideoContext`; `--with`, `--all`, `--json`, `-o`, `--no-cache` |
| `vision sections` | M1 — which producer fills each context section |
| `vision contact-sheet FILE` | M1 — tile keyframes into one labelled image; `--describe` prints the tile index |
| `vision plan new FILE` | M2 — build a straight-through `EditPlan` |
| `vision plan show PLAN` | M2 — summarise a plan's tracks and clips |
| `vision validate PLAN` | M2 — advisory checks; non-zero on render-blocking errors |
| `vision render PLAN` | M2 — render to video; `--proxy`, `--skip-validation` |
| `vision skills` | M3 — list registered skills and what context they need |
| `vision apply PLAN --skill NAME` | M3 — run a skill; `--prompt`, `--dry-run`, `--max-attempts` |

| `vision apply` | M3 |

## Conventions

- Every command that produces data takes `--json`. Agents and scripts consume
  that path; humans get the Rich rendering.
- Diagnostics go to stderr, results to stdout, so `--json` output stays parseable
  when logging is on.
- Errors render as `error: <message>` plus `hint: <remedy>` and exit non-zero.
  The hint comes from `VisionUseError.remedy` — see `errors.py`.
- No `print()`. Rich for rendering, `vision_use.log` for diagnostics.

## Doctor

`doctor` exists because native media dependencies otherwise fail deep inside a
pipeline with an obscure error. Checks are pure functions returning `Check`, kept
separate from rendering so the MCP server and tests can reuse them without
capturing terminal output.
