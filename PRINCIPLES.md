# Principles

Ten principles. They are load-bearing: `PLAN.md` and `ARCHITECTURE.md` are
derivations from them, not independent opinions.

Each principle below carries two additions beyond the original statement — the
**why**, and the **falsifier**. The falsifier matters more. A principle you
cannot violate is decoration; naming the violation makes it reviewable.

---

## Principle 1 — AI orchestrates, libraries execute

The model chooses skills, strategies, and parameters. It never manipulates pixels.

**Why:** an LLM's judgement is worth more than its arithmetic. Routing decisions
are where it adds value; per-frame maths is where it is slow, expensive, and
non-deterministic.

**Violated when:** a skill asks a model to produce numeric per-frame output, or
any pixel operation lives outside a specialised module.

---

## Principle 2 — Framework over application

VisionUse is imported, not launched. It has a CLI, but the CLI is a client of the
library like any other.

**Why:** an application accumulates assumptions about who is driving it. A
framework has to state its contracts, and stated contracts are what let an agent
— or a contributor — use it in a way the authors did not anticipate.

**Violated when:** logic lives in `vision_use.cli` that the Python API cannot
reach, or a feature only works when a terminal is attached.

---

## Principle 3 — Plugins over built-ins

A capability belongs in core only if the framework cannot function without it.

**Why:** every built-in is a permanent maintenance obligation and a
non-negotiable dependency for every user. A plugin is neither.

**Violated when:** adding a skill requires touching the runtime. That is the
test, and it is why `stabilize` ships as the first external plugin rather than as
core code — see `PLAN.md` D1.

---

## Principle 4 — Everything should work locally

No cloud call is required for any core operation. GPU is optional; a CPU path
always exists.

**Why:** video is large, often private, and frequently under a licence that
forbids uploading it. A framework that phones home is unusable for exactly the
professional work it is aimed at.

**Violated when:** a skill has no CPU path, a default configuration requires an
API key, or a test cannot run offline.

---

## Principle 5 — Open source first

Open dependencies, open formats, open development. Design happens in the
repository.

**Why:** the goal in `VISION.md` is an ecosystem, and nobody builds on a project
whose direction is decided elsewhere.

**Violated when:** a proprietary dependency sits on a required path, or an
architectural decision lands without a written rationale someone can argue with.

---

## Principle 6 — Human prompts over configuration

The interface is *"remove the shake but keep the handheld feel"*, not
`stabilization=37`.

**Why:** the parameter is not the intent. Users know what they want the footage to
look like; they do not know what 37 means, and neither does anyone else.

**Violated when:** the primary path to a capability is a numeric knob. Knobs may
exist underneath — `plan()` picks them — but they are not the interface.

---

## Principle 7 — Validation before export

Every skill validates its own output. Nothing reaches the user unmeasured.

**Why:** an agent cannot iterate on a result it cannot judge. Validation is what
makes the retry loop meaningful rather than superstitious.

**Violated when:** a skill has no `validate()` worth running, or output is
written before scoring. A skill that cannot state what would make its own output
wrong is not finished.

---

## Principle 8 — One video analysis, shared context, no duplicated work

A video is analysed once into a `VideoContext`. Every skill reads it.

**Why:** analysis dominates cost, and independent analyses disagree. Two skills
that each detect shots will disagree on boundaries, and the composition will be
wrong in ways neither skill can detect.

**Violated when:** a skill opens a decoder. If the context lacks something,
extend the context engine — that is not a licence to re-analyse.

---

## Principle 9 — Every feature must be composable

Any skill must compose with any other.

**Why:** the value is in combinations. `silence-removal` then `reframe` then
`captions` is the actual product; three tools that each own the whole file are not.

**Violated when:** a skill flattens the timeline, assumes it runs first or last,
addresses frame indices on a variable-frame-rate source, or returns a whole
`EditPlan` instead of a patch.

---

## Principle 10 — Never lock users in

Leaving must be cheap. Every document VisionUse produces is readable without it.

**Why:** a framework that is hard to leave is one people are careful about
adopting. Cheap exit is what makes adoption cheap.

**Violated when:** an output format is undocumented or proprietary, a plan cannot
be exported to a standard interchange format, or state lives somewhere the user
cannot inspect with a text editor.

---

## When principles conflict

They will. Two conflicts are already live, and both resolve the same way — the
one nearer the top of this list wins:

**Principle 3 (plugins) vs Principle 7 (validation).** A plugin author can ship a
weak `validate()`. Core cannot audit plugin quality without becoming the thing
Principle 3 forbids. Resolution: core enforces that `validate()` exists and that
`SkillMeta.produces` is honoured; it does not judge the thresholds. Plugins are
free to be bad, and the registry is free to say so.

**Principle 6 (prompts) vs Principle 4 (local).** Natural-language intent implies
a model, and the best models are remote. Resolution: prompt interpretation is a
*convenience layer*, never a required one. Every skill is reachable with explicit
parameters, so the framework stays fully usable with no model at all.

The general rule: a lower-numbered principle constrains a higher-numbered one.
If a conflict cannot be resolved that way, it is a design flaw, not a trade-off —
write it down and fix the design.