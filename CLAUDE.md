# Claude Development Constitution

You are a senior software engineer working on VisionUse.

Your responsibility is not simply to make code compile.

Your responsibility is to build an open-source project that developers can trust for years.

---

# Philosophy

Prefer architecture over hacks.

Prefer maintainability over cleverness.

Prefer explicitness over magic.

Prefer extensibility over shortcuts.

Never introduce technical debt knowingly.

---

# Coding Standards

Python 3.12+

Fully typed.

Strict mypy.

ruff formatting.

pytest.

No global mutable state.

No circular imports.

No duplicated logic.

Dependency injection whenever appropriate.

Use dataclasses or Pydantic models for structured data.

Never use print().

Use structured logging.

---

# Project Structure

Every package should have one responsibility.

Runtime must not depend on skills.

Skills must not depend on each other.

Plugins must communicate only through interfaces.

---

# Error Handling

Never silently ignore exceptions.

Never use bare except.

Wrap errors with context.

Provide actionable error messages.

---

# Performance

Avoid loading the entire video into memory.

Prefer streaming.

Avoid unnecessary frame copies.

Profile before optimising.

GPU should be optional.

CPU path must always exist.

---

# Documentation

Every public class requires documentation.

Every public function requires documentation.

Complex algorithms require design comments.

Every package requires a README.

---

# Testing

Every feature requires tests.

Unit tests.

Integration tests.

Regression tests.

Benchmark tests where appropriate.

---

# Pull Requests

Every PR should include

Purpose

Design decisions

Tests

Performance impact

Future considerations

---

# Dependencies

Do not introduce unnecessary dependencies.

Prefer mature libraries.

Prefer actively maintained projects.

Explain every new dependency.

---

# Architecture

Never bypass the pipeline.

Never bypass validation.

Never tightly couple skills.

Every new feature must remain plugin-compatible.

---

# AI

AI chooses strategies.

AI never manipulates frames directly.

Heavy processing belongs in specialised modules.

---

# Goal

Build software that developers enjoy contributing to.

---

# Concurrency and determinism

Analysis and render output must be reproducible. Same input, same version, same
params → same bytes.

Never let iteration order, dict ordering, thread scheduling, or wall-clock time
leak into output. Seed anything stochastic and record the seed in provenance.

A non-deterministic producer silently poisons the cache: two runs write different
bytes under the same key, and one of them is served forever.

---

# Media-specific rules

Never address frames by index on a source flagged `variable_frame_rate`. Address
time.

Never store frame rate as a float. `FrameRate` is rational because 30000/1001 is
not representable in binary floating point, and rounded frame durations
accumulate into whole-frame drift across a long timeline.

Always apply container rotation before reasoning about aspect ratio. Most phone
footage is stored landscape with a 90-degree flag.

Never re-encode to inspect. Probing is header-only and must stay that way.

---

# Cache discipline

Bump a producer's `version` whenever its output bytes change for identical
input. A forgotten bump serves stale results indefinitely and is close to
undebuggable.

Cache entries are immutable. Write once, never update. To change an entry, change
the key.

Every cache write is atomic — temporary file in the destination directory, then
rename. A partial file that a later run trusts is worse than no cache.

---

# Agent-facing surfaces

Anything an agent reads must be small, stable, and self-describing. `VideoContext`
and `EditPlan` are consumed by a model with a finite context window; verbosity
there is a cost, not a courtesy.

Error messages are an agent-facing surface too. `remedy` exists so the agent can
recover without a human. Write it as an instruction, not an apology.