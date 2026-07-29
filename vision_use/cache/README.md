# `vision_use.cache`

Content-addressed artefact store.

## Why this exists

The validate/retry loop in `VISION.md` re-derives the same intermediates
constantly. Without a cache a retry is indistinguishable from a cold run and the
framework's cost is unbounded — so caching here is closer to a correctness
requirement than an optimisation.

## The identity model

A cache entry is keyed on everything that could change its bytes:

| Component | Why it participates |
|---|---|
| `namespace` | Which producer made it (`analyze.shots`, `render.proxy`) |
| `version` | Bumped when a producer's output changes for identical input |
| `inputs` | Content hashes of consumed files; order is significant |
| `params` | The producer's configuration |

Identical keys are guaranteed to describe identical work, so **entries are
immutable**: written once, never updated. To change an entry, change the key.

## Atomicity

Writes go to a temporary file in the destination directory and are renamed into
place. A crashed or cancelled run can never leave a half-written file that a later
run mistakes for a valid hit — which would be worse than having no cache at all.

If two processes commit the same key concurrently, the existing entry wins. Since
entries are immutable, both are byte-identical by construction.

## Usage

```python
store = CacheStore()
key = store.key("analyze.shots", version="1", inputs=[digest], params={"threshold": 27.0})

if (hit := store.get(key)) is None:
    with store.writing(key) as tmp:
        tmp.write_bytes(expensive())
```

## Layout

`$VISION_CACHE_ROOT`, or `./.vision/cache`. Entries live at
`<namespace>/<first two hex chars>/<digest><suffix>` — the fanout keeps directory
listings fast on filesystems that degrade with very large directories.

## Rules

- Bump `version` whenever output bytes change for identical input. A forgotten
  bump serves stale results indefinitely and is close to undebuggable.
- Never mutate an entry in place.
- Never write outside `store.writing()` — that is what guarantees atomicity.
