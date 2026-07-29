"""Content-addressed cache.

The validate/retry loop described in ``VISION.md`` re-derives the same
intermediates constantly, so caching is a correctness-adjacent requirement
rather than an optimisation: without it, a retry is indistinguishable from a
cold run and the framework's cost is unbounded.
"""

from vision_use.cache.store import CacheKey, CacheStore, default_cache_root

__all__ = ["CacheKey", "CacheStore", "default_cache_root"]
