"""The context engine: analyzers that populate a :class:`VideoContext`.

An *analyzer* fills one region of the context and declares what it needs to do
so. The registry orders analyzers by those declarations, so requesting a
section pulls in exactly its prerequisites and nothing else -- the mechanism
behind Principle 8's "no duplicated work".
"""

# Imported last, and for its side effect: it populates the default registry.
# Without this, `analyze(path, [SHOTS])` would fail for anyone who had not
# separately imported the analyzers package -- the registry would be empty.
# Placed after the imports above so the submodules it depends on are loaded.
from vision_use.context import analyzers as _analyzers  # noqa: F401
from vision_use.context.analyzer import Analyzer, AnalyzerRequest, AnalyzerResult
from vision_use.context.registry import AnalyzerRegistry, registry
from vision_use.context.runner import analyze, context_digest, merge, run_analyzer

__all__ = [
    "Analyzer",
    "AnalyzerRegistry",
    "AnalyzerRequest",
    "AnalyzerResult",
    "analyze",
    "context_digest",
    "merge",
    "registry",
    "run_analyzer",
]
