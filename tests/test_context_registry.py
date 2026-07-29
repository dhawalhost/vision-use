"""Tests for analyzer registration and dependency resolution."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError
from vision_use.cache.store import CacheStore
from vision_use.context.analyzer import AnalyzerRequest, AnalyzerResult, provenance
from vision_use.context.registry import AnalyzerRegistry, default_registry
from vision_use.context.runner import merge
from vision_use.errors import ConfigurationError, VisionUseError
from vision_use.models.context import ContextSection, SectionName, VideoContext
from vision_use.models.media import FrameRate, MediaProbe, SourceInfo, VideoStreamInfo

S = ContextSection


class FakeAnalyzer:
    """A registrable analyzer that does nothing, for wiring tests."""

    version: str = "1"

    def __init__(
        self,
        name: str = "fake",
        produces: frozenset[SectionName] = frozenset({S.SHOTS}),
        requires: frozenset[SectionName] = frozenset(),
        params: Mapping[str, JsonValue] | None = None,
    ) -> None:
        self.name = name
        self.produces = produces
        self.requires = requires
        self._params: Mapping[str, JsonValue] = params or {}

    def params(self) -> Mapping[str, JsonValue]:
        return self._params

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        return AnalyzerResult()


def test_register_and_retrieve() -> None:
    registry = AnalyzerRegistry()
    analyzer = FakeAnalyzer()
    registry.register(analyzer)

    assert registry.get("fake") is analyzer
    assert "fake" in registry
    assert len(registry) == 1


def test_duplicate_registration_rejected() -> None:
    """Two plugins claiming a name must fail loudly, not depend on load order."""
    registry = default_registry([FakeAnalyzer()])

    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register(FakeAnalyzer())


def test_duplicate_allowed_with_replace() -> None:
    registry = default_registry([FakeAnalyzer()])
    replacement = FakeAnalyzer()

    registry.register(replacement, replace=True)

    assert registry.get("fake") is replacement


def test_analyzer_producing_nothing_rejected() -> None:
    with pytest.raises(ConfigurationError, match="no produced sections"):
        AnalyzerRegistry().register(FakeAnalyzer(produces=frozenset()))


def test_self_dependency_rejected() -> None:
    with pytest.raises(ConfigurationError, match="both requires and produces"):
        AnalyzerRegistry().register(
            FakeAnalyzer(produces=frozenset({S.SHOTS}), requires=frozenset({S.SHOTS}))
        )


def test_unknown_name_error_lists_known_analyzers() -> None:
    registry = default_registry([FakeAnalyzer("shots")])

    with pytest.raises(ConfigurationError) as excinfo:
        registry.get("absent")

    assert "shots" in (excinfo.value.remedy or "")


def test_unregister_is_idempotent() -> None:
    registry = default_registry([FakeAnalyzer()])
    registry.unregister("fake")
    registry.unregister("fake")

    assert len(registry) == 0


def test_resolve_pulls_in_prerequisites() -> None:
    registry = default_registry(
        [
            FakeAnalyzer("shots", frozenset({S.SHOTS})),
            FakeAnalyzer("keyframes", frozenset({S.KEYFRAMES}), frozenset({S.SHOTS})),
        ]
    )

    order = [analyzer.name for analyzer in registry.resolve([S.KEYFRAMES])]

    assert order == ["shots", "keyframes"]


def test_resolve_is_transitive() -> None:
    registry = default_registry(
        [
            FakeAnalyzer("shots", frozenset({S.SHOTS})),
            FakeAnalyzer("keyframes", frozenset({S.KEYFRAMES}), frozenset({S.SHOTS})),
            FakeAnalyzer("asr", frozenset({S.TRANSCRIPT}), frozenset({S.KEYFRAMES})),
        ]
    )

    order = [analyzer.name for analyzer in registry.resolve([S.TRANSCRIPT])]

    assert order == ["shots", "keyframes", "asr"]


def test_resolve_deduplicates_shared_prerequisites() -> None:
    registry = default_registry(
        [
            FakeAnalyzer("shots", frozenset({S.SHOTS})),
            FakeAnalyzer("keyframes", frozenset({S.KEYFRAMES}), frozenset({S.SHOTS})),
            FakeAnalyzer("audio", frozenset({S.AUDIO_ACTIVITY}), frozenset({S.SHOTS})),
        ]
    )

    order = [a.name for a in registry.resolve([S.KEYFRAMES, S.AUDIO_ACTIVITY])]

    assert order.count("shots") == 1
    assert order.index("shots") == 0


def test_resolve_ignores_probe() -> None:
    """PROBE is always present before any analyzer runs, so it needs no producer."""
    registry = default_registry([FakeAnalyzer("shots", frozenset({S.SHOTS}))])

    assert registry.resolve([S.PROBE]) == []


def test_resolve_probe_requirement_needs_no_producer() -> None:
    registry = default_registry(
        [FakeAnalyzer("shots", frozenset({S.SHOTS}), frozenset({S.PROBE}))]
    )

    assert [a.name for a in registry.resolve([S.SHOTS])] == ["shots"]


def test_resolve_empty_request() -> None:
    assert AnalyzerRegistry().resolve([]) == []


def test_unproducible_section_reports_actionable_error() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        AnalyzerRegistry().resolve([S.TRANSCRIPT])

    assert "transcript" in excinfo.value.message
    assert excinfo.value.remedy is not None


def test_dependency_cycle_detected() -> None:
    registry = default_registry(
        [
            FakeAnalyzer("a", frozenset({S.SHOTS}), frozenset({S.KEYFRAMES})),
            FakeAnalyzer("b", frozenset({S.KEYFRAMES}), frozenset({S.SHOTS})),
        ]
    )

    with pytest.raises(ConfigurationError, match="cycle"):
        registry.resolve([S.SHOTS])


def test_resolve_order_is_stable_across_requests() -> None:
    """Resolution must not depend on set iteration order."""
    registry = default_registry(
        [
            FakeAnalyzer("shots", frozenset({S.SHOTS})),
            FakeAnalyzer(
                "keyframes",
                frozenset({S.KEYFRAMES}),
                frozenset({S.SHOTS, S.AUDIO_ACTIVITY}),
            ),
            FakeAnalyzer("audio", frozenset({S.AUDIO_ACTIVITY})),
        ]
    )

    first = [a.name for a in registry.resolve([S.KEYFRAMES])]
    second = [a.name for a in registry.resolve([S.KEYFRAMES])]

    assert first == second
    assert first.index("keyframes") == len(first) - 1


def test_sections_reports_producible_set() -> None:
    registry = default_registry(
        [
            FakeAnalyzer("shots", frozenset({S.SHOTS})),
            FakeAnalyzer("audio", frozenset({S.AUDIO_ACTIVITY})),
        ]
    )

    assert registry.sections() == {S.SHOTS, S.AUDIO_ACTIVITY}


def test_producer_of_unknown_section_is_none() -> None:
    assert AnalyzerRegistry().producer_of(S.TRANSCRIPT) is None


def test_provenance_hashes_params_and_sorts_sections() -> None:
    analyzer = FakeAnalyzer(
        "multi", frozenset({S.KEYFRAMES, S.SHOTS}), params={"threshold": 0.4}
    )

    run = provenance(analyzer)

    assert run.name == "multi"
    assert run.produced == (S.KEYFRAMES, S.SHOTS)
    assert len(run.params_hash) == 64


def test_provenance_params_hash_tracks_params() -> None:
    a = provenance(FakeAnalyzer("x", params={"threshold": 0.4}))
    b = provenance(FakeAnalyzer("x", params={"threshold": 0.5}))

    assert a.params_hash != b.params_hash


def test_provenance_is_stable() -> None:
    analyzer = FakeAnalyzer("x", params={"threshold": 0.4})

    assert provenance(analyzer) == provenance(analyzer)


DIGEST = "d" * 64


def _probe(duration_s: float = 10.0) -> MediaProbe:
    return MediaProbe(
        source=SourceInfo(
            path=Path("clip.mp4"),
            sha256=DIGEST,
            size_bytes=1,
            container="mp4",
            duration_s=duration_s,
        ),
        video=VideoStreamInfo(
            index=0,
            codec="h264",
            width=1920,
            height=1080,
            frame_rate=FrameRate(num=25, den=1),
        ),
    )


def _request(context: VideoContext) -> AnalyzerRequest:
    return AnalyzerRequest(
        source=Path("clip.mp4"), context=context, cache=CacheStore(Path("/tmp/unused"))
    )


# --- plugin sections (PLAN.md D5) ------------------------------------------


class _MotionAnalyzer:
    """An analyzer as a plugin would publish it: its own section, its own shape."""

    name = "plugin.motion"
    version = "1"
    produces: frozenset[SectionName] = frozenset({"plugin.motion"})
    requires: frozenset[SectionName] = frozenset({ContextSection.PROBE})

    def params(self) -> dict[str, JsonValue]:
        return {}

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        del request
        return AnalyzerResult(extra={"plugin.motion": {"mean_px": 4.5}})


class _ShadowingAnalyzer(_MotionAnalyzer):
    """Claims a name the core owns."""

    name = "plugin.shadow"
    produces: frozenset[SectionName] = frozenset({"shots"})


class _UnnamespacedAnalyzer(_MotionAnalyzer):
    """Claims a bare name, which would collide with the next plugin to want it."""

    name = "plugin.bare"
    produces: frozenset[SectionName] = frozenset({"motion"})

    def run(self, request: AnalyzerRequest) -> AnalyzerResult:
        del request
        return AnalyzerResult(extra={"motion": {"mean_px": 4.5}})


def test_a_plugin_section_is_resolvable() -> None:
    """The registry must route a dotted name it has never heard of."""
    registry = AnalyzerRegistry()
    registry.register(_MotionAnalyzer())

    assert [item.name for item in registry.resolve(["plugin.motion"])] == ["plugin.motion"]


def test_an_unknown_plugin_section_says_to_install_the_plugin() -> None:
    """The usual cause of a missing dotted section is an uninstalled package."""
    with pytest.raises(ConfigurationError) as excinfo:
        AnalyzerRegistry().resolve(["nobody.provides.this"])

    assert "install the plugin" in (excinfo.value.remedy or "")


def test_a_plugin_section_reaches_the_context() -> None:
    context = VideoContext(probe=_probe())
    analyzer = _MotionAnalyzer()

    merged = merge(context, analyzer, analyzer.run(_request(context)))

    assert merged.section("plugin.motion") == {"mean_px": 4.5}
    assert merged.has("plugin.motion")


def test_an_unproduced_extra_key_is_ignored() -> None:
    """An analyzer cannot fill a section it never declared -- plugins included."""
    context = VideoContext(probe=_probe())
    analyzer = _MotionAnalyzer()

    merged = merge(context, analyzer, AnalyzerResult(extra={"plugin.other": {"x": 1}}))

    assert merged.section("plugin.other") is None


def test_a_plugin_cannot_shadow_a_builtin_section() -> None:
    """Silently overwriting `shots` would be the hardest failure here to trace."""
    context = VideoContext(probe=_probe())
    analyzer = _ShadowingAnalyzer()

    merged = merge(context, analyzer, AnalyzerResult(extra={"shots": "nonsense"}))

    # `shots` stays a real section; the plugin's payload never lands.
    assert merged.shots == ()
    assert merged.extra == {}


def test_an_unnamespaced_plugin_section_is_refused() -> None:
    context = VideoContext(probe=_probe())
    analyzer = _UnnamespacedAnalyzer()

    with pytest.raises(VisionUseError, match="unusable section"):
        merge(context, analyzer, analyzer.run(_request(context)))


def test_provenance_records_a_plugin_section() -> None:
    """`has()` answers from provenance, so it must be recorded verbatim."""
    analyzer = _MotionAnalyzer()

    assert provenance(analyzer).produced == ("plugin.motion",)


def test_a_context_with_a_plugin_section_round_trips() -> None:
    context = VideoContext(probe=_probe(), extra={"plugin.motion": {"mean_px": 4.5}})

    restored = VideoContext.model_validate_json(context.model_dump_json())

    assert restored == context


def test_a_context_rejects_a_shadowing_section_directly() -> None:
    with pytest.raises(ValidationError, match="built-in"):
        VideoContext(probe=_probe(), extra={"shots": []})


def test_an_empty_plugin_section_still_counts_as_analysed() -> None:
    """ "Analysed, found nothing" must stay distinct from "never analysed"."""
    context = VideoContext(probe=_probe())
    analyzer = _MotionAnalyzer()

    merged = merge(context, analyzer, AnalyzerResult(extra={"plugin.motion": {}}))

    assert merged.has("plugin.motion")
    assert merged.section("plugin.motion") == {}
