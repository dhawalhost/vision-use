"""Tests for plan construction, validation, and the plan/render commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner
from vision_use.cli.main import app
from vision_use.effects.registry import EffectRegistry
from vision_use.errors import VisionUseError
from vision_use.models.ids import clip_id, source_id, track_id
from vision_use.models.media import FrameRate
from vision_use.models.plan import Clip, EditPlan, Effect, Keyframe, Source, Timeline, Track
from vision_use.planning import (
    Severity,
    load_plan,
    save_plan,
    straight_through,
    validate_plan,
)
from vision_use.video.probe import probe

from tests.conftest import MakeClip, requires_ffmpeg

runner = CliRunner()
PAL = FrameRate(num=25, den=1)
DIGEST = "c" * 64


#: A path that certainly exists, so a fixture plan does not trip the
#: missing-source check when the test is about something else.
PRESENT = Path(__file__)


def _plan(*clips: Clip, path: Path = PRESENT, duration_s: float = 60.0) -> EditPlan:
    source = Source(id=source_id(0), path=path, sha256=DIGEST, duration_s=duration_s)
    return EditPlan(
        sources={source.id: source},
        timeline=Timeline(
            frame_rate=PAL,
            width=640,
            height=480,
            tracks=(Track(id=track_id(0), kind="video", clips=clips),),
        ),
    )


def _clip(index: int, start: float, in_s: float, out_s: float, **kwargs: object) -> Clip:
    return Clip(
        id=clip_id(index),
        source_id=source_id(0),
        source_in_s=in_s,
        source_out_s=out_s,
        timeline_start_s=start,
        **kwargs,
    )


# --- straight_through ------------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_identity_plan_covers_the_whole_source(make_clip: MakeClip) -> None:
    clip = make_clip(name="plan_a", seconds=2.0, fps="25")

    plan = straight_through(clip)

    assert plan.duration_s == pytest.approx(probe(clip).source.duration_s, abs=0.01)
    assert len(plan.timeline.clips()) == 1


@pytest.mark.integration
@requires_ffmpeg
def test_identity_plan_takes_canvas_and_rate_from_the_source(make_clip: MakeClip) -> None:
    clip = make_clip(name="plan_b", seconds=2.0, width=320, height=240, fps="25")

    plan = straight_through(clip)

    assert (plan.timeline.width, plan.timeline.height) == (320, 240)
    assert plan.timeline.frame_rate.num == 25


@pytest.mark.integration
@requires_ffmpeg
def test_identity_plan_records_whether_audio_exists(make_clip: MakeClip) -> None:
    """The compiler needs this without probing, so the plan must carry it."""
    loud = straight_through(make_clip(name="plan_loud", with_audio=True))
    silent = straight_through(make_clip(name="plan_silent", with_audio=False))

    assert loud.sources[source_id(0)].has_audio
    assert not silent.sources[source_id(0)].has_audio


@pytest.mark.integration
@requires_ffmpeg
def test_identity_plan_stores_an_absolute_path(make_clip: MakeClip) -> None:
    """A plan is moved around; a relative path would stop resolving."""
    plan = straight_through(make_clip(name="plan_abs", seconds=2.0))

    assert plan.sources[source_id(0)].path.is_absolute()


@pytest.mark.integration
@requires_ffmpeg
def test_identity_plan_validates_clean(make_clip: MakeClip) -> None:
    plan = straight_through(make_clip(name="plan_clean", seconds=2.0))

    assert validate_plan(plan).findings == ()


def test_missing_source_reports_a_remedy(tmp_path: Path) -> None:
    with pytest.raises(VisionUseError):
        straight_through(tmp_path / "absent.mp4")


# --- validation ------------------------------------------------------------


def test_gap_is_a_warning_not_an_error() -> None:
    """Gaps render as black; occasionally that is deliberate."""
    report = validate_plan(_plan(_clip(0, 0.0, 0.0, 1.0), _clip(1, 5.0, 2.0, 3.0)))

    assert report.ok
    assert any("gap" in finding.message for finding in report.warnings)


def test_unused_source_is_a_warning() -> None:
    extra = Source(id=source_id(1), path=PRESENT, sha256=DIGEST, duration_s=10.0)
    base = _plan(_clip(0, 0.0, 0.0, 1.0))
    plan = base.model_copy(update={"sources": {**base.sources, extra.id: extra}})

    report = validate_plan(plan)

    assert report.ok
    assert any("no clip uses it" in finding.message for finding in report.warnings)


def test_missing_file_is_an_error() -> None:
    report = validate_plan(_plan(_clip(0, 0.0, 0.0, 1.0), path=Path("/nope/gone.mp4")))

    assert not report.ok
    assert any("missing" in finding.message for finding in report.errors)


def test_unknown_effect_is_an_error() -> None:
    clip = _clip(0, 0.0, 0.0, 1.0, effects=(Effect(type="nobody.registered.this"),))

    report = validate_plan(_plan(clip))

    assert not report.ok
    assert any("unknown effect" in finding.message for finding in report.errors)


def test_an_empty_effect_registry_is_honoured() -> None:
    """An empty registry is falsy; `effects or default` would ignore it.

    A caller who passes their own registry is asking to validate against *that*
    one. Falling back to the global registry would validate against effects the
    caller deliberately excluded, and pass a plan they meant to reject.
    """
    clip = _clip(0, 0.0, 0.0, 1.0, effects=(Effect(type="crop", params={"width": 64}),))

    report = validate_plan(_plan(clip), effects=EffectRegistry())

    assert any("unknown effect" in finding.message for finding in report.errors)


def test_invalid_effect_params_are_an_error() -> None:
    clip = _clip(0, 0.0, 0.0, 1.0, effects=(Effect(type="crop", params={"width": 5}),))

    report = validate_plan(_plan(clip))

    assert not report.ok


def test_all_clips_disabled_is_an_error() -> None:
    report = validate_plan(_plan(_clip(0, 0.0, 0.0, 1.0, enabled=False)))

    assert not report.ok
    assert any("disabled" in finding.message for finding in report.errors)


def test_no_tracks_is_an_error() -> None:
    plan = EditPlan(timeline=Timeline(frame_rate=PAL, width=640, height=480))

    report = validate_plan(plan)

    assert not report.ok


def test_findings_carry_severity_and_remedy() -> None:
    report = validate_plan(_plan(_clip(0, 0.0, 0.0, 1.0), path=Path("/nope/gone.mp4")))

    finding = report.errors[0]
    assert finding.severity is Severity.ERROR
    assert finding.remedy


# --- round trip ------------------------------------------------------------


def test_plan_round_trips_through_disk(tmp_path: Path) -> None:
    plan = _plan(_clip(0, 0.0, 0.0, 1.0))
    target = tmp_path / "nested" / "plan.json"

    save_plan(plan, target)

    assert load_plan(target) == plan


def test_loading_a_missing_plan_reports_a_remedy(tmp_path: Path) -> None:
    with pytest.raises(VisionUseError, match="Could not read"):
        load_plan(tmp_path / "absent.json")


def test_loading_malformed_json_reports_a_remedy(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(VisionUseError, match="not a valid edit plan"):
        load_plan(broken)


def test_loading_a_structurally_invalid_plan_is_rejected(tmp_path: Path) -> None:
    """A hand-edited plan with a dangling reference must fail at load, not render."""
    plan = _plan(_clip(0, 0.0, 0.0, 1.0))
    payload = json.loads(plan.model_dump_json())
    payload["timeline"]["tracks"][0]["clips"][0]["source_id"] = "src_0009"
    target = tmp_path / "dangling.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VisionUseError, match="not a valid edit plan"):
        load_plan(target)


# --- CLI -------------------------------------------------------------------


@pytest.mark.integration
@requires_ffmpeg
def test_cli_plan_new_writes_a_plan(make_clip: MakeClip, tmp_path: Path) -> None:
    clip = make_clip(name="cli_plan", seconds=2.0)
    target = tmp_path / "plan.json"

    result = runner.invoke(app, ["plan", "new", str(clip), "-o", str(target)])

    assert result.exit_code == 0
    assert load_plan(target).duration_s > 0


@pytest.mark.integration
@requires_ffmpeg
def test_cli_plan_show_lists_clips(make_clip: MakeClip, tmp_path: Path) -> None:
    target = tmp_path / "plan.json"
    save_plan(straight_through(make_clip(name="cli_show", seconds=2.0)), target)

    result = runner.invoke(app, ["plan", "show", str(target)])

    assert result.exit_code == 0
    assert "clp_0000" in result.stdout


@pytest.mark.integration
@requires_ffmpeg
def test_cli_validate_passes_a_clean_plan(make_clip: MakeClip, tmp_path: Path) -> None:
    target = tmp_path / "plan.json"
    save_plan(straight_through(make_clip(name="cli_val", seconds=2.0)), target)

    result = runner.invoke(app, ["validate", str(target)])

    assert result.exit_code == 0
    assert "No problems" in result.stdout


def test_cli_validate_exits_nonzero_on_error(tmp_path: Path) -> None:
    target = tmp_path / "plan.json"
    save_plan(_plan(_clip(0, 0.0, 0.0, 1.0), path=Path("/nope/gone.mp4")), target)

    result = runner.invoke(app, ["validate", str(target)])

    assert result.exit_code == 1


def test_cli_render_refuses_an_invalid_plan(tmp_path: Path) -> None:
    """Failing in a second beats failing after ffmpeg has spent minutes."""
    target = tmp_path / "plan.json"
    save_plan(_plan(_clip(0, 0.0, 0.0, 1.0), path=Path("/nope/gone.mp4")), target)

    result = runner.invoke(app, ["render", str(target)])

    assert result.exit_code == 1
    assert "--skip-validation" in result.stdout + result.stderr


@pytest.mark.integration
@requires_ffmpeg
def test_cli_round_trip_preserves_the_source(make_clip: MakeClip, tmp_path: Path) -> None:
    """plan new then render is the identity edit; it must come back unchanged."""
    clip = make_clip(name="cli_round", seconds=2.0, fps="25")
    plan_path = tmp_path / "plan.json"
    out = tmp_path / "out.mp4"

    runner.invoke(app, ["plan", "new", str(clip), "-o", str(plan_path)])
    result = runner.invoke(app, ["render", str(plan_path), "-o", str(out)])

    assert result.exit_code == 0
    before, after = probe(clip), probe(out)
    assert after.video.frame_count == before.video.frame_count
    assert (after.video.width, after.video.height) == (before.video.width, before.video.height)


@pytest.mark.integration
@requires_ffmpeg
def test_cli_render_proxy_is_smaller(make_clip: MakeClip, tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    save_plan(straight_through(make_clip(name="cli_proxy", seconds=2.0)), plan_path)
    out = tmp_path / "proxy.mp4"

    result = runner.invoke(app, ["render", str(plan_path), "-o", str(out), "--proxy"])

    assert result.exit_code == 0
    assert probe(out).video.width < 320


# --- animation (PLAN.md D6) ------------------------------------------------


def _animated_crop(*frames: tuple[float, float]) -> Effect:
    return Effect(
        type="crop",
        params={"x": 0, "y": 0, "width": 64, "height": 64},
        animate={"x": tuple(Keyframe(t_s=t, value=v) for t, v in frames)},
    )


def test_an_animatable_effect_is_accepted() -> None:
    """`crop` renders animation as a time expression, so it is not refused."""
    clip = _clip(0, 0.0, 0.0, 1.0, effects=(_animated_crop((0.0, 0.0), (1.0, 20.0)),))

    report = validate_plan(_plan(clip))

    assert not any("cannot render" in finding.message for finding in report.errors)


def test_animating_an_effect_that_cannot_is_refused() -> None:
    """Rendering statically while the plan says otherwise is silently wrong."""
    clip = _clip(
        0,
        0.0,
        0.0,
        1.0,
        effects=(
            Effect(
                type="scale",
                params={"width": 64, "height": 64},
                animate={"width": (Keyframe(t_s=0.0, value=64),)},
            ),
        ),
    )

    report = validate_plan(_plan(clip))

    assert not report.ok
    assert any("cannot render" in finding.message for finding in report.errors)


def test_a_static_effect_is_still_fine() -> None:
    """The refusal must be about animation, not about crops."""
    clip = _clip(0, 0.0, 0.0, 1.0, effects=(Effect(type="crop", params={"width": 64}),))

    report = validate_plan(_plan(clip))

    assert not any("renders statically" in finding.message for finding in report.errors)


def test_keyframes_past_the_end_of_the_clip_are_a_warning() -> None:
    """They never apply, which is almost certainly not what the author meant."""
    clip = _clip(0, 0.0, 0.0, 1.0, effects=(_animated_crop((0.0, 0.0), (9.0, 20.0)),))

    report = validate_plan(_plan(clip))

    assert any("never apply" in finding.message for finding in report.warnings)


def test_keyframes_inside_the_clip_are_not_warned_about() -> None:
    clip = _clip(0, 0.0, 0.0, 4.0, effects=(_animated_crop((0.0, 0.0), (4.0, 20.0)),))

    report = validate_plan(_plan(clip))

    assert not any("never apply" in finding.message for finding in report.warnings)


@pytest.mark.integration
@requires_ffmpeg
def test_an_animated_plan_renders(make_clip: MakeClip, tmp_path: Path) -> None:
    """A moving crop renders in the ordinary single pass."""
    source = make_clip(name="anim_render", seconds=1.0)
    plan = straight_through(source)
    track = plan.timeline.tracks[0]
    clip = track.clips[0].model_copy(
        update={"effects": (_animated_crop((0.0, 0.0), (1.0, 20.0)),)}
    )
    animated = plan.model_copy(
        update={
            "timeline": plan.timeline.model_copy(
                update={"tracks": (track.model_copy(update={"clips": (clip,)}),)}
            )
        }
    )
    target = tmp_path / "plan.json"
    save_plan(animated, target)

    result = runner.invoke(app, ["render", str(target), "-o", str(tmp_path / "out.mp4")])

    assert result.exit_code == 0
    assert (tmp_path / "out.mp4").exists()
