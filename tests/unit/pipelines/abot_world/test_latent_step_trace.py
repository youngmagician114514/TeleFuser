"""Input collection boundaries for the direct whole-latent-step trace replay."""

from types import SimpleNamespace

from tools.validation.benchmark_abot_livekit_burst import ControlConfig
from tools.validation.replay_abot_latent_step_trace import build_events


def test_phase_release_keeps_latest_controls_and_cancels_empty_windows():
    events = [
        (0.0, "session_arrival", True),
        (0.9, "user_idle", False),
        (1.4, "user_active", True),
        (2.0, "session_departure", None),
    ]
    scenario = SimpleNamespace(
        seed=17,
        diagnostic_initial_control_barrier=None,
        phases=(SimpleNamespace(duration_seconds=2.0),),
        session=SimpleNamespace(control=ControlConfig(0.5, 0.0, 0.0, 0.0, 0.0, (("KeyW",),))),
        raw={
            "lifecycle_trace": {
                "kind": "explicit_session_lifecycle_v1",
                "duration_seconds": 2.0,
                "events": [
                    {
                        "offset_seconds": at,
                        "sequence": sequence,
                        "event": kind,
                        "trace_session_id": "world",
                        "input_enabled": enabled,
                    }
                    for sequence, (at, kind, enabled) in enumerate(events)
                ],
            }
        },
    )
    actual, plan = build_events(scenario, 0.5)
    assert [event.at for event in actual if event.kind == "release"] == [0.5, 1.5]
    assert all(event.controls == ("KeyW",) for event in actual if event.kind == "release")
    assert plan["phase_action_releases"] == 2
    assert build_events(scenario, 0.5) == (actual, plan)


def test_original_minimum_interval_precedes_shared_phase_collection():
    # The 1.35-second message is too early after the accepted 0.9-second
    # message, even though it would fall in a different global phase.
    scenario = SimpleNamespace(
        seed=17,
        diagnostic_initial_control_barrier=None,
        phases=(SimpleNamespace(duration_seconds=1.6),),
        session=SimpleNamespace(control=ControlConfig(0.45, 0.0, 0.0, 0.0, 0.0, (("KeyW",),))),
        raw={
            "lifecycle_trace": {
                "kind": "explicit_session_lifecycle_v1",
                "duration_seconds": 1.6,
                "events": [
                    {
                        "offset_seconds": 0.0,
                        "sequence": 0,
                        "event": "session_arrival",
                        "trace_session_id": "world",
                        "input_enabled": True,
                    },
                    {
                        "offset_seconds": 1.6,
                        "sequence": 1,
                        "event": "session_departure",
                        "trace_session_id": "world",
                        "input_enabled": None,
                    },
                ],
            }
        },
    )
    events, _ = build_events(scenario, 0.5)
    assert [event.at for event in events if event.kind == "release"] == [0.5, 1.0]
