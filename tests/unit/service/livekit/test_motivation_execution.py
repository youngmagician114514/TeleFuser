from __future__ import annotations

from telefuser.service.livekit.motivation_controller import MotivationRuntimeController
from telefuser.service.livekit.motivation_execution import MotivationExecutionBridge
from telefuser.service.livekit.motivation_scheduler import (
    GpuSchedulingState,
    MotivationProfile,
    MotivationScheduler,
    StaticMotivationProfileTable,
)


def _controller() -> MotivationRuntimeController:
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable([MotivationProfile(1, "b1_s4_w18", 0.4, 0.68, 20.0)])
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=80.0))
    return MotivationRuntimeController(scheduler, dispatch=lambda lease: None, clock=lambda: 0.0)


def test_bridge_releases_action_and_commits_on_model_output() -> None:
    controller = _controller()
    dispatched = []
    bridge = MotivationExecutionBridge(
        controller, dispatch=lambda lease, payloads: dispatched.append((lease, payloads)), clock=lambda: 0.0
    )
    bridge.register_session("session-1", owner_gpu="gpu-0", now=0.0)
    bridge.register_pipeline_session("session-1", "pipeline-1")

    assert bridge.on_control_message(
        "worker-0",
        "session-1",
        {"type": "control_state", "controls": ["W"]},
    )
    assert len(dispatched) == 1
    lease, payloads = dispatched[0]
    assert payloads[0][0] == "session-1"
    assert payloads[0][1]["controls"] == ["W"]
    assert payloads[0][1]["motivation"]["fidelity"] == "b1_s4_w18"
    assert controller.scheduler.session("session-1").in_flight is not None

    bridge.on_model_output("worker-0", "pipeline-1", {"type": "chunk"})
    assert controller.scheduler.session("session-1").in_flight is None
    assert controller.scheduler.session("session-1").slack_seconds > 1.0
    assert lease.candidate.batch_size == 1


def test_bridge_applies_one_second_gate_to_default_control_states() -> None:
    now = [0.0]
    controller = _controller()
    dispatched = []
    bridge = MotivationExecutionBridge(
        controller,
        dispatch=lambda lease, payloads: dispatched.append((lease, payloads)),
        clock=lambda: now[0],
    )
    bridge.register_session("session-1", owner_gpu="gpu-0", now=0.0)

    assert bridge.on_control_message(
        "worker-0", "session-1", {"type": "control_state", "controls": ["W"]}
    )
    now[0] = 0.4
    assert bridge.on_control_message(
        "worker-0", "session-1", {"type": "control_state", "controls": ["D"]}
    )
    assert len(dispatched) == 1
    assert dispatched[0][1][0][1]["controls"] == ["W"]

    now[0] = 1.0
    assert bridge.on_control_message(
        "worker-0", "session-1", {"type": "control_state", "controls": ["D"]}
    )
    assert len(dispatched) == 1
    # The replacement is pending and is released by the scheduler after the
    # current one-shot invocation completes.
    assert controller.scheduler.session("session-1").pending_action is not None


def test_bridge_keeps_intermediate_updates_in_the_latest_state_only() -> None:
    controller = _controller()
    dispatched = []
    bridge = MotivationExecutionBridge(
        controller,
        dispatch=lambda lease, payloads: dispatched.append((lease, payloads)),
        release_policy=lambda chunk: bool(chunk.get("release")),
        clock=lambda: 0.0,
    )
    bridge.register_session("session-1", owner_gpu="gpu-0", now=0.0)

    assert bridge.on_control_message(
        "worker-0",
        "session-1",
        {"type": "control_state", "controls": ["W"], "release": False},
    )
    assert dispatched == []
    assert controller.scheduler.session("session-1").pending_action is None

    assert bridge.on_control_message(
        "worker-0",
        "session-1",
        {"type": "control_state", "controls": ["D"], "release": True},
    )
    assert len(dispatched) == 1
    assert dispatched[0][1][0][1]["controls"] == ["D"]


def test_bridge_forwards_empty_state_to_stop_model_admission() -> None:
    controller = _controller()
    dispatched = []
    bridge = MotivationExecutionBridge(
        controller, dispatch=lambda lease, payloads: dispatched.append(payloads), clock=lambda: 0.0
    )
    bridge.register_session("session-1", owner_gpu="gpu-0", now=0.0)

    assert bridge.on_control_message(
        "worker-0",
        "session-1",
        {"type": "control_state", "controls": []},
    ) is False
    assert dispatched == []
    assert controller.scheduler.session("session-1").pending_action is None
