from __future__ import annotations

import pytest

from telefuser.service.livekit.motivation_controller import MotivationRuntimeController
from telefuser.service.livekit.motivation_execution import MotivationExecutionBridge
from telefuser.service.livekit.motivation_scheduler import (
    GpuSchedulingState,
    MotivationProfile,
    MotivationScheduler,
    MotivationSchedulerConfig,
    StaticMotivationProfileTable,
)


def _controller() -> MotivationRuntimeController:
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable([MotivationProfile(1, "b1_s4_w18", 0.4, 0.68, 20.0)])
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=80.0))
    return MotivationRuntimeController(scheduler, dispatch=lambda lease: None, clock=lambda: 0.0)


def _batch_controller(now: list[float]) -> MotivationRuntimeController:
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable(
            (
                MotivationProfile(1, "b1_s4_w18_rho0_bf16", 0.4102, 0.68, 28.13),
                MotivationProfile(2, "b2_s4_w18_rho0_bf16", 0.7386, 0.68, 37.12),
            )
        )
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=80.0))
    return MotivationRuntimeController(scheduler, dispatch=lambda lease: None, clock=lambda: now[0])


class _FakeTimer:
    instances: list["_FakeTimer"] = []

    def __init__(self, interval: float, callback, *, args=()) -> None:
        self.interval = interval
        self.callback = callback
        self.args = tuple(args)
        self.cancelled = False
        self.started = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.callback(*self.args)


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


def test_bridge_dispatch_exception_clears_lease_bookkeeping_and_retries() -> None:
    controller = _controller()

    def fail_dispatch(_lease, _payloads) -> None:
        raise RuntimeError("worker queue closed")

    bridge = MotivationExecutionBridge(controller, dispatch=fail_dispatch, clock=lambda: 0.0)
    bridge.register_session("session-1", owner_gpu="gpu-0", now=0.0)

    assert bridge.on_control_message(
        "worker-0",
        "session-1",
        {"type": "control_state", "controls": ["W"]},
    )
    assert bridge._leases == {}
    assert bridge._job_to_lease == {}
    assert controller.scheduler.session("session-1").pending_action is not None
    assert controller.scheduler.session("session-1").in_flight is None

    dispatched = []
    bridge.dispatch = lambda lease, payloads: dispatched.append((lease, payloads))
    bridge.schedule_wakeup()
    assert len(dispatched) == 1
    assert controller.scheduler.session("session-1").in_flight is not None


def test_bridge_dispatches_idle_and_waits_for_published_video_before_next_idle() -> None:
    now = [0.0]
    controller = _controller()
    dispatched = []
    bridge = MotivationExecutionBridge(
        controller,
        dispatch=lambda lease, payloads: dispatched.append((lease, payloads)),
        clock=lambda: now[0],
    )
    bridge.register_session("session-1", owner_gpu="gpu-0", now=0.0)
    bridge.register_pipeline_session("session-1", "pipeline-1")

    bridge.on_session_ready("session-1")
    assert len(dispatched) == 1
    idle_lease, idle_payloads = dispatched[0]
    assert idle_lease.jobs[0].kind == "idle"
    assert idle_payloads[0][1]["controls"] == []
    assert idle_payloads[0][1]["motivation"]["kind"] == "idle"

    bridge.on_model_output("worker-0", "pipeline-1", {"type": "chunk"})
    assert controller.scheduler.session("session-1").idle_video_outstanding is True
    assert len(dispatched) == 1

    now[0] = 0.5
    bridge.on_chunk_published("worker-0", "session-1", 12)
    assert len(dispatched) == 1

    now[0] = 1.1
    bridge.on_chunk_published("worker-0", "session-1", 12)
    assert len(dispatched) == 2
    assert dispatched[1][0].jobs[0].kind == "idle"


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


def test_bridge_preserves_falsey_injected_batch_gate() -> None:
    class _FalseyGate:
        def __bool__(self) -> bool:
            return False

        def wait_seconds(self, *, gpu_id: str, fidelity: str | None = None) -> float:
            del gpu_id, fidelity
            return 0.0

    controller = _controller()
    gate = _FalseyGate()
    bridge = MotivationExecutionBridge(
        controller,
        dispatch=lambda lease, payloads: None,
        batch_gate=gate,
        clock=lambda: 0.0,
    )

    assert bridge._batch_gate is gate
    bridge.close()


def test_bridge_caps_default_profile_window_at_heartbeat_interval(monkeypatch) -> None:
    _FakeTimer.instances.clear()
    monkeypatch.setattr("telefuser.service.livekit.motivation_execution.threading.Timer", _FakeTimer)
    now = [0.0]
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable(
            (
                MotivationProfile(1, "b1_s4_w18_rho0_bf16", 0.80, 0.68, 28.13),
                MotivationProfile(2, "b2_s4_w18_rho0_bf16", 0.10, 0.68, 37.12),
            )
        )
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=80.0))
    controller = MotivationRuntimeController(scheduler, dispatch=lambda lease: None, clock=lambda: now[0])
    bridge = MotivationExecutionBridge(
        controller,
        dispatch=lambda lease, payloads: None,
        heartbeat_interval_seconds=0.25,
        clock=lambda: now[0],
    )
    bridge.register_session("session-1", owner_gpu="gpu-0", now=0.0)

    bridge.on_control_message("worker-0", "session-1", {"type": "control_state", "controls": ["W"]})

    assert len(_FakeTimer.instances) == 1
    assert _FakeTimer.instances[0].interval == pytest.approx(0.25)
    bridge.close()


def test_bridge_uses_profile_window_for_singleton_and_dispatches_on_expiry(monkeypatch) -> None:
    _FakeTimer.instances.clear()
    monkeypatch.setattr("telefuser.service.livekit.motivation_execution.threading.Timer", _FakeTimer)
    now = [0.0]
    controller = _batch_controller(now)
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
    assert dispatched == []
    assert len(_FakeTimer.instances) == 1
    timer = _FakeTimer.instances[0]
    assert timer.started is True
    assert timer.interval == pytest.approx(2 * 0.4102 - 0.7386)
    gate_snapshot = bridge.snapshot()["batch_gate"]
    assert gate_snapshot["enabled"] is True
    assert gate_snapshot["armed"] is True
    assert gate_snapshot["wait_count"] == 1
    assert gate_snapshot["remaining_seconds"] == pytest.approx(timer.interval)

    timer.fire()

    assert len(dispatched) == 1
    assert dispatched[0][0].candidate.batch_size == 1
    bridge.close()


def test_bridge_cancels_singleton_gate_when_a_compatible_peer_arrives(monkeypatch) -> None:
    _FakeTimer.instances.clear()
    monkeypatch.setattr("telefuser.service.livekit.motivation_execution.threading.Timer", _FakeTimer)
    now = [0.0]
    controller = _batch_controller(now)
    dispatched = []
    bridge = MotivationExecutionBridge(
        controller,
        dispatch=lambda lease, payloads: dispatched.append((lease, payloads)),
        clock=lambda: now[0],
    )
    for session_id in ("session-1", "session-2"):
        bridge.register_session(session_id, owner_gpu="gpu-0", now=0.0)

    bridge.on_control_message("worker-0", "session-1", {"type": "control_state", "controls": ["W"]})
    first_timer = _FakeTimer.instances[-1]
    now[0] = 0.02
    bridge.on_control_message("worker-0", "session-2", {"type": "control_state", "controls": ["D"]})

    assert first_timer.cancelled is True
    assert len(dispatched) == 1
    assert dispatched[0][0].candidate.batch_size == 2
    bridge.close()


def test_bridge_skips_gate_when_slack_would_be_exhausted(monkeypatch) -> None:
    _FakeTimer.instances.clear()
    monkeypatch.setattr("telefuser.service.livekit.motivation_execution.threading.Timer", _FakeTimer)
    now = [0.0]
    controller = _batch_controller(now)
    controller.on_session_registered("urgent", owner_gpu="gpu-0", now=0.0, slack_seconds=0.05)
    dispatched = []
    bridge = MotivationExecutionBridge(
        controller,
        dispatch=lambda lease, payloads: dispatched.append((lease, payloads)),
        clock=lambda: now[0],
    )
    bridge.on_control_message("worker-0", "urgent", {"type": "control_state", "controls": ["W"]})

    assert len(dispatched) == 1
    assert _FakeTimer.instances == []
    bridge.close()


def test_bridge_checks_unselected_session_slack_before_waiting(monkeypatch) -> None:
    _FakeTimer.instances.clear()
    monkeypatch.setattr("telefuser.service.livekit.motivation_execution.threading.Timer", _FakeTimer)
    now = [0.0]
    controller = _batch_controller(now)
    controller.on_session_registered("urgent", owner_gpu="gpu-0", now=0.0, slack_seconds=0.05)
    controller.on_session_registered("normal", owner_gpu="gpu-0", now=0.0, slack_seconds=1.0)
    dispatched = []
    bridge = MotivationExecutionBridge(
        controller,
        dispatch=lambda lease, payloads: dispatched.append((lease, payloads)),
        clock=lambda: now[0],
    )
    bridge.on_control_message("worker-0", "normal", {"type": "control_state", "controls": ["W"]})

    assert len(dispatched) == 1
    assert dispatched[0][0].jobs[0].session_id == "normal"
    assert _FakeTimer.instances == []
    bridge.close()



def _multi_gpu_controller(now: list[float], gpu_count: int = 2) -> MotivationRuntimeController:
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable(
            (MotivationProfile(1, "b1_s4_w18", 0.4, 0.68, 20.0),)
        )
    )
    for index in range(gpu_count):
        scheduler.add_gpu(GpuSchedulingState(f"gpu-{index}", memory_free_gb=80.0))
    return MotivationRuntimeController(scheduler, dispatch=lambda lease: None, clock=lambda: now[0])


def test_bridge_drains_immediately_free_gpus_after_one_event() -> None:
    now = [0.0]
    controller = _multi_gpu_controller(now)
    dispatched = []
    bridge = MotivationExecutionBridge(
        controller,
        dispatch=lambda lease, payloads: dispatched.append((lease, payloads)),
        clock=lambda: now[0],
    )
    for index, session_id in enumerate(("session-1", "session-2")):
        bridge.register_session(session_id, owner_gpu=f"gpu-{index}", now=0.0)
    # Seed both ready action jobs before one worker-free event. With no B2
    # profile the gate is disabled, so the event should fill both GPUs.
    controller.on_action("session-1", ["W"], now=0.0)
    controller.on_action("session-2", ["D"], now=0.0)

    bridge.schedule_wakeup()

    assert len(dispatched) == 2
    assert {lease.candidate.gpu_id for lease, _ in dispatched} == {"gpu-0", "gpu-1"}
    assert bridge.snapshot()["immediate_drain"]["last_rounds"] == 2
    bridge.close()


def test_bridge_coalesces_reentrant_schedule_requests_and_bounds_drain() -> None:
    now = [0.0]
    controller = _multi_gpu_controller(now)
    dispatched = []
    bridge_ref: dict[str, MotivationExecutionBridge] = {}

    def dispatch(lease, payloads) -> None:
        dispatched.append((lease, payloads))
        # Simulate a synchronous worker-free callback from the transport.
        bridge_ref["bridge"].schedule_wakeup()

    bridge = MotivationExecutionBridge(controller, dispatch=dispatch, clock=lambda: now[0])
    bridge_ref["bridge"] = bridge
    for index in range(3):
        session_id = f"session-{index}"
        bridge.register_session(session_id, owner_gpu=f"gpu-{index % 2}", now=0.0)
        controller.on_action(session_id, ["W"], now=0.0)

    bridge.schedule_wakeup()

    # The two-GPU bound prevents recursive callback storms from dispatching an
    # unbounded sequence; the third job remains pending for a later event.
    assert len(dispatched) == 2
    assert controller.scheduler.session("session-2").pending_action is not None
    drain = bridge.snapshot()["immediate_drain"]
    assert drain["reentrant_events"] >= 2
    bridge.close()



def test_model_output_event_drains_other_free_gpus() -> None:
    now = [0.0]
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable(
            (MotivationProfile(1, "b1_s4_w18", 0.4, 0.68, 20.0),)
        ),
        config=MotivationSchedulerConfig(include_idle_jobs=False),
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=80.0))
    scheduler.add_gpu(GpuSchedulingState("gpu-1", memory_free_gb=80.0))
    controller = MotivationRuntimeController(scheduler, dispatch=lambda lease: None, clock=lambda: now[0])
    dispatched = []
    bridge = MotivationExecutionBridge(
        controller,
        dispatch=lambda lease, payloads: dispatched.append((lease, payloads)),
        clock=lambda: now[0],
    )
    for index in range(3):
        session_id = f"session-{index}"
        bridge.register_session(session_id, owner_gpu=f"gpu-{index % 2}", now=0.0)
        bridge.register_pipeline_session(session_id, f"pipeline-{index}")

    controller.on_action("session-0", ["W"], now=0.0)
    bridge.schedule_wakeup()
    assert len(dispatched) == 1
    controller.on_action("session-1", ["D"], now=0.0)
    controller.on_action("session-2", ["A"], now=0.0)

    now[0] = 0.5
    bridge.on_model_output("worker-0", "pipeline-0", {"type": "chunk"})

    assert len(dispatched) == 3
    assert {lease.candidate.gpu_id for lease, _ in dispatched[1:]} == {"gpu-0", "gpu-1"}
    bridge.close()
