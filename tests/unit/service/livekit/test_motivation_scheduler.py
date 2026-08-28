from __future__ import annotations

import threading

import pytest

from telefuser.service.livekit.async_migration import (
    AsyncMigrationManager,
    MigrationRequest,
)
from telefuser.service.livekit.motivation_scheduler import (
    GpuSchedulingState,
    MotivationProfile,
    MotivationScheduler,
    MotivationSchedulerConfig,
    StaticMotivationProfileTable,
)


def _scheduler(*, max_batch_size: int = 4) -> MotivationScheduler:
    profiles = [
        MotivationProfile(1, "high", 0.40, 0.68, 20.0),
        MotivationProfile(2, "high", 0.72, 0.68, 30.0),
        MotivationProfile(3, "high", 1.05, 0.67, 40.0),
        MotivationProfile(4, "high", 1.40, 0.66, 50.0),
        MotivationProfile(1, "low", 0.25, 0.62, 20.0),
        MotivationProfile(2, "low", 0.45, 0.62, 30.0),
    ]
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable(profiles),
        config=MotivationSchedulerConfig(max_batch_size=max_batch_size),
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=80.0))
    scheduler.add_gpu(GpuSchedulingState("gpu-1", memory_free_gb=80.0))
    return scheduler


def test_action_updates_replace_pending_without_releasing_a_second_job() -> None:
    scheduler = _scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)

    first, invalidated = scheduler.submit_action("s", ["W"], now=0.0, release=True)
    second, replacement_invalidated = scheduler.submit_action("s", ["D"], now=0.1, release=True)

    assert first is not None and second is not None
    assert second.job_id != first.job_id
    assert second.controls == ("D",)
    assert invalidated is True
    assert replacement_invalidated is False
    assert scheduler.session("s").pending_action == second


def test_action_arriving_while_running_waits_for_next_slot() -> None:
    scheduler = _scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    first, _ = scheduler.submit_action("s", ["W"], now=0.0)
    assert first is not None
    candidate = scheduler.find_best(now=0.0, include_wait=False)
    assert candidate is not None
    scheduler.reserve(candidate, now=0.0)

    next_job, invalidated = scheduler.submit_action("s", ["J"], now=0.1, release=True)

    assert next_job is not None
    assert invalidated is False
    assert scheduler.session("s").in_flight == first
    assert scheduler.session("s").pending_action == next_job


def test_idle_video_is_consumption_gated_and_does_not_get_overwritten() -> None:
    scheduler = _scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    idle = scheduler.create_idle_job("s", now=0.0)
    assert idle is not None
    candidate = scheduler.find_best(now=0.0, include_wait=False)
    assert candidate is not None and candidate.job_ids == (idle.job_id,)
    scheduler.reserve(candidate, now=0.0)

    action, _ = scheduler.submit_action("s", ["W"], now=0.1, release=True)
    scheduler.complete(candidate, completed_at=0.2)

    state = scheduler.session("s")
    assert action is not None
    assert state.pending_action == action
    assert state.idle_video_outstanding is True
    assert scheduler.create_idle_job("s", now=0.2) is None

    scheduler.set_playback_active("s", True, now=1.2)
    assert state.idle_video_outstanding is False
    # The action remains pending; it still has priority over a new sentinel.
    assert scheduler.create_idle_job("s", now=1.2) is None


def test_candidate_uses_global_slack_and_same_compatibility_key() -> None:
    scheduler = _scheduler(max_batch_size=4)
    scheduler.register_session("a", owner_gpu="gpu-0", now=0.0, slack_seconds=0.5, compatibility_key=(3,))
    scheduler.register_session("b", owner_gpu="gpu-0", now=0.0, slack_seconds=3.0, compatibility_key=(3,))
    scheduler.register_session("c", owner_gpu="gpu-0", now=0.0, slack_seconds=3.0, compatibility_key=(4,))
    for sid, controls in (("a", ["W"]), ("b", ["D"]), ("c", ["J"])):
        scheduler.submit_action(sid, controls, now=0.0)

    candidate = scheduler.find_best(now=0.0, include_wait=False)

    assert candidate is not None
    assert set(candidate.session_ids) == {"a", "b"}
    assert candidate.batch_size == 2
    assert candidate.projected_slack["c"] < 3.0


def test_candidate_is_rejected_after_pending_job_version_changes() -> None:
    scheduler = _scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    candidate = scheduler.find_best(now=0.0, include_wait=False)
    assert candidate is not None
    scheduler.submit_action("s", ["D"], now=0.1, release=True)

    assert scheduler.validate(candidate, now=0.1) is False
    with pytest.raises(RuntimeError, match="stale"):
        scheduler.reserve(candidate, now=0.1)


class _FakeMigrationBackend:
    def __init__(self) -> None:
        self.ready_event = threading.Event()
        self.committed = False
        self.aborted = False

    def begin(self, request: MigrationRequest) -> object:
        return request

    def ready(self, operation: object) -> bool:
        return self.ready_event.is_set()

    def commit(self, operation: object) -> None:
        self.committed = True

    def abort(self, operation: object) -> None:
        self.aborted = True


def test_async_migration_requires_ready_before_atomic_commit() -> None:
    backend = _FakeMigrationBackend()
    manager = AsyncMigrationManager(max_concurrent=1, clock=lambda: 2.0)
    request = MigrationRequest("s", "gpu-0", "gpu-1", 1.0, 1.2, state_bytes=1024)
    record = manager.begin(request, backend)

    assert record.state == "precopied"
    assert manager.poll("s").state == "precopied"
    with pytest.raises(RuntimeError, match="not ready"):
        manager.commit("s")

    backend.ready_event.set()
    assert manager.poll("s").state == "ready"
    completed = manager.commit("s")
    assert completed.state == "completed"
    assert backend.committed is True
