from __future__ import annotations

import threading

from telefuser.service.livekit.async_migration import AsyncMigrationManager, MigrationRequest
from telefuser.service.livekit.motivation_controller import MotivationRuntimeController
from telefuser.service.livekit.motivation_scheduler import (
    GpuSchedulingState,
    MotivationProfile,
    MotivationScheduler,
    StaticMotivationProfileTable,
)


def _make_scheduler() -> MotivationScheduler:
    profiles = [MotivationProfile(1, "high", 0.4, 0.68, 20.0)]
    scheduler = MotivationScheduler(StaticMotivationProfileTable(profiles))
    scheduler.add_gpu(GpuSchedulingState("gpu-0", free_at=0.0, memory_free_gb=80.0))
    return scheduler


def test_controller_reserves_and_completes_local_dispatch() -> None:
    scheduler = _make_scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    leases = []
    controller = MotivationRuntimeController(scheduler, dispatch=leases.append)

    lease = controller.schedule_once(now=0.0)

    assert lease is not None
    assert leases == [lease]
    assert lease.jobs[0].kind == "action"
    assert controller.on_completion(lease, completed_at=0.4)[0].session_id == "s"


class _ReadyBackend:
    def __init__(self) -> None:
        self.ready_event = threading.Event()
        self.committed = False

    def begin(self, request: MigrationRequest) -> object:
        return request

    def ready(self, operation: object) -> bool:
        return self.ready_event.is_set()

    def commit(self, operation: object) -> None:
        self.committed = True

    def abort(self, operation: object) -> None:
        return None


def test_async_search_candidate_is_revalidated_before_dispatch() -> None:
    scheduler = _make_scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    leases = []
    controller = MotivationRuntimeController(scheduler, dispatch=leases.append)

    candidate = controller.search_async(now=0.0).result(timeout=2.0)
    assert candidate is not None
    scheduler.submit_action("s", ["D"], now=0.1, release=True)

    assert controller.dispatch_candidate(candidate, now=0.1) is None
    assert leases == []
    controller.close()


def test_controller_prepares_async_migration_before_remote_dispatch() -> None:
    profiles = [MotivationProfile(1, "high", 0.4, 0.68, 20.0)]
    scheduler = MotivationScheduler(StaticMotivationProfileTable(profiles))
    scheduler.add_gpu(GpuSchedulingState("gpu-0", free_at=10.0, memory_free_gb=80.0))
    scheduler.add_gpu(GpuSchedulingState("gpu-1", free_at=0.0, memory_free_gb=80.0))
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    manager = AsyncMigrationManager(clock=lambda: 0.0)
    backend = _ReadyBackend()
    leases = []
    controller = MotivationRuntimeController(
        scheduler,
        dispatch=leases.append,
        migration_manager=manager,
        migration_backend_factory=lambda request: backend,
    )

    assert controller.schedule_once(now=0.0) is None
    assert scheduler.session("s").owner_gpu == "gpu-0"
    assert manager.active()[0].state == "precopied"

    backend.ready_event.set()
    assert controller.schedule_once(now=0.1) is None
    assert backend.committed is True
    assert scheduler.session("s").owner_gpu == "gpu-1"

    lease = controller.schedule_once(now=0.1)
    assert lease is not None
    assert lease.candidate.gpu_id == "gpu-1"
    assert leases == [lease]
