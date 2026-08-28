from __future__ import annotations

import concurrent.futures
import threading

import pytest

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


class _DeferredExecutor(concurrent.futures.Executor):
    def __init__(self) -> None:
        self._function = None
        self._future: concurrent.futures.Future[object] | None = None

    def submit(self, fn, /, *args, **kwargs):
        if args or kwargs:
            raise AssertionError("test executor only supports the wrapped search closure")
        self._function = fn
        self._future = concurrent.futures.Future()
        return self._future

    def run(self) -> None:
        assert self._function is not None and self._future is not None
        try:
            self._future.set_result(self._function())
        except BaseException as exc:
            self._future.set_exception(exc)


def test_delayed_async_search_follows_scheduler_timeline() -> None:
    scheduler = _make_scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    executor = _DeferredExecutor()
    controller = MotivationRuntimeController(
        scheduler,
        dispatch=lambda lease: None,
        search_executor=executor,
    )

    future = controller.search_async(now=0.0)
    scheduler.submit_action("s", ["D"], now=0.1, release=True)
    executor.run()

    candidate = future.result(timeout=1.0)
    assert candidate is not None
    assert candidate.job_ids == (scheduler.session("s").pending_action.job_id,)


def test_controller_can_build_from_offline_profile_table(tmp_path) -> None:
    profile_path = tmp_path / "profile.csv"
    profile_path.write_text(
        "B,S,W,rho,precision,latency_ms,latency_p95_ms,memory_GB,Q_world,config\n"
        "1,4,18,0,bf16,400,420,20,0.68,b1_s4_w18\n"
        "4,4,18,0,bf16,1400,1450,50,0.67,b4_s4_w18\n",
        encoding="utf-8",
    )
    controller = MotivationRuntimeController.from_offline_table(
        profile_path,
        gpu_states=[GpuSchedulingState("gpu-0", memory_free_gb=80.0)],
        dispatch=lambda lease: None,
    )

    controller.on_session_registered("s", owner_gpu="gpu-0", now=0.0)
    job, invalidated = controller.on_action("s", ["W"], now=0.0)

    assert job is not None
    assert invalidated is True
    assert controller.scheduler.profile_provider.profiles_for(batch_size=1, gpu_id="gpu-0")[0].quality == pytest.approx(0.68)


def test_controller_registers_gpu_at_worker_start() -> None:
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable([MotivationProfile(1, "high", 0.4, 0.68, 20.0)])
    )
    controller = MotivationRuntimeController(scheduler, dispatch=lambda lease: None)

    state = controller.on_gpu_registered("gpu-0", memory_free_gb=80.0)
    controller.on_session_registered("s", owner_gpu="gpu-0", now=0.0)

    assert state.gpu_id == "gpu-0"
    assert scheduler.gpus()[0].memory_free_gb == pytest.approx(80.0)


def test_controller_forwards_session_lifecycle_events() -> None:
    scheduler = _make_scheduler()
    controller = MotivationRuntimeController(scheduler, dispatch=lambda lease: None)

    state = controller.on_session_registered("s", owner_gpu="gpu-0", now=0.0)
    assert state.session_id == "s"
    controller.on_action("s", ["W"], now=0.0)
    controller.on_session_departed("s", now=0.1)

    assert scheduler.session("s").departed is True
    assert scheduler.session("s").pending_action is None


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
