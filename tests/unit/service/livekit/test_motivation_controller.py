from __future__ import annotations

import concurrent.futures
import threading

import pytest

from telefuser.service.livekit.async_migration import AsyncMigrationManager, MigrationRequest
from telefuser.service.livekit.migration_hysteresis import MigrationCooldownPolicy
from telefuser.service.livekit.motivation_controller import MotivationRuntimeController
from telefuser.service.livekit.motivation_diagnostics import MotivationDiagnosticsCollector
from telefuser.service.livekit.motivation_scheduler import (
    GpuSchedulingState,
    MotivationProfile,
    MotivationScheduler,
    MotivationSchedulerConfig,
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


def test_controller_rejects_physical_owner_mismatch_before_reservation() -> None:
    diagnostics = MotivationDiagnosticsCollector()
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable([MotivationProfile(1, "high", 0.4, 0.68, 20.0)]),
        diagnostics=diagnostics,
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", free_at=0.0, memory_free_gb=80.0))
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    leases = []
    controller = MotivationRuntimeController(
        scheduler,
        dispatch=leases.append,
        dispatch_owner_resolver=lambda _session_id: "gpu-1",
    )
    candidate = scheduler.find_best(now=0.0, include_wait=False)
    assert candidate is not None

    assert controller.dispatch_candidate(candidate, now=0.0) is None
    assert leases == []
    assert scheduler.session("s").pending_action is not None
    assert scheduler.session("s").in_flight is None
    assert scheduler.gpus()[0].memory_free_gb == pytest.approx(80.0)
    assert controller.diagnostics_snapshot()["dispatch_reasons"]["owner_mismatch"] == 1


def test_dispatch_exception_rolls_back_job_and_gpu_reservation() -> None:
    scheduler = _make_scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)

    def fail_dispatch(_lease) -> None:
        raise RuntimeError("transport unavailable")

    controller = MotivationRuntimeController(scheduler, dispatch=fail_dispatch)
    with pytest.raises(RuntimeError, match="transport unavailable"):
        controller.schedule_once(now=0.0)

    assert scheduler.session("s").pending_action is not None
    assert scheduler.session("s").in_flight is None
    gpu = scheduler.gpus()[0]
    assert gpu.free_at == pytest.approx(0.0)
    assert gpu.memory_free_gb == pytest.approx(80.0)

    leases = []
    controller.set_dispatch_callback(leases.append)
    retry = controller.schedule_once(now=0.0)
    assert retry is not None
    assert leases == [retry]


class _CallbackBackend:
    def __init__(self) -> None:
        self.future: concurrent.futures.Future[object] = concurrent.futures.Future()
        self.committed = False

    def begin(self, request: MigrationRequest) -> concurrent.futures.Future[object]:
        del request
        return self.future

    @staticmethod
    def ready(operation: object) -> bool:
        return isinstance(operation, concurrent.futures.Future) and operation.done()

    def commit(self, operation: object) -> None:
        assert operation is self.future
        self.future.result()
        self.committed = True

    def abort(self, operation: object) -> None:
        if isinstance(operation, concurrent.futures.Future):
            operation.cancel()

    def add_done_callback(self, callback) -> None:
        self.future.add_done_callback(callback)


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


class _ProgressiveCallbackBackend:
    def __init__(self) -> None:
        self.compute_ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        self.residual: concurrent.futures.Future[None] = concurrent.futures.Future()

    def begin(self, request: MigrationRequest) -> object:
        del request
        return self

    def ready(self, operation: object) -> bool:
        assert operation is self
        return self.compute_ready.done()

    def commit(self, operation: object) -> None:
        assert operation is self
        self.compute_ready.result()

    def done(self, operation: object) -> bool:
        assert operation is self
        return self.residual.done()

    def finalize(self, operation: object) -> None:
        assert operation is self
        self.residual.result()

    def abort(self, operation: object) -> None:
        assert operation is self
        self.residual.cancel()

    def add_ready_callback(self, callback) -> None:
        self.compute_ready.add_done_callback(callback)

    def add_done_callback(self, callback) -> None:
        self.residual.add_done_callback(callback)


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


def test_controller_defers_busy_gpu_candidate_without_dispatch() -> None:
    scheduler = _make_scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    scheduler.update_gpu("gpu-0", free_at=1.0, now=0.0)
    busy_gpu = scheduler.gpus()[0]
    candidate = scheduler.find_best(now=0.0, gpu_states=(busy_gpu,), include_wait=False)
    assert candidate is not None
    assert candidate.start_at == pytest.approx(1.0)

    leases = []
    controller = MotivationRuntimeController(scheduler, dispatch=leases.append)

    assert controller.dispatch_candidate(candidate, now=0.0) is None
    assert leases == []
    assert scheduler.session("s").pending_action is not None
    assert scheduler.session("s").in_flight is None

    lease = controller.dispatch_candidate(candidate, now=1.0)
    assert lease is not None
    assert leases == [lease]


def test_schedule_once_falls_back_to_free_gpu_after_busy_candidate() -> None:
    profile = MotivationProfile(1, "high", 0.4, 0.68, 20.0)
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable([profile]),
        config=MotivationSchedulerConfig(migration_enabled=False),
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", free_at=0.5, memory_free_gb=80.0))
    scheduler.add_gpu(GpuSchedulingState("gpu-1", free_at=0.0, memory_free_gb=80.0))
    scheduler.register_session(
        "busy", owner_gpu="gpu-0", now=0.0, slack_seconds=0.0, compatibility_key=(0,)
    )
    scheduler.register_session(
        "local", owner_gpu="gpu-1", now=0.0, slack_seconds=10.0, compatibility_key=(1,)
    )
    scheduler.submit_action("busy", ["W"], now=0.0)
    scheduler.submit_action("local", ["D"], now=0.0)
    candidate = scheduler.find_best(now=0.0, include_wait=False)
    assert candidate is not None
    assert candidate.session_ids == ("busy",)
    assert candidate.gpu_id == "gpu-0"
    assert candidate.start_at == pytest.approx(0.5)

    leases = []
    controller = MotivationRuntimeController(scheduler, dispatch=leases.append)

    lease = controller.schedule_once(now=0.0)

    assert lease is not None
    assert lease.candidate.session_ids == ("local",)
    assert lease.candidate.gpu_id == "gpu-1"
    assert leases == [lease]
    assert scheduler.session("busy").pending_action is not None


def test_controller_blocks_external_snapshot_when_internal_gpu_is_busy() -> None:
    scheduler = _make_scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    scheduler.update_gpu("gpu-0", free_at=1.0, now=0.0)
    internal_gpu = scheduler.gpus()[0]
    stale_snapshot = GpuSchedulingState(
        "gpu-0", free_at=0.0, memory_free_gb=80.0, version=internal_gpu.version
    )
    candidate = scheduler.find_best(now=0.0, gpu_states=(stale_snapshot,), include_wait=False)
    assert candidate is not None
    assert candidate.start_at == pytest.approx(0.0)

    leases = []
    controller = MotivationRuntimeController(scheduler, dispatch=leases.append)

    assert controller.dispatch_candidate(candidate, now=0.0) is None
    assert leases == []
    assert scheduler.session("s").pending_action is not None
    assert scheduler.session("s").in_flight is None


def test_controller_prepares_async_migration_before_remote_dispatch() -> None:
    profiles = [MotivationProfile(1, "high", 0.4, 0.68, 20.0)]
    scheduler = MotivationScheduler(StaticMotivationProfileTable(profiles))
    scheduler.add_gpu(GpuSchedulingState("gpu-0", free_at=10.0, memory_free_gb=80.0))
    scheduler.add_gpu(GpuSchedulingState("gpu-1", free_at=0.0, memory_free_gb=80.0))
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    manager = AsyncMigrationManager(clock=lambda: 0.0)
    backend = _ReadyBackend()
    policy = MigrationCooldownPolicy(cooldown_seconds=60.0, clock=lambda: 0.0)
    leases = []
    controller = MotivationRuntimeController(
        scheduler,
        dispatch=leases.append,
        migration_manager=manager,
        migration_backend_factory=lambda request: backend,
        migration_policy=policy,
    )

    assert controller.schedule_once(now=0.0) is None
    assert scheduler.session("s").owner_gpu == "gpu-0"
    assert manager.active()[0].state == "precopied"
    assert controller.pending_migration_sessions() == ("s",)

    backend.ready_event.set()
    assert controller.pending_migration_sessions() == ()
    lease = controller.schedule_once(now=0.1)
    assert backend.committed is True
    assert scheduler.session("s").owner_gpu == "gpu-1"
    assert policy.is_blocked("s", now=0.1)
    assert policy.snapshot(now=0.1)["commits_total"] == 1

    assert lease is not None
    assert lease.candidate.gpu_id == "gpu-1"
    assert leases == [lease]


def test_async_migration_completion_wakes_policy_search() -> None:
    profiles = [MotivationProfile(1, "high", 0.4, 0.68, 20.0)]
    scheduler = MotivationScheduler(StaticMotivationProfileTable(profiles))
    scheduler.add_gpu(GpuSchedulingState("gpu-0", free_at=10.0, memory_free_gb=80.0))
    scheduler.add_gpu(GpuSchedulingState("gpu-1", free_at=0.0, memory_free_gb=80.0))
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    manager = AsyncMigrationManager(clock=lambda: 0.0)
    backend = _CallbackBackend()
    policy = MigrationCooldownPolicy(cooldown_seconds=60.0, clock=lambda: 0.0)
    controller = MotivationRuntimeController(
        scheduler,
        dispatch=lambda lease: None,
        migration_manager=manager,
        migration_backend_factory=lambda request: backend,
        migration_policy=policy,
    )
    woken = threading.Event()
    controller.set_migration_wakeup_callback(woken.set)

    assert controller.schedule_once(now=0.0) is None
    backend.future.set_result(None)

    assert woken.wait(timeout=1.0)
    assert scheduler.session("s").owner_gpu == "gpu-1"
    assert policy.is_blocked("s", now=0.0)
    controller.close()


def test_local_candidate_progresses_while_remote_migration_waits() -> None:
    profiles = [MotivationProfile(1, "high", 0.4, 0.68, 20.0)]
    scheduler = MotivationScheduler(StaticMotivationProfileTable(profiles))
    scheduler.add_gpu(GpuSchedulingState("gpu-0", free_at=10.0, memory_free_gb=80.0))
    scheduler.add_gpu(GpuSchedulingState("gpu-1", free_at=0.0, memory_free_gb=80.0))
    scheduler.register_session("remote", owner_gpu="gpu-0", now=0.0, slack_seconds=-1.0)
    scheduler.register_session("local", owner_gpu="gpu-1", now=0.0, slack_seconds=10.0)
    scheduler.submit_action("remote", ["W"], now=0.0)
    scheduler.submit_action("local", ["D"], now=0.0)
    manager = AsyncMigrationManager(clock=lambda: 0.0)
    backend = _ReadyBackend()
    leases = []
    controller = MotivationRuntimeController(
        scheduler,
        dispatch=leases.append,
        migration_manager=manager,
        migration_backend_factory=lambda request: backend,
    )

    lease = controller.schedule_once(now=0.0)

    assert lease is not None
    assert lease.jobs[0].session_id == "local"
    assert manager.active()[0].request.session_id == "remote"
    assert scheduler.session("remote").pending_action is not None
    controller.close()


def test_failed_migration_clears_scheduler_hint() -> None:
    profiles = [MotivationProfile(1, "high", 0.4, 0.68, 20.0)]
    scheduler = MotivationScheduler(StaticMotivationProfileTable(profiles))
    scheduler.add_gpu(GpuSchedulingState("gpu-0", free_at=10.0, memory_free_gb=80.0))
    scheduler.add_gpu(GpuSchedulingState("gpu-1", free_at=0.0, memory_free_gb=80.0))
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    manager = AsyncMigrationManager(clock=lambda: 0.0)
    backend = _CallbackBackend()
    controller = MotivationRuntimeController(
        scheduler,
        dispatch=lambda lease: None,
        migration_manager=manager,
        migration_backend_factory=lambda request: backend,
    )

    assert controller.schedule_once(now=0.0) is None
    backend.future.set_exception(RuntimeError("copy failed"))
    assert controller.schedule_once(now=0.1) is None

    assert manager.active() == ()
    assert scheduler.session("s").migration_target_gpu is None
    controller.close()


def test_progressive_residual_failure_rolls_policy_owner_back_to_source() -> None:
    profile = MotivationProfile(1, "high", 0.4, 0.68, 20.0)
    scheduler = MotivationScheduler(StaticMotivationProfileTable([profile]))
    scheduler.add_gpu(GpuSchedulingState("gpu-0", free_at=10.0, memory_free_gb=80.0))
    scheduler.add_gpu(GpuSchedulingState("gpu-1", free_at=0.0, memory_free_gb=80.0))
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    manager = AsyncMigrationManager(clock=lambda: 0.0)
    backend = _ProgressiveCallbackBackend()
    controller = MotivationRuntimeController(
        scheduler,
        dispatch=lambda lease: None,
        migration_manager=manager,
        migration_backend_factory=lambda request: backend,
    )

    assert controller.schedule_once(now=0.0) is None
    backend.compute_ready.set_result(None)

    assert scheduler.session("s").owner_gpu == "gpu-1"
    assert manager.active()[0].state == "streaming"
    assert controller.pending_migration_sessions() == ()

    backend.residual.set_exception(RuntimeError("late NCCL failure"))

    assert manager.active() == ()
    assert scheduler.session("s").owner_gpu == "gpu-0"
    assert scheduler.session("s").migration_target_gpu is None
    controller.close()
