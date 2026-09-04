from __future__ import annotations

import threading

import pytest

from telefuser.service.livekit.async_migration import (
    AsyncMigrationManager,
    MigrationRequest,
)
from telefuser.service.livekit.motivation_scheduler import (
    GpuSchedulingState,
    LocalMigrationEstimator,
    MotivationProfile,
    MotivationScheduler,
    MotivationSchedulerConfig,
    SessionSchedulingState,
    StaticMotivationProfileTable,
    load_motivation_profiles_csv,
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
    assert invalidated is True
    assert scheduler.session("s").in_flight == first
    assert scheduler.session("s").pending_action == next_job


def test_pending_action_waits_out_in_flight_then_becomes_runnable() -> None:
    scheduler = _scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    first, _ = scheduler.submit_action("s", ["W"], now=0.0)
    assert first is not None
    running = scheduler.find_best(now=0.0, include_wait=False)
    assert running is not None
    scheduler.reserve(running, now=0.0)

    replacement, _ = scheduler.submit_action("s", ["J"], now=0.1, release=True)
    assert replacement is not None

    assert scheduler.find_best(now=0.1, include_wait=False) is None
    assert scheduler.session("s").pending_action == replacement

    completed = scheduler.complete(running, completed_at=0.4)
    assert completed == (first,)
    resumed = scheduler.find_best(now=0.4, include_wait=False)
    assert resumed is not None
    assert resumed.session_ids == ("s",)
    assert resumed.job_ids == (replacement.job_id,)


def test_in_flight_session_does_not_block_another_ready_session() -> None:
    scheduler = _scheduler()
    scheduler.register_session("running", owner_gpu="gpu-0", now=0.0)
    scheduler.register_session("ready", owner_gpu="gpu-1", now=0.0)
    first, _ = scheduler.submit_action("running", ["W"], now=0.0)
    assert first is not None
    running = scheduler.find_best(now=0.0, include_wait=False)
    assert running is not None
    scheduler.reserve(running, now=0.0)

    replacement, _ = scheduler.submit_action("running", ["J"], now=0.1, release=True)
    other, _ = scheduler.submit_action("ready", ["D"], now=0.1, release=True)
    assert replacement is not None and other is not None

    candidate = scheduler.find_best(now=0.1, include_wait=False)
    assert candidate is not None
    assert candidate.session_ids == ("ready",)
    assert candidate.job_ids == (other.job_id,)


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


def test_action_drops_only_a_pending_idle_sentinel() -> None:
    scheduler = _scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    idle = scheduler.create_idle_job("s", now=0.0)
    assert idle is not None

    action, _ = scheduler.submit_action("s", ["W"], now=0.1, release=True)

    assert action is not None
    assert scheduler.session("s").pending_idle is None


def test_idle_can_follow_a_completed_action_before_the_next_heartbeat() -> None:
    scheduler = _scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    action, _ = scheduler.submit_action("s", ["W"], now=0.0, release=True)
    assert action is not None
    candidate = scheduler.find_best(now=0.0, include_wait=False)
    assert candidate is not None
    scheduler.reserve(candidate, now=0.0)
    scheduler.complete(candidate, completed_at=0.1)

    idle = scheduler.create_idle_job("s", now=0.1)

    assert idle is not None
    assert scheduler.session("s").latest_controls == ("W",)


def test_action_and_another_sessions_idle_head_can_share_a_batch() -> None:
    scheduler = _scheduler(max_batch_size=2)
    scheduler.register_session("action", owner_gpu="gpu-0", now=0.0, compatibility_key=(3,))
    scheduler.register_session("idle", owner_gpu="gpu-0", now=0.0, compatibility_key=(3,))
    action, _ = scheduler.submit_action("action", ["W"], now=0.0)
    idle = scheduler.create_idle_job("idle", now=0.0)

    candidate = scheduler.find_best(now=0.0, include_wait=False)

    assert action is not None and idle is not None
    assert candidate is not None
    assert set(candidate.job_ids) == {action.job_id, idle.job_id}
    assert candidate.batch_size == 2
    assert candidate.action_count == 1


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


def test_busy_gpu_candidate_cannot_be_reserved_before_predicted_start() -> None:
    scheduler = _scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    scheduler.update_gpu("gpu-0", free_at=2.0, now=0.0)
    busy_gpu = scheduler.gpus()[0]

    candidate = scheduler.find_best(now=0.0, gpu_states=(busy_gpu,), include_wait=False)

    assert candidate is not None
    assert candidate.start_at == pytest.approx(2.0)
    assert scheduler.validate(candidate, now=0.0) is True
    with pytest.raises(RuntimeError, match="not ready"):
        scheduler.reserve(candidate, now=0.0)
    assert scheduler.session("s").pending_action is not None
    assert scheduler.session("s").in_flight is None

    scheduler.reserve(candidate, now=2.0)
    assert scheduler.session("s").in_flight is not None


def test_internal_gpu_timeline_blocks_stale_external_snapshot() -> None:
    scheduler = _scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    scheduler.update_gpu("gpu-0", free_at=2.0, now=0.0)
    internal_gpu = scheduler.gpus()[0]
    stale_snapshot = GpuSchedulingState(
        "gpu-0", free_at=0.0, memory_free_gb=80.0, version=internal_gpu.version
    )

    candidate = scheduler.find_best(now=0.0, gpu_states=(stale_snapshot,), include_wait=False)

    assert candidate is not None
    assert candidate.start_at == pytest.approx(0.0)
    assert scheduler.validate(candidate, now=0.0) is True
    assert scheduler.candidate_ready_now(candidate, now=0.0) is False
    with pytest.raises(RuntimeError, match="not ready"):
        scheduler.reserve(candidate, now=0.0)
    assert scheduler.session("s").pending_action is not None
    assert scheduler.session("s").in_flight is None


def test_departure_keeps_reserved_job_until_completion() -> None:
    scheduler = _scheduler()
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    candidate = scheduler.find_best(now=0.0, include_wait=False)
    assert candidate is not None
    scheduler.reserve(candidate, now=0.0)

    scheduler.mark_departed("s", now=0.1)
    state = scheduler.session("s")
    assert state.departed is True
    assert state.in_flight is not None
    completed = scheduler.complete(candidate, completed_at=0.4)

    assert completed[0].job_id == candidate.job_ids[0]
    assert state.in_flight is None
    assert scheduler.gpus()[0].memory_free_gb == pytest.approx(80.0)


def test_new_ready_session_invalidates_global_candidate() -> None:
    scheduler = _scheduler()
    scheduler.register_session("a", owner_gpu="gpu-0", now=0.0)
    scheduler.register_session("b", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("a", ["W"], now=0.0)
    candidate = scheduler.find_best(now=0.0, include_wait=False)
    assert candidate is not None

    scheduler.submit_action("b", ["D"], now=0.1, release=True)

    assert scheduler.validate(candidate, now=0.1) is False


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


def test_profile_loader_reads_abot_offline_table(tmp_path) -> None:
    profile_path = tmp_path / "profile.csv"
    profile_path.write_text(
        "B,S,W,rho,precision,latency_ms,latency_p95_ms,memory_GB,Q_action,Q_temporal,Q_visual,Q_world,config\n"
        "1,4,18,0,bf16,400,410,28,0.4,0.8,0.9,,b1_s4_w18_rho0_bf16\n"
        "4,4,18,0,bf16,1400,1450,50,0.5,0.8,0.9,0.7,b4_s4_w18_rho0_bf16\n"
        "8,4,18,0,bf16,2600,2700,74,0.5,0.8,0.9,0.65,b8_s4_w18_rho0_bf16\n",
        encoding="utf-8",
    )

    table = load_motivation_profiles_csv(profile_path, max_batch_size=4)

    assert [row.batch_size for row in table.profiles_for(batch_size=1, gpu_id="gpu-0")] == [1]
    assert table.profiles_for(batch_size=1, gpu_id="gpu-0")[0].quality == pytest.approx(0.7)
    assert table.profiles_for(batch_size=4, gpu_id="gpu-0")[0].quality == pytest.approx(0.7)
    assert table.profiles_for(batch_size=8, gpu_id="gpu-0") == ()


def test_profile_loader_synthesizes_missing_b3_between_b2_and_b4(tmp_path) -> None:
    profile = tmp_path / "profile.csv"
    profile.write_text(
        "B,latency_ms,memory_GB,Q_world,config,latency_p95_ms\n"
        "1,100,10,0.9,b1_lane,110\n"
        "2,180,15,0.9,b2_lane,190\n"
        "4,320,25,0.8,b4_lane,340\n",
        encoding="utf-8",
    )
    table = load_motivation_profiles_csv(profile, max_batch_size=4)
    rows = table.profiles_for(batch_size=3, gpu_id=None)
    assert len(rows) == 1
    assert rows[0].fidelity == "b3_lane"
    assert rows[0].latency_seconds == pytest.approx(0.25)
    assert rows[0].p95_seconds == pytest.approx(0.265)
    assert rows[0].quality == pytest.approx(0.8)


def test_migration_estimator_charges_first_layer_not_background_drain() -> None:
    estimator = LocalMigrationEstimator(
        first_layer_ready_seconds=0.01,
        transfer_seconds=0.008,
        drain_seconds=1.3,
    )
    session = SessionSchedulingState("s", "gpu-0", slack_seconds=1.0, quality_ema=0.8)
    estimate = estimator.estimate(session, target_gpu="gpu-1", now=5.0)
    assert estimate.required is True
    assert estimate.cost_seconds == pytest.approx(0.01)
    assert estimate.first_layer_ready_seconds == pytest.approx(0.01)
    assert estimate.transfer_seconds == pytest.approx(0.008)
    assert estimate.drain_seconds == pytest.approx(1.3)
