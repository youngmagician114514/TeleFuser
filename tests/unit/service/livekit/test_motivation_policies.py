from __future__ import annotations

import pytest

from telefuser.service.livekit.motivation_diagnostics import MotivationDiagnosticsCollector
from telefuser.service.livekit.motivation_policies import (
    FIFOPolicy,
    MotivationPolicy,
    create_scheduling_policy,
)
from telefuser.service.livekit.motivation_scheduler import (
    GpuSchedulingState,
    MotivationProfile,
    MotivationScheduler,
    MotivationSchedulerConfig,
    StaticMotivationProfileTable,
)


def _scheduler(policy_name: str, *, diagnostics=None) -> MotivationScheduler:
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable([MotivationProfile(1, "dense", 0.4, 0.8, 20.0)]),
        config=MotivationSchedulerConfig(max_batch_size=1, policy_name=policy_name),
        diagnostics=diagnostics,
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=80.0))
    return scheduler


def test_policy_registry_exposes_parallel_implementations() -> None:
    assert isinstance(create_scheduling_policy("motivation"), MotivationPolicy)
    assert isinstance(create_scheduling_policy("FIFO"), FIFOPolicy)
    with pytest.raises(ValueError, match="unknown scheduling policy"):
        create_scheduling_policy("unknown")


def test_fifo_selects_oldest_released_action_instead_of_motivation_score() -> None:
    motivation = _scheduler("motivation")
    fifo = _scheduler("fifo")
    for scheduler in (motivation, fifo):
        scheduler.register_session("old", owner_gpu="gpu-0", now=0.0, slack_seconds=3.0)
        scheduler.register_session("urgent", owner_gpu="gpu-0", now=0.0, slack_seconds=0.1)
        old_job, _ = scheduler.submit_action("old", ["W"], now=0.1)
        urgent_job, _ = scheduler.submit_action("urgent", ["D"], now=0.2)
        assert old_job is not None and urgent_job is not None

    motivation_candidate = motivation.find_best(now=0.2, include_wait=False)
    fifo_candidate = fifo.find_best(now=0.2, include_wait=False)

    assert motivation_candidate is not None and motivation_candidate.session_ids == ("urgent",)
    assert fifo_candidate is not None and fifo_candidate.session_ids == ("old",)
    assert fifo_candidate.policy_name == "fifo"


def test_fifo_is_normalized_to_singleton_action_mode() -> None:
    scheduler = _scheduler("fifo")

    assert scheduler.config.max_batch_size == 1
    assert scheduler.config.include_idle_jobs is False
    assert scheduler.config.lambda_quality == 0.0
    assert scheduler.config.fairness_delta == 0.0
    assert scheduler.config.migration_enabled is False


def test_fifo_does_not_call_the_motivation_search(monkeypatch) -> None:
    scheduler = _scheduler("fifo")
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    job, _ = scheduler.submit_action("s", ["W"], now=0.0)

    def fail(*args, **kwargs):
        raise AssertionError("FIFO must not enter the Motivation search")

    monkeypatch.setattr(scheduler, "_find_best_motivation", fail)
    candidate = scheduler.find_best(now=0.0, include_wait=False)

    assert job is not None and candidate is not None
    assert candidate.batch_size == 1
    assert candidate.score == 0.0


def test_fifo_never_dispatches_idle_only_work() -> None:
    scheduler = _scheduler("fifo")
    scheduler.register_session("idle", owner_gpu="gpu-0", now=0.0)
    idle_job = scheduler.create_idle_job("idle", now=0.0)

    candidate = scheduler.find_best(now=0.0, include_wait=True)

    assert idle_job is not None
    assert candidate is None


def test_fifo_uses_the_fixed_fidelity_without_quality_selection() -> None:
    profiles = StaticMotivationProfileTable(
        [
            MotivationProfile(1, "fixed", 0.8, 0.2, 20.0),
            MotivationProfile(1, "other", 0.1, 1.0, 20.0),
        ]
    )
    scheduler = MotivationScheduler(
        profiles,
        config=MotivationSchedulerConfig(policy_name="fifo", fifo_fidelity="fixed"),
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=80.0))
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)

    candidate = scheduler.find_best(now=0.0, include_wait=False)

    assert candidate is not None
    assert candidate.fidelity == "fixed"
    assert candidate.profile is not None and candidate.profile.quality == 0.2


def test_fifo_stays_on_the_admitted_owner_without_calling_migration_estimator() -> None:
    class FailingEstimator:
        def estimate(self, *args, **kwargs):
            raise AssertionError("FIFO must not evaluate migration candidates")

    profile = MotivationProfile(1, "fixed", 0.2, 0.2, 20.0)
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable([profile]),
        config=MotivationSchedulerConfig(policy_name="fifo", fifo_fidelity="fixed"),
        migration_estimator=FailingEstimator(),
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=80.0))
    scheduler.add_gpu(GpuSchedulingState("gpu-1", memory_free_gb=80.0))
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)

    candidate = scheduler.find_best(now=0.0, include_wait=False)

    assert candidate is not None
    assert candidate.gpu_id == "gpu-0"
    assert candidate.migration_count == 0


def test_fifo_admission_does_not_scan_profile_quality() -> None:
    class CountingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def profiles_for(self, *, batch_size: int, gpu_id: str):
            del batch_size, gpu_id
            self.calls += 1
            return (MotivationProfile(1, "dense", 0.4, 0.8, 20.0),)

    provider = CountingProvider()
    scheduler = MotivationScheduler(
        provider,
        config=MotivationSchedulerConfig(policy_name="fifo"),
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=80.0))
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)

    assert provider.calls == 0


def test_fifo_prioritizes_action_over_an_older_idle_sentinel() -> None:
    scheduler = _scheduler("fifo")
    scheduler.register_session("idle", owner_gpu="gpu-0", now=0.0)
    scheduler.register_session("action", owner_gpu="gpu-0", now=0.0)
    idle_job = scheduler.create_idle_job("idle", now=0.0)
    action_job, _ = scheduler.submit_action("action", ["W"], now=0.1)

    candidate = scheduler.find_best(now=0.1, include_wait=False)

    assert idle_job is not None and action_job is not None
    assert candidate is not None
    assert candidate.session_ids == ("action",)
    assert candidate.job_ids == (action_job.job_id,)


def test_policy_name_is_present_in_diagnostics() -> None:
    diagnostics = MotivationDiagnosticsCollector()
    scheduler = _scheduler("fifo", diagnostics=diagnostics)
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    job, _ = scheduler.submit_action("s", ["W"], now=0.0)
    candidate = scheduler.find_best(now=0.0, include_wait=False)

    assert job is not None and candidate is not None
    recent = diagnostics.snapshot()["recent_searches"][-1]
    assert recent["policy_name"] == "fifo"
