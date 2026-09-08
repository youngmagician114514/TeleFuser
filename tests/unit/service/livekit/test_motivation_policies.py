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
