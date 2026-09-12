from __future__ import annotations

import pytest

from telefuser.service.livekit.motivation_scheduler import (
    GpuSchedulingState,
    LocalMigrationEstimator,
    MotivationProfile,
    MotivationScheduler,
    MotivationSchedulerConfig,
    StaticMotivationProfileTable,
)
from telefuser.service.livekit.slackserve import (
    SlackServeTier,
    choose_bmpr_profile,
    median_quality_floor,
    pareto_frontier,
    service_credit,
    urgency_tier,
)


def _profiles() -> StaticMotivationProfileTable:
    return StaticMotivationProfileTable(
        [
            MotivationProfile(1, "b1_fast", 0.2, 0.8, 20.0),
            MotivationProfile(1, "b1_medium", 0.3, 0.9, 22.0),
            MotivationProfile(1, "b1_high", 0.5, 1.0, 24.0),
        ]
    )


def _scheduler(
    *,
    migration_enabled: bool = True,
    source_free_at: float = 0.0,
    capacity_per_worker: int = 8,
) -> MotivationScheduler:
    scheduler = MotivationScheduler(
        _profiles(),
        config=MotivationSchedulerConfig(
            policy_name="slackserve",
            max_batch_size=4,
            include_idle_jobs=False,
            migration_enabled=migration_enabled,
            slackserve_capacity_per_worker=capacity_per_worker,
        ),
        migration_estimator=LocalMigrationEstimator(first_layer_ready_seconds=0.05),
    )
    scheduler.add_gpu(
        GpuSchedulingState("gpu-0", free_at=source_free_at, memory_free_gb=80.0)
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-1", memory_free_gb=80.0))
    return scheduler


def test_slackserve_profile_primitives_match_bmpr_and_credit_rules() -> None:
    rows = _profiles().profiles_for(batch_size=1, gpu_id="gpu-0")

    assert tuple(row.fidelity for row in pareto_frontier(rows)) == (
        "b1_fast",
        "b1_medium",
        "b1_high",
    )
    assert median_quality_floor(rows) == pytest.approx(0.9)
    assert choose_bmpr_profile(rows, slack_budget_seconds=1.0).fidelity == "b1_high"
    # Recovery mode keeps the median quality floor and picks its fastest row;
    # it must not fall through to the lower-quality 0.2-second point.
    assert choose_bmpr_profile(rows, slack_budget_seconds=0.1).fidelity == "b1_medium"

    credit = service_credit(
        playout_slack_seconds=0.1,
        remaining_seconds=0.0,
        next_chunk_seconds=0.3,
    )
    assert credit == pytest.approx(-0.2)
    assert urgency_tier(credit, 0.3) is SlackServeTier.URGENT


def test_slackserve_is_a_singleton_idle_aware_policy() -> None:
    scheduler = _scheduler()

    assert scheduler.config.max_batch_size == 1
    assert scheduler.config.include_idle_jobs is True
    assert scheduler.config.lambda_quality == 0.0
    assert scheduler.config.fairness_delta == 0.0

    state = scheduler.register_session("idle", owner_gpu="gpu-0", now=0.0)
    # Paper TTFC budget is 4x the profiled highest-quality B1 chunk.
    assert state.slack_seconds == pytest.approx(2.0)
    idle = scheduler.create_idle_job("idle", now=0.0)
    candidate = scheduler.find_best(now=0.0, include_wait=False)

    assert idle is not None and candidate is not None
    assert candidate.policy_name == "slackserve"
    assert candidate.batch_size == 1
    assert candidate.job_ids == (idle.job_id,)
    assert candidate.fidelity == "b1_high"


def test_slackserve_credit_beats_fifo_age_and_bmpr_enters_recovery() -> None:
    scheduler = _scheduler(migration_enabled=False)
    scheduler.register_session("older-relaxed", owner_gpu="gpu-0", now=0.0, slack_seconds=3.0)
    scheduler.register_session("newer-urgent", owner_gpu="gpu-0", now=0.0, slack_seconds=0.1)
    older, _ = scheduler.submit_action("older-relaxed", ["W"], now=0.0)
    newer, _ = scheduler.submit_action("newer-urgent", ["D"], now=0.0)

    candidate = scheduler.find_best(now=0.0, include_wait=False)

    assert older is not None and newer is not None and candidate is not None
    assert candidate.session_ids == ("newer-urgent",)
    assert candidate.job_ids == (newer.job_id,)
    assert candidate.fidelity == "b1_medium"


def test_slackserve_does_not_call_motivation_search(monkeypatch) -> None:
    scheduler = _scheduler(migration_enabled=False)
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)

    def fail(*args, **kwargs):
        raise AssertionError("SlackServe must not enter the Motivation search")

    monkeypatch.setattr(scheduler, "_find_best_motivation", fail)
    candidate = scheduler.find_best(now=0.0, include_wait=False)

    assert candidate is not None
    assert candidate.policy_name == "slackserve"
    assert candidate.batch_size == 1


def test_slackserve_rehomes_urgent_queue_only_when_finish_improves() -> None:
    scheduler = _scheduler(source_free_at=1.0)
    for name in ("urgent-0", "urgent-1"):
        scheduler.register_session(name, owner_gpu="gpu-0", now=0.0, slack_seconds=0.1)
        scheduler.submit_action(name, ["W"], now=0.0)

    candidate = scheduler.find_best(now=0.0, include_wait=False)

    assert candidate is not None
    assert candidate.gpu_id == "gpu-1"
    assert candidate.migration_count == 1
    assert candidate.start_at == pytest.approx(0.05)
    assert candidate.finish_at == pytest.approx(0.35)
    assert candidate.fidelity == "b1_medium"


def test_slackserve_respects_rehoming_cooldown_filter() -> None:
    scheduler = _scheduler(source_free_at=1.0)
    scheduler.register_session("urgent", owner_gpu="gpu-0", now=0.0, slack_seconds=0.1)
    scheduler.submit_action("urgent", ["W"], now=0.0)
    scheduler.register_session("urgent-2", owner_gpu="gpu-0", now=0.0, slack_seconds=0.1)
    scheduler.submit_action("urgent-2", ["D"], now=0.0)

    candidate = scheduler.find_best(
        now=0.0,
        include_wait=False,
        blocked_migration_session_ids=("urgent", "urgent-2"),
    )

    assert candidate is not None
    assert candidate.gpu_id == "gpu-0"
    assert candidate.migration_count == 0


def test_slackserve_does_not_rehome_into_a_full_relaxed_worker() -> None:
    scheduler = _scheduler(source_free_at=1.0, capacity_per_worker=1)
    scheduler.register_session("urgent", owner_gpu="gpu-0", now=0.0, slack_seconds=0.1)
    scheduler.submit_action("urgent", ["W"], now=0.0)
    scheduler.register_session("urgent-2", owner_gpu="gpu-0", now=0.0, slack_seconds=0.1)
    scheduler.submit_action("urgent-2", ["D"], now=0.0)
    scheduler.register_session("relaxed", owner_gpu="gpu-1", now=0.0, slack_seconds=4.0)
    scheduler.submit_action("relaxed", ["A"], now=0.0)

    candidate = scheduler.find_best(now=0.0, include_wait=False)

    assert candidate is not None
    assert candidate.migration_count == 0


def test_slackserve_diagnostics_disclose_unsupported_elastic_sp() -> None:
    scheduler = _scheduler()

    capabilities = scheduler.diagnostics_snapshot()["slackserve_capabilities"]

    assert capabilities == {
        "service_credit": True,
        "bmpr": True,
        "rehoming": True,
        "elastic_sp": False,
        "elastic_sp_reason": (
            "ABot has no single-session SP=2 execution primitive; "
            "cross-session B2 batching is not Elastic SP"
        ),
    }
