from __future__ import annotations

import json

from telefuser.service.livekit.motivation_diagnostics import MotivationDiagnosticsCollector
from telefuser.service.livekit.motivation_scheduler import (
    GpuSchedulingState,
    MotivationProfile,
    MotivationScheduler,
    StaticMotivationProfileTable,
)


def _scheduler(collector: MotivationDiagnosticsCollector) -> MotivationScheduler:
    profiles = [
        MotivationProfile(1, "f", 0.40, 0.68, 20.0),
        MotivationProfile(2, "f", 0.72, 0.68, 30.0),
        MotivationProfile(3, "f", 1.05, 0.67, 40.0),
        MotivationProfile(4, "f", 1.40, 0.66, 50.0),
    ]
    scheduler = MotivationScheduler(
        StaticMotivationProfileTable(profiles),
        diagnostics=collector,
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", memory_free_gb=80.0))
    return scheduler


def test_search_diagnostics_exposes_batch_candidates_without_policy_coupling() -> None:
    collector = MotivationDiagnosticsCollector(recent_search_limit=1, recent_dispatch_limit=1)
    scheduler = _scheduler(collector)
    for session_id in ("a", "b"):
        scheduler.register_session(
            session_id, owner_gpu="gpu-0", now=0.0, compatibility_key=("same",)
        )
        scheduler.submit_action(session_id, ["W"], now=0.0)

    candidate = scheduler.find_best(now=0.0, include_wait=False)
    snapshot = collector.snapshot()

    assert candidate is not None
    assert candidate.batch_size == 2
    assert snapshot["search_count"] == 1
    assert snapshot["mean_ready_count"] == 2.0
    assert snapshot["enumerated_by_batch_size"]["2"] == 1
    assert snapshot["feasible_by_batch_size"]["2"] == 1
    assert snapshot["selected_batch_size"]["2"] == 1
    json.dumps(snapshot)


def test_diagnostics_recent_records_are_bounded() -> None:
    collector = MotivationDiagnosticsCollector(recent_search_limit=1, recent_dispatch_limit=1)
    scheduler = _scheduler(collector)
    scheduler.register_session("s", owner_gpu="gpu-0", now=0.0)
    scheduler.submit_action("s", ["W"], now=0.0)
    scheduler.find_best(now=0.0, include_wait=False)
    scheduler.find_best(now=0.0, include_wait=False)

    snapshot = collector.snapshot()
    assert len(snapshot["recent_searches"]) == 1
