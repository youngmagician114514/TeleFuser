"""Logical-pool dispatch uses all members without hiding cold state costs."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from telefuser.service.livekit.phase_pool import PhasePoolTask, PhaseStateCost, SharedPhaseQueue, state_copy_seconds


def _loads(ranks=(10, 27), slots=12):
    return {
        rank: SimpleNamespace(
            slots=slots,
            outstanding_seconds=0.0,
            queued_jobs=0,
            reserved_bytes=0,
            kv_budget_bytes=1000,
            seconds=lambda phase, window, rows: 0.05 * (rows / 3) ** 0.7,
        )
        for rank in ranks
    }


def test_one_refresh_worker_batches_jobs_from_multiple_denoise_sources():
    pool = SharedPhaseQueue("refresh", (27,))
    for index in range(4):
        pool.submit(PhasePoolTask(str(index), str(index), 6, 1, 1, payload={"source": 10 + index}))
    selected = pool.dispatch(_loads(), now=0, state_cost=lambda *args: PhaseStateCost(), reserve=lambda *args: None)
    assert len(selected) == 1 and selected[0].rank == 27
    assert len(selected[0].tasks) == 4
    assert len({task.payload["source"] for task in selected[0].tasks}) == 4
    assert not pool.pending and len(pool.active) == 4
    assert not pool.dispatch(_loads(), now=0, state_cost=lambda *args: PhaseStateCost(), reserve=lambda *args: None)
    for task in selected[0].tasks:
        pool.complete(task.session_id, 27)
    assert not pool.active


def test_shared_queue_uses_other_pool_members_when_one_has_no_execution_credit():
    pool, loads = SharedPhaseQueue("refresh", (10, 27)), _loads(slots=3)
    pool.submit(PhasePoolTask("a", "a", 6, 1, 1))
    first = pool.dispatch(loads, now=0, state_cost=lambda *args: PhaseStateCost(), reserve=lambda *args: None)
    pool.submit(PhasePoolTask("b", "b", 6, 1, 1))
    second = pool.dispatch(loads, now=0, state_cost=lambda *args: PhaseStateCost(), reserve=lambda *args: None)
    assert first[0].rank != second[0].rank
    with pytest.raises(RuntimeError, match="different"):
        pool.complete("a", second[0].rank)


def test_warm_affinity_is_a_cost_not_a_fixed_owner():
    def choose(cold_seconds):
        pool, loads = SharedPhaseQueue("refresh", (10, 27)), _loads()
        loads[10].outstanding_seconds = 0.4
        pool.submit(PhasePoolTask("a", "a", 18, 1, 1))
        return pool.dispatch(
            loads,
            now=0,
            state_cost=lambda task, rank: PhaseStateCost(0 if rank == 10 else cold_seconds),
            reserve=lambda *args: None,
        )[0].rank

    assert choose(0.01) == 27
    assert choose(1.0) == 10
    assert state_copy_seconds(source=10, destination=10, byte_count=100, observations={}) == 0
    assert state_copy_seconds(source=10, destination=27, byte_count=100, observations={(10, 27): (1000, 0.1)}) == 0.01


def test_cumulative_state_memory_is_reserved_before_the_next_dispatch():
    pool, loads = SharedPhaseQueue("denoise", (10,)), _loads((10,))
    for index in range(3):
        pool.submit(PhasePoolTask(str(index), str(index), 6, 2, 1))

    def reserve(task, rank):
        loads[rank].reserved_bytes += 600

    selected = pool.dispatch(loads, now=0, state_cost=lambda task, rank: PhaseStateCost(0, 600), reserve=reserve)
    assert sum(len(item.tasks) for item in selected) == 1
    assert loads[10].reserved_bytes == 600 and len(pool.pending) == 2


def test_session_fifo_exposes_one_phase_head_and_can_cancel_departed_pending_work():
    pool = SharedPhaseQueue("denoise", (10,))
    task = PhasePoolTask("a", "job1", 6, 4, 1)
    pool.submit(task)
    with pytest.raises(ValueError, match="already"):
        pool.submit(replace(task, job_id="job2"))
    pool.cancel("a")
    assert not pool.pending
