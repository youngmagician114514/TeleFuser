"""Measured partition sizing, causal channel order, and quality/load decisions."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from telefuser.service.livekit.abot_phase_policy import ABotPhasePlanner, PhasePartition, PhaseWorkerLoad
from telefuser.service.livekit.motivation_scheduler import MotivationProfile
from telefuser.service.livekit.phase_channels import PairHandoffQueue


def _loads(refresh_seconds=0.1):
    timings = {
        f"{phase}_w{window}": {3: seconds, 6: 1.5 * seconds, 12: 2 * seconds}
        for phase, seconds in (("denoise", 0.1), ("refresh", refresh_seconds))
        for window in (6, 18)
    }
    return {r: PhaseWorkerLoad(r, 12, 100_000, 10, 0, {k: dict(v) for k, v in timings.items()}) for r in range(4)}


def _planner():
    return ABotPhasePlanner(
        [
            MotivationProfile(1, "b1_s4_w18_rho0_bf16", 1.0, 1.0, 1.0),
            MotivationProfile(1, "b1_s2_w6_rho0_bf16", 1.0, 0.9, 1.0),
        ]
    )


def test_measured_phase_ratio_changes_pool_partition():
    planner = _planner()
    fast_refresh = planner.partition(_loads(0.05), resident_sessions=24, request_rate=1)
    slow_refresh = planner.partition(_loads(0.4), resident_sessions=24, request_rate=1)
    assert len(fast_refresh.refresh_ranks) == 1
    assert len(slow_refresh.refresh_ranks) == 2
    assert set(fast_refresh.denoise_ranks).isdisjoint(fast_refresh.refresh_ranks)
    assert len(fast_refresh.denoise_ranks) + len(fast_refresh.refresh_ranks) == 4


def test_partition_accounts_for_both_pools_resident_memory():
    loads = {r: replace(load, kv_budget_bytes=1500) for r, load in _loads().items()}
    partition = _planner().partition(loads, resident_sessions=24, request_rate=1)
    assert partition.profile == "b1_s2_w6_rho0_bf16"
    for pool in (partition.denoise_ranks, partition.refresh_ranks):
        assert sum(loads[r].kv_budget_bytes // loads[r].session_bytes(6) for r in pool) >= 24
    with pytest.raises(RuntimeError, match="retain"):
        _planner().partition(loads, resident_sessions=100, request_rate=1)


def test_admission_responds_to_measured_backlog_and_slack_without_changing_quality_table():
    loads, planner = _loads(), _planner()
    partition = planner.partition(loads, resident_sessions=24, request_rate=1)
    state = SimpleNamespace(slack_seconds=10.0)
    first = planner.admit(state, loads, partition)
    assert first.quality == 1.0
    loads[first.denoise_rank].outstanding_seconds = 20
    second = planner.admit(state, loads, partition)
    assert second.denoise_rank != first.denoise_rank
    state.slack_seconds = -10
    urgent = planner.admit(state, loads, partition)
    assert urgent.quality == 0.9
    loads[urgent.refresh_rank].reserved_bytes = loads[urgent.refresh_rank].kv_budget_bytes
    assert planner.admit(state, loads, partition) is None


def test_video_surplus_does_not_hide_an_overdue_action_fifo_from_quality_selection():
    loads, planner = _loads(), _planner()
    partition = planner.partition(loads, resident_sessions=24, request_rate=1)
    state = SimpleNamespace(slack_seconds=10.0)
    relaxed = planner.admit(state, loads, partition, action_age_seconds=0)
    overdue = planner.admit(state, loads, partition, action_age_seconds=4)
    assert relaxed.quality == 1.0
    assert overdue.quality == 0.9
    assert overdue.predicted_completion_seconds < relaxed.predicted_completion_seconds
    assert overdue.slack_seconds == -3.0


def test_untagged_pair_ordering_does_not_serialize_other_pairs():
    commands = []
    queue = PairHandoffQueue(lambda rank, command: commands.append((rank, command)))
    a = {"transfer_id": "a", "source": 0, "destination": 3}
    b = {"transfer_id": "b", "source": 3, "destination": 0}
    c = {"transfer_id": "c", "source": 1, "destination": 3}
    for record in (a, b, c):
        queue.submit(record)
    assert [command[1]["transfer_id"] for command in commands] == ["a", "a", "c", "c"]
    queue.complete("a", 0)
    assert len(commands) == 4
    queue.complete("a", 3)
    assert [command[1]["transfer_id"] for command in commands][-2:] == ["b", "b"]
    assert commands[-2][1]["command"] == "receive"
    queue.complete("b", 3)
    with pytest.raises(RuntimeError, match="Invalid"):
        queue.complete("b", 3)
    queue.complete("b", 0)
    queue.complete("c", 1)
    queue.complete("c", 3)
    assert not queue.pending and not queue.active


def test_idle_resident_sessions_do_not_predict_nonexistent_ready_batches():
    loads, planner = _loads(), _planner()
    partition = planner.partition(loads, resident_sessions=24, request_rate=1)
    state = SimpleNamespace(slack_seconds=10.0)
    single = planner.admit(state, loads, partition)
    for load in loads.values():
        load.sessions.update(f"idle-{i}" for i in range(8))
    resident_only = planner.admit(state, loads, partition)
    assert resident_only.denoise_seconds == single.denoise_seconds
    for load in loads.values():
        load.queued_jobs = 3
    ready_batch = planner.admit(state, loads, partition)
    assert ready_batch.denoise_seconds < single.denoise_seconds


def test_idle_arrivals_spread_by_projected_service_even_with_a_slightly_faster_worker():
    loads, planner = _loads(refresh_seconds=0.05), _planner()
    for window in (6, 18):
        loads[3].timings[f"denoise_w{window}"] = {
            rows: seconds * 0.95 for rows, seconds in loads[3].timings[f"denoise_w{window}"].items()
        }
    partition = planner.partition(loads, resident_sessions=24, request_rate=1)
    assert len(partition.denoise_ranks) == 3
    for index in range(24):
        choice = planner.admit(SimpleNamespace(slack_seconds=0.0), loads, partition)
        for rank in (choice.denoise_rank, choice.refresh_rank):
            loads[rank].sessions.add(f"session-{index}")
            loads[rank].reserved_bytes += loads[rank].session_bytes(6)
    counts = [len(loads[rank].sessions) for rank in partition.denoise_ranks]
    assert max(counts) - min(counts) <= 1


def test_existing_state_can_advance_without_headroom_for_future_arrivals():
    loads, planner = _loads(), _planner()
    partition = PhasePartition((0, 1), (2, 3), "b1_s2_w6_rho0_bf16", 1.0, 0.9, resident_sessions=8)
    for load in loads.values():
        load.kv_budget_bytes = 100
        load.reserved_bytes = 60
        load.sessions.add("a")
    choice = planner.admit(SimpleNamespace(slack_seconds=-10), loads, partition, owners=(0, 2), retained_bytes=60)
    assert choice is not None and choice.reserved_bytes == 60
    assert choice.fidelity == "b1_s2_w6_rho0_bf16"


def test_handoff_replicas_do_not_count_as_two_logical_residents():
    loads, planner = _loads(), _planner()
    partition = PhasePartition((0, 1), (2, 3), "b1_s2_w6_rho0_bf16", 1.0, 0.9, resident_sessions=6)
    for load in loads.values():
        load.kv_budget_bytes = 180
        load.reserved_bytes = 60
        load.sessions.add("a")
    # A has both source and target copies while moving. Counting it twice
    # would wrongly spend the space needed for a future distinct session.
    assert planner.admit(SimpleNamespace(slack_seconds=-10), loads, partition) is None
