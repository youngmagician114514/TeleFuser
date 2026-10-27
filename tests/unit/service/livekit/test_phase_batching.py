"""Causal GPU queue predictions and capacity recovery without enlarging FIFO."""

from collections import deque

import pytest

from telefuser.service.livekit.phase_batching import PhaseStepGroup, plan_phase_steps


def _groups(remaining=(1, 1, 1, 1)):
    return [PhaseStepGroup(str(index), 3, steps, 6, "compatible", -3.0) for index, steps in enumerate(remaining)]


def test_measured_refresh_curve_batches_overdue_heads_and_predicts_a_real_queue():
    curve = {3: 0.049272, 6: 0.083825, 9: 0.122128, 12: 0.160431}
    plan = plan_phase_steps(_groups(), max_rows=12, seconds=lambda window, rows: curve[rows])
    assert len(plan.sessions) == 4
    assert plan.clearing_seconds == pytest.approx(0.160431)
    assert plan.total_completion_seconds == pytest.approx(4 * 0.160431)


def test_step_boundary_replans_mixed_remaining_steps_without_reserving_a_whole_job():
    curve = {3: 0.057373, 6: 0.097170, 9: 0.139949, 12: 0.182728}
    groups = _groups((4, 3, 2, 1))
    total, finished = 0.0, set()
    predicted = plan_phase_steps(groups, max_rows=12, seconds=lambda window, rows: curve[rows]).clearing_seconds
    while groups:
        plan = plan_phase_steps(groups, max_rows=12, seconds=lambda window, rows: curve[rows])
        assert len(plan.sessions) >= 2
        total += plan.duration_seconds
        following = []
        for group in groups:
            remaining = group.remaining_steps - int(group.session_id in plan.sessions)
            if remaining:
                following.append(PhaseStepGroup(group.session_id, 3, remaining, 6, "compatible", -3))
            else:
                finished.add(group.session_id)
        groups = following
    assert finished == {"0", "1", "2", "3"}
    assert total == pytest.approx(predicted)
    assert total <= sum(curve[rows] for rows in (12, 9, 6, 3))


def test_batching_is_not_forced_when_the_measured_cost_is_superlinear():
    plan = plan_phase_steps(_groups(), max_rows=12, seconds=lambda window, rows: (rows / 3) ** 2)
    assert len(plan.sessions) == 1
    assert plan.clearing_seconds == 4
    assert plan.total_completion_seconds == 10


def test_slot_capacity_compatibility_and_mixed_windows_are_enforced():
    groups = [
        PhaseStepGroup("a", 3, 1, 6, "x", -3),
        PhaseStepGroup("b", 3, 1, 18, "x", -2),
        PhaseStepGroup("c", 3, 1, 6, "y", -1),
    ]
    plan = plan_phase_steps(groups, max_rows=6, seconds=lambda window, rows: window * rows**0.5)
    assert set(plan.sessions) in ({"a", "b"}, {"c"})
    assert plan.clearing_seconds >= plan.duration_seconds
    with pytest.raises(ValueError, match="capacity"):
        plan_phase_steps(groups, max_rows=2, seconds=lambda window, rows: 1.0)


def test_same_four_slot_fifo_serves_more_actions_when_beneficial_batches_are_used():
    def serve(*, singleton):
        pending = [deque() for _ in range(4)]
        now, release_index, completed, dropped = 0.0, 0, 0, 0
        curve = {3: 0.15, 6: 0.25, 9: 0.33, 12: 0.40}
        while now < 30:
            while release_index * 0.5 <= now and release_index < 60:
                for queue in pending:
                    if len(queue) == 4:
                        dropped += 1
                    else:
                        queue.append(release_index * 0.5)
                release_index += 1
            groups = [PhaseStepGroup(str(i), 3, 1, 6, "x", queue[0]) for i, queue in enumerate(pending) if queue]
            if not groups:
                now = release_index * 0.5
                continue
            if singleton:
                selected, duration = (min(groups, key=lambda group: group.deadline).session_id,), curve[3]
            else:
                plan = plan_phase_steps(groups, max_rows=12, seconds=lambda window, rows: curve[rows])
                selected, duration = plan.sessions, plan.duration_seconds
            for sid in selected:
                pending[int(sid)].popleft()
            now += duration
            completed += len(selected)
        return completed, dropped

    old_completed, old_drops = serve(singleton=True)
    completed, drops = serve(singleton=False)
    assert completed == 240 and drops == 0
    assert completed > old_completed and old_drops > drops
