"""Shared phase queues, execution credits, and costed state locality.

Logical pool membership contains transport endpoints; it has no CUDA device
ordinals. Actors keep their local devices, while each causal phase request
may select any member whose memory and state-transfer costs permit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class PhasePoolTask:
    session_id: str
    job_id: str
    window: int
    steps: int
    deadline: float
    rows: int = 3
    payload: Any = None


@dataclass(frozen=True)
class PhasePoolDispatch:
    rank: int
    tasks: tuple[PhasePoolTask, ...]
    predicted_seconds: float


@dataclass(frozen=True)
class PhaseStateCost:
    seconds: float = 0.0
    extra_bytes: int = 0


@dataclass
class SharedPhaseQueue:
    """Dispatch deadline-ordered cohorts using measured cost and state locality.

    One request per session is causally ready. Credits bound admitted dense
    rows, not the size of a hidden per-worker waiting queue. Compatible
    deadline prefixes are compared on every worker, including a cold worker
    when its state-copy cost can beat the warm owner's queue.
    """

    phase: str
    ranks: tuple[int, ...]
    pending: dict[str, PhasePoolTask] = field(default_factory=dict)
    active: dict[str, tuple[int, int]] = field(default_factory=dict)

    def submit(self, task: PhasePoolTask) -> None:
        if task.session_id in self.pending or task.session_id in self.active:
            raise ValueError(f"Session {task.session_id} already owns a {self.phase} phase request")
        self.pending[task.session_id] = task

    def cancel(self, session_id: str) -> None:
        self.pending.pop(session_id, None)

    def complete(self, session_id: str, rank: int) -> None:
        if self.active[session_id][0] != rank:
            raise RuntimeError("A phase completion came from a different execution worker")
        del self.active[session_id]

    def dispatch(
        self,
        loads: Mapping[int, Any],
        *,
        now: float,
        state_cost: Callable[[PhasePoolTask, int], PhaseStateCost | None],
        reserve: Callable[[PhasePoolTask, int], None],
    ) -> list[PhasePoolDispatch]:
        result = []
        while self.pending:
            best = None
            for rank in self.ranks:
                load = loads[rank]
                occupied = sum(rows for worker, rows in self.active.values() if worker == rank)
                available = load.slots - occupied
                if available < 3:
                    continue
                feasible = []
                for task in self.pending.values():
                    copy = state_cost(task, rank)
                    if copy is not None:
                        feasible.append((task, copy))
                feasible.sort(key=lambda pair: (pair[0].deadline, pair[0].job_id))
                selected, rows, extra_bytes = [], 0, 0
                for task, copy in feasible:
                    rows += task.rows
                    extra_bytes += copy.extra_bytes
                    if rows > available or load.reserved_bytes + extra_bytes > load.kv_budget_bytes:
                        break
                    selected.append((task, copy))
                    count = len(selected)
                    duration = (
                        sum(
                            item.rows * item.steps * load.seconds(self.phase, item.window, rows) for item, _ in selected
                        )
                        / rows
                    )
                    # Copies may overlap another worker's compute; within
                    # this candidate account for every requested state copy.
                    finish = load.outstanding_seconds + sum(cost.seconds for _, cost in selected) + duration
                    lateness = sum(max(0.0, now + finish - item.deadline) for item, _ in selected) / count
                    order = (lateness, finish / count, finish, tuple(item.job_id for item, _ in selected), rank)
                    if best is None or order < best[0]:
                        best = (order, PhasePoolDispatch(rank, tuple(item for item, _ in selected), duration))
            if best is None:
                break
            choice = best[1]
            for task in choice.tasks:
                reserve(task, choice.rank)
                del self.pending[task.session_id]
                self.active[task.session_id] = (choice.rank, task.rows)
            # Reserve before evaluating another worker/candidate. All queue
            # predictions share the same physical timeline and credit count.
            loads[choice.rank].outstanding_seconds += choice.predicted_seconds
            loads[choice.rank].queued_jobs += len(choice.tasks)
            result.append(choice)
        return result


def state_copy_seconds(
    *, source: int, destination: int, byte_count: int, observations: Mapping[tuple[int, int], tuple[int, float]]
) -> float:
    """Use actual pair transfer observations, with a conservative cold start."""
    if source == destination or not byte_count:
        return 0.0
    amount, seconds = observations.get((source, destination), (0, 0.0))
    # This is an admission estimate only. It never removes state-transfer
    # dependencies and is replaced by measured byte/time rates as they arrive.
    bandwidth = amount / seconds if amount and seconds > 0 else 20_000_000_000
    return byte_count / bandwidth
