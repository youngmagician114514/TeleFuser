"""Exercise replay admission, paired completions and teardown without GPUs."""

import json
import queue
from types import SimpleNamespace

import pytest

from telefuser.service.livekit.abot_phase_policy import ABotPhasePlanner, PhasePartition, PhaseWorkerLoad
from telefuser.service.livekit.motivation_scheduler import MotivationProfile
from tools.validation.replay_abot_async_phase_trace import PhaseReplayCoordinator
from tools.validation.replay_abot_latent_step_trace import ReplayEvent


class _Server:
    def __init__(self, delayed_decode=False, reported_state_bytes=None):
        self.events = queue.Queue()
        self.delayed_decode = delayed_decode
        self.first_decode = None
        self.jobs = 0
        self.reported_state_bytes = reported_state_bytes

    def send(self, rank, command):
        kind = command["command"]
        if kind == "create":
            self.events.put({"event": "session_ready", "rank": rank, "session_id": command["session_id"]})
        elif kind == "job":
            self.jobs += 1
            metadata = command["metadata"]
            self.events.put(self._handoff(metadata, "latent", rank, metadata["refresh_rank"]))
        elif kind == "receive":
            metadata = command["metadata"]
            common = {"session_id": metadata["session_id"], "job_id": metadata["job_id"]}
            if command["phase"] == "latent":
                self.events.put(self._handoff(metadata, "kv", rank, metadata["denoise_rank"]))
            elif command["phase"] == "kv":
                self._output(common, rank)
            elif command["phase"] == "state":
                self.events.put(dict(common, event="state_ready", rank=rank, kind=metadata["state_kind"]))
            self.events.put(dict(command, event="transfer_complete", rank=rank, total_bytes=100, total_duration_ms=1))
        elif kind == "send":
            self.events.put(dict(command, event="transfer_complete", rank=rank, total_bytes=100, total_duration_ms=1))
        elif kind == "export_state":
            metadata = command["metadata"]
            self.events.put(self._handoff(metadata, "state", rank, metadata["target_rank"]))
        elif kind in {"close", "evict"}:
            self.events.put(
                {
                    "event": "session_closed" if kind == "close" else "state_evicted",
                    "rank": rank,
                    "session_id": command["session_id"],
                }
            )

    @staticmethod
    def _handoff(metadata, phase, source, destination):
        suffix = f":{metadata['state_kind']}" if phase == "state" else ""
        return dict(
            event="handoff_ready",
            rank=source,
            source=source,
            destination=destination,
            transfer_id=f"{metadata['job_id']}:{phase}{suffix}",
            phase=phase,
            metadata=dict(metadata),
            skeleton={},
            manifest=[],
        )

    def _output(self, common, rank):
        if self.reported_state_bytes is not None:
            common = dict(common, state_bytes=self.reported_state_bytes, refresh_state_bytes=self.reported_state_bytes)
        if self.delayed_decode:
            decoded = dict(common, event="decoded", rank=rank, frames=12)
            if self.jobs == 1:
                # First RGB only arrives after the next Job dispatch.
                # A coordinator still waiting for RGB cannot pass this.
                self.first_decode = decoded
                self.events.put(dict(common, event="kv_committed", rank=rank))
            else:
                # Old RGB must not release the newer Job's reservation.
                self.events.put(self.first_decode)
                self.events.put(dict(common, event="kv_committed", rank=rank))
                self.events.put(decoded)
            return
        # RGB before KV must also credit exactly once.
        self.events.put(dict(common, event="decoded", rank=rank, frames=12))
        self.events.put(dict(common, event="kv_committed", rank=rank))


@pytest.mark.parametrize("delayed_decode", [False, True])
@pytest.mark.parametrize("reported_state_bytes", [None, 30])
def test_root_replay_completes_ordered_jobs_and_writes_existing_producer_metrics(
    tmp_path, delayed_decode, reported_state_bytes
):
    timings = {f"{phase}_w6": {3: 0.01} for phase in ("denoise", "refresh")}
    loads = {rank: PhaseWorkerLoad(rank, 3, 100000, 10, 0, timings) for rank in (0, 1)}
    planner = ABotPhasePlanner([MotivationProfile(1, "b1_s2_w6_rho0_bf16", 1.0, 0.9, 1.0)])
    admit = planner.admit
    retained = []

    def observe_admission(*args, **kwargs):
        retained.append(kwargs.get("retained_bytes", 0))
        return admit(*args, **kwargs)

    planner.admit = observe_admission
    partition = planner.partition(loads, resident_sessions=1, request_rate=1)
    args = SimpleNamespace(
        output=tmp_path,
        duration=None,
        pending_slots=4,
        collection_window=0.5,
        attention="sdpa",
        backend="eager",
        latent_slots=3,
    )
    scenario = SimpleNamespace(seed=17, session=SimpleNamespace(image_path="unused.png", prompt="world"))
    events = [
        ReplayEvent(0, "arrival", "a"),
        ReplayEvent(0.005, "release", "a", ("KeyW",)),
        ReplayEvent(0.01, "release", "a", ("KeyD",)),
        ReplayEvent(0.06, "departure", "a"),
    ]
    coordinator = PhaseReplayCoordinator(
        args,
        scenario,
        events,
        {"duration_seconds": 0.06},
        _Server(delayed_decode, reported_state_bytes),
        loads,
        planner,
        partition,
    )
    coordinator.run()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "complete"
    assert result["counts"]["action_jobs_completed"] == 2
    assert result["counts"]["generated_frames"] == 24
    assert result["counts"]["sessions_ready"] == result["counts"]["sessions_closed"] == 1
    assert result["producer"]["generated_frames"] == 24
    assert not coordinator.active and not coordinator.completed_parts
    assert not coordinator.pending_outputs
    assert not coordinator.denoise_queue.active and not coordinator.refresh_queue.active
    assert not coordinator.handoffs.active
    assert all(load.queued_jobs == 0 and load.outstanding_seconds < 1e-9 for load in loads.values())
    assert all(not load.sessions and not load.reserved_bytes for load in loads.values())
    if reported_state_bytes is not None:
        assert reported_state_bytes in retained


def test_coordinator_moves_d_and_r_state_through_shared_pools_and_cleans_reservations(tmp_path):
    timings = {f"{phase}_w6": {3: 0.01} for phase in ("denoise", "refresh")}
    loads = {rank: PhaseWorkerLoad(rank, 3, 100000, 10, 0, timings) for rank in range(4)}
    planner = ABotPhasePlanner([MotivationProfile(1, "b1_s2_w6_rho0_bf16", 1.0, 0.9, 1.0)])
    partition = PhasePartition((0, 1), (2, 3), "b1_s2_w6_rho0_bf16", 100, 0.9, 1)
    args = SimpleNamespace(
        output=tmp_path,
        duration=None,
        pending_slots=4,
        collection_window=0.5,
        attention="sdpa",
        backend="eager",
        latent_slots=3,
    )
    scenario = SimpleNamespace(seed=17, session=SimpleNamespace(image_path="unused.png", prompt="world"))
    events = [
        ReplayEvent(0, "arrival", "a"),
        ReplayEvent(0.005, "release", "a", ("KeyW",)),
        ReplayEvent(0.025, "release", "a", ("KeyD",)),
        ReplayEvent(0.05, "release", "a", ("KeyW",)),
        ReplayEvent(0.16, "departure", "a"),
    ]

    class BusyWarmWorker(_Server):
        def _output(self, common, rank):
            super()._output(common, rank)
            # Model unrelated work arriving at the current warm owners. The
            # other members can now finish earlier even with a state copy.
            d, r = coordinator.owners[common["session_id"]]
            loads[d].outstanding_seconds += 1
            loads[r].outstanding_seconds += 1

    coordinator = PhaseReplayCoordinator(
        args, scenario, events, {"duration_seconds": 0.16}, BusyWarmWorker(), loads, planner, partition
    )
    coordinator.run()
    assert coordinator.counts["action_jobs_completed"] == 3
    assert coordinator.counts["denoise_owner_changes"] > 0
    assert coordinator.counts["refresh_owner_changes"] > 0
    assert not coordinator.pending_state and not coordinator.evicting and not coordinator.state_reservations
    assert not coordinator.copy_reservations
    assert not coordinator.handoffs.active and not coordinator.handoffs.pending
    assert all(not load.sessions and not load.reserved_bytes and not load.queued_jobs for load in loads.values())
