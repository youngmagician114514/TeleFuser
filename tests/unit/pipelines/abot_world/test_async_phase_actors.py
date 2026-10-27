"""Run the real actor loops on CPU with an in-memory handoff transport."""

import copy
import queue
import threading
import time
from concurrent.futures import Future

import torch

from telefuser.service.livekit.abot_phase_policy import PhaseWorkerLoad
from telefuser.service.livekit.abot_phase_runtime import ABotPhaseWorker

from .test_latent_parallel_runtime import _stage
from .test_latent_step_runtime import _pipelines


class _Event:
    def __init__(self, **kwargs):
        self.at = time.monotonic()

    def record(self):
        self.at = time.monotonic()

    def query(self):
        return True

    def elapsed_time(self, other):
        return max(0.001, (other.at - self.at) * 1000)


class _Connection:
    def __init__(self, rank, events):
        self.rank, self.events = rank, events
        self.commands = queue.Queue()

    def send(self, record):
        self.events.put(dict(record, rank=self.rank))

    def recv(self):
        return self.commands.get(timeout=15)


@torch.inference_mode()
def test_actual_actor_loops_drain_overlapping_sessions_without_a_world_wave(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    _, denoise = _pipelines(_stage(), count=5)
    _, refresh = _pipelines(_stage(), count=0)
    refresh.taew_decode_stage.create_decode_state = object
    refresh.taew_decode_stage.warmup_first_frame = lambda *args: None
    events = queue.Queue()
    connections = [_Connection(rank, events) for rank in (0, 1)]
    timings = {f"{phase}_w18": {3: 0.01, 6: 0.015} for phase in ("denoise", "refresh")}
    actors = [
        ABotPhaseWorker(
            pipeline, connections[rank], rank=rank, load=PhaseWorkerLoad(rank, 6, 100000, 1, 0, copy.deepcopy(timings))
        )
        for rank, pipeline in enumerate((denoise, refresh))
    ]
    actors[0]._sessions.update(denoise._interactive_sessions)
    actors[0].runtime._sessions.update(denoise._interactive_sessions)

    def export(actor, payload, metadata, phase):
        # Delay refresh inputs without stopping the denoise actor. The real
        # tensor transfer runs on a communication thread in production.
        destination = actors[1 if phase == "latent" else 0]
        destination.commands.put(
            {
                "command": "import",
                "phase": phase,
                "metadata": copy.deepcopy(metadata),
                "payload": payload,
                "page_slot": None,
            }
        )

    for actor in actors:
        monkeypatch.setattr(
            actor, "_export", lambda payload, metadata, phase, a=actor: export(a, payload, metadata, phase)
        )
    threads = [threading.Thread(target=actor.run) for actor in actors]
    for thread in threads:
        thread.start()
    try:
        for chunk in range(3):
            for index, sid in enumerate(actors[0]._sessions):
                connections[0].commands.put(
                    {
                        "command": "job",
                        "metadata": {
                            "session_id": sid,
                            "job_id": f"{sid}-{chunk}",
                            "fidelity": "b1_s4_w18_rho0_bf16",
                            "actions": {"W": True},
                            "slack_seconds": 0.1,
                            "seed": 17 + index,
                            "denoise_rank": 0,
                            "refresh_rank": 1,
                        },
                    }
                )
            completed = set()
            while len(completed) < 10:
                record = events.get(timeout=15)
                assert record["event"] != "error", record
                if record["event"] in {"decoded", "kv_committed"}:
                    completed.add((record["event"], record["session_id"]))
            assert all(session.next_latent_frame == 3 * (chunk + 1) for session in actors[0]._sessions.values())
    finally:
        for connection in connections:
            connection.commands.put({"command": "stop"})
        for thread in threads:
            thread.join(timeout=15)
        assert not any(thread.is_alive() for thread in threads)


def test_decoder_owner_orders_cross_worker_arrivals_and_owns_deferred_latents():
    from types import SimpleNamespace

    actor = ABotPhaseWorker.__new__(ABotPhaseWorker)
    actor._decode_waiting = {}
    actor._decode_next = {"a": 0}
    actor._decode_states = {"a": object()}
    submitted = []

    def submit(latents, sid, state):
        submitted.append(latents.clone())
        return Future()

    actor.decode = SimpleNamespace(submit=submit)
    later = torch.ones(1, 4, 3, 2, 2)
    actor._queue_decode(dict(session_id="a", chunk_start=3), later)
    later.zero_()  # A transport receive page may be reused after import.
    assert not submitted
    actor._queue_decode(dict(session_id="a", chunk_start=0), torch.zeros_like(later))
    assert len(submitted) == 2
    assert not submitted[0].any() and submitted[1].all()
    assert actor._decode_next["a"] == 6 and not actor._decode_waiting["a"]


@torch.inference_mode()
def test_pool_actors_move_compute_state_without_moving_decoder_state(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    native, first = _pipelines(_stage(), count=1)
    pipelines = [first, *(_pipelines(first.denoise_stage, count=0)[1] for _ in range(3))]
    events = queue.Queue()
    connections = [_Connection(rank, events) for rank in range(4)]
    timings = {f"{phase}_w18": {3: 0.01} for phase in ("denoise", "refresh")}
    actors = [
        ABotPhaseWorker(
            pipeline,
            connections[rank],
            rank=rank,
            load=PhaseWorkerLoad(rank, 3, 100000, 1, 0, copy.deepcopy(timings)),
        )
        for rank, pipeline in enumerate(pipelines)
    ]
    sid = "session-0"
    session = first._interactive_sessions[sid]
    actors[0]._sessions[sid] = actors[0].runtime._sessions[sid] = session
    decoder_state = session.taew_decode_state
    actors[0]._decode_states[sid] = decoder_state
    actors[0]._decode_next[sid] = 0
    session.taew_decode_state = None

    def export(actor, payload, metadata, phase):
        key = {"latent": "refresh_rank", "kv": "denoise_rank", "decode": "decode_rank", "state": "target_rank"}
        actors[metadata[key[phase]]].commands.put(
            dict(command="import", phase=phase, payload=payload, metadata=copy.deepcopy(metadata), page_slot=None)
        )

    for actor in actors:
        actor.pool_scheduling = True
        monkeypatch.setattr(actor, "_export", lambda payload, meta, phase, a=actor: export(a, payload, meta, phase))
    threads = [threading.Thread(target=actor.run) for actor in actors]
    for thread in threads:
        thread.start()

    def wait_for(wanted):
        record = events.get(timeout=15)
        while record["event"] != wanted:
            assert record["event"] != "error", record
            record = events.get(timeout=15)
        return record

    try:
        d_rank, r_rank = 0, 2
        for chunk in range(3):
            job_id = f"{sid}-{chunk}"
            if chunk:
                for source, target, phase in ((d_rank, 1 - d_rank, "denoise"), (r_rank, 5 - r_rank, "refresh")):
                    connections[source].commands.put(
                        dict(
                            command="export_state",
                            metadata=dict(
                                session_id=sid,
                                job_id=job_id,
                                state_kind=phase,
                                target_rank=target,
                                seed=999,
                            ),
                        )
                    )
                    installed = wait_for("state_ready")
                    assert installed["rank"] == target and installed["kind"] == phase
                    connections[source].commands.put(dict(command="evict", session_id=sid))
                    assert wait_for("state_evicted")["rank"] == source
                d_rank, r_rank = 1 - d_rank, 5 - r_rank
            connections[d_rank].commands.put(
                dict(
                    command="job",
                    metadata=dict(
                        session_id=sid,
                        job_id=job_id,
                        fidelity="b1_s4_w18_rho0_bf16",
                        actions={"W": True},
                        slack_seconds=0.1,
                        seed=17,
                        denoise_rank=d_rank,
                        refresh_rank=r_rank,
                        decode_rank=0,
                    ),
                )
            )
            completed = set()
            while len(completed) < 2:
                record = events.get(timeout=15)
                assert record["event"] != "error", record
                if record["event"] in {"decoded", "kv_committed"}:
                    completed.add(record["event"])
                    assert record["rank"] == (0 if record["event"] == "decoded" else d_rank)
            native.generate_next_blocks_raw(
                [native._interactive_sessions[sid]],
                [{"W": True}],
            )
            actual = actors[d_rank]._sessions[sid]
            reference = native._interactive_sessions[sid]
            assert actual.next_latent_frame == reference.next_latent_frame == 3 * (chunk + 1)
            assert torch.equal(actual.generator.get_state(), reference.generator.get_state())
            assert actors[0]._decode_states[sid] is decoder_state
            assert actors[0]._decode_next[sid] == 3 * (chunk + 1)
            assert all(not actor._decode_states for actor in actors[1:])
    finally:
        for connection in connections:
            connection.commands.put({"command": "stop"})
        for thread in threads:
            thread.join(timeout=15)
        assert not any(thread.is_alive() for thread in threads)


def test_receive_pages_are_allocated_on_their_nccl_stream(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    current = ["compute"]
    peer_stream = object()

    @contextmanager
    def stream(value):
        previous = current[0]
        current[0] = value
        try:
            yield
        finally:
            current[0] = previous

    def acquire(manifest, device):
        assert current[0] is peer_stream
        return {}, None

    actor = ABotPhaseWorker.__new__(ABotPhaseWorker)
    actor.device = torch.device("cpu")
    actor._pages = SimpleNamespace(acquire=acquire)
    actor.commands = queue.Queue()
    actor._transport = SimpleNamespace(
        lane=SimpleNamespace(stream_for=lambda peer: peer_stream),
        transfer=lambda *args, **kwargs: SimpleNamespace(as_dict=lambda: {}),
    )
    emitted = []
    actor.emit = lambda *args, **kwargs: emitted.append(args[0])
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "stream", stream)
    actor._transfer(
        dict(
            command="receive",
            source=0,
            destination=1,
            transfer_id="job:state",
            phase="state",
            manifest=[],
            skeleton={"__dict__": {}},
        )
    )
    command = actor.commands.get_nowait()
    assert command["command"] == "import", command
    assert emitted == ["transfer_complete"]
    assert current[0] == "compute"


def test_state_ready_keeps_copy_reservation_until_import_reads_finish(monkeypatch):
    from collections import deque
    from types import SimpleNamespace

    done = [False]
    fence = SimpleNamespace(record=lambda: None, query=lambda: done[0])
    monkeypatch.setattr(torch.cuda, "Event", lambda: fence)
    actor = ABotPhaseWorker.__new__(ABotPhaseWorker)
    actor._sessions, actor._pending_events = {}, deque()
    actor.runtime = SimpleNamespace(import_canonical_state=lambda *args, **kwargs: object())
    released, emitted = [], []
    actor._pages = SimpleNamespace(release=lambda slot: released.append(slot))
    actor.emit = lambda event, **kwargs: emitted.append((event, kwargs))
    actor._command(
        dict(
            command="import",
            phase="state",
            payload={},
            page_slot="receive-pages",
            metadata=dict(session_id="a", job_id="a-1", seed=17, state_kind="denoise"),
        )
    )
    actor._poll_events()
    assert not released and not emitted
    done[0] = True
    actor._poll_events()
    assert released == ["receive-pages"]
    assert emitted == [("state_ready", dict(session_id="a", job_id="a-1", kind="denoise"))]


@torch.inference_mode()
def test_cuda_receive_does_not_overwrite_a_pending_model_read():
    from types import SimpleNamespace

    import pytest

    from telefuser.service.livekit.nccl_transfer import PersistentTensorPagePool

    if not torch.cuda.is_available():
        pytest.skip("Requires one CUDA device")
    device = torch.device("cuda:0")
    stream = torch.cuda.Stream(device=device)
    indices = torch.arange(1024, device=device, dtype=torch.int64)
    expected, observed = indices.clone(), torch.empty_like(indices)
    with torch.cuda.stream(stream):
        warm = torch.empty_like(indices)
        warm.fill_(0)
    observed.copy_(indices)
    torch.cuda.synchronize(device)
    del warm

    def transfer(leaves, **kwargs):
        with torch.cuda.stream(stream):
            leaves["page",].fill_(999)
        stream.synchronize()
        return SimpleNamespace(as_dict=lambda: {})

    actor = ABotPhaseWorker.__new__(ABotPhaseWorker)
    actor.device = device
    actor._pages = PersistentTensorPagePool()
    actor.commands = queue.Queue()
    actor.emit = lambda *args, **kwargs: None
    actor._transport = SimpleNamespace(lane=SimpleNamespace(stream_for=lambda peer: stream), transfer=transfer)
    # The default allocator can reuse indices immediately on its own stream;
    # an unrelated receive must not write that block before this read runs.
    torch.cuda._sleep(100000000)
    observed.copy_(indices)
    del indices
    actor._transfer(
        dict(
            command="receive",
            source=1,
            destination=0,
            transfer_id="job:state",
            phase="state",
            manifest=[{"path": ["page"], "shape": [1024], "dtype": "int64"}],
            skeleton={"__dict__": {"page": {"__tensor__": ["page"]}}},
        )
    )
    command = actor.commands.get_nowait()
    assert command["command"] == "import", command
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
