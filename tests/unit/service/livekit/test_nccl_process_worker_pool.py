from __future__ import annotations

import asyncio
from typing import Any

import pytest

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.nccl_process_worker_pool import (
    _MODEL_OUTPUT_PARENT_QUEUE_SIZE,
    _NCCL_INIT_PARENT_TIMEOUT_SECONDS,
    NCCLProcessLiveKitWorkerPool,
    _pump_model_outputs,
)
from telefuser.service.livekit.process_worker_pool import (
    ProcessLiveKitWorkerPool,
    ProcessWorkerSpec,
)
from telefuser.service.livekit.worker import NullWorkerEventSink


class _EventCollector:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.updated = asyncio.Event()

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)
        self.updated.set()


class _PumpAdapter:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.pull_count = 0

    async def pull_chunks(self, session_id: str):
        del session_id
        for payload in self.payloads:
            self.pull_count += 1
            yield payload

    def runtime_metrics(self) -> dict[str, int]:
        return {"active_sessions": 1}


class _PumpService:
    def runtime_metrics(self, session_id: str) -> dict[str, int]:
        del session_id
        return {"active": 1}


def _pool() -> NCCLProcessLiveKitWorkerPool:
    config = LiveKitServeConfig(
        livekit_url="wss://livekit.example",
        livekit_api_key="key",
        livekit_api_secret="secret",
        worker_mode="process-nccl",
        num_workers=2,
        worker_gpu_map="0;1",
    )
    pool = NCCLProcessLiveKitWorkerPool(
        [ProcessWorkerSpec("worker-0", ["0"]), ProcessWorkerSpec("worker-1", ["1"])],
        config=config,
        pipeline_file="pipeline.py",
        event_sink=NullWorkerEventSink(),
    )
    pool._active_workers = {"worker-0"}
    return pool


def _model_output(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "model_output",
        "worker_id": "worker-0",
        "session_id": session_id,
        "payload": payload,
    }


def _model_output_eos(session_id: str) -> dict[str, Any]:
    return {
        "type": "model_output_eos",
        "worker_id": "worker-0",
        "session_id": session_id,
    }


async def _wait_for_count(events: _EventCollector, count: int) -> None:
    while len(events.items) < count:
        events.updated.clear()
        await asyncio.wait_for(events.updated.wait(), timeout=1.0)


def test_child_pump_waits_for_credit_before_reading_next_abot_payload() -> None:
    async def run() -> None:
        events = _EventCollector()
        adapter = _PumpAdapter([{"type": "chunk", "index": 0}, {"type": "chunk", "index": 1}])
        credits = asyncio.BoundedSemaphore(_MODEL_OUTPUT_PARENT_QUEUE_SIZE)
        task = asyncio.create_task(
            _pump_model_outputs(
                adapter,
                _PumpService(),
                worker_id="worker-0",
                session_id="pipeline-1",
                credits=credits,
                events=events,
            )
        )
        await _wait_for_count(events, 1)
        await asyncio.sleep(0)
        assert adapter.pull_count == 1
        assert [item["payload"]["index"] for item in events.items] == [0]

        credits.release()
        await _wait_for_count(events, 2)
        assert adapter.pull_count == 2
        assert [item["payload"]["index"] for item in events.items] == [0, 1]

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_child_pump_emits_eos_when_abot_output_iterator_ends() -> None:
    async def run() -> None:
        events = _EventCollector()
        await _pump_model_outputs(
            _PumpAdapter([]),
            _PumpService(),
            worker_id="worker-0",
            session_id="pipeline-1",
            credits=asyncio.BoundedSemaphore(_MODEL_OUTPUT_PARENT_QUEUE_SIZE),
            events=events,
        )

        assert [(item["type"], item["worker_id"], item["session_id"]) for item in events.items] == [
            ("model_output_eos", "worker-0", "pipeline-1")
        ]

    asyncio.run(run())


def test_parent_eos_finishes_pull_and_releases_route() -> None:
    async def run() -> None:
        pool = _pool()
        sent: list[tuple[str, dict[str, Any]]] = []
        pool._send = lambda worker_id, command: sent.append((worker_id, command))
        pool.create_model_session("worker-0", "pipeline-1", {})
        sent.clear()

        chunks = pool.pull_model_chunks("pipeline-1")
        pool._dispatch_event(_model_output_eos("pipeline-1"))
        try:
            await asyncio.wait_for(chunks.__anext__(), timeout=0.2)
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("parent pull did not finish after child model-output EOF")

        assert "pipeline-1" not in pool._pipeline_routes
        assert "pipeline-1" not in pool._session_workers
        assert "pipeline-1" not in pool._model_outputs
        assert sent == [("worker-0", {"type": "model_close", "session_id": "pipeline-1"})]
        await chunks.aclose()

    asyncio.run(run())


def test_parent_queue_preserves_preview_then_replaces_stale_video_and_returns_credit() -> None:
    async def run() -> None:
        pool = _pool()
        sent: list[tuple[str, dict[str, Any]]] = []
        pool._send = lambda worker_id, command: sent.append((worker_id, command))
        pool.create_model_session("worker-0", "pipeline-1", {})
        sent.clear()

        pool._dispatch_event(_model_output("pipeline-1", {"type": "preview", "index": -1}))
        pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "index": 0}))
        assert pool._model_outputs["pipeline-1"].qsize() == 1
        assert pool._model_output_dropped["pipeline-1"] == 1

        chunks = pool.pull_model_chunks("pipeline-1")
        assert (await chunks.__anext__())["type"] == "preview"
        pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "index": 1}))
        pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "index": 2}))
        assert (await chunks.__anext__())["index"] == 2
        snapshot = pool.turboserve_snapshot()["model_output_flow_control"]
        assert snapshot["parent_queue_capacity"] == 1
        assert snapshot["max_materialized_payloads_per_session"] == 2
        assert snapshot["dropped_payloads"] == {"pipeline-1": 2}
        credits = [command for _, command in sent if command["type"] == "model_output_credit"]
        assert [command["session_id"] for command in credits] == ["pipeline-1"] * 4
        await chunks.aclose()

    asyncio.run(run())


def test_parent_queue_prioritizes_terminal_payload_over_queued_video() -> None:
    async def run() -> None:
        pool = _pool()
        pool._send = lambda worker_id, command: None
        pool.create_model_session("worker-0", "pipeline-1", {})
        pool._dispatch_event(_model_output("pipeline-1", {"type": "preview"}))
        pool._dispatch_event(_model_output("pipeline-1", {"type": "error", "error": "model failed"}))

        chunks = pool.pull_model_chunks("pipeline-1")
        payload = await chunks.__anext__()
        assert payload == {"type": "error", "error": "model failed"}
        assert pool._model_output_dropped["pipeline-1"] == 1
        await chunks.aclose()

    asyncio.run(run())


def test_initial_start_builds_one_nccl_group_after_all_workers(monkeypatch) -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = set()
        init_sizes: list[int] = []

        async def fake_parent_scale_to(self, target_workers: int) -> int:
            self._active_workers = set(list(self._specs)[:target_workers])
            return len(self._active_workers)

        async def fake_parent_start(self, *, skip_validation: bool = False) -> None:
            assert skip_validation
            # This mirrors ProcessLiveKitWorkerPool.start: its virtual
            # scale_to calls occur once for each sequential worker startup.
            await self.scale_to(1)
            await self.scale_to(2)

        async def fake_init_nccl() -> None:
            init_sizes.append(len(pool._active_workers))
            pool._nccl_ranks = {worker_id: index for index, worker_id in enumerate(sorted(pool._active_workers))}

        monkeypatch.setattr(ProcessLiveKitWorkerPool, "scale_to", fake_parent_scale_to)
        monkeypatch.setattr(ProcessLiveKitWorkerPool, "start", fake_parent_start)
        pool._init_nccl = fake_init_nccl

        await pool.start(skip_validation=True)

        assert init_sizes == [2]
        assert not pool._initializing_workers
        assert pool._nccl_ranks == {"worker-0": 0, "worker-1": 1}

    asyncio.run(run())


def test_init_nccl_uses_dedicated_parent_timeout() -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = {"worker-0", "worker-1"}
        requests: list[tuple[str, str, dict[str, object]]] = []

        async def fake_request(worker_id: str, command_type: str, **kwargs: object) -> dict[str, object]:
            requests.append((worker_id, command_type, kwargs))
            return {"result": True}

        pool._request = fake_request
        await pool._init_nccl()

        assert [(worker_id, command_type) for worker_id, command_type, _ in requests] == [
            ("worker-0", "nccl_init"),
            ("worker-1", "nccl_init"),
            ("worker-0", "nccl_warmup_collective"),
            ("worker-1", "nccl_warmup_collective"),
            ("worker-0", "nccl_warmup_peer"),
            ("worker-1", "nccl_warmup_peer"),
        ]
        init_requests = [kwargs for _, command_type, kwargs in requests if command_type == "nccl_init"]
        assert [kwargs["rank"] for kwargs in init_requests] == [0, 1]
        assert all(kwargs["world_size"] == 2 for kwargs in init_requests)
        assert all(kwargs["timeout"] == _NCCL_INIT_PARENT_TIMEOUT_SECONDS for _, _, kwargs in requests)
        assert len({kwargs["init_method"] for kwargs in init_requests}) == 1
        warmups = [kwargs for _, command_type, kwargs in requests if command_type == "nccl_warmup_peer"]
        assert warmups == [
            {
                "peer_rank": 1,
                "send_first": True,
                "timeout": _NCCL_INIT_PARENT_TIMEOUT_SECONDS,
            },
            {
                "peer_rank": 0,
                "send_first": False,
                "timeout": _NCCL_INIT_PARENT_TIMEOUT_SECONDS,
            },
        ]
        assert pool._nccl_ranks == {"worker-0": 0, "worker-1": 1}

    asyncio.run(run())


def test_parent_progress_follows_dequeued_payload_owner_after_route_change() -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = {"worker-0", "worker-1"}
        sent: list[tuple[str, dict[str, Any]]] = []
        pool._send = lambda worker_id, command: sent.append((worker_id, command))
        pool.create_model_session("worker-0", "pipeline-1", {})
        sent.clear()
        pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "frames": [object(), object()]}))
        chunks = pool.pull_model_chunks("pipeline-1")
        assert (await chunks.__anext__())["type"] == "chunk"
        pool._pipeline_routes["pipeline-1"] = "worker-1"
        assert pool.report_publisher_frame_progress(
            "pipeline-1", event="submitted", frames_delta=-1, observed_monotonic_seconds=12.0
        )
        progress = [
            (worker_id, command) for worker_id, command in sent if command["type"] == "model_publisher_frame_progress"
        ]
        assert progress == [
            (
                "worker-0",
                {
                    "type": "model_publisher_frame_progress",
                    "session_id": "pipeline-1",
                    "event": "submitted",
                    "frames_delta": -1,
                    "sequence": 0,
                    "observed_monotonic_seconds": 12.0,
                },
            )
        ]
        await chunks.aclose()
        assert "pipeline-1" not in pool._model_output_inflight_owner

    asyncio.run(run())


def test_latest_parent_queue_replacement_rebates_source_publisher_credit() -> None:
    pool = _pool()
    sent: list[tuple[str, dict[str, Any]]] = []
    pool._send = lambda worker_id, command: sent.append((worker_id, command))
    pool.create_model_session("worker-0", "pipeline-1", {})
    sent.clear()
    pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "index": 0, "frames": [object()] * 12}))
    pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "index": 1, "frames": [object()] * 12}))
    progress = [
        (worker_id, command) for worker_id, command in sent if command["type"] == "model_publisher_frame_progress"
    ]
    assert progress == [
        (
            "worker-0",
            {
                "type": "model_publisher_frame_progress",
                "session_id": "pipeline-1",
                "event": "dropped",
                "frames_delta": -12,
                "sequence": 0,
                "observed_monotonic_seconds": progress[0][1]["observed_monotonic_seconds"],
            },
        )
    ]


def test_late_control_for_released_route_is_dropped() -> None:
    pool = _pool()
    sent: list[tuple[str, dict[str, Any]]] = []
    pool._send = lambda worker_id, command: sent.append((worker_id, command))

    pool.push_model_chunk("pipeline-finished", {"type": "stop"})
    pool.push_model_batch(
        [
            ("pipeline-finished", {"type": "reset"}),
            ("pipeline-current", {"type": "control"}),
        ]
    )

    assert sent == []


def test_migration_drain_waits_for_child_queue_and_publisher_frames(monkeypatch) -> None:
    async def run() -> None:
        pool = _pool()
        calls: list[tuple[str, object]] = []
        statuses = iter(
            [
                # Child Fq remains: the parent cannot accept this snapshot.
                {"in_flight": False, "output_queue_empty": False, "publisher_unsubmitted_frames": 0},
                # Fq has crossed to the publisher, but child Fp remains.
                {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 12},
                # First zero snapshot is followed by a parent-drain barrier.
                {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 0},
                # Only a second zero status after that barrier is safe.
                {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 0},
            ]
        )

        async def fake_wait_for_model_output_drain(session_id: str, *, timeout: float) -> None:
            assert session_id == "pipeline-1"
            assert timeout > 0
            calls.append(("parent_drain", None))

        async def fake_request(worker_id: str, request_type: str, **kwargs: object) -> dict[str, object]:
            assert worker_id == "worker-0"
            assert request_type == "model_output_drain_status"
            assert kwargs["session_id"] == "pipeline-1"
            assert float(kwargs["timeout"]) > 0
            status = next(statuses)
            calls.append(("child_status", status))
            return {"result": status}

        monkeypatch.setattr(pool, "_wait_for_model_output_drain", fake_wait_for_model_output_drain)
        monkeypatch.setattr(pool, "_request", fake_request)

        await pool._drain_model_outputs_for_migration("pipeline-1", source_worker_id="worker-0", timeout=1.0)

        assert [kind for kind, _ in calls] == [
            "parent_drain",
            "child_status",
            "parent_drain",
            "child_status",
            "parent_drain",
            "child_status",
            "parent_drain",
            "child_status",
        ]
        assert [value for kind, value in calls if kind == "child_status"] == [
            {"in_flight": False, "output_queue_empty": False, "publisher_unsubmitted_frames": 0},
            {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 12},
            {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 0},
            {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 0},
        ]



def test_migration_records_transport_phase_diagnostics(monkeypatch) -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = {"worker-0", "worker-1"}
        pool._nccl_ranks = {"worker-0": 0, "worker-1": 1}
        pool._pipeline_routes["pipeline-1"] = "worker-0"
        pool._session_workers["pipeline-1"] = "worker-0"
        pool._ownership.register("pipeline-1", "worker-0")
        pool._send = lambda worker_id, command: None

        async def fake_parent_drain(session_id: str, *, timeout: float) -> None:
            assert session_id == "pipeline-1"
            assert timeout > 0

        async def fake_request(worker_id: str, request_type: str, **kwargs: object) -> dict[str, object]:
            del worker_id, kwargs
            if request_type == "model_output_drain_status":
                return {
                    "result": {
                        "in_flight": False,
                        "output_queue_empty": True,
                        "publisher_unsubmitted_frames": 0,
                    }
                }
            if request_type == "nccl_export":
                return {"result": {"tensor_manifest": [], "state_bytes": 128}}
            return {"result": True}

        monkeypatch.setattr(pool, "_wait_for_model_output_drain", fake_parent_drain)
        monkeypatch.setattr(pool, "_request", fake_request)

        ownership = await pool.migrate_session("pipeline-1", "worker-1")

        assert ownership.worker_id == "worker-1"
        diagnostics = pool.turboserve_snapshot()["migration_diagnostics"]
        assert diagnostics["attempts_total"] == 1
        assert diagnostics["success_total"] == 1
        assert diagnostics["failure_total"] == 0
        assert diagnostics["active"] == 0
        assert diagnostics["last"]["outcome"] == "success"
        assert diagnostics["last"]["state_bytes"] == 128
        assert all(diagnostics["phase_timings"][phase]["success"] == 1 for phase in (
            "drain", "pause", "export", "prepare_recv", "transfer", "commit_source", "route_commit"
        ))

    asyncio.run(run())


def test_migration_failure_records_failed_phase_and_error_kind(monkeypatch) -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = {"worker-0", "worker-1"}
        pool._nccl_ranks = {"worker-0": 0, "worker-1": 1}
        pool._pipeline_routes["pipeline-1"] = "worker-0"
        pool._session_workers["pipeline-1"] = "worker-0"
        pool._ownership.register("pipeline-1", "worker-0")
        pool._send = lambda worker_id, command: None

        async def fake_parent_drain(session_id: str, *, timeout: float) -> None:
            del session_id, timeout

        async def fake_request(worker_id: str, request_type: str, **kwargs: object) -> dict[str, object]:
            del worker_id, kwargs
            if request_type == "model_output_drain_status":
                return {
                    "result": {
                        "in_flight": False,
                        "output_queue_empty": True,
                        "publisher_unsubmitted_frames": 0,
                    }
                }
            if request_type == "nccl_export":
                return {"result": {"tensor_manifest": []}}
            if request_type == "nccl_send":
                raise RuntimeError("Worker process worker-0 is not alive")
            return {"result": True}

        monkeypatch.setattr(pool, "_wait_for_model_output_drain", fake_parent_drain)
        monkeypatch.setattr(pool, "_request", fake_request)

        with pytest.raises(RuntimeError, match="Worker process"):
            await pool.migrate_session("pipeline-1", "worker-1")

        diagnostics = pool.turboserve_snapshot()["migration_diagnostics"]
        assert diagnostics["attempts_total"] == 1
        assert diagnostics["success_total"] == 0
        assert diagnostics["failure_total"] == 1
        assert diagnostics["active"] == 0
        assert diagnostics["last_failure"]["failed_phase"] == "transfer"
        assert diagnostics["last_failure"]["error_kind"] == "worker_unavailable"
        assert diagnostics["error_counts"] == {"worker_unavailable": 1}
        assert diagnostics["phase_timings"]["transfer"]["failures"] == 1

    asyncio.run(run())


def test_source_cleanup_failure_preserves_committed_target_owner(monkeypatch) -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = {"worker-0", "worker-1"}
        pool._nccl_ranks = {"worker-0": 0, "worker-1": 1}
        pool._pipeline_routes["pipeline-1"] = "worker-0"
        pool._session_workers["pipeline-1"] = "worker-0"
        pool._ownership.register("pipeline-1", "worker-0")
        pool._send = lambda worker_id, command: None
        requests: list[str] = []

        async def fake_parent_drain(session_id: str, *, timeout: float) -> None:
            del session_id, timeout

        async def fake_request(worker_id: str, request_type: str, **kwargs: object) -> dict[str, object]:
            del worker_id, kwargs
            requests.append(request_type)
            if request_type == "model_output_drain_status":
                return {
                    "result": {
                        "in_flight": False,
                        "output_queue_empty": True,
                        "publisher_unsubmitted_frames": 0,
                    }
                }
            if request_type == "nccl_export":
                return {"result": {"tensor_manifest": [], "state_bytes": 128}}
            if request_type == "nccl_commit_source":
                raise RuntimeError("source cleanup unavailable")
            return {"result": {"groups": []}}

        monkeypatch.setattr(pool, "_wait_for_model_output_drain", fake_parent_drain)
        monkeypatch.setattr(pool, "_request", fake_request)

        ownership = await pool.migrate_session("pipeline-1", "worker-1")

        assert ownership.worker_id == "worker-1"
        assert pool._ownership.owner("pipeline-1").worker_id == "worker-1"
        assert pool._pipeline_routes["pipeline-1"] == "worker-1"
        assert "nccl_discard" not in requests
        assert "nccl_abort_source" not in requests
        snapshot = pool.turboserve_snapshot()
        assert snapshot["migration_cleanup_failures"] == 1
        assert snapshot["migration_diagnostics"]["success_total"] == 1
        assert snapshot["migration_diagnostics"]["phase_timings"]["route_commit"]["success"] == 1
        assert snapshot["migration_diagnostics"]["phase_timings"]["commit_source"]["failures"] == 1

    asyncio.run(run())


def test_migration_pause_failure_resumes_source(monkeypatch) -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = {"worker-0", "worker-1"}
        pool._nccl_ranks = {"worker-0": 0, "worker-1": 1}
        pool._pipeline_routes["pipeline-1"] = "worker-0"
        pool._session_workers["pipeline-1"] = "worker-0"
        pool._ownership.register("pipeline-1", "worker-0")
        sent: list[tuple[str, dict[str, object]]] = []
        requests: list[tuple[str, str]] = []
        pool._send = lambda worker_id, command: sent.append((worker_id, command))
        paused = False

        async def fake_parent_drain(session_id: str, *, timeout: float) -> None:
            del session_id, timeout

        async def fake_request(worker_id: str, request_type: str, **kwargs: object) -> dict[str, object]:
            nonlocal paused
            requests.append((worker_id, request_type))
            del worker_id, kwargs
            if request_type == "model_output_drain_status":
                if paused:
                    raise RuntimeError("status barrier unavailable")
                return {
                    "result": {
                        "in_flight": False,
                        "output_queue_empty": True,
                        "publisher_unsubmitted_frames": 0,
                    }
                }
            if request_type == "model_output_pause":
                paused = True
            return {"result": True}

        monkeypatch.setattr(pool, "_wait_for_model_output_drain", fake_parent_drain)
        monkeypatch.setattr(pool, "_request", fake_request)

        with pytest.raises(RuntimeError, match="status barrier unavailable"):
            await pool.migrate_session("pipeline-1", "worker-1")

        assert ("worker-0", "model_output_resume") in requests
        diagnostics = pool.turboserve_snapshot()["migration_diagnostics"]
        assert diagnostics["failure_total"] == 1
        assert diagnostics["last_failure"]["failed_phase"] == "pause"

    asyncio.run(run())


def test_nccl_start_failure_event_records_worker_exit() -> None:
    async def run() -> None:
        pool = _pool()
        startup = asyncio.get_running_loop().create_future()
        pool._startup["worker-1"] = startup

        NCCLProcessLiveKitWorkerPool._dispatch_event(
            pool,
            {
                "type": "worker_start_failed",
                "worker_id": "worker-1",
                "error": "RuntimeError: CUDA out of memory",
            },
        )

        snapshot = pool.turboserve_snapshot()["migration_diagnostics"]
        assert snapshot["worker_exits_total"] == 1
        assert snapshot["worker_exits_by_code"] == {"unknown": 1}
        assert snapshot["worker_exit_error_counts"] == {"oom": 1}
        assert snapshot["last_worker_exit"]["worker_id"] == "worker-1"
        assert snapshot["last_worker_exit"]["error_kind"] == "oom"
        assert startup.done()
        startup_error = startup.exception()
        assert isinstance(startup_error, RuntimeError)
        assert "CUDA out of memory" in str(startup_error)

    asyncio.run(run())

def test_migration_cancellation_rolls_back_state(monkeypatch) -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = {"worker-0", "worker-1"}
        pool._nccl_ranks = {"worker-0": 0, "worker-1": 1}
        pool._pipeline_routes["pipeline-1"] = "worker-0"
        pool._session_workers["pipeline-1"] = "worker-0"
        pool._ownership.register("pipeline-1", "worker-0")
        requests: list[str] = []
        pool._send = lambda worker_id, command: None

        async def fake_parent_drain(session_id: str, *, timeout: float) -> None:
            del session_id, timeout

        async def fake_request(worker_id: str, request_type: str, **kwargs: object) -> dict[str, object]:
            del worker_id, kwargs
            requests.append(request_type)
            if request_type == "model_output_drain_status":
                return {
                    "result": {
                        "in_flight": False,
                        "output_queue_empty": True,
                        "publisher_unsubmitted_frames": 0,
                    }
                }
            if request_type == "nccl_export":
                return {"result": {"tensor_manifest": [], "state_bytes": 64}}
            if request_type == "nccl_send":
                raise asyncio.CancelledError()
            return {"result": True}

        monkeypatch.setattr(pool, "_wait_for_model_output_drain", fake_parent_drain)
        monkeypatch.setattr(pool, "_request", fake_request)

        with pytest.raises(asyncio.CancelledError):
            await pool.migrate_session("pipeline-1", "worker-1")

        assert "nccl_discard" in requests
        assert "nccl_abort_source" in requests
        assert "model_output_resume" in requests
        assert "pipeline-1" not in pool._migrating_controls
        assert pool._ownership.owner("pipeline-1").worker_id == "worker-0"
        diagnostics = pool.turboserve_snapshot()["migration_diagnostics"]
        assert diagnostics["attempts_total"] == 1
        assert diagnostics["success_total"] == 0
        assert diagnostics["failure_total"] == 0
        assert diagnostics["aborted_total"] == 1
        assert diagnostics["active"] == 0
        assert diagnostics["last"]["outcome"] == "aborted"

    asyncio.run(run())


def test_progressive_migration_routes_compute_before_residual_copy_finishes(monkeypatch) -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = {"worker-0", "worker-1"}
        pool._nccl_ranks = {"worker-0": 0, "worker-1": 1}
        pool._pipeline_routes["pipeline-1"] = "worker-0"
        pool._session_workers["pipeline-1"] = "worker-0"
        pool._ownership.register("pipeline-1", "worker-0")
        pool._model_outputs["pipeline-1"] = asyncio.Queue(maxsize=1)
        sent: list[tuple[str, dict[str, object]]] = []
        pool._send = lambda worker_id, command: sent.append((worker_id, command))
        transfer_started = asyncio.Event()
        transfer_release = asyncio.Event()
        active_transfer_requests: set[str] = set()

        async def fake_parent_drain(session_id: str, *, timeout: float) -> None:
            del session_id, timeout

        async def fake_request(worker_id: str, request_type: str, **kwargs: object) -> dict[str, object]:
            del worker_id, kwargs
            if request_type == "model_output_drain_status":
                return {
                    "result": {
                        "in_flight": False,
                        "output_queue_empty": True,
                        "publisher_unsubmitted_frames": 0,
                    }
                }
            if request_type == "nccl_export":
                return {"result": {"tensor_manifest": [], "state_bytes": 128}}
            if request_type in {"nccl_send", "nccl_recv"}:
                active_transfer_requests.add(request_type)
                if len(active_transfer_requests) == 2:
                    transfer_started.set()
                await transfer_release.wait()
                return {"result": {"groups": []}}
            return {"result": True}

        monkeypatch.setattr(pool, "_wait_for_model_output_drain", fake_parent_drain)
        monkeypatch.setattr(pool, "_request", fake_request)
        compute_ready = asyncio.Event()
        migration = asyncio.create_task(
            pool.migrate_session("pipeline-1", "worker-1", on_compute_ready=compute_ready.set)
        )
        await asyncio.wait_for(transfer_started.wait(), timeout=1.0)
        transfer_id = next(iter(pool._migration_ready_waiters))
        pool._dispatch_event(
            {
                "type": "nccl_first_layer_ready",
                "worker_id": "worker-1",
                "transfer_id": transfer_id,
                "session_id": "pipeline-1",
            }
        )
        await asyncio.wait_for(compute_ready.wait(), timeout=1.0)

        assert not migration.done()
        assert pool._pipeline_routes["pipeline-1"] == "worker-1"
        assert pool._ownership.owner("pipeline-1").worker_id == "worker-0"
        pool.push_model_chunk("pipeline-1", {"type": "action", "action": ["W"]})
        assert sent[-1][0] == "worker-1"
        pool._dispatch_event(
            {
                "type": "model_output",
                "worker_id": "worker-1",
                "session_id": "pipeline-1",
                "payload": {"type": "chunk", "frames": [1]},
            }
        )
        assert pool._model_outputs["pipeline-1"].empty()

        transfer_release.set()
        ownership = await migration

        assert ownership.worker_id == "worker-1"
        assert pool._ownership.owner("pipeline-1").worker_id == "worker-1"
        queued = pool._model_outputs["pipeline-1"].get_nowait()
        assert queued.payload["frames"] == [1]
        assert "pipeline-1" not in pool._provisional_migration_controls
        assert "pipeline-1" not in pool._provisional_model_events

    asyncio.run(run())


def test_progressive_migration_failure_replays_controls_and_discards_output(monkeypatch) -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = {"worker-0", "worker-1"}
        pool._nccl_ranks = {"worker-0": 0, "worker-1": 1}
        pool._pipeline_routes["pipeline-1"] = "worker-0"
        pool._session_workers["pipeline-1"] = "worker-0"
        pool._ownership.register("pipeline-1", "worker-0")
        pool._model_outputs["pipeline-1"] = asyncio.Queue(maxsize=1)
        sent: list[tuple[str, dict[str, object]]] = []
        pool._send = lambda worker_id, command: sent.append((worker_id, command))
        transfer_started = asyncio.Event()
        transfer_release = asyncio.Event()
        active_transfer_requests: set[str] = set()

        async def fake_parent_drain(session_id: str, *, timeout: float) -> None:
            del session_id, timeout

        async def fake_request(worker_id: str, request_type: str, **kwargs: object) -> dict[str, object]:
            del worker_id, kwargs
            if request_type == "model_output_drain_status":
                return {
                    "result": {
                        "in_flight": False,
                        "output_queue_empty": True,
                        "publisher_unsubmitted_frames": 0,
                    }
                }
            if request_type == "nccl_export":
                return {"result": {"tensor_manifest": [], "state_bytes": 128}}
            if request_type in {"nccl_send", "nccl_recv"}:
                active_transfer_requests.add(request_type)
                if len(active_transfer_requests) == 2:
                    transfer_started.set()
                await transfer_release.wait()
                if request_type == "nccl_send":
                    raise RuntimeError("late NCCL failure")
                return {"result": {"groups": []}}
            return {"result": True}

        monkeypatch.setattr(pool, "_wait_for_model_output_drain", fake_parent_drain)
        monkeypatch.setattr(pool, "_request", fake_request)
        compute_ready = asyncio.Event()
        migration = asyncio.create_task(
            pool.migrate_session("pipeline-1", "worker-1", on_compute_ready=compute_ready.set)
        )
        await asyncio.wait_for(transfer_started.wait(), timeout=1.0)
        transfer_id = next(iter(pool._migration_ready_waiters))
        pool._dispatch_event(
            {
                "type": "nccl_first_layer_ready",
                "worker_id": "worker-1",
                "transfer_id": transfer_id,
                "session_id": "pipeline-1",
            }
        )
        await asyncio.wait_for(compute_ready.wait(), timeout=1.0)
        control = {"type": "action", "action": ["D"]}
        pool.push_model_chunk("pipeline-1", control)
        pool._dispatch_event(
            {
                "type": "model_output",
                "worker_id": "worker-1",
                "session_id": "pipeline-1",
                "payload": {"type": "chunk", "frames": [2]},
            }
        )

        transfer_release.set()
        with pytest.raises(RuntimeError, match="late NCCL failure"):
            await migration

        assert pool._pipeline_routes["pipeline-1"] == "worker-0"
        assert pool._ownership.owner("pipeline-1").worker_id == "worker-0"
        assert pool._model_outputs["pipeline-1"].empty()
        replayed = [
            command
            for worker_id, command in sent
            if worker_id == "worker-0" and command.get("type") == "model_push"
        ]
        assert replayed[-1]["chunk"] == control
        assert "pipeline-1" not in pool._provisional_migration_controls
        assert "pipeline-1" not in pool._provisional_model_events

    asyncio.run(run())
