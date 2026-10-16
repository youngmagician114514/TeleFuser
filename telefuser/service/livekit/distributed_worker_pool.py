"""Node-agent backed worker pool for multi-host LiveKit serving.

The coordinator owns admission and policy state, while each node agent owns
its local process-NCCL pool and joins the shared LiveKit room directly.  Only
small control messages cross the node boundary; generated media never takes a
second hop through the coordinator.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import Any

from telefuser.service.security.security_validator import SecurityLevel
from telefuser.utils.logging import logger

from .nccl_process_worker_pool import NCCLProcessLiveKitWorkerPool
from .node_agent import NodeAgentMigrationRequest, NodeAgentMigrationResult
from .process_worker_pool import ProcessWorkerSpec
from .session_registry import SessionRecord
from .token_service import LiveKitTokenService
from .turboserve import TurboServeOwnership
from .topology import NodeAgent, WorkerEndpoint
from .worker_pool import WorkerPool


class NodeAgentRpcError(RuntimeError):
    """An internal node-agent request failed."""


class NodeAgentRpcClient:
    """Small dependency-free newline JSON RPC client.

    Connections are intentionally short lived.  A per-node lock preserves
    start/dispatch/stop ordering without introducing a global coordinator
    lock, and avoids adding another runtime dependency to the serving path.
    """

    def __init__(self, endpoint: WorkerEndpoint, *, secret: str = "", timeout: float = 30.0) -> None:
        if not endpoint.address:
            raise ValueError(f"node endpoint {endpoint.endpoint_id!r} has no address")
        self.endpoint = endpoint
        self.secret = secret
        self.timeout = float(timeout)
        self._lock = asyncio.Lock()
        self._request_id = 0

    @property
    def address(self) -> tuple[str, int]:
        return self.endpoint.address or "127.0.0.1", int(self.endpoint.port or 8090)

    @property
    def node(self) -> NodeAgent:
        return NodeAgent(
            node_id=self.endpoint.node_id,
            address=self.endpoint.address or "127.0.0.1",
            rank=self.endpoint.rank or 0,
            control_port=self.endpoint.port,
        )

    async def call(self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None) -> Any:
        async with self._lock:
            self._request_id += 1
            request = {
                "id": self._request_id,
                "method": str(method),
                "params": params or {},
                "secret": self.secret,
            }
            host, port = self.address
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=timeout or self.timeout
            )
            try:
                writer.write((json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8"))
                await writer.drain()
                raw = await asyncio.wait_for(reader.readline(), timeout=timeout or self.timeout)
                if not raw:
                    raise NodeAgentRpcError(f"node agent {self.endpoint.node_id!r} closed the connection")
                response = json.loads(raw.decode("utf-8"))
                if not response.get("ok", False):
                    raise NodeAgentRpcError(str(response.get("error", "node-agent request failed")))
                return response.get("result")
            finally:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()

    async def health(self) -> dict[str, Any]:
        result = await self.call("health")
        return dict(result) if isinstance(result, dict) else {"status": result}

    async def migrate_session(self, request: NodeAgentMigrationRequest) -> NodeAgentMigrationResult:
        result = await self.call("migrate_session", {
            "pipeline_session_id": request.session_id,
            "target_worker_id": request.target_gpu,
        })
        return NodeAgentMigrationResult(
            operation_id=str(result.get("session_id", request.session_id)),
            ready=True,
            completed=True,
            metadata=dict(result),
        )


class DistributedLiveKitWorkerPool(WorkerPool):
    """Coordinator-side pool spanning local workers and remote node agents."""

    progressive_migration_supported = True

    def __init__(
        self,
        *,
        config: Any,
        pipeline_file: str,
        event_sink: Any,
        security_level: SecurityLevel | None = None,
        skip_validation: bool = False,
    ) -> None:
        self._config = config
        self._pipeline_file = pipeline_file
        self._event_sink = event_sink
        self._skip_validation = bool(skip_validation)
        self._security_level = security_level
        self._started = False
        self._closed = False
        self._session_workers: dict[str, str] = {}
        self._pipeline_sessions: dict[str, str] = {}
        self._session_start_tasks: set[asyncio.Task[Any]] = set()
        self._starting_remote_sessions: set[str] = set()
        self._remote_clients: dict[str, NodeAgentRpcClient] = {}
        self._reconcile_task: asyncio.Task[Any] | None = None

        local_endpoints = config.local_cluster_worker_endpoints()
        master, _port, _world, rank_offset = config.nccl_rendezvous(local_worker_count=len(local_endpoints))
        del master
        specs = [
            ProcessWorkerSpec(
                worker_id=endpoint.endpoint_id,
                gpu_ids=list(endpoint.gpu_ids),
                node_id=endpoint.node_id,
                node_address=endpoint.address,
                rank=rank_offset + index,
            )
            for index, endpoint in enumerate(local_endpoints)
        ]
        if not specs:
            raise ValueError(f"node {config.node_id!r} has no local worker endpoints")
        self._local = NCCLProcessLiveKitWorkerPool(
            specs,
            config=config,
            pipeline_file=pipeline_file,
            event_sink=event_sink,
            security_level=security_level,
        )
        for node in config.topology().nodes:
            if node.node_id == config.node_id:
                continue
            endpoint = WorkerEndpoint(
                worker_id="agent",
                node_id=node.node_id,
                address=node.address,
                port=node.control_port or config.node_agent_port,
                rank=node.rank,
            )
            self._remote_clients[node.node_id] = NodeAgentRpcClient(
                endpoint,
                secret=config.node_agent_secret,
                timeout=300.0,
            )

    def _endpoint_for_worker(self, worker_id: str) -> WorkerEndpoint:
        for endpoint in self._config.cluster_worker_endpoints():
            if endpoint.endpoint_id == worker_id:
                return endpoint
        raise KeyError(f"unknown distributed worker {worker_id!r}")

    def _client_for_worker(self, worker_id: str) -> NodeAgentRpcClient:
        endpoint = self._endpoint_for_worker(worker_id)
        try:
            return self._remote_clients[endpoint.node_id]
        except KeyError as exc:
            raise RuntimeError(f"worker {worker_id!r} is not hosted by this coordinator") from exc

    async def start(self, *, skip_validation: bool = False) -> None:
        if self._started:
            return
        self._skip_validation = bool(skip_validation)
        # Remote agents must be ready before local NCCL initialization: all
        # node-local child pools enter the same global communicator.
        if self._remote_clients:
            health_results = await asyncio.gather(*(client.health() for client in self._remote_clients.values()))
            for node_id, health in zip(self._remote_clients, health_results, strict=True):
                self._apply_remote_capacities(node_id, health)
        await self._local.start(skip_validation=skip_validation)
        self._started = True
        if self._remote_clients:
            self._reconcile_task = asyncio.create_task(
                self._reconcile_remote_sessions(), name="distributed-node-agent-reconcile"
            )

    async def _reconcile_remote_sessions(self) -> None:
        """Release coordinator capacity when a remote room exits unexpectedly."""
        while True:
            await asyncio.sleep(2.0)
            for node_id, client in tuple(self._remote_clients.items()):
                try:
                    health = await client.health()
                except Exception as exc:
                    logger.warning("node-agent health check failed: node=%s error=%s", node_id, exc)
                    continue
                active = {str(session_id) for session_id in health.get("active_sessions", ())}
                self._apply_remote_capacities(node_id, health)
                for received in health.get("control_received", ()):
                    self._event_sink.on_control_received(
                        str(received.get("worker_id", "")), str(received.get("session_id", ""))
                    )
                for message in health.get("control_messages", ()):
                    worker_id = str(message.get("worker_id", ""))
                    session_id = str(message.get("session_id", ""))
                    chunk = dict(message.get("chunk") or {})
                    handled = bool(self._event_sink.on_control_message(worker_id, session_id, chunk))
                    if not handled:
                        self.dispatch_batch(
                            SimpleNamespace(candidate=SimpleNamespace(gpu_id=worker_id), jobs=()),
                            [(session_id, chunk)],
                        )
                for completion in health.get("compute_completions", ()):
                    self._event_sink.on_motivation_compute_complete(
                        job_ids=tuple(str(value) for value in completion.get("job_ids", ())),
                        session_ids=tuple(str(value) for value in completion.get("session_ids", ())),
                        compatibility_keys=dict(completion.get("compatibility_keys", {})),
                    )
                for failure in health.get("dispatch_failures", ()):
                    self._event_sink.on_motivation_dispatch_failed(
                        job_ids=tuple(str(value) for value in failure.get("job_ids", ())),
                        session_ids=tuple(str(value) for value in failure.get("session_ids", ())),
                        error=str(failure.get("error", "physical batch dispatch failed")),
                    )
                for chunk in health.get("published_chunks", ()):
                    self._event_sink.on_chunk_published(
                        str(chunk.get("worker_id", "")),
                        str(chunk.get("session_id", "")),
                        int(chunk.get("frames", 0)),
                        # ``time.monotonic()`` has no cross-host epoch.  Use
                        # the coordinator's receipt time for distributed A2F
                        # accounting instead of subtracting unrelated clocks.
                        None,
                    )
                for output in health.get("model_outputs", ()):
                    self._event_sink.on_model_output(
                        str(output.get("worker_id", "")),
                        str(output.get("session_id", "")),
                        dict(output.get("payload", {})),
                        runtime_metrics=dict(output.get("runtime_metrics", {})),
                        session_runtime_metrics=dict(output.get("session_runtime_metrics", {})),
                    )
                for session_id, worker_id in tuple(self._session_workers.items()):
                    endpoint = self._endpoint_for_worker(worker_id)
                    if (
                        endpoint.node_id != node_id
                        or session_id in active
                        or session_id in self._starting_remote_sessions
                    ):
                        continue
                    self._session_workers.pop(session_id, None)
                    self._event_sink.on_session_finished(worker_id, session_id, "remote worker session exited")

    def _apply_remote_capacities(self, node_id: str, health: dict[str, Any]) -> None:
        del node_id
        callback = getattr(self._event_sink, "on_worker_capacity", None)
        if not callable(callback):
            return
        for worker_id, capacity in dict(health.get("capacities", {})).items():
            callback(str(worker_id), int(capacity), None)

    def start_session(self, record: SessionRecord) -> None:
        if not self._started:
            raise RuntimeError("distributed worker pool is not started")
        if record.worker_id is None:
            raise RuntimeError(f"Session {record.session_id} has no assigned worker")
        self._session_workers[record.session_id] = record.worker_id
        endpoint = self._endpoint_for_worker(record.worker_id)
        if endpoint.node_id == self._config.node_id:
            self._local.start_session(record)
            return
        task = asyncio.create_task(self._start_remote_session(record), name=f"remote-session-{record.session_id}")
        self._starting_remote_sessions.add(record.session_id)
        self._session_start_tasks.add(task)
        task.add_done_callback(self._session_start_tasks.discard)

    async def _start_remote_session(self, record: SessionRecord) -> None:
        client = self._client_for_worker(record.worker_id or "")
        try:
            result = await client.call("start_session", {"record": record.model_dump(mode="json")})
            pipeline_session_id = str(result["pipeline_session_id"])
            self._pipeline_sessions[pipeline_session_id] = record.session_id
            self._event_sink.on_pipeline_session(record.session_id, pipeline_session_id)
            self._event_sink.on_session_status(record.session_id, "running")
        except Exception as exc:
            self._session_workers.pop(record.session_id, None)
            self._event_sink.on_session_status(record.session_id, "failed", error=str(exc))
            self._event_sink.on_session_finished(record.worker_id or "", record.session_id, str(exc))
        finally:
            self._starting_remote_sessions.discard(record.session_id)

    def dispatch_batch(self, lease: Any, payloads: list[tuple[str, dict]]) -> None:
        expected_worker_id = getattr(getattr(lease, "candidate", None), "gpu_id", None)
        grouped: dict[str, list[tuple[str, dict]]] = {}
        for session_id, chunk in payloads:
            worker_id = self._session_workers.get(session_id)
            if worker_id is None:
                raise RuntimeError(f"Session {session_id!r} is not assigned to a distributed worker")
            if expected_worker_id is not None and worker_id != expected_worker_id:
                raise RuntimeError(
                    f"Motivation owner mismatch for {session_id!r}: candidate={expected_worker_id!r} actual={worker_id!r}"
                )
            grouped.setdefault(worker_id, []).append((session_id, dict(chunk)))
        for worker_id, items in grouped.items():
            endpoint = self._endpoint_for_worker(worker_id)
            if endpoint.node_id == self._config.node_id:
                self._local.dispatch_batch(lease, items)
            else:
                task = asyncio.create_task(
                    self._dispatch_remote(
                        worker_id,
                        expected_worker_id,
                        items,
                        job_ids=[str(job.job_id) for job in getattr(lease, "jobs", ())],
                        session_ids=[str(session_id) for session_id, _ in payloads],
                    ),
                    name=f"remote-dispatch-{worker_id}",
                )
                task.add_done_callback(self._log_task_failure)

    async def _dispatch_remote(
        self,
        worker_id: str,
        expected_worker_id: str | None,
        items: list[tuple[str, dict]],
        *,
        job_ids: list[str],
        session_ids: list[str],
    ) -> None:
        del expected_worker_id
        await self._client_for_worker(worker_id).call(
            "dispatch_batch",
            {
                "worker_id": worker_id,
                "items": items,
                "motivation_job_ids": job_ids,
                "motivation_session_ids": session_ids,
            },
        )

    @staticmethod
    def _log_task_failure(task: asyncio.Task[Any]) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error("distributed worker RPC failed: %s", task.exception())

    def dispatch_owner(self, session_id: str) -> str | None:
        return self._session_workers.get(session_id)

    async def stop_session(self, session_id: str) -> None:
        worker_id = self._session_workers.get(session_id)
        if worker_id is None:
            return
        endpoint = self._endpoint_for_worker(worker_id)
        if endpoint.node_id == self._config.node_id:
            await self._local.stop_session(session_id)
        else:
            await self._client_for_worker(worker_id).call("stop_session", {"session_id": session_id})
        self._session_workers.pop(session_id, None)
        self._starting_remote_sessions.discard(session_id)
        for pipeline_id, api_id in tuple(self._pipeline_sessions.items()):
            if api_id == session_id:
                self._pipeline_sessions.pop(pipeline_id, None)

    async def migrate_session(self, pipeline_session_id: str, target_worker_id: str, **kwargs: Any) -> TurboServeOwnership:
        del kwargs
        api_session_id = self._pipeline_sessions.get(pipeline_session_id)
        source_worker_id = self._session_workers.get(api_session_id) if api_session_id is not None else None
        if source_worker_id is None:
            return await self._local.migrate_session(pipeline_session_id, target_worker_id)
        source = self._endpoint_for_worker(source_worker_id)
        target = self._endpoint_for_worker(target_worker_id)
        if source.node_id != target.node_id:
            raise RuntimeError("cross-node session migration is disabled; placement is node-affine")
        if source.node_id == self._config.node_id:
            return await self._local.migrate_session(pipeline_session_id, target_worker_id)
        result = await self._client_for_worker(source_worker_id).call(
            "migrate_session",
            {"pipeline_session_id": pipeline_session_id, "target_worker_id": target_worker_id},
        )
        if api_session_id is not None:
            self._session_workers[api_session_id] = target_worker_id
        return TurboServeOwnership(
            session_id=str(result["session_id"]),
            worker_id=str(result["worker_id"]),
            epoch=int(result["epoch"]),
        )

    async def migrate_session_remote(self, request: NodeAgentMigrationRequest) -> NodeAgentMigrationResult:
        raise RuntimeError(
            "cross-node state transfer is intentionally disabled until a coordinated NCCL handoff is configured; "
            f"keep {request.session_id!r} on {request.source_node_id!r}"
        )

    def active_worker_count(self) -> int:
        return self._local.active_worker_count()

    async def scale_to(self, target_workers: int) -> int:
        if target_workers != self._local.active_worker_count():
            raise RuntimeError("distributed worker pools do not support partial-node autoscaling")
        return target_workers

    def turboserve_snapshot(self) -> dict[str, object]:
        snapshot = dict(self._local.turboserve_snapshot())
        snapshot["distributed"] = True
        snapshot["remote_nodes"] = sorted(self._remote_clients)
        snapshot["cross_node_migration"] = False
        return snapshot

    async def aclose(self) -> None:
        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconcile_task
            self._reconcile_task = None
        for task in tuple(self._session_start_tasks):
            task.cancel()
        if self._session_start_tasks:
            await asyncio.gather(*self._session_start_tasks, return_exceptions=True)
        await self._local.aclose()
        self._started = False
        self._closed = True


__all__ = ["DistributedLiveKitWorkerPool", "NodeAgentRpcClient", "NodeAgentRpcError"]
