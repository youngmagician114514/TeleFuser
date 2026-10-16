"""Node-local control service used by :mod:`distributed_worker_pool`."""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace
from typing import Any

from telefuser.service.security.security_validator import SecurityLevel
from telefuser.utils.logging import logger

from .config import LiveKitServeConfig
from .nccl_process_worker_pool import NCCLProcessLiveKitWorkerPool
from .process_worker_pool import ProcessWorkerSpec
from .session_registry import SessionRecord
from .token_service import LiveKitTokenService
from .turboserve import TurboServeOwnership
from .worker import NullWorkerEventSink


class _NodeAgentEventSink(NullWorkerEventSink):
    def __init__(self) -> None:
        self.pipeline_ids: dict[str, str] = {}
        self.failures: dict[str, str] = {}
        self.compute_completions: list[dict[str, Any]] = []
        self.dispatch_failures: list[dict[str, Any]] = []
        self.published_chunks: list[dict[str, Any]] = []
        self.model_outputs: list[dict[str, Any]] = []
        self.capacities: dict[str, int] = {}
        self.control_received: list[dict[str, str]] = []
        self.control_messages: list[dict[str, Any]] = []
        self._waiters: dict[str, asyncio.Future[str]] = {}

    def watch_pipeline(self, session_id: str) -> asyncio.Future[str]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        existing = self.pipeline_ids.get(session_id)
        if existing is not None:
            future.set_result(existing)
        else:
            self._waiters[session_id] = future
        return future

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        self.pipeline_ids[session_id] = pipeline_session_id
        waiter = self._waiters.pop(session_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(pipeline_session_id)

    def on_worker_capacity(self, worker_id: str, capacity: int, profile: dict[str, object] | None = None) -> None:
        del profile
        self.capacities[str(worker_id)] = int(capacity)

    def on_control_received(self, worker_id: str, session_id: str) -> None:
        self.control_received.append({"worker_id": str(worker_id), "session_id": str(session_id)})

    def on_control_message(self, worker_id: str, session_id: str, chunk: dict[str, Any]) -> bool:
        # Always forward the raw control to the coordinator.  The coordinator
        # decides whether Motivation owns it; a non-Motivation policy can send
        # the same message back through the remote worker-pool RPC.
        self.control_messages.append(
            {"worker_id": str(worker_id), "session_id": str(session_id), "chunk": dict(chunk)}
        )
        return True

    def drain_control_messages(self) -> list[dict[str, Any]]:
        values, self.control_messages = self.control_messages, []
        return values

    def drain_control_received(self) -> list[dict[str, str]]:
        values, self.control_received = self.control_received, []
        return values

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        del worker_id
        if error:
            self.failures[session_id] = error
        waiter = self._waiters.pop(session_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_exception(RuntimeError(error or f"remote session {session_id!r} finished before readiness"))

    def on_motivation_compute_complete(
        self,
        *,
        job_ids: tuple[str, ...] = (),
        session_ids: tuple[str, ...] = (),
        compatibility_keys: dict[str, str] | None = None,
    ) -> None:
        self.compute_completions.append(
            {
                "job_ids": list(job_ids),
                "session_ids": list(session_ids),
                "compatibility_keys": dict(compatibility_keys or {}),
            }
        )

    def on_motivation_dispatch_failed(
        self,
        *,
        job_ids: tuple[str, ...] = (),
        session_ids: tuple[str, ...] = (),
        error: str = "physical batch dispatch failed",
    ) -> None:
        self.dispatch_failures.append(
            {"job_ids": list(job_ids), "session_ids": list(session_ids), "error": str(error)}
        )

    def drain_compute_completions(self) -> list[dict[str, Any]]:
        values, self.compute_completions = self.compute_completions, []
        return values

    def drain_dispatch_failures(self) -> list[dict[str, Any]]:
        values, self.dispatch_failures = self.dispatch_failures, []
        return values

    def on_chunk_published(
        self, worker_id: str, session_id: str, frames: int, first_frame_at: float | None = None
    ) -> None:
        self.published_chunks.append(
            {
                "worker_id": str(worker_id),
                "session_id": str(session_id),
                "frames": int(frames),
                "first_frame_at": first_frame_at,
            }
        )

    def on_model_output(
        self,
        worker_id: str,
        session_id: str,
        payload: dict,
        runtime_metrics: dict | None = None,
        session_runtime_metrics: dict | None = None,
    ) -> None:
        # Forward only scheduler/metric metadata. PIL frames stay on the node
        # and continue directly into its LiveKit publisher.
        summary: dict[str, Any] = {
            "worker_id": str(worker_id),
            "session_id": str(session_id),
            "payload": {
                key: payload[key]
                for key in ("type", "frame_count", "scheduler", "error")
                if key in payload
            },
            "runtime_metrics": dict(runtime_metrics or {}),
            "session_runtime_metrics": dict(session_runtime_metrics or {}),
        }
        frames = payload.get("frames")
        if isinstance(frames, (list, tuple)):
            summary["payload"]["frame_count"] = len(frames)
        self.model_outputs.append(summary)

    def drain_model_outputs(self) -> list[dict[str, Any]]:
        values, self.model_outputs = self.model_outputs, []
        return values

    def drain_published_chunks(self) -> list[dict[str, Any]]:
        values, self.published_chunks = self.published_chunks, []
        return values


class NodeAgentServer:
    """Host one node-local process-NCCL pool behind a small authenticated RPC."""

    def __init__(
        self,
        *,
        config: LiveKitServeConfig,
        pipeline_file: str,
        security_level: SecurityLevel | None = None,
        skip_validation: bool = False,
    ) -> None:
        if config.cluster_world_size < 2:
            raise ValueError("NodeAgentServer is only needed for cluster_world_size >= 2")
        self.config = config
        self.pipeline_file = pipeline_file
        self._secret = config.node_agent_secret
        self._sink = _NodeAgentEventSink()
        master, _port, _world, rank_offset = config.nccl_rendezvous(
            local_worker_count=config.num_workers
        )
        del master
        specs = [
            ProcessWorkerSpec(
                worker_id=f"{config.node_id}/worker-{index}",
                gpu_ids=list(group),
                node_id=config.node_id,
                node_address=config.effective_node_address(),
                rank=rank_offset + index,
            )
            for index, group in enumerate(config.worker_gpu_groups())
        ]
        self.pool = NCCLProcessLiveKitWorkerPool(
            specs,
            config=config,
            pipeline_file=pipeline_file,
            event_sink=self._sink,
            security_level=security_level,
        )
        self._server: asyncio.AbstractServer | None = None
        self._skip_validation = bool(skip_validation)

    async def start(self) -> None:
        await self.pool.start(skip_validation=self._skip_validation)
        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self.config.host,
            port=self.config.node_agent_port,
            limit=4 * 1024 * 1024,
        )
        logger.info(
            "NodeAgent ready: node=%s address=%s port=%s workers=%s",
            self.config.node_id,
            self.config.effective_node_address(),
            self.config.node_agent_port,
            self.config.num_workers,
        )

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await self.pool.aclose()

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    return
                try:
                    request = json.loads(raw.decode("utf-8"))
                    if self._secret and request.get("secret") != self._secret:
                        raise PermissionError("invalid node-agent secret")
                    result = await self._dispatch(str(request.get("method", "")), request.get("params") or {})
                    response = {"id": request.get("id"), "ok": True, "result": result}
                except Exception as exc:
                    logger.warning("NodeAgent request failed: peer=%s error=%s", peer, exc)
                    response = {"id": request.get("id") if isinstance(request, dict) else None, "ok": False, "error": str(exc)}
                writer.write((json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8"))
                await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "health":
            snapshot = self.pool.turboserve_snapshot()
            return {
                "status": "ready",
                "node_id": self.config.node_id,
                "active_workers": snapshot.get("active_workers", []),
                "num_workers": self.config.num_workers,
                "capacities": dict(self._sink.capacities),
                "active_sessions": sorted(getattr(self.pool, "_transport_workers", {})),
                "compute_completions": self._sink.drain_compute_completions(),
                "dispatch_failures": self._sink.drain_dispatch_failures(),
                "published_chunks": self._sink.drain_published_chunks(),
                "control_messages": self._sink.drain_control_messages(),
                "control_received": self._sink.drain_control_received(),
                "model_outputs": self._sink.drain_model_outputs(),
            }
        if method == "start_session":
            record = SessionRecord.model_validate(params["record"])
            if record.worker_id is None or not record.worker_id.startswith(f"{self.config.node_id}/"):
                raise ValueError(f"session worker {record.worker_id!r} is not local to this node")
            waiter = self._sink.watch_pipeline(record.session_id)
            self.pool.start_session(record)
            pipeline_id = await asyncio.wait_for(waiter, timeout=180.0)
            return {"session_id": record.session_id, "pipeline_session_id": pipeline_id}
        if method == "dispatch_batch":
            worker_id = str(params["worker_id"])
            items = [(str(session_id), dict(chunk)) for session_id, chunk in params.get("items", [])]
            lease = SimpleNamespace(
                candidate=SimpleNamespace(gpu_id=worker_id),
                jobs=tuple(SimpleNamespace(job_id=value) for value in params.get("motivation_job_ids", ())),
            )
            self.pool.dispatch_batch(lease, items)
            return {"accepted": len(items)}
        if method == "stop_session":
            await self.pool.stop_session(str(params["session_id"]))
            return {"stopped": True}
        if method == "migrate_session":
            ownership = await self.pool.migrate_session(
                str(params["pipeline_session_id"]), str(params["target_worker_id"])
            )
            return {
                "session_id": ownership.session_id,
                "worker_id": ownership.worker_id,
                "epoch": ownership.epoch,
            }
        if method == "close":
            await self.aclose()
            return {"closed": True}
        raise ValueError(f"unknown node-agent method {method!r}")


async def run_node_agent(
    *,
    pipe_path: str,
    config: LiveKitServeConfig,
    security_level: SecurityLevel | None = None,
    skip_validation: bool = False,
) -> None:
    """Start a node agent until SIGTERM/KeyboardInterrupt."""
    agent = NodeAgentServer(
        config=config,
        pipeline_file=pipe_path,
        security_level=security_level,
        skip_validation=skip_validation,
    )
    try:
        await agent.serve_forever()
    finally:
        await agent.aclose()


__all__ = ["NodeAgentServer", "run_node_agent"]
