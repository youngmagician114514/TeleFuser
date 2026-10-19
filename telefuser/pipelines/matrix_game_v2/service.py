"""Decoupled bidirectional TeleFuser service for Matrix-Game 2.0."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from telefuser.service.livekit.model_contracts import ModelServingCapabilities
from telefuser.utils.logging import logger

from .action import MatrixGameAction
from .backend import (
    MatrixGameBackendConfig,
    MatrixGameSessionBackend,
    MatrixGameStep,
    create_matrix_game_backend,
)


_END_OF_SESSION = object()
_DEFAULT_CAPABILITIES = ModelServingCapabilities(
    model_name="matrix-game-2.0",
    output_fps=25.0,
    frames_per_job=12,
    supports_cross_session_batching=False,
    supports_session_state_transfer=False,
)


@dataclass
class _SessionState:
    backend_session: Any
    output_queue: queue.Queue[object]
    in_flight: bool = False
    pending_action: MatrixGameAction | None = None
    close_requested: bool = False
    backend_closed: bool = False
    terminal_enqueued: bool = False
    jobs_submitted: int = 0
    jobs_completed: int = 0
    last_error: str | None = None
    created_at: float = field(default_factory=time.monotonic)


class MatrixGameV2Service:
    """Expose Matrix-Game sessions through TeleFuser's bidirectional contract.

    The service owns only scheduling-safe session bookkeeping.  Model state,
    action encoding, and frame generation stay in the injected backend.  A
    session has one in-flight invocation; while it is running, the newest
    action replaces the pending action so stale controls cannot accumulate.
    """

    def __init__(
        self,
        *,
        backend: MatrixGameSessionBackend | None = None,
        backend_factory: Callable[[], MatrixGameSessionBackend] | None = None,
        backend_config: MatrixGameBackendConfig | None = None,
        output_queue_size: int = 4,
        max_workers: int | None = None,
        close_timeout: float = 120.0,
    ) -> None:
        if backend is not None and backend_factory is not None:
            raise ValueError("provide backend or backend_factory, not both")
        if output_queue_size <= 0:
            raise ValueError("output_queue_size must be positive")
        if close_timeout <= 0:
            raise ValueError("close_timeout must be positive")
        self._backend = backend
        self._backend_factory = backend_factory
        self._backend_config = backend_config or MatrixGameBackendConfig()
        self.output_queue_size = int(output_queue_size)
        self.close_timeout = float(close_timeout)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._sessions: dict[str, _SessionState] = {}
        self._sessions_lock = threading.RLock()
        self._capacity_profile: dict[str, object] | None = None
        self._started = False
        self._metrics = {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_coalesced": 0,
            "generation_errors": 0,
        }

    @property
    def serving_capabilities(self) -> ModelServingCapabilities:
        backend = self._backend
        capabilities = getattr(backend, "capabilities", None) if backend is not None else None
        return capabilities if isinstance(capabilities, ModelServingCapabilities) else _DEFAULT_CAPABILITIES

    def start(self) -> None:
        if self._started:
            return
        if self._backend is None:
            factory = self._backend_factory
            self._backend = factory() if factory is not None else create_matrix_game_backend(self._backend_config)
        self._backend.start()
        self._started = True
        logger.info("MatrixGameV2Service started: capabilities=%s", self.serving_capabilities)

    def stop(self) -> None:
        with self._sessions_lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self.close_session(session_id)
        # A close request only marks an in-flight invocation.  Wait for those
        # invocations before tearing down the model backend, otherwise a
        # backend could be stopped while its CUDA call is still using state.
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._backend is not None and self._started:
            self._backend.stop()
        self._started = False

    def configure_session_capacity(self, max_sessions: int | None = None) -> dict[str, object]:
        """Set a conservative retained-session cap without model assumptions."""

        if max_sessions is not None and max_sessions < 1:
            raise ValueError("max_sessions must be positive when provided")
        with self._sessions_lock:
            if self._sessions:
                raise RuntimeError("cannot configure session capacity with active sessions")
            if self._capacity_profile is not None:
                configured = self._capacity_profile["configured_limit"]
                if configured != max_sessions:
                    raise RuntimeError(
                        f"session capacity already configured with limit {configured}, requested {max_sessions}"
                    )
                return dict(self._capacity_profile)
            backend_limit = getattr(self._backend, "max_sessions", None)
            effective = max_sessions if max_sessions is not None else backend_limit
            if effective is None:
                effective = 1
            effective = max(1, int(effective))
            self._capacity_profile = {
                "configured_limit": max_sessions,
                "effective_capacity": effective,
                "computed_capacity": effective,
                "model_name": self.serving_capabilities.model_name,
                "supports_cross_session_batching": self.serving_capabilities.supports_cross_session_batching,
                "supports_session_state_transfer": self.serving_capabilities.supports_session_state_transfer,
            }
            return dict(self._capacity_profile)

    def session_capacity_profile(self) -> dict[str, object] | None:
        with self._sessions_lock:
            return dict(self._capacity_profile) if self._capacity_profile is not None else None

    def _effective_capacity(self) -> int:
        profile = self._capacity_profile
        if profile is not None:
            return int(profile["effective_capacity"])
        backend_limit = getattr(self._backend, "max_sessions", None)
        return max(1, int(backend_limit)) if backend_limit is not None else 1

    def create_session(self, config: dict) -> str:
        self._require_started()
        session_id = str(config.get("session_id") or config.get("id") or uuid.uuid4()).strip()
        with self._sessions_lock:
            if session_id in self._sessions:
                raise ValueError(f"session already exists: {session_id}")
            if len(self._sessions) >= self._effective_capacity():
                raise RuntimeError(
                    f"Matrix-Game retained-session capacity exhausted "
                    f"(capacity={self._effective_capacity()})"
                )
        assert self._backend is not None
        backend_config = dict(config)
        backend_config["session_id"] = session_id
        backend_session = self._backend.create_session(session_id, backend_config)
        state = _SessionState(backend_session, queue.Queue(maxsize=self.output_queue_size))
        with self._sessions_lock:
            self._sessions[session_id] = state
        return session_id

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        self._require_started()
        if str(chunk.get("type", "")).lower() in {"close", "stop", "eos"}:
            self.close_session(session_id)
            return
        with self._sessions_lock:
            state = self._sessions.get(session_id)
            if state is None or state.close_requested:
                raise KeyError(f"unknown or closed Matrix-Game session: {session_id}")
            action = MatrixGameAction.from_payload(chunk)
            state.jobs_submitted += 1
            self._metrics["jobs_submitted"] += 1
            if state.in_flight:
                state.pending_action = action
                self._metrics["jobs_coalesced"] += 1
                return
            state.in_flight = True
        self._executor.submit(self._run_action, session_id, state, action)

    def push_batch(self, items: list[tuple[str, dict]]) -> None:
        if len(items) > 1 and not self.serving_capabilities.supports_cross_session_batching:
            raise RuntimeError(
                "Matrix-Game 2.0 backend does not support cross-session batching; "
                "submit each session as an individual job"
            )
        for session_id, chunk in items:
            self.push_chunk(session_id, chunk)

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        with self._sessions_lock:
            state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"unknown Matrix-Game session: {session_id}")
        while True:
            item = await asyncio.to_thread(state.output_queue.get)
            if item is _END_OF_SESSION:
                return
            yield item  # type: ignore[misc]

    def close_session(self, session_id: str) -> None:
        with self._sessions_lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            state.close_requested = True
            state.pending_action = None
            if state.in_flight:
                return
            self._finalize_session_locked(session_id, state)

    def has_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            return session_id in self._sessions

    def runtime_metrics(self, session_id: str | None = None) -> dict[str, float | int | str] | None:
        with self._sessions_lock:
            if session_id is None:
                return dict(self._metrics)
            state = self._sessions.get(session_id)
            if state is None:
                return None
            return {
                "jobs_submitted": state.jobs_submitted,
                "jobs_completed": state.jobs_completed,
                "in_flight": int(state.in_flight),
                "pending": int(state.pending_action is not None),
                "age_seconds": max(0.0, time.monotonic() - state.created_at),
                **({"last_error": state.last_error} if state.last_error else {}),
            }

    def _run_action(self, session_id: str, state: _SessionState, action: MatrixGameAction) -> None:
        current = action
        while True:
            try:
                assert self._backend is not None
                result = self._backend.step(state.backend_session, current)
                if not isinstance(result, MatrixGameStep):
                    raise TypeError("Matrix backend step() must return MatrixGameStep")
                fps = float(result.fps or self.serving_capabilities.output_fps)
                payload = {
                    "type": "chunk",
                    "frames": list(result.frames),
                    "fps": fps,
                    "index": int(result.index),
                    "data": {"fps": fps, "index": int(result.index)},
                }
                self._put_output(state, payload)
                with self._sessions_lock:
                    state.jobs_completed += 1
                    self._metrics["jobs_completed"] += 1
            except Exception as exc:
                logger.exception("Matrix-Game session %s generation failed", session_id)
                with self._sessions_lock:
                    state.last_error = str(exc)
                    self._metrics["generation_errors"] += 1
                try:
                    self._put_output(state, {"type": "error", "error": str(exc)})
                except Exception as output_exc:
                    logger.error("Could not publish Matrix-Game error for %s: %s", session_id, output_exc)
                with self._sessions_lock:
                    state.close_requested = True
                break

            with self._sessions_lock:
                if state.close_requested:
                    break
                next_action = state.pending_action
                state.pending_action = None
                if next_action is None:
                    state.in_flight = False
                    return
            current = next_action

        with self._sessions_lock:
            state.in_flight = False
            self._finalize_session_locked(session_id, state)

    def _put_output(self, state: _SessionState, item: object) -> None:
        try:
            state.output_queue.put(item, timeout=self.close_timeout)
        except queue.Full as exc:
            raise RuntimeError("Matrix-Game output queue remained full until close timeout") from exc

    def _finalize_session_locked(self, session_id: str, state: _SessionState) -> None:
        if not state.backend_closed:
            assert self._backend is not None
            try:
                self._backend.close_session(state.backend_session)
            finally:
                state.backend_closed = True
        if not state.terminal_enqueued:
            self._put_output(state, _END_OF_SESSION)
            state.terminal_enqueued = True
        self._sessions.pop(session_id, None)

    def _require_started(self) -> None:
        if not self._started or self._backend is None:
            raise RuntimeError("MatrixGameV2Service has not been started")
