from __future__ import annotations

from concurrent.futures import Future

from telefuser.service.livekit.async_migration import AsyncMigrationManager, MigrationRequest
from telefuser.service.livekit.session_state_transfer import (
    SessionStateTransferManager,
    SessionStateTransferRequest,
)


class _Backend:
    def __init__(self) -> None:
        self.operation: Future[object] = Future()
        self.commits = 0

    def begin(self, request: SessionStateTransferRequest) -> Future[object]:
        assert request.session_id == "session-a"
        return self.operation

    @staticmethod
    def ready(operation: Future[object]) -> bool:
        return operation.done()

    def commit(self, operation: Future[object]) -> None:
        operation.result()
        self.commits += 1

    @staticmethod
    def abort(operation: Future[object]) -> None:
        operation.cancel()


class _ProgressiveBackend:
    def __init__(self) -> None:
        self.compute_ready: Future[None] = Future()
        self.residual: Future[None] = Future()

    def begin(self, request: SessionStateTransferRequest) -> object:
        del request
        return self

    def ready(self, operation: object) -> bool:
        assert operation is self
        return self.compute_ready.done()

    def commit(self, operation: object) -> None:
        assert operation is self
        self.compute_ready.result()

    def done(self, operation: object) -> bool:
        assert operation is self
        return self.residual.done()

    def finalize(self, operation: object) -> None:
        assert operation is self
        self.residual.result()

    def abort(self, operation: object) -> None:
        assert operation is self
        self.residual.cancel()


def test_session_state_transfer_manager_owns_lifecycle_only() -> None:
    backend = _Backend()
    manager = SessionStateTransferManager(clock=lambda: 1.0)
    request = SessionStateTransferRequest("session-a", "gpu-0", "gpu-1", 0.0, 0.5, state_bytes=128)

    record = manager.begin(request, backend)
    assert record.state == "precopied"
    assert manager.poll("session-a").state == "precopied"

    backend.operation.set_result(True)
    ready = manager.poll("session-a", now=2.0)
    assert ready.state == "ready"
    completed = manager.commit("session-a", now=3.0)
    assert completed.state == "completed"
    assert completed.transfer_id == record.transfer_id
    assert backend.commits == 1
    assert manager.active() == ()


def test_legacy_migration_names_remain_compatible() -> None:
    assert AsyncMigrationManager is SessionStateTransferManager
    assert MigrationRequest is SessionStateTransferRequest


def test_progressive_transfer_remains_active_until_residual_copy_completes() -> None:
    backend = _ProgressiveBackend()
    manager = SessionStateTransferManager(clock=lambda: 1.0)
    request = SessionStateTransferRequest("session-a", "gpu-0", "gpu-1", 0.0, 0.5)
    manager.begin(request, backend)

    backend.compute_ready.set_result(None)
    assert manager.poll("session-a", now=2.0).state == "ready"
    assert manager.commit("session-a", now=2.0).state == "streaming"
    assert manager.active()[0].state == "streaming"

    backend.residual.set_result(None)
    assert manager.poll("session-a", now=3.0).state == "completed"
    assert manager.active() == ()


def test_progressive_transfer_surfaces_residual_copy_failure() -> None:
    backend = _ProgressiveBackend()
    manager = SessionStateTransferManager(clock=lambda: 1.0)
    request = SessionStateTransferRequest("session-a", "gpu-0", "gpu-1", 0.0, 0.5)
    started = manager.begin(request, backend)
    backend.compute_ready.set_result(None)
    manager.poll("session-a", now=2.0)
    manager.commit("session-a", now=2.0)

    backend.residual.set_exception(RuntimeError("layer 7 transfer failed"))
    failed = manager.poll("session-a", now=3.0)

    assert failed.state == "failed"
    assert "layer 7 transfer failed" in str(failed.error)
    assert manager.active() == ()
    assert manager.get(started.transfer_id) == failed
