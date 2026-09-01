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
