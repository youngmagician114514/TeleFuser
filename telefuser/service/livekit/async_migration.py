"""Backward-compatible aliases for the Session State Transfer API.

New integrations should import from :mod:`session_state_transfer`. The old
module remains available so existing deployments and tests do not need a
flag-day migration.
"""

from __future__ import annotations

from .session_state_transfer import (
    SessionStateTransferBackend,
    SessionStateTransferManager,
    SessionStateTransferRecord,
    SessionStateTransferRequest,
    SessionStateTransferState,
    ThreadedRouterSessionStateTransferBackend,
)

MigrationState = SessionStateTransferState
MigrationRequest = SessionStateTransferRequest
AsyncMigrationBackend = SessionStateTransferBackend
MigrationRecord = SessionStateTransferRecord
AsyncMigrationManager = SessionStateTransferManager
RouterMigrationBackend = ThreadedRouterSessionStateTransferBackend

__all__ = [
    "AsyncMigrationBackend",
    "AsyncMigrationManager",
    "MigrationRecord",
    "MigrationRequest",
    "MigrationState",
    "RouterMigrationBackend",
]
