import pytest

from telefuser.service.livekit.node_agent import (
    NodeAgentDirectory,
    NodeAgentMigrationRequest,
)
from telefuser.service.livekit.session_state_transfer import SessionStateTransferRequest
from telefuser.service.livekit.topology import NodeAgent


def test_remote_migration_request_preserves_node_and_gpu_identity() -> None:
    request = SessionStateTransferRequest(
        session_id="session-a",
        source_gpu="0",
        target_gpu="3",
        requested_at=1.0,
        estimated_ready_at=1.2,
        state_bytes=128,
        source_node_id="node-0",
        target_node_id="node-1",
        source_address="h100-a",
        target_address="h100-b",
    )

    remote = NodeAgentMigrationRequest.from_transfer_request(request)

    assert remote.transport_scope == "inter_node"
    assert remote.source_node_id == "node-0"
    assert remote.target_node_id == "node-1"
    assert remote.source_gpu == "0"
    assert remote.target_gpu == "3"
    assert remote.state_bytes == 128


def test_remote_migration_request_rejects_local_transfer() -> None:
    request = SessionStateTransferRequest("session-a", "0", "1", 1.0, 1.2)

    with pytest.raises(ValueError, match="inter-node"):
        NodeAgentMigrationRequest.from_transfer_request(request)


def test_node_agent_directory_requires_topology_membership() -> None:
    class FakeClient:
        node = NodeAgent("node-0", "h100-a", ("0",))

        async def health(self):
            return {"status": "ok"}

        async def migrate_session(self, request):
            raise AssertionError("not called")

    directory = NodeAgentDirectory((NodeAgent("node-0", "h100-a", ("0",)),))
    directory.register(FakeClient())

    assert directory.registered_node_ids() == ("node-0",)
    assert directory.client("node-0").node.node_id == "node-0"
    with pytest.raises(RuntimeError, match="not registered"):
        directory.client("node-1")

