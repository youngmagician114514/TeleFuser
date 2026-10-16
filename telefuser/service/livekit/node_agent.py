"""Control-plane contracts for node-local worker agents.

The model workers remain node-local processes.  A coordinator therefore needs
one small, transport-independent request object for a cross-node state
transfer; the agent implementation can later use gRPC, an HTTP control
service, or an existing cluster RPC without changing scheduling code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol

from .session_state_transfer import SessionStateTransferRequest
from .topology import NodeAgent


@dataclass(frozen=True)
class NodeAgentMigrationRequest:
    """Control-plane description of one inter-node session migration."""

    session_id: str
    source_gpu: str
    target_gpu: str
    source_node_id: str
    target_node_id: str
    source_address: str | None
    target_address: str | None
    requested_at: float
    estimated_ready_at: float
    state_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.session_id or not self.source_gpu or not self.target_gpu:
            raise ValueError("session_id and GPU identifiers must be non-empty")
        if not self.source_node_id or not self.target_node_id:
            raise ValueError("source_node_id and target_node_id must be non-empty")
        for name in ("requested_at", "estimated_ready_at"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.estimated_ready_at < self.requested_at:
            raise ValueError("estimated_ready_at cannot precede requested_at")
        if self.state_bytes < 0:
            raise ValueError("state_bytes must be non-negative")

    @classmethod
    def from_transfer_request(cls, request: SessionStateTransferRequest) -> "NodeAgentMigrationRequest":
        if not request.crosses_node:
            raise ValueError("NodeAgentMigrationRequest requires an inter-node transfer")
        return cls(
            session_id=request.session_id,
            source_gpu=request.source_gpu,
            target_gpu=request.target_gpu,
            source_node_id=request.source_node_id,
            target_node_id=request.target_node_id,
            source_address=request.source_address,
            target_address=request.target_address,
            requested_at=request.requested_at,
            estimated_ready_at=request.estimated_ready_at,
            state_bytes=request.state_bytes,
        )

    @property
    def transport_scope(self) -> str:
        return "inter_node"


@dataclass(frozen=True)
class NodeAgentMigrationResult:
    """Minimal result returned before the scheduler commits ownership."""

    operation_id: str
    ready: bool
    completed: bool = False
    metadata: Mapping[str, Any] | None = None


class NodeAgentClient(Protocol):
    """Node-local control endpoint used by a distributed worker pool."""

    @property
    def node(self) -> NodeAgent: ...

    async def health(self) -> Mapping[str, Any]: ...

    async def migrate_session(
        self,
        request: NodeAgentMigrationRequest,
    ) -> NodeAgentMigrationResult: ...


class NodeAgentDirectory:
    """Validated node-agent registry for a coordinator process.

    The directory intentionally stores clients by stable ``node_id`` rather
    than by local worker ordinal.  This prevents ``worker-0`` on node-1 from
    colliding with ``worker-0`` on node-0 when the deployment is expanded.
    """

    def __init__(self, nodes: tuple[NodeAgent, ...]) -> None:
        if not nodes:
            raise ValueError("node-agent directory requires at least one node")
        self._nodes = {node.node_id: node for node in nodes}
        if len(self._nodes) != len(nodes):
            raise ValueError("node-agent directory contains duplicate node ids")
        self._clients: dict[str, NodeAgentClient] = {}

    def register(self, client: NodeAgentClient) -> None:
        node_id = client.node.node_id
        if node_id not in self._nodes:
            raise KeyError(f"node agent {node_id!r} is not present in the cluster topology")
        self._clients[node_id] = client

    def client(self, node_id: str) -> NodeAgentClient:
        try:
            return self._clients[node_id]
        except KeyError as exc:
            raise RuntimeError(f"node agent {node_id!r} is not registered") from exc

    def registered_node_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._clients))


__all__ = [
    "NodeAgentClient",
    "NodeAgentDirectory",
    "NodeAgentMigrationRequest",
    "NodeAgentMigrationResult",
]
