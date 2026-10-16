"""Cluster topology and worker endpoint identities for LiveKit serving.

The serving policy historically used a local GPU/worker string as its owner
identity.  That is sufficient on one host, but it loses the distinction
between an intra-node P2P transfer and an inter-node transfer when the same
GPU ordinals appear on a second host.  This module adds that distinction
without changing the legacy local identifiers.

``ClusterTopology`` is deliberately a control-plane description.  It does
not start processes or open network connections.  A node-local runtime can
therefore use the same worker pool it uses today while exposing enough
metadata for a future multi-node transport and for topology-aware placement
costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TransportScope = Literal["intra_node", "inter_node"]


def _clean_identifier(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


@dataclass(frozen=True)
class NodeAgent:
    """Control-plane identity and hardware inventory for one host."""

    node_id: str
    address: str = "127.0.0.1"
    gpu_ids: tuple[str, ...] = ()
    rank: int = 0
    control_port: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _clean_identifier(self.node_id, field_name="node_id"))
        object.__setattr__(self, "address", _clean_identifier(self.address, field_name="address"))
        gpu_ids = tuple(_clean_identifier(gpu_id, field_name="gpu_id") for gpu_id in self.gpu_ids)
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError(f"node {self.node_id!r} contains duplicate GPU ids")
        object.__setattr__(self, "gpu_ids", gpu_ids)
        if self.rank < 0:
            raise ValueError("node rank must be non-negative")
        if self.control_port is not None and not 1 <= self.control_port <= 65535:
            raise ValueError("control_port must be in [1, 65535]")


@dataclass(frozen=True)
class WorkerEndpoint:
    """Stable identity of one worker execution endpoint.

    ``worker_id`` remains local and compatible with existing routing code.
    ``endpoint_id`` is globally unique within a cluster and should be used by
    a distributed router or state-transfer backend.
    """

    worker_id: str
    node_id: str = "node-0"
    gpu_ids: tuple[str, ...] = ()
    address: str | None = None
    port: int | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_id", _clean_identifier(self.worker_id, field_name="worker_id"))
        object.__setattr__(self, "node_id", _clean_identifier(self.node_id, field_name="node_id"))
        gpu_ids = tuple(_clean_identifier(gpu_id, field_name="gpu_id") for gpu_id in self.gpu_ids)
        if len(gpu_ids) != len(set(gpu_ids)):
            raise ValueError(f"worker {self.worker_id!r} contains duplicate GPU ids")
        object.__setattr__(self, "gpu_ids", gpu_ids)
        if self.address is not None:
            object.__setattr__(self, "address", _clean_identifier(self.address, field_name="address"))
        if self.port is not None and not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if self.rank is not None and self.rank < 0:
            raise ValueError("worker rank must be non-negative")

    @property
    def endpoint_id(self) -> str:
        return f"{self.node_id}/{self.worker_id}"

    @property
    def primary_gpu_id(self) -> str | None:
        return self.gpu_ids[0] if self.gpu_ids else None

    def same_node(self, other: WorkerEndpoint) -> bool:
        return self.node_id == other.node_id

    def transport_scope(self, other: WorkerEndpoint) -> TransportScope:
        return "intra_node" if self.same_node(other) else "inter_node"


@dataclass(frozen=True)
class ClusterTopology:
    """Validated set of node agents used by placement and transfer code."""

    nodes: tuple[NodeAgent, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("cluster topology requires at least one node")
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("cluster topology contains duplicate node ids")
        ranks = [node.rank for node in self.nodes]
        if len(ranks) != len(set(ranks)):
            raise ValueError("cluster topology contains duplicate node ranks")
        seen_gpu_keys: set[tuple[str, str]] = set()
        for node in self.nodes:
            for gpu_id in node.gpu_ids:
                key = (node.node_id, gpu_id)
                if key in seen_gpu_keys:
                    raise ValueError(f"duplicate GPU endpoint {node.node_id}/{gpu_id}")
                seen_gpu_keys.add(key)

    @classmethod
    def single_node(
        cls,
        *,
        node_id: str = "node-0",
        address: str = "127.0.0.1",
        gpu_ids: tuple[str, ...] = (),
        rank: int = 0,
    ) -> ClusterTopology:
        """Build the backwards-compatible one-host topology."""
        return cls((NodeAgent(node_id=node_id, address=address, gpu_ids=gpu_ids, rank=rank),))

    @classmethod
    def from_spec(cls, spec: str) -> ClusterTopology:
        """Parse ``node@address=gpu0,gpu1;node2@address2=gpu0,...``.

        The GPU ordinals are intentionally local to each node, so ``0`` on
        node-0 and ``0`` on node-1 are valid and distinct endpoints.
        """
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("cluster topology spec must be a non-empty string")
        nodes: list[NodeAgent] = []
        for rank, raw_node in enumerate(spec.split(";")):
            raw_node = raw_node.strip()
            if not raw_node:
                continue
            try:
                identity, raw_gpus = raw_node.split("=", 1)
            except ValueError as exc:
                raise ValueError(
                    "cluster topology entries must look like node@address=gpu0,gpu1"
                ) from exc
            identity = identity.strip()
            if "@" in identity:
                node_id, address = (part.strip() for part in identity.split("@", 1))
            else:
                node_id, address = identity, "127.0.0.1"
            gpu_ids = tuple(gpu.strip() for gpu in raw_gpus.split(",") if gpu.strip())
            nodes.append(NodeAgent(node_id=node_id, address=address, gpu_ids=gpu_ids, rank=rank))
        if not nodes:
            raise ValueError("cluster topology spec contains no nodes")
        return cls(tuple(nodes))

    def node(self, node_id: str) -> NodeAgent:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def node_for_gpu(self, gpu_id: str, *, node_id: str | None = None) -> NodeAgent:
        matches = [node for node in self.nodes if gpu_id in node.gpu_ids]
        if node_id is not None:
            matches = [node for node in matches if node.node_id == node_id]
        if len(matches) != 1:
            raise KeyError(f"GPU {gpu_id!r} is not uniquely assigned in topology")
        return matches[0]

    def endpoints_for_worker_groups(
        self,
        *,
        node_id: str,
        worker_gpu_groups: tuple[tuple[str, ...], ...] | list[list[str]],
        worker_ids: tuple[str, ...] | None = None,
        first_port: int | None = None,
    ) -> tuple[WorkerEndpoint, ...]:
        """Create local worker endpoints from the existing GPU-group config."""
        node = self.node(node_id)
        groups = tuple(tuple(str(gpu) for gpu in group) for group in worker_gpu_groups)
        ids = worker_ids or tuple(f"worker-{index}" for index in range(len(groups)))
        if len(ids) != len(groups):
            raise ValueError("worker_ids and worker_gpu_groups must have the same length")
        local_gpus = set(node.gpu_ids)
        endpoints: list[WorkerEndpoint] = []
        for index, (worker_id, group) in enumerate(zip(ids, groups, strict=True)):
            if local_gpus and any(gpu not in local_gpus for gpu in group):
                raise ValueError(f"worker {worker_id!r} uses a GPU outside node {node_id!r}")
            port = None if first_port is None else first_port + index
            endpoints.append(
                WorkerEndpoint(
                    worker_id=worker_id,
                    node_id=node.node_id,
                    gpu_ids=group,
                    address=node.address,
                    port=port,
                    rank=node.rank,
                )
            )
        return tuple(endpoints)


def transport_scope(source: WorkerEndpoint, target: WorkerEndpoint) -> TransportScope:
    """Return whether a transfer crosses a node boundary."""
    return source.transport_scope(target)


__all__ = ["ClusterTopology", "NodeAgent", "TransportScope", "WorkerEndpoint", "transport_scope"]
