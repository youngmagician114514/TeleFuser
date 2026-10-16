"""Configuration for the LiveKit-backed ``telefuser stream-serve`` command."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .topology import ClusterTopology, NodeAgent


class LiveKitServeConfig(BaseSettings):
    """Runtime configuration for the LiveKit serving entrypoint."""

    model_config = SettingsConfigDict(
        env_prefix="TELEFUSER_LIVEKIT_",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="HTTP bind host")
    port: int = Field(default=8088, ge=1, le=65535, description="HTTP bind port")

    livekit_url: str = Field(default="", description="LiveKit server URL")
    livekit_api_key: str = Field(default="", description="LiveKit API key")
    livekit_api_secret: str = Field(default="", description="LiveKit API secret")

    num_workers: int = Field(default=1, ge=1, le=64, description="Number of TeleFuser LiveKit workers")
    max_sessions_per_worker: int | Literal["auto"] = Field(
        default="auto",
        description="Hardware-calculated retained sessions, optionally capped by an integer",
    )
    worker_gpu_map: str | None = Field(
        default=None,
        description="Semicolon-separated worker GPU groups, for example '0,1;2,3'",
    )
    worker_mode: Literal["in-process", "process", "process-nccl"] = Field(
        default="in-process",
        description="Worker isolation mode",
    )
    node_id: str = Field(default="node-0", description="Stable identity of the host running this service")
    node_address: str = Field(default="127.0.0.1", description="Reachable control/data address of this host")
    node_rank: int = Field(default=0, ge=0, description="Stable rank of this host within the cluster")
    node_agent_port: int = Field(
        default=8090,
        ge=1,
        le=65535,
        description="NodeAgent control port used by the coordinator and peer nodes",
    )
    node_agent_secret: str = Field(
        default="",
        description="Optional shared secret for the internal NodeAgent control plane",
    )
    node_agent_poll_interval_seconds: float = Field(
        default=0.1,
        gt=0,
        description="Coordinator polling interval for remote node-agent events",
    )
    cluster_world_size: int = Field(default=1, ge=1, le=128, description="Number of node agents in the deployment")
    cluster_topology: str | None = Field(
        default=None,
        description=(
            "Optional node map: 'node-0@host-a=0,1,...;node-1@host-b=0,1,...'. "
            "GPU ordinals are local to each node."
        ),
    )
    nccl_master_address: str | None = Field(
        default=None,
        description="Rendezvous address for a multi-node NCCL process group; node-0 address is used by default",
    )
    nccl_master_port: int = Field(default=29500, ge=1, le=65535, description="Rendezvous port for multi-node NCCL")
    nccl_world_size: int | None = Field(
        default=None,
        ge=1,
        le=4096,
        description="Optional global NCCL worker count; derived from cluster_topology when omitted",
    )

    dispatch_trace_path: str | None = Field(
        default=None,
        description="Fresh parent-process JSONL path for bounded model-dispatch audit records",
    )
    dispatch_trace_max_events: int = Field(
        default=10_000,
        ge=1,
        le=1_000_000,
        description="Maximum model-dispatch records written to the optional JSONL audit trace",
    )

    queue_size: int = Field(default=0, ge=0, le=10000, description="Maximum queued sessions")
    action_collection_window_seconds: float = Field(
        default=0.5,
        gt=0,
        description=(
            "Maximum local latest-action collection window before releasing a browser control state; "
            "this does not change the one-second model output/playback block"
        ),
    )
    action_phase_sync_enabled: bool = Field(
        default=True,
        description=(
            "Release sampled action jobs on shared collection-window phase boundaries; "
            "raw control arrival times remain independent"
        ),
    )
    autoscaling_enabled: bool = Field(default=False, description="Dynamically load configured GPU workers")
    autoscaling_min_workers: int = Field(default=1, ge=1, le=64)
    autoscaling_target_utilization: float = Field(default=0.75, gt=0, le=1)
    autoscaling_hysteresis: float = Field(default=0.10, ge=0, lt=1)
    autoscaling_cooldown_seconds: float = Field(default=30.0, ge=0)
    autoscaling_interval_seconds: float = Field(default=5.0, gt=0)
    turboserve_rebalance_enabled: bool = Field(
        default=True,
        description="Rebalance compatible in-process TurboServe sessions at chunk boundaries",
    )
    turboserve_migration_bandwidth_gbps: float = Field(
        default=24.0,
        gt=0,
        description="Conservative effective bandwidth used by the migration-aware placement model",
    )
    turboserve_migration_penalty: float = Field(
        default=1.0,
        ge=0,
        description="Relative penalty applied to estimated model-session migration time",
    )
    turboserve_scale_in_hold_seconds: float = Field(default=5.0, ge=0)
    turboserve_migration_eta: float = Field(
        default=0.01,
        ge=0,
        description=(
            "Blocking migration ETA until the target first layer is usable; "
            "transfer drain overlaps in background"
        ),
    )
    migration_state_pool_cache_gb: float = Field(
        default=1.0,
        ge=0,
        description=(
            "Per-worker byte budget for reusable NCCL session-state tensor pages; "
            "zero disables retention while keeping the same transfer protocol"
        ),
    )
    migration_incremental_enabled: bool = Field(
        default=True,
        description=(
            "Use source pre-copy plus final tensor delta for process-NCCL session migration; "
            "disable to retain the full-snapshot path"
        ),
    )
    motivation_migration_route_ready_seconds: float = Field(
        default=0.2,
        ge=0,
        description=(
            "Cold-start estimate of the end-to-end migration critical path until the target route is usable; "
            "online route-ready telemetry supersedes this prior"
        ),
    )
    turboserve_min_migration_gain_ms: float = Field(default=40.0, ge=0)
    turboserve_rebalance_iteration_limit: int = Field(default=3, ge=1, le=64)
    motivation_migration_cooldown_seconds: float = Field(
        default=60.0,
        ge=0,
        description="Minimum residence time after a Motivation-owned session migration",
    )
    motivation_placement_interval_seconds: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Periodic Motivation stage-one placement/rehoming control interval; "
            "stage-two GPU-local dispatch remains event driven"
        ),
    )
    slackserve_include_idle_jobs: bool = Field(
        default=False,
        description=(
            "Generate ABot idle-continuation jobs in the SlackServe controlled variant; "
            "the paper-faithful SlackServe baseline remains action-only by default"
        ),
    )
    control_idle_timeout: float = Field(
        default=10.0,
        gt=0,
        description="Seconds without control activity before a LingBot execution lease may yield",
    )
    session_timeout: int = Field(default=1800, ge=1, description="Maximum session lifetime in seconds")
    token_ttl: int = Field(default=3600, ge=1, description="LiveKit token TTL in seconds")
    controller_timeout: int = Field(
        default=60,
        ge=0,
        description="Seconds to keep a session after controller disconnect",
    )
    room_empty_timeout: int = Field(
        default=30,
        ge=0,
        description="Seconds to keep a session after the LiveKit room becomes empty",
    )
    role_mode: Literal["single-controller"] = Field(default="single-controller")

    default_fps: int = Field(default=16, ge=1, le=120, description="Default output video FPS")
    max_data_message_bytes: int = Field(default=12 * 1024, ge=1024, description="Maximum accepted data message size")
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS origins for the LiveKit serve API",
    )

    @field_validator("worker_gpu_map")
    @classmethod
    def validate_worker_gpu_map(cls: type[LiveKitServeConfig], value: str | None) -> str | None:
        """Normalize empty GPU maps to ``None``."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("node_id", "node_address")
    @classmethod
    def validate_node_text(cls: type[LiveKitServeConfig], value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("node_id and node_address must be non-empty")
        return normalized

    @field_validator("nccl_master_address")
    @classmethod
    def validate_nccl_master_address(cls: type[LiveKitServeConfig], value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("cluster_topology")
    @classmethod
    def validate_cluster_topology(cls: type[LiveKitServeConfig], value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        ClusterTopology.from_spec(normalized)
        return normalized

    @field_validator("max_sessions_per_worker", mode="before")
    @classmethod
    def validate_max_sessions_per_worker(cls: type[LiveKitServeConfig], value: object) -> int | str:
        """Accept ``auto`` or a positive integer capacity ceiling."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "auto":
                return normalized
            try:
                value = int(normalized)
            except ValueError as exc:
                raise ValueError("max_sessions_per_worker must be 'auto' or an integer") from exc
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 64:
            raise ValueError("max_sessions_per_worker must be 'auto' or an integer in [1, 64]")
        return value

    @model_validator(mode="after")
    def validate_autoscaling_bounds(self) -> LiveKitServeConfig:
        if self.cluster_world_size > 1 and self.cluster_topology is None:
            raise ValueError("cluster_topology is required when cluster_world_size > 1")
        if self.cluster_topology is not None:
            topology_nodes = len(ClusterTopology.from_spec(self.cluster_topology).nodes)
            if topology_nodes != self.cluster_world_size:
                raise ValueError(
                    "cluster_world_size must match cluster_topology: "
                    f"expected {topology_nodes}, got {self.cluster_world_size}"
                )
        if self.cluster_world_size == 1 and self.node_rank != 0:
            raise ValueError("node_rank must be 0 for a single-node deployment")
        if self.cluster_world_size > 1:
            if self.worker_mode != "process-nccl":
                raise ValueError("multi-node serving currently requires worker_mode='process-nccl'")
            if self.autoscaling_enabled:
                raise ValueError("multi-node serving currently requires autoscaling_enabled=False")
            if self.autoscaling_enabled and self.autoscaling_min_workers != self.num_workers:
                raise ValueError("multi-node NCCL requires all local workers to start together")
        if self.autoscaling_min_workers > self.num_workers:
            raise ValueError("autoscaling_min_workers cannot exceed num_workers")
        if self.autoscaling_enabled and self.num_workers > 1 and self.queue_size == 0:
            raise ValueError("autoscaling with multiple workers requires queue_size > 0 for cold-start admission")
        if self.worker_mode == "process-nccl":
            if self.num_workers < 2:
                raise ValueError("process-nccl requires at least two model workers")
            groups = self.worker_gpu_groups()
            if any(len(group) != 1 for group in groups):
                raise ValueError("process-nccl requires exactly one GPU id per model worker")
        return self

    def session_capacity_limit(self) -> int | None:
        """Return the operator ceiling, or ``None`` for hardware auto-sizing."""
        return self.max_sessions_per_worker if isinstance(self.max_sessions_per_worker, int) else None

    def require_livekit_credentials(self) -> None:
        """Raise when the minimum LiveKit connection settings are missing."""
        missing = []
        if not self.livekit_url:
            missing.append("livekit_url")
        if not self.livekit_api_key:
            missing.append("livekit_api_key")
        if not self.livekit_api_secret:
            missing.append("livekit_api_secret")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing LiveKit configuration: {joined}")

    def worker_gpu_groups(self) -> list[list[str]]:
        """Return one GPU-id group per configured worker."""
        if self.worker_gpu_map is None:
            return [[] for _ in range(self.num_workers)]

        groups = [
            [gpu.strip() for gpu in group.split(",") if gpu.strip()]
            for group in self.worker_gpu_map.split(";")
            if group.strip()
        ]
        if len(groups) != self.num_workers:
            raise ValueError(f"worker_gpu_map defines {len(groups)} worker groups, but num_workers={self.num_workers}")
        return groups

    def topology(self) -> ClusterTopology:
        """Return the configured cluster topology, defaulting to one local node."""
        if self.cluster_topology is not None:
            topology = ClusterTopology.from_spec(self.cluster_topology)
            topology.node(self.node_id)
            if self.cluster_world_size != len(topology.nodes):
                raise ValueError(
                    "cluster_world_size must match cluster_topology: "
                    f"expected {len(topology.nodes)}, got {self.cluster_world_size}"
                )
            node = topology.node(self.node_id)
            if node.rank != self.node_rank:
                raise ValueError(
                    f"node_rank={self.node_rank} does not match topology rank {node.rank} for {self.node_id!r}"
                )
            return topology
        local_gpus = tuple(gpu for group in self.worker_gpu_groups() for gpu in group)
        return ClusterTopology.single_node(
            node_id=self.node_id,
            address=self.node_address,
            gpu_ids=local_gpus,
            rank=self.node_rank,
        )

    def local_node(self) -> NodeAgent:
        """Return this process's node record from the validated topology."""
        return self.topology().node(self.node_id)

    def cluster_worker_endpoints(self) -> tuple["WorkerEndpoint", ...]:
        """Return globally unique one-GPU worker endpoints for a cluster.

        The legacy single-node path intentionally keeps its original
        ``worker-N`` identifiers.  A multi-node deployment uses stable
        ``node-id/worker-N`` endpoint ids so local GPU ordinals can repeat on
        different hosts without colliding in admission or policy state.
        """
        from .topology import WorkerEndpoint

        topology = self.topology()
        if len(topology.nodes) == 1 and self.cluster_world_size == 1:
            groups = self.worker_gpu_groups()
            node = topology.nodes[0]
            return tuple(
                WorkerEndpoint(
                    worker_id=f"worker-{index}",
                    node_id=node.node_id,
                    gpu_ids=tuple(group),
                    address=node.address,
                    port=self.node_agent_port,
                    rank=node.rank,
                )
                for index, group in enumerate(groups)
            )
        endpoints: list[WorkerEndpoint] = []
        for node in topology.nodes:
            for index, gpu_id in enumerate(node.gpu_ids):
                endpoints.append(
                    WorkerEndpoint(
                        worker_id=f"worker-{index}",
                        node_id=node.node_id,
                        gpu_ids=(gpu_id,),
                        address=node.address,
                        port=node.control_port or self.node_agent_port,
                        rank=node.rank,
                    )
                )
        return tuple(endpoints)

    def local_cluster_worker_endpoints(self) -> tuple["WorkerEndpoint", ...]:
        """Return only endpoints hosted by this process's node."""
        return tuple(endpoint for endpoint in self.cluster_worker_endpoints() if endpoint.node_id == self.node_id)

    def effective_node_address(self) -> str:
        """Return the topology address when one was supplied, else node_address."""
        if self.cluster_topology is not None:
            return self.local_node().address
        return self.node_address

    def nccl_rendezvous(self, *, local_worker_count: int) -> tuple[str, int, int, int]:
        """Return ``(master, port, world_size, local_rank_offset)``.

        Single-node deployments retain the caller's existing ephemeral
        rendezvous behavior.  Multi-node deployments use the explicit
        cluster topology and stable node rank so each host can launch its
        local eight-worker shard into one 16-rank process group.
        """
        if local_worker_count < 1:
            raise ValueError("local_worker_count must be positive")
        if self.cluster_world_size == 1 and self.cluster_topology is None:
            return "127.0.0.1", 0, local_worker_count, 0
        topology = self.topology()
        if len(topology.nodes) == 1:
            return topology.nodes[0].address, self.nccl_master_port, local_worker_count, 0
        counts = [len(node.gpu_ids) for node in topology.nodes]
        if any(count == 0 for count in counts):
            world_size = self.nccl_world_size or local_worker_count * len(topology.nodes)
            offset = self.node_rank * local_worker_count
        else:
            if any(count != local_worker_count for count in counts):
                raise ValueError("multi-node NCCL currently requires equal local worker counts on every node")
            world_size = self.nccl_world_size or sum(counts)
            offset = sum(counts[: self.node_rank])
        required_world_size = sum(counts) if all(counts) else local_worker_count * len(topology.nodes)
        if world_size < required_world_size:
            raise ValueError("nccl_world_size is smaller than the configured node worker population")
        master = self.nccl_master_address or topology.nodes[0].address
        return master, self.nccl_master_port, world_size, offset
