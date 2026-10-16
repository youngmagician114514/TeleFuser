"""Entrypoint helpers for ``telefuser stream-serve``."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import uvicorn

from telefuser._logo import TELEFUSER_LOGO
from telefuser.service.security.security_validator import SecurityLevel
from telefuser.utils.logging import logger

from .app import create_livekit_app
from .config import LiveKitServeConfig
from .migration_hysteresis import MigrationCooldownPolicy
from .motivation_controller import MotivationRuntimeController
from .motivation_diagnostics import MotivationDiagnosticsCollector
from .motivation_scheduler import (
    GpuSchedulingState,
    LocalMigrationEstimator,
    MotivationSchedulerConfig,
)
from .runtime import LiveKitServeRuntime


def run_stream_server(
    *,
    pipe_path: str,
    host: str | None = None,
    port: int | None = None,
    livekit_url: str | None = None,
    livekit_api_key: str | None = None,
    livekit_api_secret: str | None = None,
    num_workers: int | None = None,
    max_sessions_per_worker: int | str | None = None,
    worker_gpu_map: str | None = None,
    node_id: str | None = None,
    node_address: str | None = None,
    node_rank: int | None = None,
    node_agent_port: int | None = None,
    node_agent_secret: str | None = None,
    cluster_world_size: int | None = None,
    cluster_topology: str | None = None,
    nccl_master_address: str | None = None,
    nccl_master_port: int | None = None,
    nccl_world_size: int | None = None,
    queue_size: int | None = None,
    autoscaling_enabled: bool | None = None,
    autoscaling_min_workers: int | None = None,
    autoscaling_target_utilization: float | None = None,
    autoscaling_hysteresis: float | None = None,
    autoscaling_cooldown_seconds: float | None = None,
    autoscaling_interval_seconds: float | None = None,
    control_idle_timeout: float | None = None,
    session_timeout: int | None = None,
    token_ttl: int | None = None,
    controller_timeout: int | None = None,
    room_empty_timeout: int | None = None,
    worker_mode: str | None = None,
    skip_validation: bool = False,
    security_level: str | None = None,
    motivation_profile: str | None = None,
    motivation_memory_free_gb: float = 80.0,
    motivation_max_batch_size: int = 4,
    motivation_lambda_quality: float = 2.0,
    motivation_pending_action_slots: int = 4,
    motivation_migration: bool = True,
    motivation_policy: str = "motivation",
) -> None:
    """Run the LiveKit-backed streaming HTTP API."""
    config_kwargs: dict[str, Any] = _drop_none(
        {
            "host": host,
            "port": port,
            "livekit_url": livekit_url,
            "livekit_api_key": livekit_api_key,
            "livekit_api_secret": livekit_api_secret,
            "num_workers": num_workers,
            "max_sessions_per_worker": max_sessions_per_worker,
            "worker_gpu_map": worker_gpu_map,
            "node_id": node_id,
            "node_address": node_address,
            "node_rank": node_rank,
            "node_agent_port": node_agent_port,
            "node_agent_secret": node_agent_secret,
            "cluster_world_size": cluster_world_size,
            "cluster_topology": cluster_topology,
            "nccl_master_address": nccl_master_address,
            "nccl_master_port": nccl_master_port,
            "nccl_world_size": nccl_world_size,
            "queue_size": queue_size,
            "autoscaling_enabled": autoscaling_enabled,
            "autoscaling_min_workers": autoscaling_min_workers,
            "autoscaling_target_utilization": autoscaling_target_utilization,
            "autoscaling_hysteresis": autoscaling_hysteresis,
            "autoscaling_cooldown_seconds": autoscaling_cooldown_seconds,
            "autoscaling_interval_seconds": autoscaling_interval_seconds,
            "control_idle_timeout": control_idle_timeout,
            "session_timeout": session_timeout,
            "token_ttl": token_ttl,
            "controller_timeout": controller_timeout,
            "room_empty_timeout": room_empty_timeout,
            "worker_mode": worker_mode,
        }
    )
    config = LiveKitServeConfig(**config_kwargs)
    config.require_livekit_credentials()

    motivation_controller = None
    if motivation_profile is not None:
        normalized_policy = str(motivation_policy).strip().lower()
        if normalized_policy == "slackserver":
            normalized_policy = "slackserve"
        endpoints = config.cluster_worker_endpoints()
        gpu_states = [
            GpuSchedulingState(
                endpoint.endpoint_id if config.cluster_world_size > 1 else endpoint.worker_id,
                node_id=endpoint.node_id,
                endpoint_address=endpoint.address,
                endpoint_port=endpoint.port if config.cluster_world_size > 1 else None,
                memory_free_gb=motivation_memory_free_gb,
            )
            for endpoint in endpoints
        ]
        diagnostics = MotivationDiagnosticsCollector(
            recent_search_limit=64,
            recent_dispatch_limit=64,
        )
        motivation_controller = MotivationRuntimeController.from_offline_table(
            motivation_profile,
            gpu_states=gpu_states,
            dispatch=lambda _lease: None,
            scheduler_config=MotivationSchedulerConfig(
                max_batch_size=motivation_max_batch_size,
                lambda_quality=motivation_lambda_quality,
                pending_action_slots=motivation_pending_action_slots,
                # FIFO is a static-owner baseline.  The scheduler also
                # normalizes this defensively, but keeping the entrypoint
                # value explicit makes runtime metadata and tests unambiguous.
                migration_enabled=motivation_migration and normalized_policy != "fifo",
                # DistributedLiveKitWorkerPool implements the same state
                # transfer protocol across nodes; the scheduler only charges
                # the topology-specific transfer estimate.
                allow_cross_node_migration=True,
                cross_node_migration_extra_seconds=config.motivation_migration_cross_node_extra_seconds,
                policy_name=normalized_policy,
                load_balance_capacity_per_worker=config.session_capacity_limit() or 12,
                turboserve_capacity_per_worker=config.session_capacity_limit() or 12,
                turboserve_target_utilization=config.autoscaling_target_utilization,
                turboserve_migration_eta=config.turboserve_migration_eta,
                turboserve_min_gain_ms=config.turboserve_min_migration_gain_ms,
                turboserve_rebalance_iteration_limit=config.turboserve_rebalance_iteration_limit,
                slackserve_capacity_per_worker=config.session_capacity_limit() or 12,
                slackserve_include_idle_jobs=config.slackserve_include_idle_jobs,
            ),
            # Charge the complete route-ready critical path. Residual layer
            # copy and source cleanup overlap target compute and are excluded.
            migration_estimator=LocalMigrationEstimator(
                first_layer_ready_seconds=config.motivation_migration_route_ready_seconds,
            ),
            # Motivation and SlackServe keep a 60-second residence cooldown;
            # TurboServe already has its own min-gain/bounded-rebalance guard.
            migration_policy=(
                MigrationCooldownPolicy(
                    cooldown_seconds=config.motivation_migration_cooldown_seconds,
                )
                if normalized_policy in {"motivation", "slackserve"}
                else None
            ),
            diagnostics=diagnostics,
        )

    runtime = LiveKitServeRuntime(
        config=config,
        pipeline_file=pipe_path,
        skip_validation=skip_validation,
        security_level=security_level,
        motivation_controller=motivation_controller,
    )
    app = create_livekit_app(runtime)

    try:
        print(TELEFUSER_LOGO)
        logger.info(
            "Starting TeleFuser LiveKit server on %s:%s, workers=%s, pipeline=%s, "
            "skip_validation=%s, security_level=%s",
            config.host,
            config.port,
            config.num_workers,
            pipe_path,
            skip_validation,
            security_level,
        )
        uvicorn.run(app, host=config.host, port=config.port, log_level="warning")
    except KeyboardInterrupt:
        logger.info("LiveKit server interrupted by user")
    except Exception as exc:
        logger.error(f"LiveKit server failed: {exc}")
        sys.exit(1)
    finally:
        try:
            asyncio.run(runtime.aclose())
        except Exception as exc:
            logger.warning(f"Error during LiveKit server cleanup: {exc}")


def run_node_agent(
    *,
    pipe_path: str,
    host: str | None = None,
    livekit_url: str | None = None,
    livekit_api_key: str | None = None,
    livekit_api_secret: str | None = None,
    num_workers: int | None = None,
    max_sessions_per_worker: int | str | None = None,
    worker_gpu_map: str | None = None,
    node_id: str | None = None,
    node_address: str | None = None,
    node_rank: int | None = None,
    node_agent_port: int | None = None,
    node_agent_secret: str | None = None,
    cluster_world_size: int | None = None,
    cluster_topology: str | None = None,
    nccl_master_address: str | None = None,
    nccl_master_port: int | None = None,
    nccl_world_size: int | None = None,
    worker_mode: str = "process-nccl",
    skip_validation: bool = False,
    security_level: str | None = None,
) -> None:
    """Run a node-local worker agent for a distributed coordinator."""
    config = LiveKitServeConfig(
        host=host or "0.0.0.0",
        livekit_url=livekit_url or "",
        livekit_api_key=livekit_api_key or "",
        livekit_api_secret=livekit_api_secret or "",
        num_workers=num_workers or 1,
        max_sessions_per_worker=max_sessions_per_worker or "auto",
        worker_gpu_map=worker_gpu_map,
        node_id=node_id or "node-0",
        node_address=node_address or "127.0.0.1",
        node_rank=node_rank or 0,
        node_agent_port=node_agent_port or 8090,
        node_agent_secret=node_agent_secret or "",
        cluster_world_size=cluster_world_size or 1,
        cluster_topology=cluster_topology,
        nccl_master_address=nccl_master_address,
        nccl_master_port=nccl_master_port or 29500,
        nccl_world_size=nccl_world_size,
        worker_mode=worker_mode,
    )
    config.require_livekit_credentials()
    from .node_agent_server import run_node_agent as _run_node_agent

    level = None if security_level is None else SecurityLevel[security_level.upper()]
    try:
        asyncio.run(
            _run_node_agent(
                pipe_path=pipe_path,
                config=config,
                security_level=level,
                skip_validation=skip_validation,
            )
        )
    except KeyboardInterrupt:
        logger.info("NodeAgent interrupted by user")


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
