"""Adapter from LiveKit workers to TeleFuser stream pipeline services."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from telefuser.service.core.config import ServerConfig
from telefuser.service.core.stream_pipeline_service import StreamPipelineService
from telefuser.service.security.security_validator import SecurityLevel


class LiveKitPipelineAdapter:
    """Thin wrapper around ``StreamPipelineService`` for LiveKit workers."""

    def __init__(self, *, security_level: SecurityLevel | None = None, config: ServerConfig | None = None) -> None:
        self.stream_service = StreamPipelineService(security_level=security_level, config=config)

    def start(
        self,
        pipeline_file: str,
        *,
        skip_validation: bool = False,
        gpu_num: int = 1,
        gpu_ids: list[str] | None = None,
    ) -> None:
        """Load and start a stream pipeline on the assigned CUDA devices."""
        if not self.stream_service.start_service(
            pipeline_file, skip_validation=skip_validation, gpu_num=gpu_num, gpu_ids=gpu_ids
        ):
            raise RuntimeError(f"Failed to start LiveKit stream pipeline: {pipeline_file}")

    @property
    def stream_mode(self) -> str | None:
        """Return the detected TeleFuser stream interaction mode."""
        return self.stream_service.stream_mode

    @property
    def uses_relative_rope(self) -> bool | None:
        """Return the concrete ABot DiT's Relative-RoPE capability."""
        denoise_stage = self.denoise_stage
        dit = getattr(denoise_stage, "dit", None)
        value = getattr(dit, "use_relative_rope", None)
        return bool(value) if value is not None else None

    @property
    def denoise_stage(self) -> object | None:
        """Expose the loaded pipeline's RoPE capability to ABot scheduling.

        The ABot service uses this read-only capability when constructing a
        batch key. Process-NCCL workers wrap the concrete service in
        ``StreamPipelineService``; hiding ``denoise_stage`` makes the service
        fail closed as Absolute-RoPE and unnecessarily separates sessions at
        different latent cursors, even though Relative-RoPE supports them.
        """
        service = getattr(self.stream_service, "service", None)
        pipeline = getattr(service, "pipeline", None)
        return getattr(pipeline, "denoise_stage", None)

    async def aclose(self) -> None:
        """Stop the wrapped stream service."""
        await self.stream_service.aclose()

    def create_session(self, config: dict) -> str:
        return self.stream_service.create_session(config)

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        self.stream_service.push_chunk(session_id, chunk)

    def push_batch(self, items: list[tuple[str, dict]]) -> None:
        """Apply policy-selected controls while holding the service boundary."""
        push_batch = getattr(self.stream_service, "push_batch", None)
        if not callable(push_batch):
            # The wrapper owns transport lifecycle while the concrete pipeline
            # service owns the atomic scheduling boundary. ABot exposes
            # push_batch on that nested service; checking only the wrapper
            # silently fell back to one push_chunk per member.
            push_batch = getattr(getattr(self.stream_service, "service", None), "push_batch", None)
        if callable(push_batch):
            push_batch(items)
            return
        for session_id, chunk in items:
            self.stream_service.push_chunk(session_id, chunk)

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        async for chunk in self.stream_service.pull_chunks(session_id):
            yield chunk

    def enable_publisher_frame_tracking(self, session_id: str) -> bool:
        """Enable the wrapped service's optional real-time frame feedback."""

        return self.stream_service.enable_publisher_frame_tracking(session_id)

    def report_publisher_frame_progress(
        self, session_id: str, *, event: str, frames_delta: int, sequence: int, observed_monotonic_seconds: float
    ) -> bool:
        """Forward one idempotent publisher progress update when supported."""

        return self.stream_service.report_publisher_frame_progress(
            session_id,
            event=event,
            frames_delta=frames_delta,
            sequence=sequence,
            observed_monotonic_seconds=observed_monotonic_seconds,
        )

    async def stream_task(self, config: dict) -> AsyncGenerator[dict, None]:
        """Yield chunks from a server-push service."""
        async for chunk in self.stream_service.stream_task(config):
            yield chunk

    def close_session(self, session_id: str) -> None:
        self.stream_service.close_session(session_id)

    def configure_session_capacity(self, max_sessions: int | None) -> dict[str, object] | None:
        """Configure and return the loaded pipeline's optional capacity profile."""
        return self.stream_service.configure_session_capacity(max_sessions)

    def runtime_metrics(self, session_id: str | None = None) -> dict[str, float | int | str] | None:
        """Return optional model-service scheduling measurements.

        ``session_id`` selects per-pipeline-session facts when the wrapped
        service supports them. Calling without an identifier preserves the
        aggregate placement-metrics contract used by existing worker pools.
        Older third-party services may expose only a no-argument
        ``runtime_metrics`` method; in that case the per-session request
        gracefully falls back to the aggregate snapshot.
        """
        service = getattr(self.stream_service, "service", None)
        metrics = getattr(service, "runtime_metrics", None)
        if not callable(metrics):
            return None
        try:
            value = metrics(session_id) if session_id is not None else metrics()
        except TypeError:
            # Keep adapters for older pipeline services usable while allowing
            # ABot-style services to expose ``runtime_metrics(session_id)``.
            if session_id is None:
                return None
            try:
                value = metrics()
            except Exception:
                return None
        except Exception:
            return None
        return dict(value) if isinstance(value, dict) else None
