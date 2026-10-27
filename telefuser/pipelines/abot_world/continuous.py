"""Opt-in interactive driver for iteration-level ABot continuous batching."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .step_batching import ABotWorldStepBatcher

if TYPE_CHECKING:
    from PIL import Image

    from .interactive import ABotWorldInteractivePipeline, ABotWorldInteractiveSession

    PostprocessSubmitter = Callable[[torch.Tensor], Future[list[list[Image.Image]]]]
    DecodeSubmitter = Callable[[torch.Tensor, ABotWorldInteractiveSession], Future["ABotWorldDecodedBatch"]]
else:
    # The alias is used only in deferred annotations; keep module import safe
    # when PIL is not imported by a CPU-only caller.
    PostprocessSubmitter = Callable
    DecodeSubmitter = Callable


@dataclass(frozen=True)
class ABotWorldDecodedBatch:
    """One asynchronously decoded session row.

    The decode queue returns one row per submitted session even when the
    underlying TAeW invocation batches several rows. Keeping the row boundary
    here lets the service preserve per-session output ordering.
    """

    decoded: torch.Tensor
    frames_per_session: int


@dataclass(frozen=True)
class ABotWorldCompletedChunk:
    """A completed chunk; only the output rank owns decoded RGB frames."""

    session_id: str
    decoded: torch.Tensor
    frames_per_session: int
    # Optional service-owned conversion. The model runtime submits one
    # decoded batch to the service executor; every session keeps its row index
    # so publication can remain ordered without copying or converting on the
    # scheduler thread.
    postprocess_future: Future[list[list[Image.Image]]] | None = None
    postprocess_batch_index: int = 0
    # Optional asynchronous TAeW decode.  It is appended after the legacy
    # fields so existing positional callers keep their behavior.
    decode_future: Future[ABotWorldDecodedBatch] | None = None


class ABotWorldContinuousRuntime:
    """Submit session chunks and advance one compatible DiT step per call.

    This driver owns its submitted sessions until their chunks complete. Do not
    concurrently run, migrate, resize, or close those sessions through another
    pipeline API. Multi-GPU callers must drive all ranks in the same order.
    """

    def __init__(
        self,
        pipeline: ABotWorldInteractivePipeline,
        *,
        max_batch_size: int,
        postprocess_submitter: PostprocessSubmitter | None = None,
        decode_submitter: DecodeSubmitter | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.batcher = ABotWorldStepBatcher(pipeline.denoise_stage, max_batch_size=max_batch_size)
        self._sessions: dict[str, ABotWorldInteractiveSession] = {}
        self._postprocess_submitter = postprocess_submitter
        self._decode_submitter = decode_submitter

    @property
    def has_pending(self) -> bool:
        return self.batcher.has_pending

    @torch.inference_mode()
    def submit(
        self,
        session: ABotWorldInteractiveSession,
        actions: Mapping[str, bool] | None = None,
        *,
        control_latent_frames: int = 3,
        timestep_positions: tuple[int, ...] | None = None,
    ) -> None:
        """Capture one chunk using the session's existing window and state."""
        pipeline = self.pipeline
        with pipeline._execution_lock:
            pipeline._require_session(session)
            if session.session_id in self._sessions:
                raise ValueError("Session already has a submitted continuous-batching chunk")
            if control_latent_frames not in {1, 2, 3}:
                raise ValueError("control_latent_frames must be 1, 2, or 3")
            if session.migration_transfer_in_progress:
                raise ValueError("Finish the session state transfer before continuous-batching admission")
            shape = session.first_frame_latent.shape
            noise = torch.randn(
                (1, shape[1], control_latent_frames, shape[3], shape[4]),
                generator=session.generator,
                device=pipeline.device,
                dtype=torch.float32,
            ).to(pipeline.torch_dtype)
            action = pipeline.build_action_context(
                actions,
                latent_frames=control_latent_frames,
                height=pipeline.config.height,
                width=pipeline.config.width,
                device=pipeline.device,
                dtype=pipeline.torch_dtype,
            )
            task = self.batcher.begin(
                session.session_id,
                noise,
                session.prompt_emb,
                action,
                session.first_frame_latent if session.next_latent_frame == 0 else None,
                session.self_cache,
                session.cross_cache,
                session.next_latent_frame,
                session.generator,
                session.scheduler,
                timestep_positions=timestep_positions,
            )
            pipeline._release_cuda_graph(session.session_id)
            self.batcher.submit(task)
            self._sessions[session.session_id] = session
            session.step_task_active = True
            session.last_activity_at = time.monotonic()

    @torch.inference_mode()
    def step(self) -> list[ABotWorldCompletedChunk]:
        """Advance one DiT step batch, then batch-decode any finished chunks."""
        pipeline = self.pipeline
        with pipeline._execution_lock:
            if any(session.closed for session in self._sessions.values()):
                raise RuntimeError("A continuous-batching session was closed before its chunk completed")
            completed = self.batcher.step()
            if not completed:
                return []
            sessions = [self._sessions[task.session_id] for task in completed]
            latents = torch.cat([task.latent for task in completed])
            decode_futures: list[Future[ABotWorldDecodedBatch] | None] = [None] * len(sessions)
            if pipeline.is_output_rank and self._decode_submitter is not None:
                # Submit rows independently.  The service-owned decode queue
                # coalesces rows from different waves while retaining one
                # causal decoder state per session.
                for row, session in enumerate(sessions):
                    decode_futures[row] = self._decode_submitter(latents[row : row + 1], session)
                decoded = latents.new_empty((len(sessions), 3, 0, 0, 0))
                frames = 0
            elif pipeline.is_output_rank:
                decoded = pipeline.taew_decode_stage.decode_chunks(
                    latents, [session.taew_decode_state for session in sessions]
                )
                frames = int(decoded.shape[2])
            else:
                decoded = latents.new_empty((len(sessions), 3, 0, 0, 0))
                frames = 0
            if self._decode_submitter is None:
                runtime = pipeline.denoise_stage.latent_parallel_runtime
                if runtime is not None:
                    count = torch.tensor(frames, device=latents.device, dtype=torch.long)
                    runtime.broadcast_(count)
                    frames = int(count.item())
            postprocess_future = None
            if self._postprocess_submitter is not None and pipeline.is_output_rank and self._decode_submitter is None:
                postprocess_future = self._postprocess_submitter(decoded)
            results = []
            for row, (task, session) in enumerate(zip(completed, sessions, strict=True)):
                session.next_latent_frame += task.full_shape[2]
                if self._decode_submitter is None:
                    session.emitted_frames += frames
                session.step_task_active = False
                del self._sessions[task.session_id]
                results.append(
                    ABotWorldCompletedChunk(
                        session.session_id,
                        decoded[row : row + 1],
                        frames if pipeline.is_output_rank else 0,
                        postprocess_future,
                        row,
                        decode_futures[row],
                    )
                )
            return results
