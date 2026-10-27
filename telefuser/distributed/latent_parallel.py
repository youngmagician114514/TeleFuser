"""Persistent frame-aligned sequence communication for ABot latent parallelism."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist


class LatentParallelRuntime:
    """Reuse NCCL buffers while sharding the current latent frames by rank.

    Model weights and historical KV remain replicated on every rank. Only the
    current K/V block is exchanged inside a DiT layer; the latent itself stays
    sharded across all denoising steps and is gathered once at chunk end.
    """

    def __init__(self, *, group: dist.ProcessGroup | None = None) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("latent parallelism requires an initialized torch.distributed process group")
        self.group = group
        self.world_size = dist.get_world_size(group=group)
        self.rank = dist.get_rank(group=group)
        if self.world_size < 2:
            raise ValueError("latent parallelism requires at least two ranks")
        self._buffers: dict[str, torch.Tensor] = {}

    def _buffer(self, name: str, shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
        buffer = self._buffers.get(name)
        if (
            buffer is None
            or buffer.shape != shape
            or buffer.dtype != reference.dtype
            or buffer.device != reference.device
        ):
            buffer = torch.empty(shape, dtype=reference.dtype, device=reference.device)
            self._buffers[name] = buffer
        return buffer

    @property
    def is_output_rank(self) -> bool:
        """Only rank zero decodes and publishes the generated video."""
        return self.rank == 0

    def broadcast_(self, value: torch.Tensor) -> None:
        """Use the output rank's conditioning consistently across replicas."""
        source = 0 if self.group is None else dist.get_global_rank(self.group, 0)
        dist.broadcast(value, src=source, group=self.group)

    def frame_slice(self, value: torch.Tensor) -> torch.Tensor:
        """Return this rank's contiguous frame shard from ``[B,C,F,H,W]``."""
        if value.ndim != 5:
            raise ValueError("latent parallel frame slicing expects [B,C,F,H,W]")
        frames = value.shape[2]
        if frames < 1 or frames % self.world_size:
            raise ValueError(f"latent frame count {frames} is not divisible by world size {self.world_size}")
        frames_per_rank = frames // self.world_size
        start = self.rank * frames_per_rank
        return value[:, :, start : start + frames_per_rank].contiguous()

    def _gather_rank_major(self, value: torch.Tensor, name: str) -> torch.Tensor:
        value = value.contiguous()
        stacked = self._buffer(name, (self.world_size * value.shape[0], *value.shape[1:]), value)
        dist.all_gather_into_tensor(stacked, value, group=self.group)
        return stacked.view(self.world_size, *value.shape)

    def gather_current_kv(
        self,
        local_key: torch.Tensor,
        local_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather rank-local current K/V into rank-major sequence order."""
        if local_key.ndim != 4 or local_value.shape != local_key.shape:
            raise ValueError("current K/V must have matching [B,S,H,D] shapes")
        key_wait = self.gather_current_async(local_key, name="key")
        value_wait = self.gather_current_async(local_value, name="value")
        return key_wait(), value_wait()

    def gather_current_async(self, value: torch.Tensor, *, name: str) -> Callable[[], torch.Tensor]:
        """Submit K or V exchange and return a stream-ordered completion callback.

        A layer submits K, computes V and submits V, then computes Q before
        consuming the callbacks. NCCL can overlap these projections. A returned
        tensor is scratch storage, consumed before the following layer reuses it.
        """
        value = value.contiguous()
        shape = (self.world_size * value.shape[0], *value.shape[1:])
        stacked = self._buffer(f"{name}_rank_major", shape, value)
        work = dist.all_gather_into_tensor(stacked, value, group=self.group, async_op=True)

        def wait() -> torch.Tensor:
            work.wait()
            batch, sequence, heads, head_dim = value.shape
            rank_major = stacked.view(self.world_size, batch, sequence, heads, head_dim)
            full = self._buffer(f"{name}_sequence", (batch, self.world_size * sequence, heads, head_dim), value)
            full.view(batch, self.world_size, sequence, heads, head_dim).copy_(rank_major.transpose(0, 1))
            return full

        return wait

    def gather_frames(self, local_latent: torch.Tensor) -> torch.Tensor:
        """Gather ``[B,C,F_LOCAL,H,W]`` into ``[B,C,F,H,W]`` once per chunk."""
        if local_latent.ndim != 5:
            raise ValueError("latent parallel output gathering expects [B,C,F,H,W]")
        gathered = self._gather_rank_major(local_latent, "latent_output")
        batch, channels, frames, height, width = local_latent.shape
        # Callers may retain earlier chunks, so never return a view of the
        # reusable collective storage (not even for singleton batches).
        return (
            gathered.permute(1, 2, 0, 3, 4, 5).reshape(batch, channels, self.world_size * frames, height, width).clone()
        )

    def clear_buffers(self) -> None:
        """Release persistent communication buffers when a pipeline is closed."""
        self._buffers.clear()
