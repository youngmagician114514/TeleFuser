"""Reference latent/sequence parallel primitives for ABot-World.

This module is intentionally opt-in and is not imported by the serving path.
It provides a correctness-first prototype for the mentor's proposed mapping:
one continuation latent frame per rank in a three-rank group.  The prototype
keeps historical KV replicated and exchanges only the current K/V shard plus
the attention output shard.  That makes its communication accounting explicit
and gives us a lower-cost reference against a full-KV all-gather upper bound.

The production ABot causal cache is not modified here.  Integrating this path
requires a model-level decision about KV ownership, cache cursor updates and
the VAE output owner, so this module must remain opt-in until its numerical and
performance tests pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class LatentParallelConfig:
    """Static shape contract for one ABot continuation chunk."""

    world_size: int = 3
    latent_frames: int = 3
    frame_tokens: int = 390
    hidden_size: int = 3072
    num_heads: int = 24
    head_dim: int = 128
    dtype_bytes: int = 2

    def validate(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if self.latent_frames < 1:
            raise ValueError("latent_frames must be positive")
        if self.latent_frames % self.world_size:
            raise ValueError("latent_frames must be divisible by world_size for frame-aligned sharding")
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim")
        if self.dtype_bytes not in {2, 4}:
            raise ValueError("dtype_bytes must be 2 or 4")

    @property
    def frames_per_rank(self) -> int:
        self.validate()
        return self.latent_frames // self.world_size

    @property
    def tokens_per_rank(self) -> int:
        return self.frames_per_rank * self.frame_tokens

    def communication_bytes_per_layer(self, batch_size: int = 1) -> dict[str, int]:
        """Return logical peer traffic for the reference attention path.

        The prototype all-gathers current K/V and gathers the output shard.
        Historical KV is assumed replicated and is therefore excluded from the
        per-layer communication count.  The result is a logical traffic count,
        not a hardware wire-byte guarantee; NCCL protocol overhead is measured
        separately by the microbenchmark.
        """

        self.validate()
        peers = self.world_size - 1
        current_kv_local = batch_size * self.tokens_per_rank * self.num_heads * self.head_dim * self.dtype_bytes * 2
        output_local = batch_size * self.tokens_per_rank * self.hidden_size * self.dtype_bytes
        return {
            "current_kv_peer_bytes": current_kv_local * peers,
            "output_peer_bytes": output_local * peers,
            "total_peer_bytes": (current_kv_local + output_local) * peers,
            "current_kv_local_bytes": current_kv_local,
            "output_local_bytes": output_local,
        }


class LatentParallelCommunicator:
    """Frame-aligned sequence communicator for a distributed prototype."""

    def __init__(
        self,
        config: LatentParallelConfig,
        *,
        group: dist.ProcessGroup | None = None,
        rank: int | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.group = group
        self.world_size = dist.get_world_size(group) if dist.is_initialized() else config.world_size
        self.rank = dist.get_rank(group) if dist.is_initialized() else (0 if rank is None else int(rank))
        if self.world_size != config.world_size:
            raise ValueError(f"distributed world size {self.world_size} != config world size {config.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} is outside world size {self.world_size}")

    def _require_distributed(self) -> None:
        if not dist.is_initialized():
            raise RuntimeError("LatentParallelCommunicator requires an initialized process group")

    def shard_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return this rank's contiguous frame-aligned token shard."""

        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, sequence, hidden]")
        expected = self.config.latent_frames * self.config.frame_tokens
        if tokens.shape[1] != expected:
            raise ValueError(f"expected {expected} tokens, got {tokens.shape[1]}")
        start = self.rank * self.config.tokens_per_rank
        end = start + self.config.tokens_per_rank
        return tokens[:, start:end].contiguous()

    def gather_tokens(self, local_tokens: torch.Tensor) -> torch.Tensor:
        """Gather frame-aligned token shards in rank order."""

        self._require_distributed()
        if local_tokens.ndim != 3 or local_tokens.shape[1] != self.config.tokens_per_rank:
            raise ValueError("local_tokens must be [batch, tokens_per_rank, hidden]")
        # all_gather_into_tensor concatenates along the first dimension.  Fold
        # rank into batch here, then restore rank-major order before returning
        # the logical `[B, world_size * S, H]` sequence.
        gathered = torch.empty(
            (self.world_size * local_tokens.shape[0], *local_tokens.shape[1:]),
            dtype=local_tokens.dtype,
            device=local_tokens.device,
        )
        dist.all_gather_into_tensor(gathered, local_tokens, group=self.group)
        gathered = gathered.reshape(self.world_size, *local_tokens.shape)
        return gathered.transpose(0, 1).reshape(
            local_tokens.shape[0], self.config.latent_frames * self.config.frame_tokens, local_tokens.shape[2]
        )

    def gather_current_kv(
        self,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather current K/V shards, preserving `[B,S,H,D]` layout."""

        self._require_distributed()
        expected = local_k.ndim == 4 and local_k.shape[1] == self.config.tokens_per_rank
        if not expected or local_v.shape != local_k.shape:
            raise ValueError("local K/V must have matching [batch, tokens_per_rank, heads, head_dim] shapes")
        shape = (self.world_size * local_k.shape[0], *local_k.shape[1:])
        gathered_k = torch.empty(shape, dtype=local_k.dtype, device=local_k.device)
        gathered_v = torch.empty(shape, dtype=local_v.dtype, device=local_v.device)
        dist.all_gather_into_tensor(gathered_k, local_k, group=self.group)
        dist.all_gather_into_tensor(gathered_v, local_v, group=self.group)
        gathered_k = gathered_k.reshape(self.world_size, *local_k.shape).transpose(0, 1)
        gathered_v = gathered_v.reshape(self.world_size, *local_v.shape).transpose(0, 1)
        full_shape = (local_k.shape[0], self.config.latent_frames * self.config.frame_tokens, *local_k.shape[2:])
        return gathered_k.reshape(full_shape), gathered_v.reshape(full_shape)

    def causal_attention(
        self,
        local_q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        *,
        history_k: torch.Tensor | None = None,
        history_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention for this rank's query shard.

        `history_k`/`history_v` are replicated across ranks and contain the
        already committed causal window. Current K/V are exchanged once per
        layer. The mask allows every historical token and only the current
        tokens up to each local query position.
        """

        self._require_distributed()
        current_k, current_v = self.gather_current_kv(local_k, local_v)
        if history_k is None:
            history_k = current_k.new_empty((*current_k.shape[:1], 0, *current_k.shape[2:]))
        if history_v is None:
            history_v = current_v.new_empty((*current_v.shape[:1], 0, *current_v.shape[2:]))
        if history_k.shape != history_v.shape or history_k.ndim != 4:
            raise ValueError("history K/V must have matching [batch, history, heads, head_dim] shapes")
        keys = torch.cat([history_k, current_k], dim=1)
        values = torch.cat([history_v, current_v], dim=1)
        q = local_q.transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)
        history_tokens = history_k.shape[1]
        local_offset = self.rank * self.config.tokens_per_rank
        query_positions = local_offset + torch.arange(
            self.config.tokens_per_rank, device=local_q.device, dtype=torch.long
        )
        key_positions = torch.arange(keys.shape[1], device=local_q.device, dtype=torch.long)
        allow = key_positions.unsqueeze(0) <= (history_tokens + query_positions).unsqueeze(1)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=allow, dropout_p=0.0)
        return output.transpose(1, 2).contiguous()


def latent_parallel_projection(config: LatentParallelConfig, *, layers: int = 30, steps: int = 4) -> dict[str, Any]:
    """Return shape-derived communication facts for reports and tests."""

    per_layer = config.communication_bytes_per_layer()
    peer_total = per_layer["total_peer_bytes"] * layers * steps
    return {
        "world_size": config.world_size,
        "latent_frames": config.latent_frames,
        "frames_per_rank": config.frames_per_rank,
        "frame_tokens": config.frame_tokens,
        "tokens_per_rank": config.tokens_per_rank,
        "per_layer": per_layer,
        "layers": layers,
        "denoising_steps": steps,
        "peer_bytes_per_chunk": peer_total,
        "peer_gb_per_chunk": peer_total / 1e9,
    }
