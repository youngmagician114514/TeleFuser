"""Persistent two-rank channels for denoise/refresh state handoff."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist


class ABotPhaseTransfer:
    """Transfer packed tensors only between the two phase owners.

    All world ranks construct channels in the same order. Unrelated GPUs do
    not participate in handoff collectives. A channel is reused across chunks;
    it does not recreate a session or a process group for each transition.
    """

    def __init__(self, denoise_ranks: Sequence[int], refresh_ranks: Sequence[int]) -> None:
        self.rank = dist.get_rank()
        self.channels = {
            (source, destination): dist.new_group(sorted((source, destination)))
            for source in denoise_ranks
            for destination in refresh_ranks
        }
        self.sent_bytes = 0
        self.latent_bytes = 0
        self.kv_bytes = 0
        self.handoffs = 0
        self.logical_latent_bytes = 0
        self.logical_kv_bytes = 0

    def transfer(
        self, value: torch.Tensor | None, *, source: int, destination: int, kind: str, byte_count: int
    ) -> torch.Tensor | None:
        """Return a stream-ordered tensor on the destination rank."""
        pair = (source, destination) if (source, destination) in self.channels else (destination, source)
        if kind == "latent":
            self.logical_latent_bytes += byte_count
        else:
            self.logical_kv_bytes += byte_count
        if self.rank in (source, destination):
            dist.broadcast(value, src=source, group=self.channels[pair])
            if self.rank == source:
                size = value.numel() * value.element_size()
                self.sent_bytes += size
                if kind == "latent":
                    self.latent_bytes += size
                else:
                    self.kv_bytes += size
                self.handoffs += 1
        return value

    def close(self) -> None:
        for group in self.channels.values():
            if group != dist.GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(group)
        self.channels.clear()
