"""Independent NCCL communicators for concurrently progressing phase peers."""

from __future__ import annotations

import threading
from datetime import timedelta
from typing import Any, Sequence

import torch
import torch.distributed as dist

from .nccl_transfer import (
    PersistentNCCLTransferLane,
    TensorTransferGroup,
    TensorTransferReport,
    transfer_tensor_leaves_nccl_streamed,
)


class PhasePeerTransport:
    """Isolate peer order instead of sharing one NCCL launch timeline.

    Different phase pairs may submit in different orders. Per-pair CPU
    ordering alone cannot make those orders consistent on a single world
    communicator. Initialize pair communicators once, outside trace time.
    """

    def __init__(self, device: torch.device, groups: dict[int, Any]) -> None:
        self.groups = groups
        self.lane = PersistentNCCLTransferLane(device)
        self.locks = {peer: threading.Lock() for peer in groups}

    @classmethod
    def bootstrap(cls, device: torch.device) -> PhasePeerTransport:
        rank, world = dist.get_rank(), dist.get_world_size()
        groups = {}
        for source in range(world):
            for destination in range(source + 1, world):
                group = dist.new_group((source, destination), backend="nccl", timeout=timedelta(minutes=5))
                if rank in (source, destination):
                    # Warm the actual P2P channel, whose communicator and GPU
                    # buffers can differ from the collective channel. Include
                    # their residency in the subsequent memory calibration.
                    value = torch.zeros(64 << 20, device=device, dtype=torch.uint8)
                    peer = destination if rank == source else source
                    operation = dist.isend if rank == source else dist.irecv
                    for work in dist.batch_isend_irecv([dist.P2POp(operation, value, peer, group=group)]):
                        work.wait()
                    torch.cuda.current_stream(device).synchronize()
                    groups[destination if rank == source else source] = group
        return cls(device, groups)

    def transfer(
        self,
        leaves: dict[tuple, torch.Tensor],
        *,
        peer: int,
        send: bool,
        groups: Sequence[TensorTransferGroup],
        ready: torch.cuda.Event | None = None,
    ) -> TensorTransferReport:
        return transfer_tensor_leaves_nccl_streamed(
            leaves,
            peer_rank=peer,
            send=send,
            groups=groups,
            stream=self.lane.stream_for(peer),
            submission_lock=self.locks[peer],
            source_ready_event=ready,
            process_group=self.groups[peer],
        )

    def close(self) -> None:
        for peer in sorted(self.groups):
            dist.destroy_process_group(self.groups[peer])
        self.groups.clear()
