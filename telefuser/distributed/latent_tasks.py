"""Exchange current K/V for whole-latent tasks placed on arbitrary GPU ranks."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist


class LatentTaskExchange:
    """Route task rows without partitioning the spatial tokens of a latent.

    All ranks execute the same dependency wave. A rank may own zero, one, or
    several latents of a session. For an explicitly split dependency group,
    only current K/V crosses ranks inside a layer; historical KV stays with
    the owners of the corresponding tasks. The normal colocated path routes
    locally and performs no per-layer collective. The one-rank path performs
    the same routing without a process group.
    """

    def __init__(self, *, group: dist.ProcessGroup | None = None, local: bool = False) -> None:
        self.group = group
        self.distributed = not local and dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size(group) if self.distributed else 1
        self.rank = dist.get_rank(group) if self.distributed else 0
        self.frames_per_chunk = 1
        self.local_rows: tuple[int, ...] = ()
        self._buffers: dict[str, torch.Tensor] = {}
        self._signature: tuple | None = None
        self.attention_groups: tuple[tuple[int, int, tuple[int, ...]], ...] = ()
        # The normal path keeps a session's sibling latents on one rank.  In
        # that case attention can route current K/V locally and avoid making
        # every DiT layer a world-wide collective.  Explicit split placement
        # remains supported as an elastic fallback.
        self.local_sibling_rows: tuple[int, ...] = ()
        self.local_sibling_only = False
        self._local_sibling_indices: torch.Tensor | None = None

    def configure_attention(self, layout: Sequence[tuple[int, ...]]) -> None:
        """Describe contiguous local rows with equal window and cache cursors."""
        groups = []
        for row, global_row in enumerate(self.local_rows):
            metadata = layout[global_row]
            if groups and groups[-1][2] == metadata:
                begin, _, previous = groups[-1]
                groups[-1] = (begin, row + 1, previous)
            else:
                groups.append((row, row + 1, metadata))
        self.attention_groups = tuple(groups)

    def _buffer(self, name: str, shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
        value = self._buffers.get(name)
        if value is None or value.shape != shape or value.dtype != reference.dtype or value.device != reference.device:
            value = torch.empty(shape, dtype=reference.dtype, device=reference.device)
            self._buffers[name] = value
        return value

    def configure(
        self,
        owners: Sequence[int],
        *,
        frames: int,
        device: torch.device,
        row_order: Sequence[int] | None = None,
    ) -> None:
        """Prepare routing once per wave; rows are chunk-major, then latent-major."""
        owners = tuple(owners)
        if frames < 1 or not owners or len(owners) % frames:
            raise ValueError("A latent wave must contain complete positive-size dependency groups")
        if any(owner < 0 or owner >= self.world_size for owner in owners):
            raise ValueError("Latent task owner is outside the process group")
        order = tuple(range(len(owners))) if row_order is None else tuple(row_order)
        if sorted(order) != list(range(len(owners))):
            raise ValueError("A latent row order must be a permutation of the complete global wave")
        signature = (owners, frames, device, order)
        if signature == self._signature:
            return
        self._signature = signature
        self.frames_per_chunk = frames
        self.local_rows = tuple(row for row in order if owners[row] == self.rank)
        counts = [0] * self.world_size
        positions = [0] * len(owners)
        for row in order:
            owner = owners[row]
            positions[row] = counts[owner]
            counts[owner] += 1
        self._padded_rows = max(counts)
        self._global_offsets = torch.tensor(
            [owner * self._padded_rows + position for owner, position in zip(owners, positions, strict=True)],
            device=device,
            dtype=torch.long,
        )
        sibling_rows = [
            sibling
            for row in self.local_rows
            for sibling in range(row // frames * frames, (row // frames + 1) * frames)
        ]
        self._sibling_offsets = torch.tensor(sibling_rows, device=device, dtype=torch.long)
        self.frame_indices = torch.tensor([row % frames for row in self.local_rows], device=device, dtype=torch.long)
        # This decision must be identical on every rank.  A per-rank decision
        # would let a rank containing only a colocated group skip the collective
        # while another rank containing a split group still enters all-gather.
        # That mixed collective sequence is both incorrect and a common source
        # of hard-to-diagnose Gloo/NCCL hangs.
        groups_colocated = all(len(set(owners[begin : begin + frames])) == 1 for begin in range(0, len(owners), frames))
        local_positions = {row: index for index, row in enumerate(self.local_rows)}
        local_sibling_rows: list[int] = []
        if groups_colocated:
            for row in self.local_rows:
                begin = row // frames * frames
                positions = [local_positions[sibling] for sibling in range(begin, begin + frames)]
                local_sibling_rows.extend(positions)
        self.local_sibling_only = groups_colocated
        self.local_sibling_rows = tuple(local_sibling_rows) if groups_colocated else ()
        self._local_sibling_indices = (
            torch.tensor(self.local_sibling_rows, device=device, dtype=torch.long) if groups_colocated else None
        )

    def _gather_async(self, value: torch.Tensor, *, name: str) -> Callable[[], torch.Tensor]:
        padded = self._buffer(f"{name}_send", (self._padded_rows, *value.shape[1:]), value)
        padded[: len(self.local_rows)].copy_(value)
        # Padding is never selected by the routing table.
        if self.world_size == 1:
            return lambda: padded.index_select(0, self._global_offsets)
        received = self._buffer(f"{name}_receive", (self.world_size * self._padded_rows, *value.shape[1:]), value)
        work = dist.all_gather_into_tensor(received, padded, group=self.group, async_op=True)

        def wait() -> torch.Tensor:
            work.wait()
            return received.index_select(0, self._global_offsets)

        return wait

    def gather_current_async(self, value: torch.Tensor, *, name: str) -> Callable[[], torch.Tensor]:
        """Return each local task's complete current chunk K/V, in latent order."""
        if self.local_sibling_only:
            offsets = self._local_sibling_indices
            if offsets is None or offsets.device != value.device:
                offsets = torch.tensor(self.local_sibling_rows, device=value.device, dtype=torch.long)

            def route_local() -> torch.Tensor:
                selected = value.index_select(0, offsets)
                return selected.reshape(len(self.local_rows), self.frames_per_chunk * value.shape[1], *value.shape[2:])

            return route_local
        wait = self._gather_async(value, name=name)

        def route() -> torch.Tensor:
            full = wait()
            selected = full.index_select(0, self._sibling_offsets)
            return selected.reshape(len(self.local_rows), self.frames_per_chunk * value.shape[1], *value.shape[2:])

        return route

    def gather_outputs(self, value: torch.Tensor) -> torch.Tensor:
        """Collect completed whole-latent rows once, before video decoding."""
        return self._gather_async(value, name="output")()

    def broadcast_(self, value: torch.Tensor, owner: int = 0) -> None:
        if self.world_size > 1:
            source = owner if self.group is None else dist.get_global_rank(self.group, owner)
            dist.broadcast(value, src=source, group=self.group)

    def clear(self) -> None:
        self._buffers.clear()
        self._signature = None
        self.local_sibling_rows = ()
        self.local_sibling_only = False
        self._local_sibling_indices = None
