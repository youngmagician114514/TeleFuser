"""Worker-owned physical KV pages and session-owned logical frame tables.

One self-attention page stores one complete latent frame across DiT layers.
Attention gathers only the current layer into the existing attention backend;
there is no persistent dense execution copy of all historical layers.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

import torch


class KVPagePool:
    """Allocate, reuse and account for physical pages, including SST receives."""

    def __init__(self, *, max_bytes: int | None = None, max_cached_bytes: int = 1 << 30) -> None:
        self.max_bytes = max_bytes
        self.max_cached_bytes = max_cached_bytes
        self._pages: dict[int, torch.Tensor] = {}
        self._addresses: dict[int, int] = {}
        self._free: dict[tuple, list[int]] = defaultdict(list)
        self._idle: set[int] = set()
        self._ready: dict[int, torch.cuda.Event] = {}
        self._next_id = 0
        self._lock = threading.RLock()
        self.allocated_bytes = self.cached_bytes = 0
        self.peak_bytes = 0

    @staticmethod
    def _key(shape: tuple, dtype: torch.dtype, device: torch.device) -> tuple:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return tuple(shape), dtype, device

    def acquire(self, shape: tuple, *, dtype: torch.dtype, device: torch.device) -> int:
        with self._lock:
            key = self._key(shape, dtype, device)
            if self._free[key]:
                page_id = self._free[key].pop()
                self._idle.remove(page_id)
                ready = self._ready.pop(page_id, None)
                if ready is not None:
                    torch.cuda.current_stream(device).wait_event(ready)
                self.cached_bytes -= self._pages[page_id].numel() * self._pages[page_id].element_size()
                return page_id
            amount = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
            if self.max_bytes is not None:
                self.trim_free(max(0, self.max_bytes - self.allocated_bytes + self.cached_bytes - amount))
                if self.allocated_bytes + amount > self.max_bytes:
                    raise RuntimeError("Physical KV page budget exhausted before CUDA allocation")
            value = torch.empty(shape, dtype=dtype, device=device)
            return self._install(value)

    def _install(self, value: torch.Tensor) -> int:
        page_id = self._next_id
        self._next_id += 1
        self._pages[page_id] = value
        self._addresses[value.data_ptr()] = page_id
        self.allocated_bytes += value.numel() * value.element_size()
        self.peak_bytes = max(self.peak_bytes, self.allocated_bytes)
        return page_id

    def adopt(self, value: torch.Tensor, *, copy: bool) -> int:
        with self._lock:
            if not copy and value.data_ptr() in self._addresses:
                return self._addresses[value.data_ptr()]
            page_id = self.acquire(tuple(value.shape), dtype=value.dtype, device=value.device)
            self._pages[page_id].copy_(value)
            return page_id

    def page(self, page_id: int) -> torch.Tensor:
        return self._pages[page_id]

    def release(self, page_id: int) -> None:
        self.release_many((page_id,))

    def release_many(self, page_ids: list[int] | tuple[int, ...]) -> None:
        with self._lock:
            if not page_ids:
                return
            value = self._pages[page_ids[0]]
            ready = None
            if value.device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(value.device))
            for page_id in page_ids:
                if page_id in self._idle:
                    raise RuntimeError("A physical KV page was released twice")
                value = self._pages[page_id]
                if ready is not None:
                    value.record_stream(torch.cuda.current_stream(value.device))
                    self._ready[page_id] = ready
                key = self._key(tuple(value.shape), value.dtype, value.device)
                self._free[key].append(page_id)
                self._idle.add(page_id)
                self.cached_bytes += value.numel() * value.element_size()
            self.trim_free(self.max_cached_bytes)

    def trim_free(self, target_bytes: int = 0) -> None:
        with self._lock:
            while self.cached_bytes > target_bytes and self._idle:
                # A logically free page can still have a GPU reader. Keep it
                # inside the physical budget until its fence completes; a
                # matching acquire can instead reuse it with a stream wait.
                page_id = next(
                    (page_id for page_id in self._idle if page_id not in self._ready or self._ready[page_id].query()),
                    None,
                )
                if page_id is None:
                    break
                self._idle.remove(page_id)
                value = self._pages.pop(page_id)
                key = self._key(tuple(value.shape), value.dtype, value.device)
                self._free[key].remove(page_id)
                self._addresses.pop(value.data_ptr())
                self._ready.pop(page_id, None)
                amount = value.numel() * value.element_size()
                self.allocated_bytes -= amount
                self.cached_bytes -= amount

    def snapshot(self) -> dict[str, int]:
        return dict(
            allocated_bytes=self.allocated_bytes,
            live_bytes=self.allocated_bytes - self.cached_bytes,
            cached_bytes=self.cached_bytes,
            live_pages=len(self._pages) - len(self._idle),
            cached_pages=len(self._idle),
            peak_bytes=self.peak_bytes,
        )


class PagedSessionKV:
    """Translate logical retained frames into worker-local physical page IDs."""

    def __init__(self, pool: KVPagePool, reference: torch.Tensor, *, layers: int, frame_tokens: int) -> None:
        self.pool, self.layers, self.frame_tokens = pool, layers, frame_tokens
        self.dtype, self.device = reference.dtype, reference.device
        self.head_shape = tuple(reference.shape[-2:])
        self.table: list[int] = []
        self.cross_page: int | None = None
        self.cross_ready = [False] * layers
        self.cross_length = 0
        self.global_end = 0

    def truncate(self, frames: int, sink: int) -> None:
        if len(self.table) <= frames:
            return
        keep = min(sink, frames)
        tail = frames - keep
        removed = self.table[keep : len(self.table) - tail if tail else len(self.table)]
        self.table = self.table[:keep] + (self.table[-tail:] if tail else [])
        self.pool.release_many(removed)

    def prepare(self, *, current_end: int, window: int, sink: int, frames: int, context_length: int) -> None:
        if current_end > self.global_end:
            self.truncate(window - frames, sink)
            for _ in range(frames):
                self.table.append(
                    self.pool.acquire(
                        (2, self.layers, self.frame_tokens, *self.head_shape), dtype=self.dtype, device=self.device
                    )
                )
            self.global_end = current_end
        if self.cross_page is None:
            self.cross_page = self.pool.acquire(
                (2, self.layers, context_length, *self.head_shape), dtype=self.dtype, device=self.device
            )
            self.cross_length = context_length
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            for page_id in (*self.table, self.cross_page):
                self.pool.page(page_id).record_stream(stream)

    def write_layer(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        frames = key.shape[1] // self.frame_tokens
        for index, page_id in enumerate(self.table[-frames:]):
            page = self.pool.page(page_id)
            begin = index * self.frame_tokens
            page[0, layer].copy_(key[0, begin : begin + self.frame_tokens])
            page[1, layer].copy_(value[0, begin : begin + self.frame_tokens])

    def layer_pages(self, layer: int, field: int) -> list[torch.Tensor]:
        return [self.pool.page(page_id)[field, layer] for page_id in self.table]

    def cross(self, layer: int, field: int) -> torch.Tensor:
        return self.pool.page(self.cross_page)[field, layer].unsqueeze(0)

    def clean_delta(self, frames: int) -> torch.Tensor:
        return torch.stack(
            [
                torch.cat([self.pool.page(page_id)[field, layer] for page_id in self.table[-frames:]]).unsqueeze(0)
                for layer in range(self.layers)
                for field in (0, 1)
            ]
        )

    def commit_delta(self, delta: torch.Tensor) -> None:
        for layer in range(self.layers):
            self.write_layer(layer, delta[layer * 2], delta[layer * 2 + 1])

    def export(self) -> dict[str, Any]:
        return dict(
            paged_self=[self.pool.page(page_id) for page_id in self.table],
            paged_cross=None if self.cross_page is None else self.pool.page(self.cross_page),
            paged_cross_ready=tuple(self.cross_ready),
            paged_cross_length=self.cross_length,
        )

    def resident_bytes(self) -> int:
        return sum(
            self.pool.page(page_id).numel() * self.pool.page(page_id).element_size()
            for page_id in (*self.table, *((self.cross_page,) if self.cross_page is not None else ()))
        )

    def install(self, payload: Any, *, copy: bool) -> None:
        table = [self.pool.adopt(value, copy=copy) for value in payload["paged_self"]]
        cross = payload["paged_cross"]
        cross_page = None if cross is None else self.pool.adopt(cross, copy=copy)
        self.close()
        self.table, self.cross_page = table, cross_page
        self.cross_ready = list(payload["paged_cross_ready"])
        self.cross_length = int(payload["paged_cross_length"])
        self.global_end = int(payload["cursor"][0])

    def close(self) -> None:
        self.pool.release_many(self.table + ([] if self.cross_page is None else [self.cross_page]))
        self.table.clear()
        self.cross_page = None
