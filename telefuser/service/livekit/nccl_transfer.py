"""Tensor-manifest helpers for chunk-boundary NCCL session migration.

The control plane transports only a small Python metadata object.  All retained
model tensors are described by that object, allocated on the target GPU, and
then copied directly with ``torch.distributed`` point-to-point NCCL operations.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TensorTransferGroup:
    """An ordered subset of a session tensor manifest."""

    name: str
    paths: tuple[tuple[Any, ...], ...]
    layer_index: int | None = None


@dataclass(frozen=True)
class TensorTransferGroupStat:
    """Bounded timing and byte telemetry for one transfer group."""

    name: str
    layer_index: int | None
    bytes: int
    duration_ms: float


@dataclass(frozen=True)
class TensorTransferReport:
    """Result of one complete ordered point-to-point transfer."""

    total_bytes: int
    total_duration_ms: float
    groups: tuple[TensorTransferGroupStat, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "total_duration_ms": self.total_duration_ms,
            "groups": [asdict(group) for group in self.groups],
        }


TransferGroupCallback = Callable[[TensorTransferGroup, torch.cuda.Event | None], None]


class LayerTransferProgress:
    """Thread-safe layer readiness shared by an NCCL receiver and eager DiT."""

    def __init__(self, layer_count: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        if not isinstance(layer_count, int) or isinstance(layer_count, bool) or layer_count < 1:
            raise ValueError("layer_count must be a positive integer")
        self.layer_count = layer_count
        self._clock = clock
        self._created_at = clock()
        self._available = [threading.Event() for _ in range(layer_count)]
        self._cuda_events: list[torch.cuda.Event | None] = [None] * layer_count
        self._complete = threading.Event()
        self._failure: BaseException | None = None
        self._first_layer_ready_at: float | None = None
        self._transfer_complete_at: float | None = None
        self._first_compute_requested_at: float | None = None
        self._host_wait_seconds = 0.0
        self._wait_calls = 0
        self._complete_host_wait_seconds = 0.0
        self._complete_wait_calls = 0
        self._lock = threading.RLock()

    @property
    def first_layer_ready(self) -> threading.Event:
        return self._available[0]

    @property
    def complete(self) -> bool:
        with self._lock:
            return self._complete.is_set() and self._failure is None

    def mark_layer_ready(self, layer_index: int, event: torch.cuda.Event | None) -> None:
        if not 0 <= layer_index < self.layer_count:
            raise IndexError(layer_index)
        with self._lock:
            if self._available[layer_index].is_set():
                raise RuntimeError(f"layer {layer_index} readiness was recorded twice")
            self._cuda_events[layer_index] = event
            now = self._clock()
            if layer_index == 0:
                self._first_layer_ready_at = now
            self._available[layer_index].set()

    def mark_complete(self) -> None:
        with self._lock:
            self._transfer_complete_at = self._clock()
            self._complete.set()

    def mark_failed(self, error: BaseException) -> None:
        with self._lock:
            self._failure = error
            for available in self._available:
                available.set()
            self._complete.set()

    def wait_layer(
        self,
        layer_index: int,
        *,
        stream: torch.cuda.Stream | None = None,
        timeout: float | None = None,
    ) -> float:
        """Wait until a layer has arrived, then order its CUDA stream dependency."""
        if not 0 <= layer_index < self.layer_count:
            raise IndexError(layer_index)
        started = self._clock()
        with self._lock:
            if self._first_compute_requested_at is None:
                self._first_compute_requested_at = started
        if not self._available[layer_index].wait(timeout):
            raise TimeoutError(f"timed out waiting for migrated layer {layer_index}")
        waited = max(0.0, self._clock() - started)
        with self._lock:
            self._host_wait_seconds += waited
            self._wait_calls += 1
            failure = self._failure
            event = self._cuda_events[layer_index]
        if failure is not None:
            raise RuntimeError(f"layer transfer failed before layer {layer_index} became ready") from failure
        if event is not None:
            target_stream = stream or torch.cuda.current_stream(event.device)
            target_stream.wait_event(event)
        return waited

    def wait_complete(self, *, timeout: float | None = None) -> float:
        """Wait for decoder-tail state before the post-DiT decode stage."""

        started = self._clock()
        if not self._complete.wait(timeout):
            raise TimeoutError("timed out waiting for migrated decoder state")
        waited = max(0.0, self._clock() - started)
        with self._lock:
            self._complete_host_wait_seconds += waited
            self._complete_wait_calls += 1
            failure = self._failure
        if failure is not None:
            raise RuntimeError("layer transfer failed before decoder state became ready") from failure
        return waited

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            first_ready_ms = (
                None
                if self._first_layer_ready_at is None
                else max(0.0, self._first_layer_ready_at - self._created_at) * 1000.0
            )
            complete_ms = (
                None
                if self._transfer_complete_at is None
                else max(0.0, self._transfer_complete_at - self._created_at) * 1000.0
            )
            residual_ms = 0.0
            if self._first_compute_requested_at is not None and self._first_layer_ready_at is not None:
                residual_ms = max(0.0, self._first_layer_ready_at - self._first_compute_requested_at) * 1000.0
            return {
                "layer_count": self.layer_count,
                "ready_layers": sum(available.is_set() for available in self._available),
                "complete": self._complete.is_set() and self._failure is None,
                "failed": self._failure is not None,
                "first_layer_ready_ms": first_ready_ms,
                "transfer_complete_ms": complete_ms,
                "first_compute_residual_wait_ms": residual_ms,
                "host_wait_ms": self._host_wait_seconds * 1000.0,
                "wait_calls": self._wait_calls,
                "complete_host_wait_ms": self._complete_host_wait_seconds * 1000.0,
                "complete_wait_calls": self._complete_wait_calls,
            }


def flatten_tensor_tree(
    value: Any, *, path: tuple[Any, ...] = ()
) -> tuple[Any, list[dict[str, Any]], dict[tuple[Any, ...], torch.Tensor]]:
    """Separate a nested tree into scalar skeleton, tensor manifest, and leaves."""
    manifest: list[dict[str, Any]] = []
    leaves: dict[tuple[Any, ...], torch.Tensor] = {}

    def visit(item: Any, item_path: tuple[Any, ...]) -> Any:
        if isinstance(item, torch.Tensor):
            tensor = item.detach()
            manifest.append(
                {
                    "path": list(item_path),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                }
            )
            leaves[item_path] = tensor
            return {"__tensor__": list(item_path)}
        if isinstance(item, dict):
            return {"__dict__": {key: visit(child, item_path + (key,)) for key, child in item.items()}}
        if isinstance(item, list):
            return {"__list__": [visit(child, item_path + (index,)) for index, child in enumerate(item)]}
        if isinstance(item, tuple):
            return {"__tuple__": [visit(child, item_path + (index,)) for index, child in enumerate(item)]}
        return {"__value__": item}

    return visit(value, path), manifest, leaves


def allocate_tensor_tree_leaves(
    manifest: list[dict[str, Any]], device: torch.device
) -> dict[tuple[Any, ...], torch.Tensor]:
    """Allocate target GPU tensors from a source manifest."""
    dtype_table = {
        name.removeprefix("torch."): value for name, value in vars(torch).items() if isinstance(value, torch.dtype)
    }
    leaves: dict[tuple[Any, ...], torch.Tensor] = {}
    for entry in manifest:
        dtype_name = str(entry["dtype"])
        if dtype_name not in dtype_table:
            raise ValueError(f"Unsupported NCCL tensor dtype {dtype_name!r}")
        leaves[tuple(entry["path"])] = torch.empty(tuple(entry["shape"]), dtype=dtype_table[dtype_name], device=device)
    return leaves


def build_layer_transfer_groups(manifest: Sequence[dict[str, Any]]) -> tuple[TensorTransferGroup, ...]:
    """Prioritize DiT bootstrap/cache state and defer decoder-only tensors."""
    preamble: list[tuple[Any, ...]] = []
    decoder_tail: list[tuple[Any, ...]] = []
    layer_paths: dict[int, list[tuple[Any, ...]]] = {}
    for entry in manifest:
        path = tuple(entry["path"])
        if (
            len(path) >= 2
            and path[0] in {"self_cache", "cross_cache"}
            and isinstance(path[1], int)
            and not isinstance(path[1], bool)
            and path[1] >= 0
        ):
            layer_paths.setdefault(path[1], []).append(path)
        elif path and path[0] in {"vae_feat_cache", "taew_decode_state"}:
            # These tensors are not touched until the post-DiT decode stage.
            # Sending them after the cache layers materially shortens the
            # first-layer compute-ready boundary without delaying decode.
            decoder_tail.append(path)
        else:
            preamble.append(path)
    groups: list[TensorTransferGroup] = []
    if preamble:
        groups.append(TensorTransferGroup("preamble", tuple(preamble)))
    groups.extend(
        TensorTransferGroup(f"layer_{layer_index:02d}", tuple(layer_paths[layer_index]), layer_index)
        for layer_index in sorted(layer_paths)
    )
    if decoder_tail:
        groups.append(TensorTransferGroup("decoder_tail", tuple(decoder_tail)))
    return tuple(groups)


def rebuild_tensor_tree(skeleton: Any, leaves: dict[tuple[Any, ...], torch.Tensor]) -> Any:
    """Rebuild a nested value produced by :func:`flatten_tensor_tree`."""
    if "__tensor__" in skeleton:
        return leaves[tuple(skeleton["__tensor__"])]
    if "__dict__" in skeleton:
        return {key: rebuild_tensor_tree(value, leaves) for key, value in skeleton["__dict__"].items()}
    if "__list__" in skeleton:
        return [rebuild_tensor_tree(value, leaves) for value in skeleton["__list__"]]
    if "__tuple__" in skeleton:
        return tuple(rebuild_tensor_tree(value, leaves) for value in skeleton["__tuple__"])
    if "__value__" in skeleton:
        return skeleton["__value__"]
    raise ValueError("Invalid NCCL tensor-tree skeleton")


def transfer_tensor_leaves_nccl(
    leaves: dict[tuple[Any, ...], torch.Tensor],
    *,
    peer_rank: int,
    send: bool,
) -> int:
    """Synchronously send or receive a manifest's GPU leaves over NCCL."""
    group = TensorTransferGroup(
        "full_state",
        tuple(sorted(leaves, key=lambda value: tuple(map(str, value)))),
    )
    return transfer_tensor_leaves_nccl_streamed(
        leaves,
        peer_rank=peer_rank,
        send=send,
        groups=(group,),
    ).total_bytes


def transfer_tensor_leaves_nccl_streamed(
    leaves: dict[tuple[Any, ...], torch.Tensor],
    *,
    peer_rank: int,
    send: bool,
    groups: Sequence[TensorTransferGroup],
    stream: torch.cuda.Stream | None = None,
    on_group_complete: TransferGroupCallback | None = None,
) -> TensorTransferReport:
    """Transfer ordered state groups on an optional dedicated CUDA stream.

    Both peers must supply identical groups.  Completing each group before
    submitting the next gives the target a stable layer-ready boundary while
    the caller's worker thread remains free to overlap later groups with model
    computation on another CUDA stream.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("NCCL process group is not initialized")
    expected_paths = set(leaves)
    grouped_paths = [path for group in groups for path in group.paths]
    if len(grouped_paths) != len(set(grouped_paths)) or set(grouped_paths) != expected_paths:
        raise ValueError("NCCL transfer groups must contain every tensor path exactly once")

    started = time.monotonic()
    stats: list[TensorTransferGroupStat] = []
    stream_context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with stream_context:
        if stream is not None and send:
            # Source tensors were produced on the worker's compute stream and
            # must be ordered before NCCL reads them. Receive buffers are fresh
            # allocations, so waiting on the target default stream would only
            # serialize SST behind unrelated inference work on that GPU.
            stream.wait_stream(torch.cuda.default_stream(stream.device))
        for group in groups:
            group_started = time.monotonic()
            ordered = [leaves[path].contiguous() for path in group.paths]
            ops = [dist.P2POp(dist.isend if send else dist.irecv, tensor, peer_rank) for tensor in ordered]
            if ops:
                requests = dist.batch_isend_irecv(ops)
                for request in requests:
                    request.wait()
            ready_event = None
            if stream is not None:
                ready_event = torch.cuda.Event()
                ready_event.record(stream)
            if on_group_complete is not None:
                on_group_complete(group, ready_event)
            byte_count = sum(tensor.numel() * tensor.element_size() for tensor in ordered)
            stats.append(
                TensorTransferGroupStat(
                    name=group.name,
                    layer_index=group.layer_index,
                    bytes=byte_count,
                    duration_ms=(time.monotonic() - group_started) * 1000.0,
                )
            )
        if stream is not None:
            stream.synchronize()
    return TensorTransferReport(
        total_bytes=sum(group.bytes for group in stats),
        total_duration_ms=(time.monotonic() - started) * 1000.0,
        groups=tuple(stats),
    )
