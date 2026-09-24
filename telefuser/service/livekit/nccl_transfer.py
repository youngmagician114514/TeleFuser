"""Tensor-manifest helpers for chunk-boundary NCCL session migration.

The control plane transports only a small Python metadata object.  All retained
model tensors are described by that object, allocated on the target GPU, and
then copied directly with ``torch.distributed`` point-to-point NCCL operations.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
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
class TensorTreeLayout:
    """Reusable tensor/scalar layout for repeated session snapshots.

    The layout contains no tensor values.  It can therefore be retained per
    session while the model mutates its KV tensors, and reused to rebuild only
    the current leaf mapping and scalar metadata at the next chunk boundary.
    """

    template: Any
    manifest: tuple[dict[str, Any], ...]


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


@dataclass
class PersistentTensorSlot:
    """A reusable collection of page-sized tensor allocations.

    The slot deliberately contains no session metadata.  It is only an
    ownership token for target-side allocations, so a completed transfer can
    release the tensors back to the worker-local pool without keeping the
    Python session object alive.
    """

    pages: tuple[tuple[tuple[str, tuple[int, ...], str], torch.Tensor], ...]
    released: bool = False


class PersistentTensorPagePool:
    """Reuse target allocations across migrations without changing tensor ABI.

    NCCL still copies the current state, but allocation and CUDA allocator
    churn are removed from the migration hot path.  Pages are keyed by
    ``(dtype, shape, device)`` and can therefore be reused even when two
    sessions have different manifest paths.  A bounded byte budget prevents
    the cache from becoming an untracked second session-capacity pool.
    """

    def __init__(self, *, max_cached_bytes: int = 1 << 30) -> None:
        if max_cached_bytes < 0:
            raise ValueError("max_cached_bytes must be non-negative")
        self.max_cached_bytes = int(max_cached_bytes)
        self._free: dict[tuple[str, tuple[int, ...], str], deque[torch.Tensor]] = {}
        self._free_order: deque[tuple[tuple[str, tuple[int, ...], str], torch.Tensor, int]] = deque()
        self._cached_bytes = 0
        self._lock = threading.RLock()

    @staticmethod
    def _key(entry: dict[str, Any], device: torch.device) -> tuple[str, tuple[int, ...], str]:
        return (
            str(entry["dtype"]),
            tuple(int(value) for value in entry["shape"]),
            str(device),
        )

    def acquire(
        self,
        manifest: Sequence[dict[str, Any]],
        device: torch.device,
    ) -> tuple[dict[tuple[Any, ...], torch.Tensor], PersistentTensorSlot]:
        """Acquire one tensor for each manifest path, reusing free pages."""

        leaves: dict[tuple[Any, ...], torch.Tensor] = {}
        pages: list[tuple[tuple[str, tuple[int, ...], str], torch.Tensor]] = []
        with self._lock:
            for entry in manifest:
                key = self._key(entry, device)
                bucket = self._free.get(key)
                tensor = bucket.pop() if bucket else None
                if tensor is not None:
                    bytes_used = tensor.numel() * tensor.element_size()
                    self._cached_bytes -= bytes_used
                    # Remove the corresponding eviction index now.  Tensor
                    # equality is element-wise, so compare by identity.
                    for index, (_, candidate, _) in enumerate(self._free_order):
                        if candidate is tensor:
                            del self._free_order[index]
                            break
                else:
                    tensor = torch.empty(tuple(key[1]), dtype=_dtype_from_name(key[0]), device=device)
                path = tuple(entry["path"])
                leaves[path] = tensor
                pages.append((key, tensor))
        return leaves, PersistentTensorSlot(tuple(pages))

    def release(self, slot: PersistentTensorSlot | None) -> None:
        """Return a slot to the pool; release is idempotent."""

        if slot is None:
            return
        with self._lock:
            if slot.released:
                return
            slot.released = True
            for key, tensor in slot.pages:
                bytes_used = tensor.numel() * tensor.element_size()
                if self.max_cached_bytes == 0:
                    continue
                self._free.setdefault(key, deque()).append(tensor)
                self._free_order.append((key, tensor, bytes_used))
                self._cached_bytes += bytes_used
            self._trim_locked()

    def _trim_locked(self) -> None:
        while self._cached_bytes > self.max_cached_bytes and self._free_order:
            key, tensor, bytes_used = self._free_order.popleft()
            bucket = self._free.get(key)
            if not bucket:
                continue
            # A tensor can have been acquired after it was indexed.  Remove
            # only if this exact object is still free; otherwise leave the
            # accounting to the newer index entry.
            index = next((index for index, candidate in enumerate(bucket) if candidate is tensor), None)
            if index is None:
                continue
            del bucket[index]
            self._cached_bytes -= bytes_used
            if not bucket:
                self._free.pop(key, None)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "cached_bytes": int(self._cached_bytes),
                "cached_pages": sum(len(bucket) for bucket in self._free.values()),
                "max_cached_bytes": self.max_cached_bytes,
            }


class PersistentNCCLTransferLane:
    """Reuse one CUDA transfer stream per peer on a worker.

    NCCL P2P operations on an unordered worker pair still require the parent
    pair lock because the process group has no reliable application tag.  The
    lock protects ordering, while this lane removes the unrelated stream
    allocation/destruction cost from every migration.  Different peers retain
    independent streams and can overlap normally.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self._streams: dict[int, torch.cuda.Stream] = {}
        self._lock = threading.Lock()

    def stream_for(self, peer_rank: int) -> torch.cuda.Stream:
        """Return a persistent stream for one NCCL peer."""

        peer = int(peer_rank)
        with self._lock:
            stream = self._streams.get(peer)
            if stream is None:
                stream = torch.cuda.Stream(device=self.device)
                self._streams[peer] = stream
            return stream

    def snapshot(self) -> dict[str, int | str]:
        """Expose bounded diagnostics without synchronizing any stream."""

        with self._lock:
            return {"device": str(self.device), "peer_streams": len(self._streams)}


def _dtype_from_name(name: str) -> torch.dtype:
    """Resolve the manifest dtype without relying on ``getattr(torch, ...)``."""

    normalized = name.removeprefix("torch.")
    dtype = getattr(torch, normalized, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported NCCL tensor dtype {name!r}")
    return dtype


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


def build_tensor_tree_layout(
    value: Any, *, path: tuple[Any, ...] = ()
) -> tuple[TensorTreeLayout, Any, list[dict[str, Any]], dict[tuple[Any, ...], torch.Tensor]]:
    """Build a reusable layout and flatten the current value in one traversal."""

    manifest: list[dict[str, Any]] = []
    leaves: dict[tuple[Any, ...], torch.Tensor] = {}

    def visit(item: Any, item_path: tuple[Any, ...]) -> tuple[Any, Any]:
        if isinstance(item, torch.Tensor):
            manifest_entry = {
                "path": list(item_path),
                "shape": list(item.shape),
                "dtype": str(item.dtype).removeprefix("torch."),
            }
            manifest.append(manifest_entry)
            leaves[item_path] = item.detach()
            marker = {"__tensor__": list(item_path)}
            return marker, marker
        if isinstance(item, dict):
            template: dict[Any, Any] = {}
            skeleton: dict[Any, Any] = {}
            for key, child in item.items():
                child_template, child_skeleton = visit(child, item_path + (key,))
                template[key] = child_template
                skeleton[key] = child_skeleton
            return {"__dict__": template}, {"__dict__": skeleton}
        if isinstance(item, list):
            template = []
            skeleton = []
            for index, child in enumerate(item):
                child_template, child_skeleton = visit(child, item_path + (index,))
                template.append(child_template)
                skeleton.append(child_skeleton)
            return {"__list__": template}, {"__list__": skeleton}
        if isinstance(item, tuple):
            template = []
            skeleton = []
            for index, child in enumerate(item):
                child_template, child_skeleton = visit(child, item_path + (index,))
                template.append(child_template)
                skeleton.append(child_skeleton)
            return {"__tuple__": template}, {"__tuple__": skeleton}
        return {"__scalar__": list(item_path)}, {"__value__": item}

    template, skeleton = visit(value, path)
    layout = TensorTreeLayout(template=template, manifest=tuple(dict(entry) for entry in manifest))
    return layout, skeleton, manifest, leaves


def flatten_tensor_tree_with_layout(
    value: Any,
    layout: TensorTreeLayout,
    *,
    path: tuple[Any, ...] = (),
) -> tuple[Any, list[dict[str, Any]], dict[tuple[Any, ...], torch.Tensor]]:
    """Flatten a current state using a previously validated tensor layout.

    A shape/layout change raises ``ValueError`` so callers can rebuild the
    layout once.  This makes incremental metadata reuse fail closed instead of
    silently pairing a stale target manifest with a changed model state.
    """

    leaves: dict[tuple[Any, ...], torch.Tensor] = {}
    manifest_by_path = {tuple(entry["path"]): entry for entry in layout.manifest}

    def visit(item: Any, template: Any, item_path: tuple[Any, ...]) -> Any:
        if isinstance(template, dict) and "__tensor__" in template:
            if not isinstance(item, torch.Tensor):
                raise ValueError(f"tensor layout changed at {item_path!r}")
            expected_path = tuple(template["__tensor__"])
            if expected_path != item_path:
                raise ValueError(f"tensor path changed at {item_path!r}")
            expected = manifest_by_path.get(item_path)
            if expected is None:
                raise ValueError(f"tensor manifest is missing path {item_path!r}")
            if tuple(int(value) for value in expected["shape"]) != tuple(item.shape):
                raise ValueError(f"tensor shape changed at {item_path!r}")
            if str(item.dtype).removeprefix("torch.") != str(expected["dtype"]):
                raise ValueError(f"tensor dtype changed at {item_path!r}")
            leaves[item_path] = item.detach()
            return {"__tensor__": list(item_path)}
        if isinstance(template, dict) and "__scalar__" in template:
            return {"__value__": item}
        if isinstance(template, dict) and "__dict__" in template:
            if not isinstance(item, dict) or set(item) != set(template["__dict__"]):
                raise ValueError(f"mapping layout changed at {item_path!r}")
            return {
                "__dict__": {
                    key: visit(item[key], child, item_path + (key,))
                    for key, child in template["__dict__"].items()
                }
            }
        if isinstance(template, dict) and "__list__" in template:
            if not isinstance(item, list) or len(item) != len(template["__list__"]):
                raise ValueError(f"list layout changed at {item_path!r}")
            return {
                "__list__": [
                    visit(child, child_template, item_path + (index,))
                    for index, (child, child_template) in enumerate(zip(item, template["__list__"]))
                ]
            }
        if isinstance(template, dict) and "__tuple__" in template:
            if not isinstance(item, tuple) or len(item) != len(template["__tuple__"]):
                raise ValueError(f"tuple layout changed at {item_path!r}")
            return {
                "__tuple__": [
                    visit(child, child_template, item_path + (index,))
                    for index, (child, child_template) in enumerate(zip(item, template["__tuple__"]))
                ]
            }
        raise ValueError(f"invalid tensor layout at {item_path!r}")

    skeleton = visit(value, layout.template, path)
    return skeleton, [dict(entry) for entry in layout.manifest], leaves


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


def select_incremental_tensor_delta(
    base_leaves: dict[tuple[Any, ...], torch.Tensor],
    current_leaves: dict[tuple[Any, ...], torch.Tensor],
    manifest: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], torch.Tensor]]:
    """Return changed tensor leaves while preserving the current manifest order.

    The target already owns ``base_leaves`` from the pre-copy.  A changed leaf
    is sent as a complete leaf, which keeps the existing NCCL tensor ABI and
    gives model backends a safe fallback when they cannot expose page-level
    dirty tracking.  Shape or dtype changes are treated as changes and are
    validated by the receiver against the full manifest.
    """

    changed: set[tuple[Any, ...]] = set()
    for path, tensor in current_leaves.items():
        previous = base_leaves.get(path)
        if previous is None:
            changed.add(path)
            continue
        if tuple(previous.shape) != tuple(tensor.shape) or previous.dtype != tensor.dtype:
            changed.add(path)
            continue
        if not torch.equal(previous, tensor):
            changed.add(path)
    delta_manifest = [
        dict(entry)
        for entry in manifest
        if tuple(entry.get("path", ())) in changed
    ]
    return delta_manifest, {path: tensor for path, tensor in current_leaves.items() if path in changed}


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
    submission_lock: threading.Lock | None = None,
) -> TensorTransferReport:
    """Transfer ordered state groups on an optional dedicated CUDA stream.

    Both peers must supply identical groups.  Groups are submitted in manifest
    order, and per-group CUDA events provide a stable layer-ready boundary
    while the caller's worker thread remains free to overlap later groups with
    model computation on another CUDA stream.  A single final stream fence
    still defines complete-transfer readiness.
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
                # ``batch_isend_irecv`` updates process-group coalescing state
                # that is not thread-safe.  Serialize only this short
                # submission call; request completion and CUDA transfers may
                # continue concurrently with other GPU pairs.
                lock_context = submission_lock if submission_lock is not None else contextlib.nullcontext()
                with lock_context:
                    requests = dist.batch_isend_irecv(ops)
                for request in requests:
                    request.wait()
            # ``Work.wait()`` guarantees that the NCCL operation has been
            # submitted.  The event recorded below is the device-side fence
            # consumed by the target compute stream; a host synchronize here
            # would serialize every layer and erase the intended overlap.
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
