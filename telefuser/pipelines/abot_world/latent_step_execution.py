"""Persistent task-row storage and replaceable whole-latent step execution."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Literal

import torch


def clone_cache(cache: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in entry.items()}
        for entry in cache
    ]


class LatentStepArena:
    """Keep fixed-capacity KV and input storage across changes in task membership.

    Dense attention still requires copying incoming historical KV when a task
    enters a different cohort. Stable cohorts do not copy history per step.
    Departing tasks retain only their own row, not the entire old allocation.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cohort: tuple = ()
        self.self_cache: list[dict[str, Any]] = []
        self.cross_cache: list[dict[str, Any]] = []
        self._storage: dict[str, list[dict[str, Any]]] = {}
        self._inputs: dict[str, torch.Tensor] = {}
        self.generation = 0
        self.rebindings = 0
        self.cache_copy_bytes = 0

    def _preserve(self, task: Any, attribute: str) -> None:
        """Detach a dense row without copying its uninitialized capacity."""
        cache = getattr(task, attribute)
        for entry in cache:
            valid = task.valid_tokens if attribute == "self_cache" else entry["sequence_length"]
            saved = {}
            for name, value in entry.items():
                if name in {"k", "v"}:
                    capacity = getattr(task, "capacity_tokens", 0) if attribute == "self_cache" else 0
                    saved[name] = value.new_empty((value.shape[0], capacity or value.shape[1], *value.shape[2:]))
                    saved[name][:, :valid].copy_(value[:, :valid])
                    self.cache_copy_bytes += value[:, :valid].numel() * value.element_size()
                else:
                    saved[name] = value
            entry.update(saved)

    def detach(self, keeping: tuple = ()) -> None:
        preserved = {}
        for task in self.cohort:
            if task not in keeping and not getattr(task, "discarded", False):
                for attribute in ("self_cache", "cross_cache"):
                    group = getattr(task, "cache_group", None)
                    key = (group, attribute)
                    if group is not None and key in preserved:
                        # Whole sibling dependencies advance together and
                        # retain identical KV. Share their read-only history
                        # off-arena; bind stages independent rows before writes.
                        for entry, source in zip(getattr(task, attribute), preserved[key], strict=True):
                            for name in ("k", "v"):
                                entry[name] = source[name]
                    else:
                        self._preserve(task, attribute)
                        if group is not None:
                            preserved[key] = getattr(task, attribute)
        self.cohort = ()
        self.self_cache, self.cross_cache = [], []

    def bind(self, tasks: list) -> None:
        cohort = tuple(tasks)
        if len(tasks) > self.capacity:
            raise ValueError("A latent-step wave exceeds the local task-slot capacity")
        if not tasks:
            self.detach()
            self.self_cache, self.cross_cache = [], []
            return
        first = tasks[0]
        self_capacity = max(getattr(task, "capacity_tokens", 0) or task.self_cache[0]["k"].shape[1] for task in tasks)
        previous = self._storage.get("self_cache")
        if previous:
            value = previous[0]["k"]
            reference = first.self_cache[0]["k"]
            if (
                value is not None
                and value.shape[2:] == reference.shape[2:]
                and value.dtype == reference.dtype
                and value.device == reference.device
            ):
                # Keep the high-water allocation across W changes. Repeated
                # shrinking/growing fragments large KV blocks and also forces
                # unchanged members to copy their complete visible history.
                self_capacity = max(self_capacity, value.shape[1])
        signature = tuple(
            (
                attribute,
                name,
                (self_capacity, *value.shape[2:]) if attribute == "self_cache" else value.shape[1:],
                value.dtype,
                value.device,
            )
            for attribute in ("self_cache", "cross_cache")
            for entry in getattr(first, attribute)
            for name, value in entry.items()
            if name in {"k", "v"}
        )
        changed = signature != getattr(self, "_signature", None)
        if cohort == self.cohort and not changed:
            return
        previous_rows = {id(task): row for row, task in enumerate(self.cohort)}
        self.detach(keeping=cohort)
        moving = [row for row, task in enumerate(tasks) if changed or previous_rows.get(id(task)) != row]
        indices = torch.tensor(moving, device=first.self_cache[0]["k"].device, dtype=torch.long) if moving else None
        if changed:
            self._signature = signature
            self.generation += 1
            self._storage.clear()
            self._inputs.clear()
            for attribute in ("self_cache", "cross_cache"):
                self._storage[attribute] = [
                    {name: (None if name in {"k", "v"} else value) for name, value in entry.items()}
                    for entry in getattr(first, attribute)
                ]
        for attribute in ("self_cache", "cross_cache"):
            views = []
            for layer, storage in enumerate(self._storage[attribute]):
                entry = {}
                for name, value in storage.items():
                    reference = getattr(first, attribute)[layer][name]
                    if name in {"k", "v"}:
                        # Stage one field at a time: a large cohort must not
                        # temporarily hold a third copy of every KV layer.
                        if value is None:
                            shape = (
                                (self.capacity, self_capacity, *reference.shape[2:])
                                if attribute == "self_cache"
                                else (self.capacity, *reference.shape[1:])
                            )
                            value = reference.new_empty(shape)
                            storage[name] = value
                        entry[name] = value[: len(tasks)]
                        if moving:
                            valid = (
                                max(tasks[row].valid_tokens for row in moving)
                                if attribute == "self_cache"
                                else max(tasks[row].cross_cache[layer]["sequence_length"] for row in moving)
                            )
                            sources = []
                            for row in moving:
                                source = getattr(tasks[row], attribute)[layer][name]
                                if source.shape[1] < valid:
                                    padded = source.new_empty((1, valid, *source.shape[2:]))
                                    padded[:, : source.shape[1]].copy_(source)
                                    source = padded
                                sources.append(source[:, :valid])
                            staged = torch.cat(sources)
                            entry[name][:, :valid].index_copy_(0, indices, staged)
                            self.cache_copy_bytes += 2 * staged.numel() * staged.element_size()
                        for row, task in enumerate(tasks):
                            getattr(task, attribute)[layer][name] = entry[name][row : row + 1]
                    elif isinstance(value, torch.Tensor):
                        # Cursor values are supplied as host wave metadata;
                        # canonical tensors are published at state release.
                        entry[name] = reference
                    else:
                        entry[name] = reference
                if attribute == "cross_cache":
                    spans: list[tuple[int, int]] = []
                    for row, task in enumerate(tasks):
                        if not task.cross_cache[layer]["is_init"]:
                            if spans and spans[-1][1] == row:
                                spans[-1] = (spans[-1][0], row + 1)
                            else:
                                spans.append((row, row + 1))
                    entry["initialize_rows"] = tuple(spans)
                    entry["is_init"] = not spans
                views.append(entry)
            setattr(self, attribute, views)
        self.cohort = cohort
        self.rebindings += 1

    def inputs(self, values: dict[str, list[torch.Tensor]], count: int) -> dict[str, torch.Tensor]:
        result = {}
        for name, rows in values.items():
            reference = rows[0]
            shape = (self.capacity, *reference.shape[1:])
            buffer = self._inputs.get(name)
            if buffer is None or buffer.shape != shape or buffer.dtype != reference.dtype:
                buffer = reference.new_empty(shape)
                self._inputs[name] = buffer
                self.generation += 1
            for row, value in enumerate(rows[:count]):
                buffer[row : row + 1].copy_(value)
            result[name] = buffer[:count]
        return result

    def scatter(self, tasks: list, *, context_length: int) -> None:
        for row, task in enumerate(tasks):
            for attribute in ("self_cache", "cross_cache"):
                for layer, entry in enumerate(getattr(self, attribute)):
                    target = getattr(task, attribute)[layer]
                    for name in ("k", "v"):
                        target[name] = entry[name][row : row + 1]
                    if attribute == "cross_cache":
                        # Graph replay performs device writes, not the Python
                        # metadata assignments made while capturing forward.
                        entry["is_init"], entry["sequence_length"] = True, context_length
                        target["is_init"], target["sequence_length"] = True, context_length


class PagedJobKVArena(LatentStepArena):
    """Query routing only; the physical state lives in worker-owned pages."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self._groups: tuple[tuple[int, ...], ...] = ()
        self._layouts: dict[tuple, dict[str, torch.Tensor]] = {}

    def configure_attention(self) -> None:
        """Publish immutable query-to-state routing for this physical wave."""
        device = self.self_cache[0]["k"].device
        layout_key = (self._groups, device)
        routing = self._layouts.get(layout_key)
        if routing is None:
            order = torch.tensor([row for rows in self._groups for row in rows], device=device, dtype=torch.long)
            routing = {
                "query_rows": order,
                "restore_rows": torch.argsort(order),
                "context_rows": torch.tensor([rows[0] for rows in self._groups], device=device, dtype=torch.long),
            }
            self._layouts[layout_key] = routing
        for cache in (self.self_cache, self.cross_cache):
            for entry in cache:
                entry.update(routing)
                entry["sibling_frames"] = len(self._groups[0])

    def bind(self, tasks: list) -> None:
        groups = {}
        for row, task in enumerate(tasks):
            groups.setdefault(task.cache_group, []).append(row)
        self._groups = tuple(tuple(sorted(rows, key=lambda row: tasks[row].latent_index)) for rows in groups.values())
        if not self._groups:
            self.detach()
            return
        frames = len(self._groups[0])
        if len(tasks) > self.capacity or any(
            tuple(tasks[row].latent_index for row in rows) != tuple(range(frames)) for rows in self._groups
        ):
            raise ValueError("Paged KV requires complete sibling dependency groups within query capacity")
        self.cohort = tuple(tasks)
        states = tuple(tasks[rows[0]].self_cache[0]["paged_state"] for rows in self._groups)
        self.self_cache, self.cross_cache = (
            [
                {"k": entry["k"], "v": entry["v"], "paged_jobs": states, "layer_index": layer}
                for layer, entry in enumerate(getattr(tasks[0], attribute))
            ]
            for attribute in ("self_cache", "cross_cache")
        )
        self.rebindings += 1

    def detach(self, keeping: tuple = ()) -> None:
        self.cohort = ()
        self.self_cache, self.cross_cache = [], []

    def scatter(self, tasks: list, *, context_length: int) -> None:
        for task in tasks:
            for entry in task.cross_cache:
                entry["is_init"], entry["sequence_length"] = True, context_length


class LatentStepExecutor:
    """Use eager or cached per-step CUDA Graphs over the same model operation.

    This is a graph execution backend, not an always-running DiT megakernel.
    The interface isolates that future backend from task admission and state.
    All ranks must select the same backend and execute the same dependency wave.
    """

    def __init__(self, backend: Literal["eager", "cuda_graph"] = "eager", *, max_graphs: int = 4) -> None:
        if backend not in {"eager", "cuda_graph"} or max_graphs < 1:
            raise ValueError("backend must be eager or cuda_graph, and max_graphs must be positive")
        self.backend = backend
        self.max_graphs = max_graphs
        self._graphs: OrderedDict[tuple, tuple] = OrderedDict()
        self._generation: object = None
        self.calls = self.captures = self.replays = 0

    def run(
        self, model: Any, *, key: tuple, generation: tuple, arena: LatentStepArena, refreshing: bool, **kwargs: Any
    ) -> torch.Tensor:
        self.calls += 1

        def forward() -> torch.Tensor:
            with torch.autocast(
                kwargs["x"].device.type,
                dtype=kwargs["x"].dtype,
                enabled=kwargs["x"].device.type == "cuda" and not refreshing,
            ):
                return model.forward_latent_tasks(**kwargs)

        # Page tables and physical addresses can change at any phase boundary.
        # Keep the existing eager fallback until graph slots own stable page
        # indirection buffers; capturing transient page pointers is unsafe.
        if self.backend == "eager" or isinstance(arena, PagedJobKVArena):
            return forward()
        if kwargs["x"].device.type != "cuda":
            raise ValueError("The cuda_graph latent-step backend requires CUDA tensors")
        # Invalidation follows global wave layout, not local allocation events.
        # Empty ranks may allocate their KV arena later than busy ranks; using
        # that local event could make some ranks capture while others replay.
        if generation != self._generation:
            self._graphs.clear()
            self._generation = generation
        # Routing tensors are part of a graph's static input layout. Values of
        # x, timestep, prompt, action and KV change in their persistent buffers.
        if key in self._graphs:
            graph, output, _keepalive = self._graphs.pop(key)
            self._graphs[key] = (graph, output, _keepalive)
            graph.replay()
            self.replays += 1
            return output
        backup = [clone_cache(cache) for cache in (arena.self_cache, arena.cross_cache)]

        def restore() -> None:
            for cache, saved in zip((arena.self_cache, arena.cross_cache), backup, strict=True):
                for entry, reference in zip(cache, saved, strict=True):
                    for name, value in reference.items():
                        if isinstance(value, torch.Tensor):
                            entry[name].copy_(value)
                        else:
                            entry[name] = value

        stream = torch.cuda.Stream(device=kwargs["x"].device)
        stream.wait_stream(torch.cuda.current_stream(kwargs["x"].device))
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(stream):
                forward()
                restore()
                with torch.cuda.graph(graph, stream=stream):
                    output = forward()
                restore()
            torch.cuda.current_stream(kwargs["x"].device).wait_stream(stream)
        except Exception:
            torch.cuda.current_stream(kwargs["x"].device).wait_stream(stream)
            restore()
            raise
        # Keep external communication allocations and route tables alive even
        # if the next wave selects a different batch or ownership layout.
        exchange = kwargs["exchange"]
        keepalive = (
            *exchange._buffers.values(),
            exchange._global_offsets,
            exchange._sibling_offsets,
            exchange.frame_indices,
            *(
                arena.self_cache[0][name]
                for name in ("query_rows", "restore_rows", "context_rows")
                if arena.self_cache and name in arena.self_cache[0]
            ),
        )
        self._graphs[key] = (graph, output, keepalive)
        if len(self._graphs) > self.max_graphs:
            self._graphs.popitem(last=False)
        graph.replay()
        self.captures += 1
        self.replays += 1
        return output

    def clear(self) -> None:
        self._graphs.clear()
