"""Whole-latent, single-step task scheduling across an arbitrary GPU pool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any, Literal

import torch

from telefuser.distributed.latent_tasks import LatentTaskExchange

from .continuous import ABotWorldCompletedChunk, DecodeSubmitter, PostprocessSubmitter
from .fidelity import ABotWorldFidelity
from .latent_step_execution import LatentStepArena, LatentStepExecutor, PagedJobKVArena
from .paged_kv import KVPagePool, PagedSessionKV
from .phase_transfer import ABotPhaseTransfer
from .step_batching import ABotWorldStepBatcher, ABotWorldStepTask

if TYPE_CHECKING:
    from PIL import Image

    from .interactive import ABotWorldInteractivePipeline, ABotWorldInteractiveSession


@dataclass(frozen=True)
class LatentStepKey:
    """The scheduling unit: one complete latent at one diffusion step."""

    session_id: str
    chunk_start: int
    latent_index: int
    step_index: int
    refreshing: bool
    owner: int


@dataclass(eq=False)
class _LatentState:
    session_id: str
    latent_index: int
    self_cache: list[dict[str, Any]]
    cross_cache: list[dict[str, Any]]
    valid_tokens: int = 0
    latent: torch.Tensor | None = None
    capacity_tokens: int = 0
    discarded: bool = False
    cache_group: object | None = None


class ABotWorldLatentStepRuntime:
    """Batch whole latent-step rows, keeping their state on the assigned rank.

    A dependency wave includes every sibling latent required by attention.
    Siblings can share a GPU or use different GPUs, and different sessions can
    use different denoising steps. This does not spatially split a latent.
    All ranks drive the same submit/step/release sequence. A session remains
    owned by this runtime between chunks; explicit release materializes its
    ordinary pipeline state, instead of broadcasting historical KV per step.
    An optional service decode submitter receives one refreshed latent row
    after the GPU boundary.  The service may batch those rows across waves and
    serialize each TAeW stream; frame conversion then runs in the existing
    output executor and can overlap the next DiT wave.
    """

    def __init__(
        self,
        pipeline: ABotWorldInteractivePipeline,
        *,
        max_batch_size: int,
        backend: Literal["eager", "cuda_graph"] = "eager",
        max_graphs: int = 4,
        postprocess_submitter: PostprocessSubmitter | None = None,
        decode_submitter: DecodeSubmitter | None = None,
        refresh_ranks: Sequence[int] = (),
        local_execution: bool = False,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive; it counts local latent task slots")
        self.pipeline = pipeline
        self.local_execution = local_execution
        self.exchange = LatentTaskExchange(local=local_execution)
        self._output_exchange = LatentTaskExchange(local=local_execution)
        refresh = tuple(sorted(refresh_ranks))
        invalid_ranks = any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank >= self.exchange.world_size
            for rank in refresh
        )
        if len(set(refresh)) != len(refresh) or invalid_ranks or len(refresh) == self.exchange.world_size:
            raise ValueError("refresh_ranks must be distinct valid ranks and leave at least one denoise GPU")
        if refresh and max_batch_size < 3:
            raise ValueError("Phase-separated execution needs three local slots for a colocated sibling group")
        self.refresh_ranks = refresh
        self.denoise_ranks = tuple(rank for rank in range(self.exchange.world_size) if rank not in refresh)
        self.phase_transfer = ABotPhaseTransfer(self.denoise_ranks, refresh) if refresh else None
        self.max_batch_size = max_batch_size
        self.arena = PagedJobKVArena(max_batch_size) if local_execution else LatentStepArena(max_batch_size)
        self.kv_pages = KVPagePool()
        self._paged_states: dict[str, PagedSessionKV] = {}
        self.executor = LatentStepExecutor(backend, max_graphs=max_graphs)
        self._postprocess_submitter = postprocess_submitter
        self._decode_submitter = decode_submitter
        self._builder = ABotWorldStepBatcher(pipeline.denoise_stage, max_batch_size=1)
        self._sessions: dict[str, ABotWorldInteractiveSession] = {}
        self._requests: dict[str, ABotWorldStepTask] = {}
        self._cursors: dict[str, tuple[int, int]] = {}
        self._owners: dict[str, tuple[int, ...]] = {}
        self._refresh_owners: dict[str, tuple[int, ...]] = {}
        self._refresh_cursors: dict[str, tuple[int, int]] = {}
        self._refresh_prepared: set[str] = set()
        self._retained_capacities: dict[str, int] = {}
        self._canonical_capacities: dict[str, int] = {}
        self._local: dict[tuple[str, int], _LatentState] = {}
        self._slot_layout: tuple[tuple[tuple[str, int], ...], ...] = ((),) * self.exchange.world_size
        self._storage_layout: tuple[int, ...] = ()
        self._storage_epoch = 0
        self._next_owner = 0
        self.last_tasks: tuple[LatentStepKey, ...] = ()
        self.last_local_tasks: tuple[LatentStepKey, ...] = ()
        self.last_windows: dict[str, int] = {}
        self.last_batch_size = 0
        self.dispatch_count = 0
        self.compute_forward_count = 0
        self.latent_step_items = 0
        self.logical_kv_send_bytes = 0
        self.logical_output_send_bytes = 0

    @property
    def has_pending(self) -> bool:
        return bool(self._requests)

    @property
    def is_output_rank(self) -> bool:
        return self.exchange.rank == 0

    @torch.inference_mode()
    def create_session(
        self, image: Image.Image, prompt: str, *, seed: int = 42, session_id: str | None = None
    ) -> ABotWorldInteractiveSession:
        """Create an owned session; allocate KV only after placement is known."""
        session = self.pipeline._create_interactive_session(
            image, prompt, seed=seed, session_id=session_id, cache_batch_size=0
        )
        session.latent_runtime_owner = self
        self._sessions[session.session_id] = session
        return session

    def _retain(
        self,
        session: ABotWorldInteractiveSession,
        owners: tuple[int, ...] | None = None,
        *,
        capacity_tokens: int,
        valid_tokens: int,
        refresh_owners: tuple[int, ...] | None = None,
    ) -> None:
        if owners is None:
            # Keep a session's three sibling latents together.  This is the
            # cheap/default path: step batching remains local to a worker and
            # does not require an all-rank K/V exchange at every DiT layer.
            # The policy may still pass an explicitly split placement when
            # local capacity is exhausted.
            owner = self.denoise_ranks[self._next_owner % len(self.denoise_ranks)]
            owners = (owner, owner, owner)
            self._next_owner += 1
        refresh_owner = None
        if self.refresh_ranks:
            refresh_owner = (
                refresh_owners[0]
                if refresh_owners
                else min(
                    self.refresh_ranks,
                    key=lambda rank: (sum(placement[0] == rank for placement in self._refresh_owners.values()), rank),
                )
            )
            self._refresh_owners[session.session_id] = (refresh_owner,) * 3
            self._refresh_cursors[session.session_id] = self._cursors.get(session.session_id, (0, valid_tokens))
        first_local = True
        cache_group = object()
        if self.local_execution:
            self._ensure_paged_session(session)
        for index, owner in enumerate(owners):
            if owner != self.exchange.rank and refresh_owner != self.exchange.rank:
                continue
            # Local sibling queries share one Job history. Explicit distributed
            # sibling placement retains the existing rank-local dense backend.
            source = self._local[session.session_id, 0] if self.local_execution and not first_local else session
            caches = ([dict(entry) for entry in source.self_cache], [dict(entry) for entry in source.cross_cache])
            for attribute, cache in zip(("self_cache", "cross_cache"), caches, strict=True):
                # Siblings have identical history at phase admission and keep
                # one shared state row throughout local phase execution.
                if self.local_execution:
                    continue
                for entry in cache:
                    for name, value in entry.items():
                        if not isinstance(value, torch.Tensor):
                            continue
                        if name in {"k", "v"}:
                            capacity = capacity_tokens if attribute == "self_cache" else value.shape[1]
                            if first_local and value.shape[0] and (capacity == value.shape[1] or self.local_execution):
                                continue
                            saved = value.new_empty((1, capacity, *value.shape[2:]))
                            valid = valid_tokens if attribute == "self_cache" else entry["sequence_length"]
                            if valid:
                                saved[:, :valid].copy_(value[:, :valid])
                            entry[name] = saved
                        else:
                            entry[name] = value.clone()
            self._local[session.session_id, index] = _LatentState(
                session.session_id,
                index,
                *caches,
                valid_tokens=valid_tokens,
                capacity_tokens=capacity_tokens,
                cache_group=cache_group,
            )
            first_local = False
        # Keep layout and cursor metadata, but no redundant full session KV on
        # ranks that do not own it. Ordinary pipeline calls are blocked by lease.
        for cache in (session.self_cache, session.cross_cache):
            for entry in cache:
                for name in ("k", "v"):
                    entry[name] = entry[name].new_empty((0, *entry[name].shape[1:]))
        self._sessions[session.session_id] = session
        self._owners[session.session_id] = owners
        session.latent_runtime_owner = self

    def _ensure_paged_session(self, session: ABotWorldInteractiveSession) -> PagedSessionKV:
        state = self._paged_states.get(session.session_id)
        if state is not None:
            return state
        shape = session.first_frame_latent.shape
        patch = self.pipeline.denoise_stage.dit.patch_size
        tokens = (shape[-2] // patch[1]) * (shape[-1] // patch[2])
        state = PagedSessionKV(
            self.kv_pages, session.self_cache[0]["k"], layers=len(session.self_cache), frame_tokens=tokens
        )
        global_end, valid = self._cursors.get(
            session.session_id,
            (
                int(session.self_cache[0]["global_end_index"].item()),
                int(session.self_cache[0]["local_end_index"].item()),
            ),
        )
        if valid:
            state.prepare(
                current_end=global_end,
                window=valid // tokens,
                sink=0,
                frames=valid // tokens,
                context_length=session.prompt_emb.shape[1],
            )
            for layer, entry in enumerate(session.self_cache):
                state.write_layer(layer, entry["k"][:, :valid], entry["v"][:, :valid])
        state.global_end = global_end
        for attribute in ("self_cache", "cross_cache"):
            for layer, entry in enumerate(getattr(session, attribute)):
                if attribute == "cross_cache" and entry["is_init"]:
                    if state.cross_page is None:
                        state.prepare(
                            current_end=global_end,
                            window=1,
                            sink=0,
                            frames=0,
                            context_length=entry["sequence_length"],
                        )
                    for field, name in enumerate(("k", "v")):
                        state.cross(layer, field).copy_(entry[name][:, : state.cross_length])
                    state.cross_ready[layer] = True
                for name in ("k", "v"):
                    entry[name] = entry[name].new_empty((0, *entry[name].shape[1:]))
                entry["paged_state"] = state
        self._paged_states[session.session_id] = state
        return state

    @torch.inference_mode()
    def create_conditioned_session(
        self,
        session_id: str,
        first_frame: torch.Tensor,
        prompt: torch.Tensor,
        *,
        seed: int = 42,
        initialize_decode: bool = True,
    ) -> ABotWorldInteractiveSession:
        """Install immutable conditioning on a phase actor without re-encoding."""
        from .interactive import ABotWorldInteractiveSession

        pipeline = self.pipeline
        self_cache, cross_cache = pipeline.denoise_stage._new_cache(0, first_frame.shape[-2], first_frame.shape[-1])
        session = ABotWorldInteractiveSession(
            session_id=session_id,
            first_frame_latent=first_frame.clone(),
            prompt_emb=prompt.clone(),
            self_cache=self_cache,
            cross_cache=cross_cache,
            scheduler=pipeline.denoise_stage._scheduler(),
            generator=torch.Generator(device=pipeline.device).manual_seed(seed),
        )
        if initialize_decode:
            session.taew_decode_state = pipeline.taew_decode_stage.create_decode_state()
            pipeline.taew_decode_stage.warmup_first_frame(session.taew_decode_state, session.first_frame_latent)
        with pipeline._lifecycle_lock:
            if session_id in pipeline._interactive_sessions:
                raise ValueError(f"Session {session_id!r} already exists")
            pipeline._interactive_sessions[session_id] = session
        self._sessions[session_id] = session
        session.latent_runtime_owner = self
        return session

    @torch.inference_mode()
    def export_canonical_state(self, session_id: str) -> dict[str, Any]:
        """Export clean parked compute state without transferring decoder state.

        Only phase boundaries may move a session. Borrow immutable prefixes;
        the transfer retains their allocation until destination installation.
        """
        if session_id in self._requests or session_id in self._owners:
            raise RuntimeError("Canonical state transfer requires an idle parked session")
        session = self._sessions[session_id]
        cursor = self._cursors.get(session_id, (0, 0))
        if self.local_execution:
            paged = self._ensure_paged_session(session)
            return dict(
                first_frame=session.first_frame_latent,
                prompt=session.prompt_emb,
                generator_state=session.generator.get_state().numpy().tobytes().hex(),
                next_latent_frame=session.next_latent_frame,
                cursor=cursor,
                capacity_tokens=session.self_cache[0]["k"].shape[1] if paged.table else 0,
                cross_capacity_tokens=session.cross_cache[0]["k"].shape[1],
                **paged.export(),
            )
        caches = {}
        for attribute in ("self_cache", "cross_cache"):
            caches[attribute] = [
                {
                    name: (
                        value[:, : cursor[1] if attribute == "self_cache" else entry["sequence_length"]]
                        if name in {"k", "v"}
                        else value.clone()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for name, value in entry.items()
                    if name != "paged_state"
                }
                for entry in getattr(session, attribute)
            ]
        payload = dict(
            first_frame=session.first_frame_latent,
            prompt=session.prompt_emb,
            # Generator state is already host metadata, not a GPU KV page.
            # Keep it on the control path rather than staging it through NCCL.
            generator_state=session.generator.get_state().numpy().tobytes().hex(),
            next_latent_frame=session.next_latent_frame,
            cursor=cursor,
            capacity_tokens=(session.self_cache[0]["k"].shape[1] if session.self_cache[0]["k"].shape[0] else 0),
            cross_capacity_tokens=session.cross_cache[0]["k"].shape[1],
            **caches,
        )
        return payload

    @torch.inference_mode()
    def import_canonical_state(
        self, session_id: str, payload: Mapping[str, Any], *, seed: int, adopt_pages: bool = False
    ) -> Any:
        """Install SST payload while leaving a separately owned VAE stream alone."""
        if session_id in self._requests or session_id in self._owners:
            raise RuntimeError("Cannot install canonical state over active compute")
        session = self._sessions.get(session_id)
        if session is None:
            session = self.create_conditioned_session(
                session_id, payload["first_frame"], payload["prompt"], seed=seed, initialize_decode=False
            )
        capacity = int(payload["capacity_tokens"])
        if self.local_execution and "paged_self" in payload:
            state = self._ensure_paged_session(session)
            state.install(payload, copy=not adopt_pages)
            for attribute in ("self_cache", "cross_cache"):
                for layer, entry in enumerate(getattr(session, attribute)):
                    tokens = capacity if attribute == "self_cache" else int(payload["cross_capacity_tokens"])
                    for name in ("k", "v"):
                        entry[name] = entry[name].new_empty((0, tokens, *entry[name].shape[2:]))
                    if attribute == "self_cache":
                        entry["global_end_index"].fill_(int(payload["cursor"][0]))
                        entry["local_end_index"].fill_(int(payload["cursor"][1]))
                    else:
                        entry["is_init"], entry["sequence_length"] = state.cross_ready[layer], state.cross_length
            session.generator.set_state(
                torch.frombuffer(bytearray.fromhex(payload["generator_state"]), dtype=torch.uint8)
            )
            session.next_latent_frame = int(payload["next_latent_frame"])
            self._cursors[session_id] = tuple(payload["cursor"])
            self._canonical_capacities[session_id] = capacity
            return session
        for attribute in ("self_cache", "cross_cache"):
            cache = []
            for source in payload[attribute]:
                entry = {}
                for name, value in source.items():
                    if name in {"k", "v"}:
                        tokens = capacity if attribute == "self_cache" else int(payload["cross_capacity_tokens"])
                        shape = (1, tokens, *value.shape[2:])
                        saved = value.new_empty(shape)
                        if value.numel():
                            saved[:, : value.shape[1]].copy_(value)
                        entry[name] = saved
                    else:
                        entry[name] = value.clone() if isinstance(value, torch.Tensor) else value
                cache.append(entry)
            setattr(session, attribute, cache)
        session.generator.set_state(torch.frombuffer(bytearray.fromhex(payload["generator_state"]), dtype=torch.uint8))
        session.next_latent_frame = int(payload["next_latent_frame"])
        self._cursors[session_id] = tuple(payload["cursor"])
        self._canonical_capacities[session_id] = capacity
        return session

    @torch.inference_mode()
    def submit(
        self,
        session: ABotWorldInteractiveSession,
        actions: Mapping[str, bool] | None = None,
        *,
        control_latent_frames: int = 3,
        timestep_positions: tuple[int, ...] | None = None,
        fidelity: ABotWorldFidelity | None = None,
        owners: tuple[int, ...] | None = None,
        refresh_owners: tuple[int, ...] | None = None,
    ) -> None:
        pipeline = self.pipeline
        with pipeline._execution_lock:
            if session.latent_runtime_owner is not self:
                pipeline._require_session(session)
            if session.closed or session.session_id in self._requests:
                raise ValueError("Session is closed or already has unfinished latent-step work")
            if control_latent_frames not in {1, 2, 3}:
                raise ValueError("control_latent_frames must be 1, 2, or 3")
            if self.refresh_ranks and control_latent_frames != 3:
                raise ValueError("Phase-separated execution currently requires a complete three-latent chunk")
            if session.migration_transfer_in_progress:
                raise ValueError("Finish state transfer before latent-step admission")
            if self.local_execution:
                self._ensure_paged_session(session)
            if owners is not None:
                if len(owners) != 3 or any(
                    not isinstance(owner, int)
                    or isinstance(owner, bool)
                    or owner < 0
                    or owner >= self.exchange.world_size
                    for owner in owners
                ):
                    raise ValueError("Provide one valid GPU owner for each of the three whole latents")
                if session.session_id in self._owners and owners != self._owners[session.session_id]:
                    raise ValueError("Retained latent owners cannot change without an explicit state transfer")
                if self.refresh_ranks and (len(set(owners)) != 1 or owners[0] not in self.denoise_ranks):
                    raise ValueError("Phase-separated denoise owners must colocate siblings on a denoise GPU")
            if refresh_owners is not None:
                if (
                    len(refresh_owners) != 3
                    or len(set(refresh_owners)) != 1
                    or refresh_owners[0] not in self.refresh_ranks
                ):
                    raise ValueError("refresh_owners must colocate three siblings on a refresh GPU")
                existing = self._refresh_owners.get(session.session_id)
                if existing is not None and refresh_owners != existing:
                    raise ValueError("A persistent refresh replica cannot change owner during a session")
            if fidelity is not None:
                if timestep_positions is not None:
                    raise ValueError("Pass fidelity or timestep_positions, not both")
                timestep_positions = fidelity.denoise_step_positions
                frame_tokens = (session.first_frame_latent.shape[-2] // pipeline.denoise_stage.dit.patch_size[1]) * (
                    session.first_frame_latent.shape[-1] // pipeline.denoise_stage.dit.patch_size[2]
                )
                if fidelity.local_attn_size * frame_tokens > session.self_cache[0]["k"].shape[1]:
                    raise ValueError("Fidelity window exceeds the session's retained KV allocation")
            if timestep_positions is not None and (
                not timestep_positions
                or timestep_positions != tuple(sorted(set(timestep_positions)))
                or not 0 <= timestep_positions[0] <= timestep_positions[-1] < 4
            ):
                raise ValueError("timestep_positions must be increasing indices in [0, 3]")
            placement = owners if owners is not None else self._owners.get(session.session_id)
            peak_rows = (
                max(placement[:control_latent_frames].count(rank) for rank in range(self.exchange.world_size))
                if placement is not None
                else (control_latent_frames + self.exchange.world_size - 1) // self.exchange.world_size
            )
            if peak_rows > self.max_batch_size:
                raise ValueError("Local latent task slots cannot fit one complete attention dependency group")
            if fidelity is not None:
                self._trim_history(session, fidelity, frame_tokens)
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
            task = self._builder.begin(
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
                shard_latents=False,
                cache_indices=self._cursors.get(session.session_id),
            )
            if fidelity is not None:
                task.window = (fidelity.local_attn_size, fidelity.sink_size)
            capacity_tokens = session.self_cache[0]["k"].shape[1]
            if task.window[0] > 0:
                capacity_tokens = min(capacity_tokens, task.window[0] * task.frame_tokens)
            if session.session_id not in self._owners:
                self._cursors[session.session_id] = task.global_end, task.local_end
                self._retain(
                    session,
                    owners,
                    capacity_tokens=capacity_tokens,
                    valid_tokens=task.local_end,
                    refresh_owners=refresh_owners,
                )
            self._retained_capacities[session.session_id] = capacity_tokens
            self._cursors[session.session_id] = task.global_end, task.local_end
            for index in range(control_latent_frames):
                local = self._local.get((session.session_id, index))
                if local is not None:
                    local.latent = task.latent[:, :, index : index + 1]
                    # The refresh replica contains clean history only. Its
                    # next append is independent of the denoise cursor.
                    if not self.refresh_ranks or self.exchange.rank in self.denoise_ranks:
                        local.valid_tokens = task.local_end
                    local.capacity_tokens = capacity_tokens
            self._requests[session.session_id] = task
            session.step_task_active = True
            pipeline._release_cuda_graph(session.session_id)

    def _trim_history(self, session: ABotWorldInteractiveSession, fidelity: ABotWorldFidelity, tokens: int) -> None:
        """Drop excess history once at a Job boundary without resizing the arena.

        A later window increase cannot resurrect discarded history. Keeping
        allocation capacity independent of the visible window permits mixed
        quality rows without padding their actual attention computations.
        """
        indices = self._cursors.get(session.session_id)
        if indices is None:
            indices = (
                int(session.self_cache[0]["global_end_index"].item()),
                int(session.self_cache[0]["local_end_index"].item()),
            )
        global_end, valid = indices
        retained = min(valid, fidelity.local_attn_size * tokens)
        if retained == valid:
            return
        if self.local_execution:
            self._ensure_paged_session(session).truncate(fidelity.local_attn_size, fidelity.sink_size)
            self._cursors[session.session_id] = global_end, retained
            for (sid, _), state in self._local.items():
                if sid == session.session_id:
                    state.valid_tokens = retained
            return
        local = [state for (sid, _), state in self._local.items() if sid == session.session_id]
        caches = [state.self_cache for state in local] if local else [session.self_cache]
        caches = list({cache[0]["k"].data_ptr(): cache for cache in caches}.values())
        sink = min(fidelity.sink_size * tokens, retained)
        tail = retained - sink
        for cache in caches:
            for entry in cache:
                for name in ("k", "v"):
                    entry[name][:, sink:retained].copy_(entry[name][:, valid - tail : valid].clone())
        self._cursors[session.session_id] = global_end, retained
        if session.session_id in self._refresh_cursors:
            self._refresh_cursors[session.session_id] = global_end, retained
        for state in local:
            state.valid_tokens = retained

    def task_owners(self, task: ABotWorldStepTask) -> tuple[int, ...]:
        """Return physical owners for the task's current phase."""
        mapping = self._refresh_owners if self.refresh_ranks and task.refreshing else self._owners
        return mapping[task.session_id][: task.full_shape[2]]

    @torch.inference_mode()
    def export_denoised(self, session_id: str) -> torch.Tensor:
        """Expose final latents to another actor without advancing local refresh."""
        task = self._requests[session_id]
        if self.exchange.world_size != 1 or not task.refreshing:
            raise RuntimeError("Denoise export requires a completed local denoise phase")
        return torch.cat([self._local[session_id, index].latent for index in range(task.full_shape[2])], dim=2)

    @torch.inference_mode()
    def submit_refresh(
        self,
        session: ABotWorldInteractiveSession,
        latents: torch.Tensor,
        actions: Mapping[str, bool],
        *,
        fidelity: ABotWorldFidelity,
        chunk_start: int,
    ) -> None:
        """Admit only the clean-latent refresh transition on a local actor."""
        if self.exchange.world_size != 1 or session.next_latent_frame != chunk_start:
            raise RuntimeError("Refresh input must match this actor's causal session cursor")
        self.submit(session, actions, fidelity=fidelity)
        task = self._requests[session.session_id]
        task.step_index = len(task.timesteps)
        for index in range(task.full_shape[2]):
            self._local[session.session_id, index].latent = latents[:, :, index : index + 1].clone()

    @torch.inference_mode()
    def export_clean_delta(self, session_id: str, frames: int = 3) -> torch.Tensor:
        """Pack one canonical clean tail, not three historical KV replicas."""
        if self.local_execution:
            return self._paged_states[session_id].clean_delta(frames)
        _, end = self._cursors[session_id]
        session = self._sessions[session_id]
        patch = self.pipeline.denoise_stage.dit.patch_size
        shape = session.first_frame_latent.shape
        tokens = frames * (shape[-2] // patch[1]) * (shape[-1] // patch[2])
        return torch.stack(
            [
                entry[name][:, end - tokens : end]
                for entry in self._local[session_id, 0].self_cache
                for name in ("k", "v")
            ]
        )

    @torch.inference_mode()
    def ensure_local_capacity(self, session_id: str, fidelity: ABotWorldFidelity) -> None:
        """Grow a parked canonical row only when its next quality requires it."""
        if self.exchange.world_size != 1 or session_id in self._owners:
            raise RuntimeError("Capacity changes require a parked local session")
        session = self._sessions[session_id]
        shape = session.first_frame_latent.shape
        patch = self.pipeline.denoise_stage.dit.patch_size
        tokens = (shape[-2] // patch[1]) * (shape[-1] // patch[2])
        capacity = fidelity.local_attn_size * tokens
        if self.local_execution:
            self._ensure_paged_session(session)
            for entry in session.self_cache:
                for name in ("k", "v"):
                    value = entry[name]
                    entry[name] = value.new_empty((0, max(capacity, value.shape[1]), *value.shape[2:]))
            self._canonical_capacities[session_id] = session.self_cache[0]["k"].shape[1]
            return
        valid = self._cursors.get(session_id, (0, 0))[1]
        for entry in session.self_cache:
            for name in ("k", "v"):
                value = entry[name]
                if value.shape[1] >= capacity:
                    continue
                expanded = value.new_empty((value.shape[0], capacity, *value.shape[2:]))
                if valid and value.shape[0]:
                    expanded[:, :valid].copy_(value[:, :valid])
                entry[name] = expanded
        if session.self_cache[0]["k"].shape[0]:
            self._canonical_capacities[session_id] = session.self_cache[0]["k"].shape[1]

    @torch.inference_mode()
    def park_session(self, session_id: str) -> None:
        """Compact inactive sibling rows into one persistent local clean/noisy row.

        A denoised head may wait for remote clean KV while parked. It remains
        causally in flight and cannot be invoked by the ordinary pipeline.
        Local sibling queries occupy one shared Job history row while active.
        """
        task = self._requests.get(session_id)
        if self.exchange.world_size != 1 or (task is not None and not task.refreshing):
            raise RuntimeError("Only a completed local phase can park its sibling state")
        session = self._sessions[session_id]
        source = self._local[session_id, 0]
        global_end, local_end = self._cursors[session_id]
        if self.local_execution:
            for entry in session.self_cache:
                entry["global_end_index"].fill_(global_end)
                entry["local_end_index"].fill_(local_end)
            for entry in session.cross_cache:
                entry["is_init"], entry["sequence_length"] = True, session.prompt_emb.shape[1]
            for index in range(3):
                self._local.pop((session_id, index)).discarded = True
            del self._owners[session_id]
            return
        for attribute in ("self_cache", "cross_cache"):
            for entry, row in zip(getattr(session, attribute), getattr(source, attribute), strict=True):
                valid = local_end if attribute == "self_cache" else row["sequence_length"]
                capacity = (
                    max(source.capacity_tokens, self._canonical_capacities.get(session_id, 0))
                    if attribute == "self_cache"
                    else row["k"].shape[1]
                )
                for name in ("k", "v"):
                    saved = row[name].new_empty((1, capacity, *row[name].shape[2:]))
                    saved[:, :valid].copy_(row[name][:, :valid])
                    entry[name] = saved
                if attribute == "self_cache":
                    entry["global_end_index"].fill_(global_end)
                    entry["local_end_index"].fill_(local_end)
                else:
                    entry["is_init"], entry["sequence_length"] = row["is_init"], valid
        for index in range(3):
            self._local.pop((session_id, index)).discarded = True
        self._canonical_capacities[session_id] = session.self_cache[0]["k"].shape[1]
        del self._owners[session_id]

    @torch.inference_mode()
    def commit_external_refresh(self, session_id: str, delta: torch.Tensor) -> None:
        """Make clean state visible before admitting this session's next Job."""
        task = self._requests[session_id]
        if self.exchange.world_size != 1 or not task.refreshing:
            raise RuntimeError("Clean KV must commit to the corresponding local denoise head")
        if self.local_execution:
            self._paged_states[session_id].commit_delta(delta)
            session = self._sessions[session_id]
            session.next_latent_frame += task.full_shape[2]
            session.step_task_active = False
            task.complete = True
            del self._requests[session_id]
            return
        end = task.local_end
        tokens = task.full_shape[2] * task.frame_tokens
        local = [
            self._local[session_id, index] for index in range(task.full_shape[2]) if (session_id, index) in self._local
        ]
        caches = [state.self_cache for state in local] if local else [self._sessions[session_id].self_cache]
        caches = list({cache[0]["k"].data_ptr(): cache for cache in caches}.values())
        for cache in caches:
            for layer, entry in enumerate(cache):
                for field, name in enumerate(("k", "v")):
                    entry[name][:, end - tokens : end].copy_(delta[layer * 2 + field])
        for state in local:
            state.valid_tokens = end
        session = self._sessions[session_id]
        session.next_latent_frame += task.full_shape[2]
        session.step_task_active = False
        task.complete = True
        del self._requests[session_id]

    def _compatible(self, task: ABotWorldStepTask) -> tuple:
        key = task.latent_batch_key()
        # Different phases can share a global dispatch only on disjoint GPU
        # pools. Each GPU still executes one homogeneous physical batch.
        return key[:5] + key[6:] if self.refresh_ranks else key

    def _prepare_refresh(self, task: ABotWorldStepTask) -> None:
        """Move final latents; the refresh pool already retains clean history."""
        key = task.session_id
        if not self.refresh_ranks or not task.refreshing or key in self._refresh_prepared:
            return
        source, destination = self._owners[key][0], self._refresh_owners[key][0]
        frames = task.full_shape[2]
        if self.exchange.rank == source:
            values = torch.cat([self._local[key, index].latent for index in range(frames)])
        elif self.exchange.rank == destination:
            values = task.latent.new_empty((frames, task.full_shape[1], 1, *task.full_shape[3:]))
        else:
            values = None
        self.phase_transfer.transfer(
            values,
            source=source,
            destination=destination,
            kind="latent",
            byte_count=prod(task.full_shape) * task.latent.element_size(),
        )
        if self.exchange.rank == destination:
            for index in range(frames):
                self._local[key, index].latent = values[index : index + 1]
        # Refresh appends to its own previous clean cursor; denoise has
        # already appended the current chunk's noisy KV on another GPU.
        task.global_end, task.local_end = self._refresh_cursors[key]
        self._refresh_prepared.add(key)

    def _return_refresh_delta(self, task: ABotWorldStepTask) -> None:
        """Return just the clean chunk KV, once, before the next causal Job."""
        if not self.refresh_ranks:
            return
        key = task.session_id
        source, destination = self._refresh_owners[key][0], self._owners[key][0]
        tokens = task.full_shape[2] * task.frame_tokens
        begin, end = task.local_end - tokens, task.local_end
        if self.exchange.rank == source:
            cache = self._local[key, 0].self_cache
            delta = torch.stack([entry[name][:, begin:end] for entry in cache for name in ("k", "v")])
        elif self.exchange.rank == destination:
            reference = task.self_cache[0]["k"]
            delta = reference.new_empty((len(task.self_cache) * 2, 1, tokens, *reference.shape[2:]))
        else:
            delta = None
        reference = task.self_cache[0]["k"]
        self.phase_transfer.transfer(
            delta,
            source=source,
            destination=destination,
            kind="kv",
            byte_count=len(task.self_cache) * 2 * tokens * prod(reference.shape[2:]) * reference.element_size(),
        )
        if self.exchange.rank == destination:
            for index in range(3):
                local = self._local[key, index]
                for layer, entry in enumerate(local.self_cache):
                    for field, name in enumerate(("k", "v")):
                        entry[name][:, begin:end].copy_(delta[layer * 2 + field])
                local.valid_tokens = task.local_end
        self._refresh_cursors[key] = task.global_end, task.local_end
        self._refresh_prepared.discard(key)

    def _wave(self, session_ids: Sequence[str] | None = None) -> list[ABotWorldStepTask]:
        if session_ids is not None:
            if not session_ids or len(set(session_ids)) != len(session_ids):
                raise ValueError("A planned wave must contain distinct complete dependency groups")
            selected = [self._requests[session_id] for session_id in session_ids]
            key = self._compatible(selected[0])
            counts = [0] * self.exchange.world_size
            for task in selected:
                if self._compatible(task) != key:
                    raise ValueError("A planned wave contains incompatible latent-step groups")
                for owner in self.task_owners(task):
                    counts[owner] += 1
            if max(counts) > self.max_batch_size:
                raise ValueError("A planned wave exceeds the per-rank latent-slot capacity")
            return sorted(selected, key=lambda task: task.attention_key())
        candidates = list(self._requests.values())
        anchor = next((task for task in candidates if task.step_index == 0), candidates[0])
        key = self._compatible(anchor)
        selected, counts = [], [0] * self.exchange.world_size
        for task in candidates:
            if self._compatible(task) != key:
                continue
            owners = self.task_owners(task)
            proposed = [count + owners.count(rank) for rank, count in enumerate(counts)]
            if max(proposed) <= self.max_batch_size:
                selected.append(task)
                counts = proposed
        # Make equal attention layouts contiguous locally. Projection and FFN
        # still execute once for the entire physical batch.
        return sorted(selected, key=lambda task: task.attention_key())

    def _stable_row_order(self) -> tuple[int, ...]:
        """Keep retained rows in their dense slots without changing placement.

        Every rank derives the same permutation from the global wave history;
        no additional collective is needed to agree on routing or graph keys.
        """
        order = []
        layouts = []
        for rank, previous in enumerate(self._slot_layout):
            current = {
                (key.session_id, key.latent_index): index
                for index, key in enumerate(self.last_tasks)
                if key.owner == rank
            }
            rows = [None] * len(current)
            for slot, key in enumerate(previous[: len(current)]):
                if key in current:
                    rows[slot] = current.pop(key)
            incoming = iter(current.values())
            rows = tuple(next(incoming) if row is None else row for row in rows)
            order.extend(rows)
            layouts.append(tuple((self.last_tasks[row].session_id, self.last_tasks[row].latent_index) for row in rows))
        self._slot_layout = tuple(layouts)
        return tuple(order)

    def _sample(
        self,
        wave: list[ABotWorldStepTask],
        local: list[_LatentState],
        prediction: torch.Tensor,
        timestep: torch.Tensor,
    ) -> None:
        """Advance batched samples without copying GPU timesteps to the CPU.

        Noise is still drawn once per complete session chunk, on every rank,
        in the original task order. CPU scalar sigmas retain the native BF16
        promotion and rounding of each multiply and the subsequent addition.
        """
        noise = {}
        for task in wave:
            task.step_index += 1
            if not task.refreshing:
                noise[task.session_id] = torch.randn(
                    task.full_shape, generator=task.generator, dtype=prediction.dtype, device=task.latent.device
                )
        if not local:
            return
        tasks = [self._requests[state.session_id] for state in local]
        sigmas = torch.tensor(
            [float(task.sampling_sigmas[task.step_index - 1]) for task in tasks],
            device=prediction.device,
            dtype=torch.float64,
        )
        latents = torch.cat([state.latent for state in local])
        updated = self.pipeline.denoise_stage._x0_prediction(
            prediction, latents, timestep, tasks[0].scheduler, sigma=sigmas
        )
        groups = {}
        for row, task in enumerate(tasks):
            if not task.refreshing:
                sigma = task.sampling_sigmas[task.step_index]
                groups.setdefault(float(sigma), (sigma, []))[1].append(row)
        for sigma, rows in groups.values():
            indices = torch.tensor(rows, device=prediction.device, dtype=torch.long)
            samples = updated.index_select(0, indices)
            fresh = torch.cat(
                [
                    noise[local[row].session_id][:, :, local[row].latent_index : local[row].latent_index + 1]
                    for row in rows
                ]
            )
            updated.index_copy_(0, indices, (1 - sigma) * samples + sigma * fresh)
        for row, (state, task) in enumerate(zip(local, tasks, strict=True)):
            state.latent = updated[row : row + 1]
            if task.first_frame is not None and state.latent_index == 0:
                state.latent.copy_(task.first_frame)

    def ready_latent_steps(self) -> tuple[LatentStepKey, ...]:
        """Expose current row heads, including their attention dependencies."""
        return tuple(
            LatentStepKey(task.session_id, task.current_start, index, task.step_index, task.refreshing, owner)
            for task in self._requests.values()
            if task.session_id in self._owners
            for index, owner in enumerate(self.task_owners(task))
        )

    def step_latent_tasks(self, keys: Sequence[LatentStepKey]) -> list[ABotWorldCompletedChunk]:
        """Advance selected row heads once, preserving same-layer sibling KV.

        ABot uses bidirectional attention inside a chunk. Missing a sibling's
        current-layer KV changes the model, so a physical invocation must
        close that dependency group. This does not reserve later steps.
        """
        sessions = tuple(dict.fromkeys(key.session_id for key in keys))
        expected = {key for key in self.ready_latent_steps() if key.session_id in sessions}
        if not keys or len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError("Latent-step selection must contain current, complete sibling dependencies")
        return self.step(sessions)

    @torch.inference_mode()
    def step(self, session_ids: Sequence[str] | None = None) -> list[ABotWorldCompletedChunk]:
        with self.pipeline._execution_lock:
            if not self.has_pending:
                self.last_tasks = self.last_local_tasks = ()
                self.last_batch_size = 0
                return []
            if any(session.closed for session in self._sessions.values()):
                raise RuntimeError("A session owned by the latent-step runtime was closed")
            wave = self._wave(session_ids)
            for task in wave:
                self._prepare_refresh(task)
            anchor = wave[0]
            local_anchor = next((task for task in wave if self.exchange.rank in self.task_owners(task)), anchor)
            frames = anchor.full_shape[2]
            refreshing = {task.session_id for task in wave if task.refreshing}
            self.last_windows = {task.session_id: task.window[0] for task in wave}
            self.last_tasks = tuple(
                LatentStepKey(task.session_id, task.current_start, index, task.step_index, task.refreshing, owner)
                for task in wave
                for index, owner in enumerate(self.task_owners(task))
            )
            storage_layout = tuple(
                max(
                    (
                        min(task.self_cache[0]["k"].shape[1], task.window[0] * task.frame_tokens)
                        if task.window[0] > 0
                        else task.self_cache[0]["k"].shape[1]
                        for task in wave
                        if rank in self.task_owners(task)
                    ),
                    default=0,
                )
                for rank in range(self.exchange.world_size)
            )
            previous_layout = self._storage_layout or (0,) * self.exchange.world_size
            storage_layout = tuple(max(old, new) for old, new in zip(previous_layout, storage_layout, strict=True))
            if storage_layout != self._storage_layout:
                # Arena pointers may change when W changes. Invalidate on the
                # same global boundary even on ranks with no local allocation.
                self._storage_layout = storage_layout
                self._storage_epoch += 1
            row_order = self._stable_row_order()
            self.exchange.configure(
                [key.owner for key in self.last_tasks], frames=frames, device=anchor.latent.device, row_order=row_order
            )
            self.last_local_tasks = tuple(self.last_tasks[row] for row in self.exchange.local_rows)
            self.last_batch_size = len(self.last_local_tasks)
            self.dispatch_count += 1
            self.compute_forward_count += int(bool(self.last_batch_size))
            self.latent_step_items += self.last_batch_size
            attention_layout = [task.attention_key() for task in wave]
            if self.pipeline.denoise_stage.dit.use_relative_rope:
                # Relative RoPE uses local retained positions. Cache updates
                # depend on end-start, not the absolute chunk number.
                attention_layout = [
                    (window, sink, 0, global_end - start, local_end)
                    for window, sink, start, global_end, local_end in attention_layout
                ]
            self.exchange.configure_attention([layout for layout in attention_layout for _ in range(frames)])
            local = [self._local[key.session_id, key.latent_index] for key in self.last_local_tasks]
            if self.local_execution:
                for task in wave:
                    self._paged_states[task.session_id].prepare(
                        current_end=task.current_end,
                        window=task.window[0],
                        sink=task.window[1],
                        frames=frames,
                        context_length=task.prompt.shape[1],
                    )
            self.arena.bind(local)
            if isinstance(self.arena, PagedJobKVArena):
                self.arena.configure_attention()
            values = {name: [] for name in ("x", "timestep", "context", "act_context")}
            for key in self.last_local_tasks:
                task = self._requests[key.session_id]
                state = self._local[key.session_id, key.latent_index]
                timestep = (
                    torch.zeros_like(task.timesteps[:1])
                    if task.refreshing
                    else task.timesteps[task.step_index : task.step_index + 1]
                ).view(1, 1)
                if task.first_frame is not None and key.latent_index == 0 and not task.refreshing:
                    timestep = torch.zeros_like(timestep)
                values["x"].append(state.latent.to(self.pipeline.torch_dtype))
                values["timestep"].append(timestep)
                values["context"].append(task.prompt)
                values["act_context"].append(task.action[:, :, key.latent_index : key.latent_index + 1])
            if not local:
                values = {
                    "x": [anchor.latent[:, :, :1].to(self.pipeline.torch_dtype)],
                    "timestep": [anchor.timesteps[:1].view(1, 1)],
                    "context": [anchor.prompt],
                    "act_context": [anchor.action[:, :, :1]],
                }
            inputs = self.arena.inputs(values, len(local))
            stage = self.pipeline.denoise_stage
            model = stage.dit
            graph_key = (
                anchor.latent_batch_key(),
                local_anchor.refreshing,
                tuple(attention_layout),
                tuple(tuple(entry["is_init"] for entry in task.cross_cache) for task in wave),
                tuple(key.owner for key in self.last_tasks),
                row_order,
                len(local),
            )
            generation = (
                self._storage_epoch,
                anchor.full_shape[1:],
                anchor.prompt.shape[1:],
                anchor.timesteps.dtype,
                tuple((entry["k"].shape[1:], entry["k"].dtype) for entry in anchor.self_cache + anchor.cross_cache),
            )
            prediction = self.executor.run(
                model,
                key=graph_key,
                generation=generation,
                arena=self.arena,
                refreshing=local_anchor.refreshing,
                **inputs,
                kv_cache=self.arena.self_cache,
                crossattn_cache=self.arena.cross_cache,
                current_start=local_anchor.current_start * local_anchor.frame_tokens,
                cache_indices=(local_anchor.global_end, local_anchor.local_end),
                exchange=self.exchange,
            )
            padded_rows = self.exchange._padded_rows
            # The colocated path never invokes the per-layer collective, so
            # its communication accounting must stay zero as well.  Keep the
            # estimate only for an explicitly split dependency group.
            if not self.exchange.local_sibling_only:
                self.logical_kv_send_bytes += (
                    2
                    * model.num_layers
                    * padded_rows
                    * anchor.frame_tokens
                    * model.dim
                    * inputs["x"].element_size()
                    * (self.exchange.world_size - 1)
                )
            self.arena.scatter(local, context_length=anchor.prompt.shape[1])
            denoising = [task for task in wave if task.session_id not in refreshing]
            if denoising:
                denoise_local = [state for state in local if state.session_id not in refreshing]
                self._sample(denoising, denoise_local, prediction, inputs["timestep"])
            local_sessions = {key.session_id for key in self.last_local_tasks}
            for task in wave:
                delta = task.current_end - task.global_end
                capacity = task.self_cache[0]["k"].shape[1]
                capacity = min(capacity, task.window[0] * task.frame_tokens)
                evicted = max(0, task.local_end + frames * task.frame_tokens - capacity) if delta > 0 else 0
                task.local_end += delta - evicted
                task.global_end = task.current_end
                self._cursors[task.session_id] = task.global_end, task.local_end
                for entry in task.cross_cache:
                    entry["is_init"], entry["sequence_length"] = True, task.prompt.shape[1]
                for index in range(frames):
                    state = self._local.get((task.session_id, index))
                    if state is not None and task.session_id in local_sessions:
                        state.valid_tokens = task.local_end
            if not refreshing:
                return []
            # Clean KV is committed on the denoise owner before a successor
            # Job can be admitted. Only this chunk's tail crosses the pair.
            wave = [task for task in wave if task.session_id in refreshing]
            for task in wave:
                self._return_refresh_delta(task)
            output_local = [state for state in local if state.session_id in refreshing]
            output_indices = [index for index, key in enumerate(self.last_tasks) if key.refreshing]
            output_positions = {original: index for index, original in enumerate(output_indices)}
            self._output_exchange.configure(
                [owner for task in wave for owner in self.task_owners(task)],
                frames=frames,
                device=anchor.latent.device,
                row_order=[output_positions[index] for index in row_order if index in output_positions],
            )
            completed = torch.cat([state.latent for state in output_local]) if output_local else inputs["x"][:0]
            all_latents = self._output_exchange.gather_outputs(completed)
            padded_rows = self._output_exchange._padded_rows
            self.logical_output_send_bytes += (
                padded_rows * completed[0:1].numel() * completed.element_size() * (self.exchange.world_size - 1)
                if output_local
                else padded_rows
                * anchor.latent[:, :, :1].numel()
                * inputs["x"].element_size()
                * (self.exchange.world_size - 1)
            )
            latents = all_latents.reshape(len(wave), frames, *all_latents.shape[1:]).squeeze(3).permute(0, 2, 1, 3, 4)
            sessions = [self._sessions[task.session_id] for task in wave]
            decode_futures = [None] * len(wave)
            if self._decode_submitter is not None:
                # A row is the unit submitted to the service-owned TAeW
                # queue.  The queue batches rows across waves and serializes
                # updates to each session's streaming decoder state.
                if self.is_output_rank:
                    for row, session in enumerate(sessions):
                        decode_futures[row] = self._decode_submitter(latents[row : row + 1], session)
                decoded = latents.new_empty((len(wave), 3, 0, 0, 0))
                output_frames = 0
                postprocess_future = None
            else:
                decoded = (
                    self.pipeline.taew_decode_stage.decode_chunks(
                        latents, [session.taew_decode_state for session in sessions]
                    )
                    if self.is_output_rank
                    else latents.new_empty((len(wave), 3, 0, 0, 0))
                )
                postprocess_future = (
                    self._postprocess_submitter(decoded)
                    if self._postprocess_submitter is not None and self.is_output_rank
                    else None
                )
                count = torch.tensor(decoded.shape[2], device=latents.device, dtype=torch.long)
                self.exchange.broadcast_(count)
                output_frames = int(count.item())
            results = []
            for row, (task, session) in enumerate(zip(wave, sessions, strict=True)):
                if frames < 3:
                    self._fill_inactive_siblings(task)
                session.next_latent_frame += frames
                if self._decode_submitter is None:
                    session.emitted_frames += output_frames
                session.step_task_active = False
                task.complete = True
                del self._requests[task.session_id]
                results.append(
                    ABotWorldCompletedChunk(
                        task.session_id,
                        decoded[row : row + 1],
                        output_frames if self.is_output_rank else 0,
                        postprocess_future,
                        row,
                        decode_futures[row],
                    )
                )
            return results

    def _fill_inactive_siblings(self, task: ABotWorldStepTask) -> None:
        """Prime sibling replicas after an optional one/two-latent chunk.

        The usual three-latent path needs no historical state broadcast. A
        shorter chunk did not execute its absent siblings, so copy only valid
        state before those siblings can participate in a subsequent chunk.
        """
        if self.local_execution:
            return  # All siblings refer to the same logical page table.
        source_rank = self._owners[task.session_id][0]
        source = self._local.get((task.session_id, 0))
        for attribute, count in (("self_cache", task.local_end), ("cross_cache", task.prompt.shape[1])):
            for layer, entry in enumerate(getattr(self._sessions[task.session_id], attribute)):
                for name in ("k", "v"):
                    payload = entry[name].new_empty((1, count, *entry[name].shape[2:]))
                    if self.exchange.rank == source_rank:
                        payload.copy_(getattr(source, attribute)[layer][name][:, :count])
                    self.exchange.broadcast_(payload, source_rank)
                    for index in range(task.full_shape[2], 3):
                        target = self._local.get((task.session_id, index))
                        if target is not None:
                            target_entry = getattr(target, attribute)[layer]
                            target_entry[name][:, :count].copy_(payload)
                            if attribute == "cross_cache":
                                target_entry["is_init"], target_entry["sequence_length"] = True, count
                            else:
                                target.valid_tokens = task.local_end

    @torch.inference_mode()
    def release_session(self, session_id: str, *, materialize: bool = True) -> None:
        """Materialize canonical state only when returning ownership to the pipeline."""
        with self.pipeline._execution_lock:
            if session_id in self._requests:
                raise RuntimeError("Finish this session's latent-step work before releasing its state")
            session = self._sessions[session_id]
            paged = self._paged_states.pop(session_id, None)
            if paged is not None and materialize:
                global_end, local_end = self._cursors.get(session_id, (0, 0))
                for layer, entry in enumerate(session.self_cache):
                    for field, name in enumerate(("k", "v")):
                        value = entry[name]
                        dense = value.new_zeros((1, *value.shape[1:]))
                        if paged.table:
                            dense[:, :local_end].copy_(torch.cat(paged.layer_pages(layer, field)).unsqueeze(0))
                        entry[name] = dense
                    entry["global_end_index"].fill_(global_end)
                    entry["local_end_index"].fill_(local_end)
                for layer, entry in enumerate(session.cross_cache):
                    for field, name in enumerate(("k", "v")):
                        value = entry[name]
                        dense = value.new_zeros((1, *value.shape[1:]))
                        if paged.cross_page is not None:
                            dense[:, : paged.cross_length].copy_(paged.cross(layer, field))
                        entry[name] = dense
                    entry["is_init"], entry["sequence_length"] = paged.cross_ready[layer], paged.cross_length
            if paged is not None:
                paged.close()
                for cache in (session.self_cache, session.cross_cache):
                    for entry in cache:
                        entry.pop("paged_state", None)
            # Keep surviving slot identities until the next bind. Clearing the
            # cohort here would leave unselected survivors with untracked views
            # that a later wave could silently overwrite.
            if materialize:
                owner = self._owners.get(session_id, (0,))[0]
                local = self._local.get((session_id, 0))
                global_end, local_end = self._cursors.get(session_id, (0, 0))
                attributes = (
                    ()
                    if paged is not None
                    or (
                        self.exchange.world_size == 1
                        and session_id not in self._owners
                        and session.self_cache[0]["k"].shape[0] == 1
                    )
                    else ("self_cache", "cross_cache")
                )
                for attribute in attributes:
                    for layer, entry in enumerate(getattr(session, attribute)):
                        if attribute == "self_cache":
                            entry["global_end_index"].fill_(global_end)
                            entry["local_end_index"].fill_(local_end)
                        for name in ("k", "v"):
                            entry[name] = entry[name].new_zeros((1, *entry[name].shape[1:]))
                            if self.exchange.rank == owner and local is not None:
                                source = getattr(local, attribute)[layer]
                                valid = local.valid_tokens if attribute == "self_cache" else source["sequence_length"]
                                entry[name][:, :valid].copy_(source[name][:, :valid])
                            self.exchange.broadcast_(entry[name], owner)
            else:
                self.pipeline.close_interactive_session(session)
            for index in range(3):
                local = self._local.pop((session_id, index), None)
                if local is not None:
                    local.discarded = True
            self._owners.pop(session_id, None)
            self._refresh_owners.pop(session_id, None)
            self._refresh_cursors.pop(session_id, None)
            self._refresh_prepared.discard(session_id)
            self._retained_capacities.pop(session_id, None)
            self._canonical_capacities.pop(session_id, None)
            del self._sessions[session_id]
            self._cursors.pop(session_id, None)
            session.latent_runtime_owner = None

    def close(self, *, materialize: bool = True) -> None:
        if self.has_pending:
            raise RuntimeError("Finish latent-step work before closing the runtime")
        if materialize:
            for session_id in tuple(self._sessions):
                self.release_session(session_id)
        else:
            for state in self._paged_states.values():
                state.close()
            self._paged_states.clear()
            for session in self._sessions.values():
                self.pipeline.close_interactive_session(session)
                session.latent_runtime_owner = None
            self._sessions.clear()
            self._owners.clear()
            self._refresh_owners.clear()
            self._refresh_cursors.clear()
            self._refresh_prepared.clear()
            self._retained_capacities.clear()
            self._canonical_capacities.clear()
            self._local.clear()
            self._cursors.clear()
        self.executor.clear()
        self.kv_pages.trim_free()
        # Discarded tasks do not need independent KV allocations at teardown.
        self.arena.detach(keeping=self.arena.cohort)
        self.arena._storage.clear()
        self.arena._inputs.clear()
        self.exchange.clear()
        self._output_exchange.clear()
        if self.phase_transfer is not None:
            self.phase_transfer.close()
