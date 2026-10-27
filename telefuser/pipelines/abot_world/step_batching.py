"""Resumable DiT steps and iteration-level batching for ABot-World.

A session's current latent group advances together: attention exchanges its
current K/V at every layer. Different sessions may use different diffusion
steps. Scheduling occurs between complete DiT forwards, not between kernels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .denoising import ABotWorldDenoisingStage


@dataclass(eq=False)
class ABotWorldStepTask:
    """One session chunk whose sampling state survives scheduling boundaries."""

    session_id: str
    latent: torch.Tensor
    prompt: torch.Tensor
    action: torch.Tensor
    first_frame: torch.Tensor | None
    self_cache: list[dict[str, Any]]
    cross_cache: list[dict[str, Any]]
    generator: torch.Generator
    scheduler: Any
    timesteps: torch.Tensor
    full_shape: tuple[int, ...]
    current_start: int
    frame_tokens: int
    global_end: int
    local_end: int
    window: tuple[int, int]
    relative_rope: bool
    sharded: bool
    step_index: int = 0
    complete: bool = False
    sampling_sigmas: tuple[torch.Tensor, ...] = ()

    @property
    def refreshing(self) -> bool:
        return self.step_index == len(self.timesteps)

    @property
    def current_end(self) -> int:
        return (self.current_start + self.full_shape[2]) * self.frame_tokens

    def batch_key(self) -> tuple[Any, ...]:
        """Bucket by execution shape/cache layout, allowing different timesteps."""
        return (
            self.full_shape[1:],
            self.latent.dtype,
            self.latent.device,
            self.prompt.shape[1:],
            self.window,
            self.local_end,
            self.current_end - self.global_end,
            self.current_start if not self.relative_rope else None,
            self.refreshing,
            self.sharded,
            tuple((entry["is_init"], entry["sequence_length"]) for entry in self.cross_cache),
        )

    def latent_batch_key(self) -> tuple[Any, ...]:
        """Share projections/FFN while attention handles per-row cache layouts."""
        return (
            self.full_shape[1:],
            self.latent.dtype,
            self.latent.device,
            self.prompt.shape[1:],
            self.relative_rope,
            self.refreshing,
            self.sharded,
        )

    def attention_key(self) -> tuple[int, ...]:
        """Keep equal attention layouts contiguous without padding their windows."""
        return (*self.window, self.current_start * self.frame_tokens, self.global_end, self.local_end)


class ABotWorldStepBatcher:
    """Keep a bounded active cohort and re-form batches at DiT step boundaries.

    New active chunks first execute their cache-appending step, then can join
    compatible continuations already at later steps. Stable cohorts keep their
    collated KV allocation; membership changes require dense KV copies. This
    preserves the existing attention backend and is not a paged-KV kernel.
    """

    def __init__(self, stage: ABotWorldDenoisingStage, *, max_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.stage = stage
        self.max_batch_size = max_batch_size
        self._waiting: deque[ABotWorldStepTask] = deque()
        self._active: list[ABotWorldStepTask] = []
        self._session_ids: set[str] = set()
        self._cohort: tuple[ABotWorldStepTask, ...] = ()
        self._self_cache: list[dict[str, Any]] = []
        self._cross_cache: list[dict[str, Any]] = []
        self.last_batch_size = 0
        self.last_step_indices: tuple[int, ...] = ()

    @property
    def has_pending(self) -> bool:
        return bool(self._waiting or self._active)

    def begin(
        self,
        session_id: str,
        latent: torch.Tensor,
        prompt: torch.Tensor,
        action: torch.Tensor,
        first_frame: torch.Tensor | None,
        self_cache: list[dict[str, Any]],
        cross_cache: list[dict[str, Any]],
        current_start: int,
        generator: torch.Generator,
        scheduler: Any,
        *,
        timestep_positions: tuple[int, ...] | None = None,
        shard_latents: bool = True,
        cache_indices: tuple[int, int] | None = None,
    ) -> ABotWorldStepTask:
        """Capture one chunk without running a DiT forward or advancing RNG."""
        if latent.ndim != 5 or latent.shape[0] != 1:
            raise ValueError("A step task requires one session's [1,C,F,H,W] latent")
        runtime = self.stage.latent_parallel_runtime
        shape = tuple(latent.shape)
        sharded = shard_latents and runtime is not None and shape[2] > 1
        timesteps = self.stage._official_denoising_timesteps(scheduler)
        if timestep_positions is not None:
            positions = tuple(timestep_positions)
            if (
                not positions
                or positions != tuple(sorted(set(positions)))
                or not 0 <= positions[0] <= positions[-1] < 4
            ):
                raise ValueError("timestep_positions must be increasing indices in [0, 3]")
            timesteps = timesteps[list(positions)]
        timesteps = timesteps.to(self.stage.device)
        first = first_frame if current_start == 0 else None
        if first is not None:
            latent = latent.clone()
            latent[:, :, :1].copy_(first)
        if sharded:
            latent, action = runtime.frame_slice(latent), runtime.frame_slice(action)
            if runtime.rank != 0:
                first = None
        model = self.stage.dit
        if cache_indices is None:
            cache_indices = (
                int(self_cache[0]["global_end_index"].item()),
                int(self_cache[0]["local_end_index"].item()),
            )
        task = ABotWorldStepTask(
            session_id,
            latent,
            prompt,
            action,
            first,
            self_cache,
            cross_cache,
            generator,
            scheduler,
            timesteps,
            shape,
            current_start,
            (shape[-2] // model.patch_size[1]) * (shape[-1] // model.patch_size[2]),
            *cache_indices,
            (model.local_attn_size, model.sink_size),
            model.use_relative_rope,
            sharded,
        )
        official = (0, 250, 500, 750)
        task.sampling_sigmas = tuple(scheduler.sigmas[official[index]] for index in (timestep_positions or range(4)))
        return task

    def submit(self, task: ABotWorldStepTask) -> None:
        """Admit at most one causally ready chunk per session key."""
        if task.complete or task.session_id in self._session_ids:
            raise ValueError("Session already has pending step work, or task is complete")
        self._session_ids.add(task.session_id)
        self._waiting.append(task)

    @torch.inference_mode()
    def step(self) -> list[ABotWorldStepTask]:
        """Execute one physical DiT batch and return chunks whose KV is final."""
        while self._waiting and len(self._active) < self.max_batch_size:
            self._active.append(self._waiting.popleft())
        if not self._active:
            self.last_batch_size, self.last_step_indices = 0, ()
            return []
        anchor = next((task for task in self._active if task.step_index == 0), self._active[0])
        key = anchor.batch_key()
        tasks = [task for task in self._active if task.batch_key() == key]
        self.last_batch_size = len(tasks)
        self.last_step_indices = tuple(task.step_index for task in tasks)
        self._advance(tasks)
        completed = [task for task in tasks if task.complete]
        for task in completed:
            self._active.remove(task)
            self._session_ids.remove(task.session_id)
        if not self.has_pending:
            self._release_cohort()
        return completed

    def _release_cohort(self, keeping: tuple[ABotWorldStepTask, ...] = ()) -> None:
        # Detach departing rows so an inactive session cannot retain an entire
        # old batch allocation. Stable active cohorts avoid this full-KV copy.
        if len(self._cohort) > 1:
            for task in self._cohort:
                if task not in keeping:
                    for cache in (task.self_cache, task.cross_cache):
                        for entry in cache:
                            for name in ("k", "v"):
                                entry[name] = entry[name].clone()
        self._cohort, self._self_cache, self._cross_cache = (), [], []

    def _prepare_caches(self, tasks: list[ABotWorldStepTask]) -> None:
        cohort = tuple(tasks)
        if cohort == self._cohort:
            return
        self._release_cohort(keeping=cohort)
        self._cohort = cohort
        for attribute, destination in (("self_cache", self._self_cache), ("cross_cache", self._cross_cache)):
            for index, reference in enumerate(getattr(tasks[0], attribute)):
                entry = dict(reference)
                if len(tasks) > 1:
                    for name in ("k", "v"):
                        entry[name] = torch.cat([getattr(task, attribute)[index][name] for task in tasks])
                    for name, value in reference.items():
                        if isinstance(value, torch.Tensor) and name not in {"k", "v"}:
                            entry[name] = value.clone()
                destination.append(entry)

    def _advance(self, tasks: list[ABotWorldStepTask]) -> None:
        self._prepare_caches(tasks)
        anchor = tasks[0]
        model = self.stage.dit
        if (model.local_attn_size, model.sink_size) != anchor.window:
            model.set_causal_attention_window(*anchor.window)
        timesteps = torch.cat(
            [
                (
                    torch.zeros_like(task.timesteps[:1])
                    if task.refreshing
                    else task.timesteps[task.step_index : task.step_index + 1]
                )
                .view(1, 1)
                .expand(1, task.latent.shape[2])
                .clone()
                for task in tasks
            ]
        )
        for row, task in enumerate(tasks):
            if task.first_frame is not None and not task.refreshing:
                timesteps[row, 0] = 0
        current = torch.cat([task.latent for task in tasks])
        with torch.autocast(
            self.stage.device.type,
            dtype=self.stage.torch_dtype,
            enabled=self.stage.device.type == "cuda" and not anchor.refreshing,
        ):
            prediction = model(
                x=current.to(self.stage.torch_dtype),
                timestep=timesteps,
                context=torch.cat([task.prompt for task in tasks]),
                act_context=torch.cat([task.action for task in tasks]),
                kv_cache=self._self_cache,
                crossattn_cache=self._cross_cache,
                current_start=anchor.current_start * anchor.frame_tokens,
                latent_parallel=anchor.sharded,
                cache_indices=(anchor.global_end, anchor.local_end),
            )
        if anchor.refreshing:
            runtime = self.stage.latent_parallel_runtime
            output = runtime.gather_frames(current) if anchor.sharded else current
            for row, task in enumerate(tasks):
                task.latent = output[row : row + 1]
                task.complete = True
        else:
            for row, task in enumerate(tasks):
                value = self.stage._x0_prediction(
                    prediction[row : row + 1], task.latent, timesteps[row : row + 1], task.scheduler
                )
                task.step_index += 1
                if not task.refreshing:
                    noise = torch.randn(
                        task.full_shape, generator=task.generator, dtype=value.dtype, device=value.device
                    )
                    if task.sharded:
                        noise = self.stage.latent_parallel_runtime.frame_slice(noise)
                    value = task.scheduler.add_noise(value, noise, task.timesteps[task.step_index])
                if task.first_frame is not None:
                    value[:, :, :1].copy_(task.first_frame)
                task.latent = value
        for row, task in enumerate(tasks):
            delta = task.current_end - task.global_end
            tokens = task.full_shape[2] * task.frame_tokens
            capacity = task.self_cache[0]["k"].shape[1]
            evicted = max(0, task.local_end + tokens - capacity) if delta > 0 else 0
            task.local_end += delta - evicted
            task.global_end = task.current_end
            for attribute, batched in (("self_cache", self._self_cache), ("cross_cache", self._cross_cache)):
                for index, entry in enumerate(batched):
                    target = getattr(task, attribute)[index]
                    for name, value in entry.items():
                        if name in {"k", "v"}:
                            target[name] = value[row : row + 1]
                        elif attribute == "self_cache" and name in {"global_end_index", "local_end_index"}:
                            target[name].fill_(task.global_end if name == "global_end_index" else task.local_end)
                        else:
                            target[name] = value
