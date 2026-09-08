#!/usr/bin/env python3
"""Build a minimal single-GPU offline profile for ABot-World.

The benchmark reuses the native interactive pipeline.  Every measured run starts
from the same image, prompt, action sequence, and seed, then emits the same fixed
number of LF=3 continuation blocks.  It records wall-clock and CUDA stage timing,
per-DiT-call CUDA timing, allocated GPU memory, generated videos, and lightweight
offline quality proxies relative to the four-step 480p baseline.

The quality numbers are regression/fidelity proxies, not semantic, action, VBench,
or WorldModelBench scores.  The manifest deliberately retains the inputs needed by
a future evaluator.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.utils.video import save_video

DEFAULT_PROMPT = "A smooth first-person exploration through a vivid natural landscape."
DEFAULT_ACTIONS: tuple[dict[str, bool], ...] = (
    {"W": True},
    {"W": True, "J": True},
    {"D": True},
)


@dataclass(frozen=True)
class ProfileConfig:
    """One session-static ABot fidelity setting."""

    name: str
    resolution: str
    height: int
    width: int
    denoise_step_positions: tuple[int, ...]
    precision: str = "bf16"

    @property
    def spatial_tokens_per_latent_frame(self) -> int:
        # Wan VAE spatial compression is 16 and ABot DiT spatial patching is 2.
        return (self.height // 32) * (self.width // 32)


PROFILES: dict[str, ProfileConfig] = {
    "high": ProfileConfig("high", "480p", 480, 832, (0, 1, 2, 3)),
    # Nominal heights 360 and 240 are invalid because both dimensions must be
    # divisible by 32.  These retain the baseline aspect ratio to within 1.2%.
    "medium": ProfileConfig("medium", "360p-compatible", 352, 608, (0, 1, 2, 3)),
    "low": ProfileConfig("low", "240p-compatible", 224, 384, (0, 1, 2, 3)),
    "steps2": ProfileConfig("steps2", "480p", 480, 832, (0, 3)),
}


@dataclass
class TimedBlock:
    config: str
    run: int
    block: int
    actions: str
    frames: int
    latency_ms: float
    dit_ms: float
    vae_decode_ms: float
    postprocess_ms: float
    denoise_step_ms: list[float]
    cache_commit_ms: float


class DiTCallTimer:
    """Measure each eager DiT forward with CUDA events and module hooks."""

    def __init__(self, dit: torch.nn.Module) -> None:
        self._pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._pending: list[torch.cuda.Event] = []
        self._pre_handle = dit.register_forward_pre_hook(self._before)
        self._post_handle = dit.register_forward_hook(self._after)

    def _before(self, _module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
        started = torch.cuda.Event(enable_timing=True)
        started.record()
        self._pending.append(started)

    def _after(self, _module: torch.nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
        if not self._pending:
            raise RuntimeError("DiT timing hook observed an unmatched forward completion")
        finished = torch.cuda.Event(enable_timing=True)
        finished.record()
        self._pairs.append((self._pending.pop(), finished))

    def begin_block(self) -> None:
        if self._pending:
            raise RuntimeError("DiT timing hook retained an incomplete forward")
        self._pairs.clear()

    def end_block(self) -> list[float]:
        if self._pending:
            raise RuntimeError("DiT timing hook retained an incomplete forward")
        return [started.elapsed_time(finished) for started, finished in self._pairs]

    def close(self) -> None:
        self._pre_handle.remove()
        self._post_handle.remove()


def _load_example_loader() -> Any:
    path = Path(__file__).resolve().parents[2] / "examples/abot_world/_loader.py"
    spec = importlib.util.spec_from_file_location("abot_offline_profile_loader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ABot example loader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_profiles(value: str) -> list[ProfileConfig]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names).difference(PROFILES))
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"profiles must be a nonempty comma-separated subset of {','.join(PROFILES)}; got {value!r}"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("profiles must not contain duplicates")
    return [PROFILES[name] for name in names]


def _parse_actions(value: str) -> tuple[dict[str, bool], ...]:
    """Parse a comma-separated sequence such as ``W,W+J,D``."""
    blocks: list[dict[str, bool]] = []
    valid = set(ABotWorldInteractivePipeline._ACTION_ORDER)
    for raw_block in value.split(","):
        keys = [key.strip().upper() for key in raw_block.split("+") if key.strip()]
        unknown = sorted(set(keys).difference(valid))
        if not keys or unknown:
            raise argparse.ArgumentTypeError(f"invalid action block {raw_block!r}; valid keys are {sorted(valid)}")
        blocks.append({key: True for key in keys})
    if not blocks:
        raise argparse.ArgumentTypeError("actions must contain at least one block")
    return tuple(blocks)


def _action_label(actions: Mapping[str, bool]) -> str:
    return "+".join(key for key in ABotWorldInteractivePipeline._ACTION_ORDER if actions.get(key))


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered),
        "p50": ordered[(len(ordered) - 1) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _configure_profile(
    pipeline: ABotWorldInteractivePipeline,
    profile: ProfileConfig,
    official_timesteps: Callable[[Any], torch.Tensor],
) -> list[float]:
    if profile.precision != "bf16":
        raise ValueError("This checkpoint loader only exposes the existing BF16 DiT path")
    pipeline.config.height = profile.height
    pipeline.config.width = profile.width
    pipeline.denoise_stage.configure_cuda_graph(False)

    def selected_timesteps(scheduler: Any) -> torch.Tensor:
        source = official_timesteps(scheduler)
        positions = torch.tensor(profile.denoise_step_positions, device=source.device, dtype=torch.long)
        return source.index_select(0, positions)

    pipeline.denoise_stage._official_denoising_timesteps = selected_timesteps
    scheduler = pipeline.denoise_stage._scheduler()
    return [float(item) for item in selected_timesteps(scheduler).cpu().tolist()]


def _cache_facts(session: Any, profile: ProfileConfig) -> dict[str, Any]:
    tensors = [
        value
        for cache_name in ("self_cache", "cross_cache")
        for layer in getattr(session, cache_name)
        for value in layer.values()
        if isinstance(value, torch.Tensor)
    ]
    self_k = session.self_cache[0]["k"]
    return {
        "spatial_tokens_per_latent_frame": profile.spatial_tokens_per_latent_frame,
        "temporal_window_latent_frames": int(session.self_cache[0]["k"].shape[1])
        // profile.spatial_tokens_per_latent_frame,
        "self_k_shape_per_layer": list(self_k.shape),
        "session_cache_bytes": int(sum(value.numel() * value.element_size() for value in tensors)),
    }


def _warm_up(
    pipeline: ABotWorldInteractivePipeline,
    timer: DiTCallTimer,
    image: Image.Image,
    prompt: str,
    actions: Mapping[str, bool],
    seed: int,
    profile: ProfileConfig,
) -> None:
    session = pipeline.create_interactive_session(
        image,
        prompt,
        seed=seed,
        session_id=f"offline-profile-{profile.name}-warmup",
    )
    try:
        timer.begin_block()
        frames = pipeline.generate_next_block(session, actions, control_latent_frames=3)
        torch.cuda.synchronize(pipeline.device)
        dit_call_ms = timer.end_block()
        if len(frames) != 12 or len(dit_call_ms) != len(profile.denoise_step_positions) + 1:
            raise RuntimeError(
                f"warm-up contract mismatch: {len(frames)} frames and {len(dit_call_ms)} DiT calls for {profile.name}"
            )
    finally:
        pipeline.close_interactive_session(session)
        del session
        gc.collect()
        torch.cuda.empty_cache()


def _run_once(
    pipeline: ABotWorldInteractivePipeline,
    timer: DiTCallTimer,
    image: Image.Image,
    prompt: str,
    actions: Sequence[Mapping[str, bool]],
    seed: int,
    profile: ProfileConfig,
    run_index: int,
) -> tuple[list[Image.Image], list[TimedBlock], dict[str, Any], float, dict[str, Any]]:
    device = torch.device(pipeline.device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    session = pipeline.create_interactive_session(
        image,
        prompt,
        seed=seed,
        session_id=f"offline-profile-{profile.name}-run-{run_index}",
    )
    cache_facts = _cache_facts(session, profile)
    create_metrics = dict(pipeline.last_stage_metrics())
    frames: list[Image.Image] = []
    blocks: list[TimedBlock] = []
    try:
        for block_index, block_actions in enumerate(actions, start=1):
            timer.begin_block()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            block_frames = pipeline.generate_next_block(session, block_actions, control_latent_frames=3)
            torch.cuda.synchronize(device)
            latency_ms = (time.perf_counter() - started) * 1000.0
            dit_call_ms = timer.end_block()
            metrics = dict(pipeline.last_stage_metrics())
            if len(block_frames) != 12:
                raise RuntimeError(f"{profile.name} emitted {len(block_frames)} frames instead of 12")
            if len(dit_call_ms) != len(profile.denoise_step_positions) + 1:
                raise RuntimeError(
                    f"{profile.name} emitted {len(dit_call_ms)} DiT calls instead of "
                    f"{len(profile.denoise_step_positions) + 1} (denoise steps plus cache commit)"
                )
            frames.extend(block_frames)
            blocks.append(
                TimedBlock(
                    config=profile.name,
                    run=run_index,
                    block=block_index,
                    actions=_action_label(block_actions),
                    frames=len(block_frames),
                    latency_ms=latency_ms,
                    dit_ms=float(metrics["denoise_seconds"]) * 1000.0,
                    vae_decode_ms=float(metrics["vae_decode_seconds"]) * 1000.0,
                    postprocess_ms=float(metrics["postprocess_seconds"]) * 1000.0,
                    denoise_step_ms=dit_call_ms[:-1],
                    cache_commit_ms=dit_call_ms[-1],
                )
            )
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    finally:
        pipeline.close_interactive_session(session)
        del session
        gc.collect()
        torch.cuda.empty_cache()
    return frames, blocks, create_metrics, float(peak_bytes), cache_facts


def _quality_size(size: tuple[int, int], max_side: int = 320) -> tuple[int, int]:
    width, height = size
    scale = min(1.0, max_side / max(width, height))
    return max(16, round(width * scale)), max(16, round(height * scale))


def _frames_to_tensor(frames: Sequence[Image.Image], size: tuple[int, int]) -> torch.Tensor:
    arrays = [
        np.asarray(frame.convert("RGB").resize(size, Image.Resampling.BICUBIC), dtype=np.float32) for frame in frames
    ]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).div_(255.0)


def _ssim(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Compute mean local RGB SSIM using a dependency-free 11x11 box window."""
    if reference.shape != candidate.shape or reference.ndim != 4:
        raise ValueError(f"SSIM tensors must have the same NCHW shape, got {reference.shape} and {candidate.shape}")
    kernel = 11 if min(reference.shape[-2:]) >= 11 else max(3, min(reference.shape[-2:]) // 2 * 2 - 1)
    mu_ref = F.avg_pool2d(reference, kernel, stride=1)
    mu_cand = F.avg_pool2d(candidate, kernel, stride=1)
    var_ref = F.avg_pool2d(reference.square(), kernel, stride=1) - mu_ref.square()
    var_cand = F.avg_pool2d(candidate.square(), kernel, stride=1) - mu_cand.square()
    covariance = F.avg_pool2d(reference * candidate, kernel, stride=1) - mu_ref * mu_cand
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_ref * mu_cand + c1) * (2 * covariance + c2)) / (
        (mu_ref.square() + mu_cand.square() + c1) * (var_ref + var_cand + c2)
    )
    return float(value.mean())


def _intrinsic_quality(frames: Sequence[Image.Image]) -> dict[str, float | str]:
    if len(frames) < 2:
        raise ValueError("quality metrics require at least two frames")
    size = _quality_size(frames[0].size)
    video = _frames_to_tensor(frames, size)
    temporal_l1 = float((video[1:] - video[:-1]).abs().mean())
    temporal_ssim = _ssim(video[:-1], video[1:])
    gray = 0.299 * video[:, 0] + 0.587 * video[:, 1] + 0.114 * video[:, 2]
    laplacian = -4 * gray[:, 1:-1, 1:-1]
    laplacian += gray[:, :-2, 1:-1] + gray[:, 2:, 1:-1]
    laplacian += gray[:, 1:-1, :-2] + gray[:, 1:-1, 2:]
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(np.asarray(frame.convert("RGB"), dtype=np.uint8).tobytes())
    return {
        "temporal_ssim": temporal_ssim,
        "temporal_l1": temporal_l1,
        "laplacian_variance": float(laplacian.var()),
        "pixel_sha256": digest.hexdigest(),
    }


def _reference_quality(reference: Sequence[Image.Image], candidate: Sequence[Image.Image]) -> dict[str, float]:
    count = min(len(reference), len(candidate))
    if count < 1:
        raise ValueError("reference comparison requires nonempty videos")
    size = _quality_size(reference[0].size)
    ref = _frames_to_tensor(reference[:count], size)
    cand = _frames_to_tensor(candidate[:count], size)
    mse = float((ref - cand).square().mean())
    return {
        "reference_psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        "reference_ssim": _ssim(ref, cand),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows:
        return
    fields = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pareto_configs(profile_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    frontier: list[str] = []
    for candidate in profile_rows:
        dominated = any(
            float(other["latency_ms"]) <= float(candidate["latency_ms"])
            and float(other["quality_score"]) >= float(candidate["quality_score"])
            and (
                float(other["latency_ms"]) < float(candidate["latency_ms"])
                or float(other["quality_score"]) > float(candidate["quality_score"])
            )
            for other in profile_rows
            if other is not candidate
        )
        if not dominated:
            frontier.append(str(candidate["config"]))
    return frontier


def _write_analysis(
    output: Path,
    profile_rows: Sequence[Mapping[str, Any]],
    cache_by_config: Mapping[str, Mapping[str, Any]],
    reference_name: str,
) -> None:
    by_name = {str(row["config"]): row for row in profile_rows}
    baseline = by_name.get("high", profile_rows[0])
    resolution_rows = [row for row in profile_rows if int(row["denoise_steps"]) == 4]
    lines = [
        "# ABot-World minimal offline profile",
        "",
        "All timings are eager, singleton, LF=3 generation blocks on one GPU. `latency_ms` is the mean "
        "wall time per 12-frame block; FPS is generated frames divided by generation wall time.",
        "",
        "| Config | Actual size | Steps | Latency (ms) | FPS | Peak allocated GPU GiB | Reference SSIM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in profile_rows:
        lines.append(
            f"| {row['config']} | {row['height']}x{row['width']} | {row['denoise_steps']} | "
            f"{float(row['latency_ms']):.2f} | {float(row['fps']):.2f} | "
            f"{float(row['gpu_memory_gib']):.3f} | {float(row['quality_score']):.4f} |"
        )
    lines.extend(["", "## Answers", ""])
    reductions = []
    for row in resolution_rows:
        if row is baseline:
            continue
        reduction = (1.0 - float(row["latency_ms"]) / float(baseline["latency_ms"])) * 100.0
        reductions.append(f"{row['config']} {reduction:.1f}% lower")
    lines.append(
        "1. **Does lower resolution significantly reduce latency?** "
        + (
            "Measured block-latency changes versus high: " + ", ".join(reductions) + "."
            if reductions
            else "Not measured."
        )
    )
    lines.append(
        "2. **Does it affect KV/cache or temporal state?** Yes for tensor shape and bytes: the self-attention KV "
        "capacity is `18 × spatial_tokens_per_frame`, so resolution changes every layer's K/V shape. The temporal "
        "window remains 18 latent frames, and counters/action semantics do not change. A retained session cannot "
        "hot-switch resolution because its KV and streaming decoder state have the old spatial shape."
    )
    frontier = _pareto_configs(profile_rows)
    lines.append(
        f"3. **Is there a latency-quality Pareto frontier?** Under the baseline-fidelity SSIM proxy "
        f"(reference `{reference_name}`), the non-dominated configurations are: {', '.join(frontier)}. "
        "This small proxy cannot establish semantic or action quality; inspect the saved videos before a serving "
        "choice."
    )
    lines.append(
        "4. **Should resolution be a fidelity dimension?** Yes as a session-static/admission-time dimension: the "
        "pipeline and existing TAeW decoder run at all measured divisible shapes without model-structure changes. "
        "It is not currently safe as a per-block hot-switch dimension; mixed-resolution sessions also cannot share "
        "a batch."
    )
    lines.extend(
        [
            "",
            "## Compatibility and quality scope",
            "",
            "- No VAE decode modification was made. The existing dynamic TAeW streaming decoder was reused.",
            "- Nominal 360p/240p are represented by 352x608 and 224x384 because both dimensions must be "
            "divisible by 32.",
            "- `precision=bf16` describes DiT/T5. Existing loading keeps image VAE/TAeW in FP32. No ABot model-FP8 "
            "checkpoint/path exists; the optional Sage kernel's FP8 PV arithmetic is not a model precision profile.",
            "- Reference PSNR/SSIM is frame-aligned regression similarity after bicubic resizing to the baseline size. "
            "Temporal SSIM, temporal L1, and Laplacian variance are diagnostic proxies, not VBench/WorldModelBench.",
            "",
            "## Cache facts",
            "",
        ]
    )
    for name, facts in cache_by_config.items():
        lines.append(
            f"- `{name}`: {facts['spatial_tokens_per_latent_frame']} tokens/latent-frame, "
            f"K per layer `{facts['self_k_shape_per_layer']}`, "
            f"session K/V tensors {int(facts['session_cache_bytes']) / 2**30:.3f} GiB."
        )
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_model = project_root.parent / "model_zoo/ABot-World-0-5B-LF"
    default_image = project_root.parent / "ABot-World/web_client/datasets/images/84b90ad568b693d2.png"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=default_model)
    parser.add_argument("--image", type=Path, default=default_image)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profiles", type=_parse_profiles, default=list(PROFILES.values()))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--actions", type=_parse_actions, default=DEFAULT_ACTIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    videos_dir = output / "videos"
    videos_dir.mkdir()
    plan = {
        "script": Path(__file__).name,
        "model_root": str(args.model_root.resolve()),
        "initial_image": str(args.image.resolve()),
        "prompt": args.prompt,
        "seed": args.seed,
        "actions": [_action_label(item) for item in args.actions],
        "runs_per_config": args.runs,
        "control_latent_frames": 3,
        "frames_per_block": 12,
        "blocks_per_rollout": len(args.actions),
        "playback_fps": args.fps,
        "device_id": args.device_id,
        "profiles": [asdict(profile) for profile in args.profiles],
        "precision_components": {"dit": "bf16", "text_encoder": "bf16", "image_vae": "fp32", "taew": "fp32"},
        "cuda_graph_enabled": False,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This profile requires CUDA")
    if args.device_id >= torch.cuda.device_count():
        raise ValueError(f"device-id {args.device_id} is outside visible CUDA devices [0,{torch.cuda.device_count()})")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    os.environ["TELEFUSER_ABOT_CUDA_GRAPH_ENABLED"] = "0"
    torch.cuda.set_device(args.device_id)
    loader = _load_example_loader()
    image = Image.open(args.image).convert("RGB")
    first_profile = args.profiles[0]
    print(f"loading ABot pipeline on cuda:{args.device_id}", flush=True)
    pipeline = loader.get_pipeline(
        model_root=args.model_root,
        height=first_profile.height,
        width=first_profile.width,
        device_id=args.device_id,
        pipeline_class=ABotWorldInteractivePipeline,
    )
    official_timesteps = pipeline.denoise_stage._official_denoising_timesteps
    timer = DiTCallTimer(pipeline.denoise_stage.dit)

    all_blocks: list[TimedBlock] = []
    run_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    cache_by_config: dict[str, dict[str, Any]] = {}
    reference_frames: list[Image.Image] | None = None
    reference_name = first_profile.name
    try:
        for profile_index, profile in enumerate(args.profiles):
            selected_timesteps = _configure_profile(pipeline, profile, official_timesteps)
            print(
                f"profile={profile.name} size={profile.height}x{profile.width} "
                f"steps={len(profile.denoise_step_positions)} warmup",
                flush=True,
            )
            _warm_up(pipeline, timer, image, args.prompt, args.actions[0], args.seed + 10_000, profile)
            for run_index in range(1, args.runs + 1):
                print(f"profile={profile.name} run={run_index}/{args.runs}", flush=True)
                frames, blocks, create_metrics, peak_bytes, cache_facts = _run_once(
                    pipeline,
                    timer,
                    image,
                    args.prompt,
                    args.actions,
                    args.seed,
                    profile,
                    run_index,
                )
                cache_by_config[profile.name] = cache_facts
                video_path = (videos_dir / f"{profile.name}_run{run_index}.mp4").resolve()
                save_video(frames, str(video_path), fps=args.fps, quality=9)
                if profile_index == 0 and run_index == 1:
                    reference_frames = [frame.copy() for frame in frames]
                assert reference_frames is not None
                quality = _intrinsic_quality(frames) | _reference_quality(reference_frames, frames)
                block_latencies = [block.latency_ms for block in blocks]
                rollout_ms = sum(block_latencies)
                flat_steps = [value for block in blocks for value in block.denoise_step_ms]
                row = {
                    "config": profile.name,
                    "run": run_index,
                    "resolution": profile.resolution,
                    "height": profile.height,
                    "width": profile.width,
                    "denoise_steps": len(profile.denoise_step_positions),
                    "selected_timesteps": json.dumps(selected_timesteps),
                    "precision": profile.precision,
                    "latency_ms": statistics.fmean(block_latencies),
                    "rollout_latency_ms": rollout_ms,
                    "fps": len(frames) / (rollout_ms / 1000.0),
                    "dit_ms": statistics.fmean(block.dit_ms for block in blocks),
                    "denoise_step_ms": statistics.fmean(flat_steps),
                    "cache_commit_ms": statistics.fmean(block.cache_commit_ms for block in blocks),
                    "vae_decode_ms": statistics.fmean(block.vae_decode_ms for block in blocks),
                    "postprocess_ms": statistics.fmean(block.postprocess_ms for block in blocks),
                    "vae_encode_ms": float(create_metrics["vae_encode_seconds"]) * 1000.0,
                    "text_encode_ms": float(create_metrics["text_encode_seconds"]) * 1000.0,
                    "gpu_memory_bytes": int(peak_bytes),
                    "gpu_memory_gib": peak_bytes / 2**30,
                    "video_path": str(video_path),
                    **quality,
                }
                run_rows.append(row)
                quality_rows.append(
                    {
                        "video_path": str(video_path),
                        "config_name": profile.name,
                        "prompt": args.prompt,
                        "initial_image": str(args.image.resolve()),
                        "actions": json.dumps([_action_label(item) for item in args.actions]),
                        "seed": args.seed,
                        "reference_config": reference_name,
                        "reference_video_path": str((videos_dir / f"{reference_name}_run1.mp4").resolve()),
                        **quality,
                    }
                )
                manifest_rows.append(
                    {
                        "video_path": str(video_path),
                        "config_name": profile.name,
                        "prompt": args.prompt,
                        "initial_image": str(args.image.resolve()),
                        "actions": json.dumps([_action_label(item) for item in args.actions]),
                        "seed": args.seed,
                        "run": run_index,
                    }
                )
                all_blocks.extend(blocks)
    finally:
        timer.close()
        pipeline.denoise_stage._official_denoising_timesteps = official_timesteps
        pipeline.close()
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    block_rows: list[dict[str, Any]] = []
    for block in all_blocks:
        block_rows.append(
            {
                **{key: value for key, value in asdict(block).items() if key != "denoise_step_ms"},
                "denoise_step_ms": json.dumps(block.denoise_step_ms),
                "denoise_step_mean_ms": statistics.fmean(block.denoise_step_ms),
            }
        )
    _write_csv(output / "blocks.csv", block_rows)
    _write_csv(output / "runs.csv", run_rows)
    _write_csv(output / "quality.csv", quality_rows)
    _write_csv(output / "quality_manifest.csv", manifest_rows)

    profile_rows: list[dict[str, Any]] = []
    for profile in args.profiles:
        rows = [row for row in run_rows if row["config"] == profile.name]
        latencies = [float(row["latency_ms"]) for row in rows]
        fps_values = [float(row["fps"]) for row in rows]
        memory_values = [float(row["gpu_memory_gib"]) for row in rows]
        reference_ssim = [float(row["reference_ssim"]) for row in rows]
        deterministic = len({str(row["pixel_sha256"]) for row in rows}) == 1
        latency_summary = _summary(latencies)
        representative = rows[0]
        profile_rows.append(
            {
                "config": profile.name,
                "resolution": profile.resolution,
                "height": profile.height,
                "width": profile.width,
                "denoise_steps": len(profile.denoise_step_positions),
                "precision": profile.precision,
                "latency_ms": latency_summary["mean"],
                "latency_std_ms": latency_summary["std"],
                "latency_p95_ms": latency_summary["p95"],
                "fps": statistics.fmean(fps_values),
                "dit_ms": statistics.fmean(float(row["dit_ms"]) for row in rows),
                "denoise_step_ms": statistics.fmean(float(row["denoise_step_ms"]) for row in rows),
                "cache_commit_ms": statistics.fmean(float(row["cache_commit_ms"]) for row in rows),
                "vae_decode_ms": statistics.fmean(float(row["vae_decode_ms"]) for row in rows),
                "gpu_memory": f"{max(memory_values):.3f} GiB",
                "gpu_memory_gib": max(memory_values),
                "quality_score": statistics.fmean(reference_ssim),
                "reference_psnr_db": statistics.fmean(float(row["reference_psnr_db"]) for row in rows),
                "temporal_ssim": statistics.fmean(float(row["temporal_ssim"]) for row in rows),
                "laplacian_variance": statistics.fmean(float(row["laplacian_variance"]) for row in rows),
                "deterministic_across_runs": deterministic,
                "video_path": representative["video_path"],
            }
        )
    required_fields = [
        "config",
        "resolution",
        "denoise_steps",
        "precision",
        "latency_ms",
        "fps",
        "gpu_memory",
        "video_path",
    ]
    remaining_fields = [key for key in profile_rows[0] if key not in required_fields]
    _write_csv(output / "profile.csv", profile_rows, required_fields + remaining_fields)
    _write_analysis(output, profile_rows, cache_by_config, reference_name)
    result = {
        "output_dir": str(output),
        "profile_csv": str(output / "profile.csv"),
        "analysis": str(output / "analysis.md"),
        "configs": [profile.name for profile in args.profiles],
    }
    (output / "results.json").write_text(
        json.dumps({"plan": plan, "profiles": profile_rows, "runs": run_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
