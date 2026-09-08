#!/usr/bin/env python3
"""Profile ABot-World latency and offline quality at fixed latent/KV shape.

The experiment uses the native interactive pipeline and varies only native
batch size and the number of selected denoising calls.  Resolution, KV window,
attention implementation, checkpoint precision, prompt, initial image, action
sequence, and RNG seed stay fixed.

Action scores are deliberately absent unless an external evaluator or reviewer
supplies them.  The repository does not ship an ABot action evaluator, and a
baseline-similarity score is not evidence that an action was executed correctly.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.models import vgg16

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.utils.video import save_video

DEFAULT_PROMPT = "A smooth first-person exploration through a vivid natural landscape."
DEFAULT_ACTIONS: tuple[dict[str, bool], ...] = (
    {"W": True},
    {"W": True, "J": True},
    {"D": True},
)
HEIGHT = 480
WIDTH = 832
KV_WINDOW = 18
SINK_FRAMES = 6
FRAMES_PER_BLOCK = 12
CONTROL_LATENT_FRAMES = 3
VGG_CHECKPOINT = "vgg16-397923af.pth"


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    denoise_step_positions: tuple[int, ...]

    @property
    def denoise_steps(self) -> int:
        return len(self.denoise_step_positions)

    @property
    def name(self) -> str:
        return f"b{self.batch_size}_s{self.denoise_steps}_w{KV_WINDOW}_dense_bf16"


CONFIGS = (
    ExperimentConfig(1, (0, 1, 2, 3)),
    ExperimentConfig(2, (0, 1, 2, 3)),
    ExperimentConfig(1, (0, 3)),
    ExperimentConfig(2, (0, 3)),
)


def _load_example_loader() -> Any:
    path = Path(__file__).resolve().parents[2] / "examples/abot_world/_loader.py"
    spec = importlib.util.spec_from_file_location("abot_quality_profile_loader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ABot example loader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_positive_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _parse_steps(value: str) -> list[tuple[int, ...]]:
    allowed = {4: (0, 1, 2, 3), 2: (0, 3)}
    values = _parse_positive_ints(value)
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(f"supported denoise step counts are {sorted(allowed)}, got {unknown}")
    return [allowed[item] for item in values]


def _parse_actions(value: str) -> tuple[dict[str, bool], ...]:
    valid = set(ABotWorldInteractivePipeline._ACTION_ORDER)
    blocks: list[dict[str, bool]] = []
    for raw_block in value.split(","):
        keys = [key.strip().upper() for key in raw_block.split("+") if key.strip()]
        unknown = sorted(set(keys).difference(valid))
        if not keys or unknown:
            raise argparse.ArgumentTypeError(f"invalid action block {raw_block!r}; valid keys are {sorted(valid)}")
        blocks.append({key: True for key in keys})
    return tuple(blocks)


def _action_label(actions: Mapping[str, bool]) -> str:
    return "+".join(key for key in ABotWorldInteractivePipeline._ACTION_ORDER if actions.get(key))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    fields = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _configure_steps(
    pipeline: ABotWorldInteractivePipeline,
    positions: tuple[int, ...],
    official_timesteps: Callable[[Any], torch.Tensor],
) -> list[float]:
    pipeline.config.height = HEIGHT
    pipeline.config.width = WIDTH
    pipeline.config.local_attn_size = KV_WINDOW
    pipeline.config.sink_size = SINK_FRAMES
    pipeline.denoise_stage.configure_cuda_graph(False)
    pipeline.denoise_stage.dit.set_causal_attention_window(KV_WINDOW, SINK_FRAMES)

    def selected_timesteps(scheduler: Any) -> torch.Tensor:
        source = official_timesteps(scheduler)
        indices = torch.tensor(positions, device=source.device, dtype=torch.long)
        return source.index_select(0, indices)

    pipeline.denoise_stage._official_denoising_timesteps = selected_timesteps
    return [float(item) for item in selected_timesteps(pipeline.denoise_stage._scheduler()).cpu().tolist()]


def _create_sessions(
    pipeline: ABotWorldInteractivePipeline,
    image: Image.Image,
    prompt: str,
    seed: int,
    config: ExperimentConfig,
    run: int,
    tag: str,
) -> list[Any]:
    # Every lane deliberately receives the same seed.  This isolates numerical
    # batching effects instead of conflating B with a different diffusion path.
    return [
        pipeline.create_interactive_session(
            image,
            prompt,
            seed=seed,
            session_id=f"quality-{config.name}-{tag}-r{run}-lane{lane}",
        )
        for lane in range(config.batch_size)
    ]


def _close_sessions(pipeline: ABotWorldInteractivePipeline, sessions: Sequence[Any]) -> None:
    for session in sessions:
        pipeline.close_interactive_session(session)
    gc.collect()
    torch.cuda.empty_cache()


def _warm_up(
    pipeline: ABotWorldInteractivePipeline,
    image: Image.Image,
    prompt: str,
    seed: int,
    actions: Mapping[str, bool],
    config: ExperimentConfig,
) -> None:
    sessions = _create_sessions(pipeline, image, prompt, seed + 10_000, config, 0, "warmup")
    try:
        outputs = pipeline.generate_next_blocks(
            sessions,
            [actions for _ in sessions],
            control_latent_frames=CONTROL_LATENT_FRAMES,
        )
        torch.cuda.synchronize(pipeline.device)
        if any(len(frames) != FRAMES_PER_BLOCK for frames in outputs):
            raise RuntimeError(f"warm-up emitted unexpected frame counts: {[len(frames) for frames in outputs]}")
    finally:
        _close_sessions(pipeline, sessions)


def _run_once(
    pipeline: ABotWorldInteractivePipeline,
    image: Image.Image,
    prompt: str,
    seed: int,
    actions: Sequence[Mapping[str, bool]],
    config: ExperimentConfig,
    run: int,
    selected_timesteps: Sequence[float],
    videos_dir: Path,
    fps: float,
) -> list[dict[str, Any]]:
    device = torch.device(pipeline.device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    sessions = _create_sessions(pipeline, image, prompt, seed, config, run, "measured")
    all_frames: list[list[Image.Image]] = [[] for _ in sessions]
    block_latencies: list[float] = []
    stage_metrics: list[dict[str, Any]] = []
    try:
        for block_actions in actions:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            outputs = pipeline.generate_next_blocks(
                sessions,
                [block_actions for _ in sessions],
                control_latent_frames=CONTROL_LATENT_FRAMES,
            )
            torch.cuda.synchronize(device)
            block_latencies.append((time.perf_counter() - started) * 1000.0)
            stage_metrics.append(dict(pipeline.last_stage_metrics()))
            if any(len(frames) != FRAMES_PER_BLOCK for frames in outputs):
                raise RuntimeError(
                    f"measured run emitted unexpected frame counts: {[len(frames) for frames in outputs]}"
                )
            for lane, frames in enumerate(outputs):
                all_frames[lane].extend(frames)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        rollout_latency_ms = sum(block_latencies)
        rows = []
        for lane, frames in enumerate(all_frames):
            video_path = (videos_dir / f"{config.name}_run{run}_session{lane}.mp4").resolve()
            save_video(frames, str(video_path), fps=fps, quality=9)
            rows.append(
                {
                    "config": config.name,
                    "B": config.batch_size,
                    "S": config.denoise_steps,
                    "W": KV_WINDOW,
                    "rho": "dense",
                    "Q": "bf16",
                    "run": run,
                    "session": lane,
                    "seed": seed,
                    "actions": json.dumps([_action_label(item) for item in actions]),
                    "selected_timesteps": json.dumps(selected_timesteps),
                    "height": HEIGHT,
                    "width": WIDTH,
                    "latent_shape": "[1, 48, 3, 30, 52]",
                    "kv_shape_per_layer": "[1, 7020, 24, 128]",
                    "latency_ms": statistics.fmean(block_latencies),
                    "latency_p95_ms": max(block_latencies),
                    "rollout_latency_ms": rollout_latency_ms,
                    "FPS": config.batch_size * FRAMES_PER_BLOCK / (statistics.fmean(block_latencies) / 1000.0),
                    "per_user_FPS": FRAMES_PER_BLOCK / (statistics.fmean(block_latencies) / 1000.0),
                    "memory_bytes": peak_bytes,
                    "memory_gib": peak_bytes / 2**30,
                    "dit_ms": statistics.fmean(float(item["denoise_seconds"]) * 1000.0 for item in stage_metrics),
                    "vae_decode_ms": statistics.fmean(
                        float(item["vae_decode_seconds"]) * 1000.0 for item in stage_metrics
                    ),
                    "video_path": str(video_path),
                }
            )
        return rows
    finally:
        _close_sessions(pipeline, sessions)


def _generate(args: argparse.Namespace, output: Path) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("generation requires CUDA")
    if args.device_id >= torch.cuda.device_count():
        raise ValueError(f"device-id {args.device_id} is outside visible CUDA devices")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    videos_dir = output / "videos"
    videos_dir.mkdir(parents=True)
    os.environ["TELEFUSER_ABOT_CUDA_GRAPH_ENABLED"] = "0"
    torch.cuda.set_device(args.device_id)
    image = Image.open(args.image).convert("RGB")
    configs = [ExperimentConfig(batch, positions) for positions in args.steps for batch in args.batch_sizes]
    loader = _load_example_loader()
    pipeline = loader.get_pipeline(
        model_root=args.model_root,
        height=HEIGHT,
        width=WIDTH,
        device_id=args.device_id,
        pipeline_class=ABotWorldInteractivePipeline,
    )
    official_timesteps = pipeline.denoise_stage._official_denoising_timesteps
    rows: list[dict[str, Any]] = []
    try:
        for config in configs:
            selected_timesteps = _configure_steps(pipeline, config.denoise_step_positions, official_timesteps)
            print(f"config={config.name} warm-up", flush=True)
            _warm_up(pipeline, image, args.prompt, args.seed, args.actions[0], config)
            for run in range(1, args.runs + 1):
                print(f"config={config.name} run={run}/{args.runs}", flush=True)
                rows.extend(
                    _run_once(
                        pipeline,
                        image,
                        args.prompt,
                        args.seed,
                        args.actions,
                        config,
                        run,
                        selected_timesteps,
                        videos_dir,
                        args.fps,
                    )
                )
    finally:
        pipeline.denoise_stage._official_denoising_timesteps = official_timesteps
        pipeline.close()
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    _write_csv(output / "raw_runs.csv", rows)
    return rows


def _load_video(path: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_image().convert("RGB"))
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return frames


def _quality_size(size: tuple[int, int], max_side: int = 320) -> tuple[int, int]:
    width, height = size
    scale = min(1.0, max_side / max(width, height))
    return max(16, round(width * scale)), max(16, round(height * scale))


def _frames_to_tensor(frames: Sequence[Image.Image], size: tuple[int, int]) -> torch.Tensor:
    arrays = [np.asarray(frame.resize(size, Image.Resampling.BICUBIC), dtype=np.float32) for frame in frames]
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).div_(255.0)


def _ssim(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.shape != candidate.shape or reference.ndim != 4:
        raise ValueError(f"SSIM requires equal NCHW tensors, got {reference.shape} and {candidate.shape}")
    kernel = 11 if min(reference.shape[-2:]) >= 11 else max(3, min(reference.shape[-2:]) // 2 * 2 - 1)
    mu_ref = F.avg_pool2d(reference, kernel, stride=1)
    mu_cand = F.avg_pool2d(candidate, kernel, stride=1)
    var_ref = F.avg_pool2d(reference.square(), kernel, stride=1) - mu_ref.square()
    var_cand = F.avg_pool2d(candidate.square(), kernel, stride=1) - mu_cand.square()
    covariance = F.avg_pool2d(reference * candidate, kernel, stride=1) - mu_ref * mu_cand
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_ref * mu_cand + c1) * (2 * covariance + c2)) / (
        (mu_ref.square() + mu_cand.square() + c1) * (var_ref + var_cand + c2)
    )
    return float(score.mean())


def _load_vgg_features(device: torch.device) -> torch.nn.Module:
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / VGG_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"offline VGG16 checkpoint is missing: {checkpoint}; quality evaluation will not download weights"
        )
    model = vgg16(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    features = model.features[:23].eval().to(device)
    for parameter in features.parameters():
        parameter.requires_grad_(False)
    return features


def _vgg_embeddings(
    frames: Sequence[Image.Image],
    extractor: torch.nn.Module,
    device: torch.device,
    batch_size: int = 12,
) -> torch.Tensor:
    video = _frames_to_tensor(frames, (384, 224))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    video = (video - mean) / std
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(video), batch_size):
            features = extractor(video[start : start + batch_size].to(device))
            pooled = F.adaptive_avg_pool2d(features, (4, 4)).flatten(1)
            embeddings.append(F.normalize(pooled, dim=1).cpu())
    return torch.cat(embeddings)


def _laplacian_flicker(video: torch.Tensor) -> float:
    gray = 0.299 * video[:, 0] + 0.587 * video[:, 1] + 0.114 * video[:, 2]
    laplacian = -4 * gray[:, 1:-1, 1:-1]
    laplacian += gray[:, :-2, 1:-1] + gray[:, 2:, 1:-1]
    laplacian += gray[:, 1:-1, :-2] + gray[:, 1:-1, 2:]
    return float((laplacian[1:] - laplacian[:-1]).abs().mean())


def _make_contact_sheet(frames: Sequence[Image.Image], path: Path, actions: Sequence[str]) -> None:
    samples = (0, 3, 7, 11)
    thumb_width = 300
    thumb_height = round(thumb_width * HEIGHT / WIDTH)
    label_height = 30
    canvas = Image.new("RGB", (thumb_width * len(samples), (thumb_height + label_height) * len(actions)), "white")
    draw = ImageDraw.Draw(canvas)
    for block, action in enumerate(actions):
        y = block * (thumb_height + label_height)
        draw.text((8, y + 7), f"block {block + 1}: {action}", fill="black")
        for column, offset in enumerate(samples):
            index = block * FRAMES_PER_BLOCK + offset
            thumb = frames[index].resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (column * thumb_width, y + label_height))
            draw.text(
                (column * thumb_width + 8, y + label_height + 6),
                f"f{index}",
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
    canvas.save(path)


def _load_action_scores(path: Path | None) -> dict[tuple[str, int, int], float]:
    if path is None:
        return {}
    scores: dict[tuple[str, int, int], float] = {}
    for row in _read_csv(path):
        raw = row.get("Q_action", "").strip()
        if not raw:
            continue
        score = float(raw)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Q_action must be in [0,1], got {score}")
        scores[(row["config"], int(row["run"]), int(row["session"]))] = score
    return scores


def _evaluate(
    args: argparse.Namespace,
    output: Path,
    raw_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference_row = next(
        row
        for row in raw_rows
        if int(row["B"]) == 1 and int(row["S"]) == 4 and int(row["run"]) == 1 and int(row["session"]) == 0
    )
    reference_path = Path(str(reference_row["video_path"]))
    reference_frames = _load_video(reference_path)
    quality_device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    extractor = _load_vgg_features(quality_device)
    reference_embeddings = _vgg_embeddings(reference_frames, extractor, quality_device)
    quality_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    action_scores = _load_action_scores(args.action_scores)
    contact_dir = output / "contact_sheets"
    contact_dir.mkdir(exist_ok=True)
    actions = [_action_label(item) for item in args.actions]
    representatives: set[str] = set()
    try:
        for raw in raw_rows:
            video_path = Path(str(raw["video_path"]))
            frames = _load_video(video_path)
            if len(frames) != len(reference_frames):
                raise ValueError(
                    f"frame count mismatch: {video_path} has {len(frames)}, reference has {len(reference_frames)}"
                )
            size = _quality_size(reference_frames[0].size)
            reference_pixels = _frames_to_tensor(reference_frames, size)
            candidate_pixels = _frames_to_tensor(frames, size)
            embeddings = _vgg_embeddings(frames, extractor, quality_device)
            adjacent_cosine = (embeddings[:-1] * embeddings[1:]).sum(dim=1)
            reference_cosine = (reference_embeddings * embeddings).sum(dim=1)
            temporal_ssim = _ssim(candidate_pixels[:-1], candidate_pixels[1:])
            reference_ssim = _ssim(reference_pixels, candidate_pixels)
            q_temporal = float(adjacent_cosine.mean().clamp(0.0, 1.0))
            feature_reference = float(reference_cosine.mean().clamp(0.0, 1.0))
            q_visual = 0.5 * reference_ssim + 0.5 * feature_reference
            key = (str(raw["config"]), int(raw["run"]), int(raw["session"]))
            q_action = action_scores.get(key)
            q_world = None if q_action is None else 0.5 * q_action + 0.3 * q_temporal + 0.2 * q_visual
            result = {
                **raw,
                "Q_action": "" if q_action is None else q_action,
                "Q_action_status": "unavailable_no_evaluator" if q_action is None else "externally_supplied",
                "Q_temporal": q_temporal,
                "Q_visual": q_visual,
                "Q_world": "" if q_world is None else q_world,
                "temporal_feature_cosine_p05": float(torch.quantile(adjacent_cosine, 0.05)),
                "temporal_ssim": temporal_ssim,
                "temporal_l1": float((candidate_pixels[1:] - candidate_pixels[:-1]).abs().mean()),
                "temporal_laplacian_flicker": _laplacian_flicker(candidate_pixels),
                "reference_ssim": reference_ssim,
                "reference_vgg_cosine": feature_reference,
                "LPIPS": "",
                "CLIP_similarity": "",
                "reference_video_path": str(reference_path),
            }
            quality_rows.append(result)
            config = str(raw["config"])
            contact_path = (contact_dir / f"{config}.png").resolve()
            if config not in representatives:
                _make_contact_sheet(frames, contact_path, actions)
                representatives.add(config)
            action_rows.append(
                {
                    "config": config,
                    "B": raw["B"],
                    "S": raw["S"],
                    "run": raw["run"],
                    "session": raw["session"],
                    "video_path": str(video_path),
                    "contact_sheet": str(contact_path),
                    "actions": json.dumps(actions),
                    "question": "Does the generated video correctly execute each supplied movement/camera action?",
                    "Q_action": "" if q_action is None else q_action,
                    "status": "pending_external_vlm_or_human_review" if q_action is None else "scored",
                    "reviewer": "",
                    "notes": "",
                }
            )
    finally:
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_csv(output / "quality_runs.csv", quality_rows)
    _write_csv(output / "action_review.csv", action_rows)
    return quality_rows, action_rows


def _pareto(rows: Sequence[Mapping[str, Any]], quality_key: str) -> list[str]:
    frontier = []
    for candidate in rows:
        dominated = any(
            float(other["latency_ms"]) <= float(candidate["latency_ms"])
            and float(other[quality_key]) >= float(candidate[quality_key])
            and (
                float(other["latency_ms"]) < float(candidate["latency_ms"])
                or float(other[quality_key]) > float(candidate[quality_key])
            )
            for other in rows
            if other is not candidate
        )
        if not dominated:
            frontier.append(str(candidate["config"]))
    return frontier


def _profile(output: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    profile_rows: list[dict[str, Any]] = []
    configs = list(dict.fromkeys(str(row["config"]) for row in rows))
    for config in configs:
        group = [row for row in rows if row["config"] == config]
        q_actions = [float(row["Q_action"]) for row in group if str(row["Q_action"]).strip()]
        q_worlds = [float(row["Q_world"]) for row in group if str(row["Q_world"]).strip()]
        first = group[0]
        representative = next(row for row in group if int(row["run"]) == 1 and int(row["session"]) == 0)
        profile_rows.append(
            {
                "B": int(first["B"]),
                "S": int(first["S"]),
                "W": int(first["W"]),
                "rho": first["rho"],
                "Q": first["Q"],
                "latency_ms": _mean(group, "latency_ms"),
                "FPS": _mean(group, "FPS"),
                "memory": f"{max(float(row['memory_gib']) for row in group):.3f} GiB",
                "Q_action": statistics.fmean(q_actions) if len(q_actions) == len(group) else "",
                "Q_temporal": _mean(group, "Q_temporal"),
                "Q_visual": _mean(group, "Q_visual"),
                "Q_world": statistics.fmean(q_worlds) if len(q_worlds) == len(group) else "",
                "video_path": representative["video_path"],
                "config": config,
                "latency_p95_ms": max(float(row["latency_p95_ms"]) for row in group),
                "per_user_FPS": _mean(group, "per_user_FPS"),
                "memory_gib": max(float(row["memory_gib"]) for row in group),
                "reference_ssim": _mean(group, "reference_ssim"),
                "reference_vgg_cosine": _mean(group, "reference_vgg_cosine"),
                "temporal_ssim": _mean(group, "temporal_ssim"),
                "temporal_laplacian_flicker": _mean(group, "temporal_laplacian_flicker"),
                "Q_action_status": "available" if len(q_actions) == len(group) else "unavailable_no_evaluator",
                "Q_world_status": "available" if len(q_worlds) == len(group) else "not_composed_without_Q_action",
                "runs": len({int(row["run"]) for row in group}),
                "videos": len(group),
            }
        )
    _write_csv(output / "profile.csv", profile_rows)
    return profile_rows


def _write_rubric(output: Path) -> None:
    text = """# ABot action consistency review rubric

`Q_action` is intentionally empty until a reviewer or external VLM evaluates the MP4, not only the contact sheet.

Question: **Does the generated video correctly execute each supplied movement/camera action?**

- Frames 0–11, `W`: forward locomotion; the viewpoint should advance through the scene.
- Frames 12–23, `W+J`: forward locomotion plus yaw left; both translation and left camera rotation should be present.
- Frames 24–35, `D`: right strafe; viewpoint translation should be rightward without being mistaken for pure yaw.

Score every segment in `[0,1]`, then use their arithmetic mean as the video
`Q_action`. Inspect the MP4 at native speed; the contact sheet is navigation aid
only. Record the evaluator/model version and notes. Populate `action_review.csv`
(or a copy) and rerun with `--evaluate-only --action-scores PATH` to compose
`Q_world`.
"""
    (output / "action_rubric.md").write_text(text, encoding="utf-8")


def _write_analysis(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    by_key = {(int(row["B"]), int(row["S"])): row for row in rows}
    b1s4 = by_key[(1, 4)]
    b1s2 = by_key[(1, 2)]
    b2s4 = by_key[(2, 4)]
    b2s2 = by_key[(2, 2)]
    step_reduction = (1.0 - float(b1s2["latency_ms"]) / float(b1s4["latency_ms"])) * 100.0
    visual_drop = float(b1s4["Q_visual"]) - float(b1s2["Q_visual"])
    temporal_drop = float(b1s4["Q_temporal"]) - float(b1s2["Q_temporal"])
    b2_gain_s4 = float(b2s4["FPS"]) / float(b1s4["FPS"]) - 1.0
    b2_gain_s2 = float(b2s2["FPS"]) / float(b1s2["FPS"]) - 1.0
    batch_visual_delta_s4 = float(b2s4["Q_visual"]) - float(b1s4["Q_visual"])
    batch_visual_delta_s2 = float(b2s2["Q_visual"]) - float(b1s2["Q_visual"])
    batch_temporal_delta_s4 = float(b2s4["Q_temporal"]) - float(b1s4["Q_temporal"])
    batch_temporal_delta_s2 = float(b2s2["Q_temporal"]) - float(b1s2["Q_temporal"])
    proxy_frontier = _pareto(rows, "Q_visual")
    lines = [
        "# ABot-World fixed-shape latency–quality profile",
        "",
        "All configurations retain 480×832 output, latent `[1,48,3,30,52]`, "
        "18-frame KV, dense attention, and BF16 DiT/T5. Every run uses the same prompt, "
        "initial image, action sequence (`W`, `W+J`, `D`), and seed. B=2 lanes also use "
        "the same seed to isolate batching numerics.",
        "",
        "| Config | B | S | Latency ms | Aggregate FPS | Per-user FPS | Peak GiB | "
        "Q temporal | Q visual | Q action | Q world |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['B']} | {row['S']} | {float(row['latency_ms']):.2f} | "
            f"{float(row['FPS']):.2f} | {float(row['per_user_FPS']):.2f} | {float(row['memory_gib']):.3f} | "
            f"{float(row['Q_temporal']):.4f} | {float(row['Q_visual']):.4f} | "
            f"{row['Q_action'] or 'N/A'} | {row['Q_world'] or 'N/A'} |"
        )
    lines.extend(
        [
            "",
            "## Metric contract",
            "",
            "- `Q_temporal`: mean cosine similarity of consecutive ImageNet VGG16 relu4_3 embeddings. Temporal SSIM, "
            "pixel L1, 5th-percentile feature cosine, and Laplacian flicker are retained as diagnostics.",
            "- `Q_visual`: `0.5 × frame-aligned SSIM + 0.5 × VGG16 feature cosine` against the B=1/S=4 reference. "
            "It is an auxiliary baseline-fidelity score, not semantic world correctness.",
            "- `Q_action`: unavailable. Neither repository contains an ABot action evaluator, and no "
            "offline CLIP/DINO/VLM checkpoint is installed. `action_review.csv` and contact "
            "sheets are emitted for an external VLM/human review.",
            "- `Q_world`: intentionally not composed while `Q_action` is unavailable. Once supplied, the script uses "
            "`0.5 Q_action + 0.3 Q_temporal + 0.2 Q_visual`.",
            "- LPIPS and CLIP columns remain empty because their packages/checkpoints are "
            "unavailable. No downloads or new "
            "quantization/evaluation libraries were introduced.",
            "",
            "## Answers",
            "",
            f"1. **Effective latency knobs.** Under the fixed-shape constraint, reducing S "
            "from 4 to 2 lowers B=1 block "
            f"latency by {step_reduction:.1f}%. B is a packing knob rather than fidelity: B=2 changes aggregate FPS by "
            f"{b2_gain_s4 * 100:.1f}% at S=4 and {b2_gain_s2 * 100:.1f}% at S=2 versus their B=1 rows. W, rho, "
            "precision, resolution, and token pruning were not varied.",
            f"2. **Largest quality effect.** Only S is a tested fidelity knob. S=2 changes Q_visual by "
            f"{-visual_drop:+.4f} and Q_temporal by {-temporal_drop:+.4f} versus S=4 at "
            "B=1. No ranking across untested "
            "knobs is claimed, and action quality remains unscored. The higher S=2 "
            "Q_temporal is not evidence of better dynamics: adjacent-frame similarity can "
            "increase when motion is reduced, which is why it must be paired with Q_action.",
            f"3. **Pareto frontier.** A world-quality frontier cannot be claimed without "
            "Q_action/Q_world. The diagnostic "
            f"latency–Q_visual non-dominated set is: {', '.join(proxy_frontier)}.",
            "4. **Batch effect.** B changes batch latency, aggregate throughput, per-user "
            "effective FPS, and memory, but is not supposed to change a session's generative "
            f"objective. At S=4, B=2 changes Q_visual by {batch_visual_delta_s4:+.6f} and "
            f"Q_temporal by {batch_temporal_delta_s4:+.6f}; at S=2 the changes are "
            f"{batch_visual_delta_s2:+.6f} and {batch_temporal_delta_s2:+.6f}. These same-seed "
            "differences are numerical batch-path drift, not a fidelity benefit.",
            "5. **Scheduler fairness.** Yes. Comparing requests only by latency would favor "
            "reduced S even when action fidelity is unknown. Scheduler admission/reward should "
            "condition quality budgets on action mix, scene, rollout horizon, and "
            "batch composition, and should not use Q_world until Q_action has valid coverage.",
            "",
            "## Scope and fairness",
            "",
            "- Three independent session recreations are measured per `(B,S)`; each rollout "
            "contains three 12-frame blocks.",
            "- B=2 quality aggregates both lanes (six videos/config); B=1 aggregates three videos/config.",
            "- Generation uses the existing eager interactive batch path. No model/pipeline "
            "source, checkpoint, latent shape, "
            "KV shape, VAE, or attention implementation was changed.",
            "- FPS in `profile.csv` is aggregate generated FPS. `per_user_FPS` is provided separately.",
            "",
        ]
    )
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=root.parent / "model_zoo/ABot-World-0-5B-LF")
    parser.add_argument(
        "--image",
        type=Path,
        default=root.parent / "ABot-World/web_client/datasets/images/84b90ad568b693d2.png",
    )
    parser.add_argument("--batch-sizes", type=_parse_positive_ints, default=[1, 2])
    parser.add_argument("--steps", type=_parse_steps, default=[(0, 1, 2, 3), (0, 3)])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--actions", type=_parse_actions, default=DEFAULT_ACTIONS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--action-scores", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    if set(args.batch_sizes).difference({1, 2}):
        parser.error("this fixed-shape MVP supports --batch-sizes 1,2")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def main() -> None:
    args = _parse_args()
    output = args.output_dir.resolve()
    if args.evaluate_only:
        raw_path = output / "raw_runs.csv"
        if not raw_path.is_file():
            raise FileNotFoundError(f"--evaluate-only requires {raw_path}")
        raw_rows: list[dict[str, Any]] = _read_csv(raw_path)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        plan = {
            "script": Path(__file__).name,
            "model_root": str(args.model_root.resolve()),
            "initial_image": str(args.image.resolve()),
            "prompt": args.prompt,
            "seed": args.seed,
            "actions": [_action_label(item) for item in args.actions],
            "runs": args.runs,
            "batch_sizes": args.batch_sizes,
            "step_positions": args.steps,
            "height": HEIGHT,
            "width": WIDTH,
            "latent_shape": [1, 48, 3, 30, 52],
            "KV_window": KV_WINDOW,
            "KV_shape_per_layer": [1, 7020, 24, 128],
            "rho": "dense",
            "precision": {"DiT": "bf16", "T5": "bf16", "VAE": "fp32", "TAEW": "fp32"},
            "cuda_graph": False,
            "quality_features": f"ImageNet VGG16 relu4_3 ({VGG_CHECKPOINT}, local checkpoint only)",
        }
        (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raw_rows = _generate(args, output)
    quality_rows, _ = _evaluate(args, output, raw_rows)
    profile_rows = _profile(output, quality_rows)
    _write_rubric(output)
    _write_analysis(output, profile_rows)
    print(f"wrote {output / 'profile.csv'}", flush=True)


if __name__ == "__main__":
    main()
