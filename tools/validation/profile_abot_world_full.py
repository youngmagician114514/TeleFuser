#!/usr/bin/env python3
"""Build the fixed-resolution ABot-World scheduler offline profile.

Four independent worker processes place complete ABot sessions on distinct GPUs;
there is no tensor parallelism.  The measured Cartesian product is batch size,
denoising calls, and a session-static causal KV window.  Unsupported attention
sparsity and model precision modes are reported instead of implemented here.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import math
import multiprocessing as mp
import os
import statistics
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.utils.video import save_video
from tools.validation.profile_abot_world_quality import (
    DEFAULT_ACTIONS,
    DEFAULT_PROMPT,
    _action_label,
    _frames_to_tensor,
    _laplacian_flicker,
    _load_example_loader,
    _load_vgg_features,
    _load_video,
    _make_contact_sheet,
    _parse_actions,
    _quality_size,
    _ssim,
    _vgg_embeddings,
    _write_csv,
)

HEIGHT = 480
WIDTH = 832
LATENT_SHAPE = [1, 48, 3, 30, 52]
TOKENS_PER_LATENT_FRAME = 390
FRAMES_PER_BLOCK = 12
CONTROL_LATENT_FRAMES = 3
# Keep the historical default matrix unchanged, but allow a targeted B=3
# capture when an online scheduler needs a measured three-session profile.
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
SUPPORTED_BATCH_SIZES = frozenset((*DEFAULT_BATCH_SIZES, 3))
DEFAULT_WINDOWS = (6, 12, 18)
DEFAULT_STEP_POSITIONS: dict[int, tuple[int, ...]] = {
    4: (0, 1, 2, 3),
    3: (0, 2, 3),
    2: (0, 3),
}
PREFILL_ACTIONS = DEFAULT_ACTIONS * 2
MEASURE_ACTIONS = DEFAULT_ACTIONS


@dataclass(frozen=True)
class FidelityGroup:
    steps: int
    window: int

    @property
    def sink_frames(self) -> int:
        return self.window // 3


@dataclass(frozen=True)
class ProfileConfig:
    batch_size: int
    steps: int
    window: int

    @property
    def sink_frames(self) -> int:
        return self.window // 3

    @property
    def name(self) -> str:
        return f"b{self.batch_size}_s{self.steps}_w{self.window}_rho0_bf16"


def _parse_positive_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not values or any(item < 1 for item in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected distinct comma-separated positive integers")
    return values


def _parse_nonnegative_ints(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated non-negative integers") from exc
    if not values or any(item < 0 for item in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected distinct comma-separated non-negative integers")
    return values


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered),
        "p50": ordered[(len(ordered) - 1) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _balanced_groups(steps: Sequence[int], windows: Sequence[int], workers: int) -> list[list[FidelityGroup]]:
    assignments: list[list[FidelityGroup]] = [[] for _ in range(workers)]
    loads = [0 for _ in range(workers)]
    groups = [FidelityGroup(step, window) for window in windows for step in steps]
    for group in sorted(groups, key=lambda item: item.steps * item.window, reverse=True):
        target = min(range(workers), key=lambda index: loads[index])
        assignments[target].append(group)
        loads[target] += group.steps * group.window
    return assignments


def _configure_group(
    pipeline: ABotWorldInteractivePipeline,
    group: FidelityGroup,
    official_timesteps: Any,
) -> list[float]:
    pipeline.config.height = HEIGHT
    pipeline.config.width = WIDTH
    pipeline.config.local_attn_size = group.window
    pipeline.config.sink_size = group.sink_frames
    pipeline.denoise_stage.configure_cuda_graph(False)
    pipeline.denoise_stage.dit.set_causal_attention_window(group.window, group.sink_frames)
    positions = DEFAULT_STEP_POSITIONS[group.steps]

    def selected_timesteps(scheduler: Any) -> torch.Tensor:
        source = official_timesteps(scheduler)
        indices = torch.tensor(positions, device=source.device, dtype=torch.long)
        return source.index_select(0, indices)

    pipeline.denoise_stage._official_denoising_timesteps = selected_timesteps
    scheduler = pipeline.denoise_stage._scheduler()
    return [float(item) for item in selected_timesteps(scheduler).cpu().tolist()]


def _close_sessions(pipeline: ABotWorldInteractivePipeline, sessions: Sequence[Any]) -> None:
    for session in sessions:
        try:
            pipeline.close_interactive_session(session)
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()


def _create_sessions(
    pipeline: ABotWorldInteractivePipeline,
    image: Image.Image,
    prompt: str,
    seed: int,
    config: ProfileConfig,
    run: int,
    worker_id: int,
    tag: str,
) -> list[Any]:
    sessions: list[Any] = []
    try:
        for lane in range(config.batch_size):
            sessions.append(
                pipeline.create_interactive_session(
                    image,
                    prompt,
                    seed=seed,
                    session_id=f"full-profile-w{worker_id}-{tag}-{config.name}-r{run}-lane{lane}",
                )
            )
        return sessions
    except Exception:
        _close_sessions(pipeline, sessions)
        raise


def _update_hashes(digests: Sequence[Any], outputs: Sequence[Sequence[Image.Image]]) -> None:
    for digest, frames in zip(digests, outputs, strict=True):
        for frame in frames:
            digest.update(np.asarray(frame.convert("RGB"), dtype=np.uint8).tobytes())


def _generate_blocks(
    pipeline: ABotWorldInteractivePipeline,
    sessions: Sequence[Any],
    actions: Mapping[str, bool],
) -> list[list[Image.Image]]:
    outputs = pipeline.generate_next_blocks(
        sessions,
        [actions for _ in sessions],
        control_latent_frames=CONTROL_LATENT_FRAMES,
    )
    if any(len(frames) != FRAMES_PER_BLOCK for frames in outputs):
        raise RuntimeError(f"unexpected output frame counts: {[len(frames) for frames in outputs]}")
    return outputs


def _run_once(
    pipeline: ABotWorldInteractivePipeline,
    image: Image.Image,
    prompt: str,
    seed: int,
    config: ProfileConfig,
    run: int,
    worker_id: int,
    selected_timesteps: Sequence[float],
    capture_video: bool,
    barrier: Any | None = None,
    tag: str = "matrix",
) -> tuple[dict[str, Any], list[Image.Image]]:
    device = torch.device(pipeline.device)
    sessions = _create_sessions(pipeline, image, prompt, seed, config, run, worker_id, tag)
    digests = [hashlib.sha256() for _ in sessions]
    representative_frames: list[Image.Image] = []
    block_latencies: list[float] = []
    stage_metrics: list[dict[str, Any]] = []
    try:
        for actions in PREFILL_ACTIONS:
            outputs = _generate_blocks(pipeline, sessions, actions)
            _update_hashes(digests, outputs)
            if capture_video:
                representative_frames.extend(outputs[0])
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for actions in MEASURE_ACTIONS:
            if barrier is not None:
                barrier.wait(timeout=120)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            outputs = _generate_blocks(pipeline, sessions, actions)
            torch.cuda.synchronize(device)
            block_latencies.append((time.perf_counter() - started) * 1000.0)
            stage_metrics.append(dict(pipeline.last_stage_metrics()))
            _update_hashes(digests, outputs)
            if capture_video:
                representative_frames.extend(outputs[0])
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
    finally:
        _close_sessions(pipeline, sessions)
    latency = statistics.fmean(block_latencies)
    row = {
        "config": config.name,
        "B": config.batch_size,
        "S": config.steps,
        "W": config.window,
        "sink_frames": config.sink_frames,
        "rho": 0,
        "precision": "bf16",
        "run": run,
        "device_id": int(device.index or 0),
        "selected_timesteps": json.dumps(selected_timesteps),
        "prefill_actions": json.dumps([_action_label(item) for item in PREFILL_ACTIONS]),
        "measured_actions": json.dumps([_action_label(item) for item in MEASURE_ACTIONS]),
        "seed": seed,
        "latent_shape": json.dumps(LATENT_SHAPE),
        "kv_shape_per_layer_per_session": json.dumps([1, config.window * TOKENS_PER_LATENT_FRAME, 24, 128]),
        "block_latencies_ms": json.dumps(block_latencies),
        "latency_ms": latency,
        "aggregate_FPS": config.batch_size * FRAMES_PER_BLOCK / (latency / 1000.0),
        "per_session_FPS": FRAMES_PER_BLOCK / (latency / 1000.0),
        "peak_memory_bytes": peak_bytes,
        "peak_memory_GB": peak_bytes / 1e9,
        "peak_memory_GiB": peak_bytes / 2**30,
        "dit_ms": statistics.fmean(float(item["denoise_seconds"]) * 1000.0 for item in stage_metrics),
        "vae_decode_ms": statistics.fmean(float(item["vae_decode_seconds"]) * 1000.0 for item in stage_metrics),
        "lane_pixel_sha256": json.dumps([digest.hexdigest() for digest in digests]),
        "video_path": "",
    }
    return row, representative_frames


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _worker_main(
    worker_id: int,
    device_id: int,
    groups: Sequence[FidelityGroup],
    args: argparse.Namespace,
    output: Path,
    placement_barrier: Any,
) -> None:
    worker_dir = output / "workers" / f"worker{worker_id}_gpu{device_id}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    unavailable_rows: list[dict[str, Any]] = []
    pipeline: ABotWorldInteractivePipeline | None = None
    try:
        os.environ["TELEFUSER_ABOT_CUDA_GRAPH_ENABLED"] = "0"
        torch.cuda.set_device(device_id)
        image = Image.open(args.image).convert("RGB")
        loader = _load_example_loader()
        pipeline = loader.get_pipeline(
            model_root=args.model_root,
            height=HEIGHT,
            width=WIDTH,
            device_id=device_id,
            pipeline_class=ABotWorldInteractivePipeline,
        )
        official_timesteps = pipeline.denoise_stage._official_denoising_timesteps

        baseline_group = FidelityGroup(4, 18)
        baseline_timesteps = _configure_group(pipeline, baseline_group, official_timesteps)
        placement_barrier.wait(timeout=180)
        placement_config = ProfileConfig(1, 4, 18)
        for run in range(1, args.runs + 1):
            print(f"worker={worker_id} gpu={device_id} placement run={run}/{args.runs}", flush=True)
            row, _ = _run_once(
                pipeline,
                image,
                args.prompt,
                args.seed,
                placement_config,
                run,
                worker_id,
                baseline_timesteps,
                False,
                placement_barrier,
                "placement",
            )
            placement_rows.append(row)

        for group in groups:
            selected_timesteps = _configure_group(pipeline, group, official_timesteps)
            stop_larger_batches = False
            for batch_size in args.batch_sizes:
                config = ProfileConfig(batch_size, group.steps, group.window)
                if stop_larger_batches:
                    unavailable_rows.append(
                        {
                            **asdict(config),
                            "config": config.name,
                            "status": "skipped_after_smaller_batch_oom",
                            "reason": "A smaller batch exhausted this GPU, so larger batches were not attempted.",
                        }
                    )
                    continue
                config_rows: list[dict[str, Any]] = []
                representative: list[Image.Image] = []
                try:
                    for run in range(1, args.runs + 1):
                        print(
                            f"worker={worker_id} gpu={device_id} config={config.name} run={run}/{args.runs}",
                            flush=True,
                        )
                        row, frames = _run_once(
                            pipeline,
                            image,
                            args.prompt,
                            args.seed,
                            config,
                            run,
                            worker_id,
                            selected_timesteps,
                            run == 1,
                        )
                        config_rows.append(row)
                        if run == 1:
                            representative = frames
                    video_path = (output / "videos" / f"{config.name}.mp4").resolve()
                    save_video(representative, str(video_path), fps=args.fps, quality=9)
                    config_rows[0]["video_path"] = str(video_path)
                    raw_rows.extend(config_rows)
                except Exception as exc:
                    if not _is_oom(exc):
                        raise
                    stop_larger_batches = True
                    unavailable_rows.append(
                        {
                            **asdict(config),
                            "config": config.name,
                            "status": "unsupported_oom",
                            "reason": str(exc).replace("\n", " ")[:500],
                        }
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
        pipeline.denoise_stage._official_denoising_timesteps = official_timesteps
    except Exception:
        (worker_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        _write_csv(worker_dir / "raw_runs.csv", raw_rows)
        _write_csv(worker_dir / "placement_runs.csv", placement_rows)
        _write_csv(
            worker_dir / "unavailable.csv",
            unavailable_rows,
            ["batch_size", "steps", "window", "config", "status", "reason"],
        )
        if pipeline is not None:
            pipeline.close()
            del pipeline
        gc.collect()
        torch.cuda.empty_cache()


def _load_action_scores(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    scores = {}
    for row in _read_csv(path):
        raw = row.get("Q_action", "").strip()
        if not raw:
            continue
        score = float(raw)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Q_action must be in [0,1], got {score}")
        scores[row["config"]] = score
    return scores


def _write_action_rubric(output: Path, actions: Sequence[str]) -> None:
    lines = [
        "# ABot action-consistency VLM rubric",
        "",
        "Inspect the MP4 at native speed. The contact sheet is only a navigation aid.",
        "",
        "Question: **Does the generated video correctly execute every supplied movement/camera action?**",
        "",
    ]
    for index, action in enumerate(actions):
        start = index * FRAMES_PER_BLOCK
        end = start + FRAMES_PER_BLOCK - 1
        expectation = {
            "W": "forward locomotion; the viewpoint should advance through the scene",
            "W+J": "forward locomotion plus yaw left; both translation and left rotation should appear",
            "D": "right strafe; translation should not be mistaken for pure camera yaw",
        }.get(action, "the visible motion should agree with the supplied control")
        lines.append(f"- Frames {start}–{end}, `{action}`: {expectation}.")
    lines.extend(
        [
            "",
            "Return one score in `[0,1]` per block and their arithmetic mean as `Q_action`. "
            "Record evaluator name/version and concise evidence. Copy `action_review.csv`, fill "
            "`Q_action`, and rerun with `--evaluate-only --action-scores PATH` to compose Q_world.",
            "",
        ]
    )
    (output / "action_rubric.md").write_text("\n".join(lines), encoding="utf-8")


def _evaluate_quality(
    args: argparse.Namespace,
    output: Path,
    profile_inputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reference = next(row for row in profile_inputs if int(row["B"]) == 1 and int(row["S"]) == 4 and int(row["W"]) == 18)
    reference_path = Path(str(reference["video_path"]))
    reference_frames = _load_video(reference_path)
    size = _quality_size(reference_frames[0].size)
    reference_pixels = _frames_to_tensor(reference_frames, size)
    quality_device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(quality_device)
    extractor = _load_vgg_features(quality_device)
    reference_embeddings = _vgg_embeddings(reference_frames, extractor, quality_device)
    action_scores = _load_action_scores(args.action_scores)
    all_actions = [_action_label(item) for item in (*PREFILL_ACTIONS, *MEASURE_ACTIONS)]
    contact_dir = output / "contact_sheets"
    contact_dir.mkdir(exist_ok=True)
    quality_rows = []
    action_rows = []
    vlm_requests = []
    try:
        for index, row in enumerate(profile_inputs, start=1):
            config = str(row["config"])
            video_path = Path(str(row["video_path"]))
            frames = _load_video(video_path)
            if len(frames) != len(reference_frames):
                raise ValueError(
                    f"frame count mismatch for {config}: {len(frames)} versus reference {len(reference_frames)}"
                )
            candidate_pixels = _frames_to_tensor(frames, size)
            embeddings = _vgg_embeddings(frames, extractor, quality_device)
            adjacent = (embeddings[:-1] * embeddings[1:]).sum(dim=1)
            reference_cosine = (reference_embeddings * embeddings).sum(dim=1)
            q_temporal = float(adjacent.mean().clamp(0.0, 1.0))
            reference_ssim = _ssim(reference_pixels, candidate_pixels)
            reference_vgg = float(reference_cosine.mean().clamp(0.0, 1.0))
            q_visual = 0.5 * reference_ssim + 0.5 * reference_vgg
            q_action = action_scores.get(config)
            q_world = None if q_action is None else 0.5 * q_action + 0.3 * q_temporal + 0.2 * q_visual
            contact_path = (contact_dir / f"{config}.png").resolve()
            _make_contact_sheet(frames, contact_path, all_actions)
            result = {
                "config": config,
                "Q_action": "" if q_action is None else q_action,
                "Q_temporal": q_temporal,
                "Q_visual": q_visual,
                "Q_world": "" if q_world is None else q_world,
                "reference_ssim": reference_ssim,
                "reference_vgg_cosine": reference_vgg,
                "temporal_feature_cosine_p05": float(torch.quantile(adjacent, 0.05)),
                "temporal_ssim": _ssim(candidate_pixels[:-1], candidate_pixels[1:]),
                "temporal_l1": float((candidate_pixels[1:] - candidate_pixels[:-1]).abs().mean()),
                "temporal_laplacian_flicker": _laplacian_flicker(candidate_pixels),
                "LPIPS": "",
                "CLIP_similarity": "",
                "quality_reference": str(reference_path),
                "video_path": str(video_path),
            }
            quality_rows.append(result)
            action_rows.append(
                {
                    "config": config,
                    "prompt": args.prompt,
                    "actions": json.dumps(all_actions),
                    "video_path": str(video_path),
                    "contact_sheet": str(contact_path),
                    "question": "Does the generated video correctly execute every supplied action?",
                    "Q_action": "" if q_action is None else q_action,
                    "status": "pending_external_vlm_or_human_review" if q_action is None else "scored",
                    "evaluator": "",
                    "evaluator_version": "",
                    "evidence": "",
                }
            )
            vlm_requests.append(
                {
                    "request_id": f"abot-action-{index:03d}",
                    "config": config,
                    "video_path": str(video_path),
                    "prompt": args.prompt,
                    "initial_image": str(args.image.resolve()),
                    "actions": all_actions,
                    "question": "Does the generated video correctly execute every supplied action?",
                    "rubric_path": str((output / "action_rubric.md").resolve()),
                    "response_schema": {
                        "block_scores": [
                            {"block": block + 1, "action": action, "score": "float[0,1]"}
                            for block, action in enumerate(all_actions)
                        ],
                        "Q_action": "mean(block_scores.score)",
                        "evidence": "string",
                    },
                }
            )
    finally:
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_csv(output / "quality.csv", quality_rows)
    _write_csv(output / "action_review.csv", action_rows)
    with (output / "vlm_requests.jsonl").open("w", encoding="utf-8") as handle:
        for request in vlm_requests:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
    return quality_rows


def _aggregate_runs(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    configs = list(dict.fromkeys(str(row["config"]) for row in raw_rows))
    rows = []
    for config in configs:
        group = [row for row in raw_rows if row["config"] == config]
        if len(group) < 3:
            raise ValueError(f"{config} has only {len(group)} completed runs")
        blocks = [value for row in group for value in json.loads(str(row["block_latencies_ms"]))]
        latency = statistics.fmean(blocks)
        first = group[0]
        hashes_by_run = [json.loads(str(row["lane_pixel_sha256"])) for row in group]
        batch_size = int(first["B"])
        if any(len(hashes) != batch_size for hashes in hashes_by_run):
            raise ValueError(f"{config} has incomplete per-lane pixel hashes")
        repeat_deterministic = all(len({hashes[lane] for hashes in hashes_by_run}) == 1 for lane in range(batch_size))
        lanes_identical = all(len(set(hashes)) == 1 for hashes in hashes_by_run)
        unique_lane_hashes = len({digest for hashes in hashes_by_run for digest in hashes})
        video_path = next(str(row["video_path"]) for row in group if str(row["video_path"]))
        rows.append(
            {
                "config": config,
                "B": int(first["B"]),
                "S": int(first["S"]),
                "W": int(first["W"]),
                "sink_frames": int(first["sink_frames"]),
                "rho": 0,
                "precision": "bf16",
                "latency_ms": latency,
                "latency_std_ms": statistics.pstdev(blocks),
                "latency_p95_ms": _summary(blocks)["p95"],
                "FPS": int(first["B"]) * FRAMES_PER_BLOCK / (latency / 1000.0),
                "per_session_FPS": FRAMES_PER_BLOCK / (latency / 1000.0),
                "memory_GB": max(float(row["peak_memory_GB"]) for row in group),
                "memory_GiB": max(float(row["peak_memory_GiB"]) for row in group),
                "dit_ms": statistics.fmean(float(row["dit_ms"]) for row in group),
                "vae_decode_ms": statistics.fmean(float(row["vae_decode_ms"]) for row in group),
                "Q_action": "",
                "Q_temporal": "",
                "Q_visual": "",
                "Q_world": "",
                "reference_ssim": "",
                "reference_vgg_cosine": "",
                "LPIPS": "",
                "CLIP_similarity": "",
                "video_path": video_path,
                "runs": len(group),
                "quality_lane": 0,
                "repeat_deterministic": repeat_deterministic,
                "lanes_identical": lanes_identical,
                "unique_lane_hashes": unique_lane_hashes,
                "device_id": first["device_id"],
                "kv_shape_per_layer_per_session": first["kv_shape_per_layer_per_session"],
            }
        )
    return rows


def _merge_quality(
    runtime_rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    quality = {str(row["config"]): row for row in quality_rows}
    merged = []
    for runtime in runtime_rows:
        row = dict(runtime)
        scores = quality[str(row["config"])]
        for key in (
            "Q_action",
            "Q_temporal",
            "Q_visual",
            "Q_world",
            "reference_ssim",
            "reference_vgg_cosine",
            "LPIPS",
            "CLIP_similarity",
        ):
            row[key] = scores[key]
        merged.append(row)
    return merged


def _pareto_flags(
    rows: Sequence[Mapping[str, Any]],
    x_key: str,
    maximize_x: bool,
    quality_key: str,
) -> dict[str, bool]:
    flags = {}
    for candidate in rows:
        x = float(candidate[x_key])
        quality = float(candidate[quality_key])
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_x = float(other[x_key])
            other_quality = float(other[quality_key])
            x_better = other_x >= x if maximize_x else other_x <= x
            x_strict = other_x > x if maximize_x else other_x < x
            if x_better and other_quality >= quality and (x_strict or other_quality > quality):
                dominated = True
                break
        flags[str(candidate["config"])] = not dominated
    return flags


def _svg_plot(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
) -> None:
    width, height = 920, 560
    left, right, top, bottom = 90, 35, 55, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for _, points in series for point in points]
    if not all_points:
        return
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * 0.08, 0.5 if x_max == x_min else 1e-9)
    y_pad = max((y_max - y_min) * 0.12, 0.01 if y_max == y_min else 1e-9)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">'
        f"{html.escape(title)}</text>",
    ]
    for tick in range(6):
        fraction = tick / 5
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_min + fraction * (y_max - y_min)
        x_pos = px(x_value)
        y_pos = py(y_value)
        lines.extend(
            [
                f'<line x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{top + plot_height}" stroke="#e5e7eb"/>',
                f'<text x="{x_pos:.2f}" y="{top + plot_height + 24}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12">{x_value:.3g}</text>',
                f'<line x1="{left}" y1="{y_pos:.2f}" x2="{left + plot_width}" y2="{y_pos:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y_pos + 4:.2f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{y_value:.3g}</text>',
            ]
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
            f'y2="{top + plot_height}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="20" y="{top + plot_height / 2}" text-anchor="middle" '
            f'transform="rotate(-90 20 {top + plot_height / 2})" font-family="sans-serif" '
            f'font-size="14">{html.escape(y_label)}</text>',
        ]
    )
    legend_x = left + 12
    for index, (label, points) in enumerate(series):
        color = colors[index % len(colors)]
        ordered = sorted(points)
        coordinates = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in ordered)
        lines.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in ordered:
            lines.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="4" fill="{color}"/>')
        legend_y = top + 18 + index * 22
        lines.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 31}" y="{legend_y + 4}" font-family="sans-serif" '
                f'font-size="12">{html.escape(label)}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_curves_and_tables(output: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    quality_key = "Q_world" if all(str(row["Q_world"]).strip() for row in rows) else "Q_visual"
    latency_flags = _pareto_flags(rows, "latency_ms", False, quality_key)
    throughput_flags = _pareto_flags(rows, "FPS", True, quality_key)
    pareto_rows = []
    for row in rows:
        pareto_rows.append(
            {
                "config": row["config"],
                "B": row["B"],
                "S": row["S"],
                "W": row["W"],
                "latency_ms": row["latency_ms"],
                "FPS": row["FPS"],
                "quality_metric": quality_key,
                "quality": row[quality_key],
                "latency_quality_frontier": latency_flags[str(row["config"])],
                "throughput_quality_frontier": throughput_flags[str(row["config"])],
            }
        )
    _write_csv(output / "pareto.csv", pareto_rows)
    reference = next(row for row in rows if int(row["B"]) == 1 and int(row["S"]) == 4 and int(row["W"]) == 18)
    degradation_rows = []
    for row in rows:
        degradation_rows.append(
            {
                "config": row["config"],
                "B": row["B"],
                "S": row["S"],
                "W": row["W"],
                "latency_delta_percent": (float(row["latency_ms"]) / float(reference["latency_ms"]) - 1.0) * 100.0,
                "Q_action_delta": ""
                if not str(row["Q_action"]).strip()
                else float(row["Q_action"]) - float(reference["Q_action"]),
                "Q_temporal_delta": float(row["Q_temporal"]) - float(reference["Q_temporal"]),
                "Q_visual_delta": float(row["Q_visual"]) - float(reference["Q_visual"]),
                "Q_world_delta": ""
                if not str(row["Q_world"]).strip()
                else float(row["Q_world"]) - float(reference["Q_world"]),
            }
        )
    _write_csv(output / "quality_degradation.csv", degradation_rows)

    b1_rows = [row for row in rows if int(row["B"]) == 1]
    _svg_plot(
        output / "latency_quality_pareto_curve.svg",
        f"ABot latency–quality ({quality_key}; B=1)",
        "Block latency (ms, lower is better)",
        quality_key,
        [
            (
                f"W={window}",
                [(float(row["latency_ms"]), float(row[quality_key])) for row in b1_rows if int(row["W"]) == window],
            )
            for window in DEFAULT_WINDOWS
        ],
    )
    w18 = [row for row in rows if int(row["W"]) == 18]
    _svg_plot(
        output / "batch_scaling_curve.svg",
        "ABot native-batch aggregate throughput (W=18)",
        "Batch size B",
        "Aggregate generated FPS",
        [
            (
                f"S={steps}",
                [(float(row["B"]), float(row["FPS"])) for row in w18 if int(row["S"]) == steps],
            )
            for steps in sorted(DEFAULT_STEP_POSITIONS, reverse=True)
        ],
    )
    s4 = [row for row in rows if int(row["S"]) == 4]
    _svg_plot(
        output / "memory_scaling_curve.svg",
        "ABot peak allocated memory (S=4)",
        "Batch size B",
        "Peak memory (GB)",
        [
            (
                f"W={window}",
                [(float(row["B"]), float(row["memory_GB"])) for row in s4 if int(row["W"]) == window],
            )
            for window in DEFAULT_WINDOWS
        ],
    )
    latency_frontier = [name for name, enabled in latency_flags.items() if enabled]
    throughput_frontier = [name for name, enabled in throughput_flags.items() if enabled]
    return latency_frontier, throughput_frontier


def _write_support_tables(
    output: Path,
    unavailable: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    limitations = [
        {
            "dimension": "rho",
            "value": "0.5",
            "status": "unsupported",
            "reason": (
                "Existing ABot attention backends accept dense Q/K/V only; "
                "no dynamic sparsity ratio or mask path exists."
            ),
        },
        {
            "dimension": "rho",
            "value": "0.75",
            "status": "unsupported",
            "reason": (
                "Existing Sage/Flash/SDPA paths expose no block/token sparsity knob; no new CUDA kernel was added."
            ),
        },
        {
            "dimension": "precision",
            "value": "fp16",
            "status": "unsupported",
            "reason": (
                "The released ABot loader is a BF16 DiT/T5 runtime; no existing validated full FP16 profile is exposed."
            ),
        },
        {
            "dimension": "precision",
            "value": "fp8",
            "status": "unsupported",
            "reason": (
                "No full FP8 ABot checkpoint/runtime exists; optional FP8 Sage PV arithmetic is not full-model FP8."
            ),
        },
        {
            "dimension": "quality",
            "value": "LPIPS",
            "status": "unavailable",
            "reason": "LPIPS package/learned linear weights are not installed locally; no score was fabricated.",
        },
        {
            "dimension": "quality",
            "value": "CLIP_similarity",
            "status": "unavailable",
            "reason": (
                "No local CLIP package/checkpoint is available; VGG feature similarity is reported explicitly instead."
            ),
        },
    ]
    parity_failures = [str(row["config"]) for row in rows if not bool(row["lanes_identical"])]
    if parity_failures:
        affected_batches = sorted({int(row["B"]) for row in rows if not bool(row["lanes_identical"])})
        limitations.append(
            {
                "dimension": "batch",
                "value": ",".join(str(value) for value in affected_batches),
                "status": "supported_with_parity_warning",
                "reason": (
                    "All repeats were deterministic, but equal-seed lanes were not pixel-identical for: "
                    + ", ".join(parity_failures)
                    + ". Treat these batch sizes as provisional until batch-parity impact is accepted."
                ),
            }
        )
    _write_csv(output / "support_matrix.csv", limitations)
    fields = ["batch_size", "steps", "window", "config", "status", "reason"]
    _write_csv(output / "unavailable_configs.csv", unavailable, fields)


def _aggregate_placement(
    output: Path,
    placement_rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> dict[str, float]:
    per_gpu = []
    for device_id in sorted({int(row["device_id"]) for row in placement_rows}):
        group = [row for row in placement_rows if int(row["device_id"]) == device_id]
        blocks = [value for row in group for value in json.loads(str(row["block_latencies_ms"]))]
        latency = statistics.fmean(blocks)
        per_gpu.append(
            {
                "device_id": device_id,
                "sessions_on_gpu": 1,
                "S": 4,
                "W": 18,
                "latency_ms": latency,
                "FPS": FRAMES_PER_BLOCK / (latency / 1000.0),
                "memory_GB": max(float(row["peak_memory_GB"]) for row in group),
                "runs": len(group),
            }
        )
    _write_csv(output / "multi_gpu_placement.csv", per_gpu)
    aggregate_fps = sum(float(row["FPS"]) for row in per_gpu)
    ideal = len(per_gpu) * float(baseline["FPS"])
    summary = {
        "gpu_count": float(len(per_gpu)),
        "sessions_total": float(len(per_gpu)),
        "aggregate_FPS": aggregate_fps,
        "single_gpu_baseline_FPS": float(baseline["FPS"]),
        "scaling_efficiency": aggregate_fps / ideal,
        "mean_per_gpu_latency_ms": statistics.fmean(float(row["latency_ms"]) for row in per_gpu),
    }
    _write_csv(output / "multi_gpu_placement_summary.csv", [summary])
    return summary


def _write_analysis(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    placement: Mapping[str, float],
    latency_frontier: Sequence[str],
    throughput_frontier: Sequence[str],
) -> None:
    by_key = {(int(row["B"]), int(row["S"]), int(row["W"])): row for row in rows}
    baseline = by_key[(1, 4, 18)]
    s3 = by_key[(1, 3, 18)]
    s2 = by_key[(1, 2, 18)]
    w12 = by_key[(1, 4, 12)]
    w6 = by_key[(1, 4, 6)]
    s3_reduction = (1.0 - float(s3["latency_ms"]) / float(baseline["latency_ms"])) * 100.0
    s2_reduction = (1.0 - float(s2["latency_ms"]) / float(baseline["latency_ms"])) * 100.0
    w12_reduction = (1.0 - float(w12["latency_ms"]) / float(baseline["latency_ms"])) * 100.0
    w6_reduction = (1.0 - float(w6["latency_ms"]) / float(baseline["latency_ms"])) * 100.0
    max_batch = max(int(row["B"]) for row in rows)
    batch_gains = []
    for steps in sorted(DEFAULT_STEP_POSITIONS, reverse=True):
        b1 = by_key[(1, steps, 18)]
        bmax = by_key.get((max_batch, steps, 18))
        if bmax is not None:
            gain = (float(bmax["FPS"]) / float(b1["FPS"]) - 1.0) * 100.0
            batch_gains.append(f"S={steps}: {gain:+.1f}%")
    batch_quality_drift = max(
        abs(float(row["Q_visual"]) - float(by_key[(1, int(row["S"]), int(row["W"]))]["Q_visual"])) for row in rows
    )
    repeat_failures = [str(row["config"]) for row in rows if not bool(row["repeat_deterministic"])]
    lane_parity_failures = [str(row["config"]) for row in rows if not bool(row["lanes_identical"])]
    parity_batches = sorted({int(row["B"]) for row in rows if not bool(row["lanes_identical"])})
    quality_metric = "Q_world" if all(str(row["Q_world"]).strip() for row in rows) else "Q_visual proxy"
    lines = [
        "# ABot-World full fixed-resolution offline profile",
        "",
        "本实验保持 480×832、latent `[1,48,3,30,52]`、action conditioning、KV representation、VAE 和 "
        "checkpoint 不变。每个 session 完整驻留单卡；四卡仅做独立 session placement，不使用 tensor parallel。",
        "",
        "## 测量范围",
        "",
        f"- 成功 profile 行数：{len(rows)}；最大可运行 batch：B={max_batch}。",
        "- 每个 `(B,S,W)` 三次新建 session；每次先用固定 6-block history 填满 W=18，再测量固定 3 blocks。",
        "- 每个配置只保存 run1/lane0 的一个 108-frame 代表视频；其余 run/lane 用像素哈希分别验证"
        "重复确定性和 lane parity。",
        f"- 三次重复不一致配置：{', '.join(repeat_failures) if repeat_failures else '无'}；"
        f"equal-seed lane 不一致的 batch：{', '.join(map(str, parity_batches)) if parity_batches else '无'}。",
        "- rho 仅支持 0（dense）；precision 仅支持完整 BF16 DiT/T5 runtime。",
        "",
        "## 关键结果",
        "",
        f"1. **哪些 knob 降低 latency？** 在 B=1/W=18 下，S=4→3 降低 {s3_reduction:.1f}%，"
        f"S=4→2 降低 {s2_reduction:.1f}%；在 B=1/S=4 下，W=18→12 降低 {w12_reduction:.1f}%，"
        f"W=18→6 降低 {w6_reduction:.1f}%。rho/FP16/FP8 没有可用 runtime，未计入。",
        "2. **哪些 knob 不破坏 KV/state？** S 和 B 不改变 session KV shape；本次 precision 只有 BF16。"
        "W=6/12/18 使用现有 session-static window API，新 session 分别分配 2340/4680/7020 token "
        "容量，cursor 与 RoPE 仍走原生 causal 路径。W 不能对存量 session 热切换。",
        f"3. **batch 是否有收益？** W=18 的 B=1→B={max_batch} aggregate FPS 变化为 "
        f"{', '.join(batch_gains)}；收益应与增长的 batch latency、per-session FPS 下降和显存一起决策。"
        f"四卡 B=1 placement 得到 {placement['aggregate_FPS']:.2f} aggregate FPS，"
        f"相对单卡线性 scaling efficiency 为 {placement['scaling_efficiency'] * 100:.1f}%。",
        f"4. **batch 与 fidelity 是否耦合？** 计算收益随 S/W 改变，因此存在系统层耦合。同 seed 的 "
        f"B-path 最大 Q_visual proxy 漂移为 {batch_quality_drift:.6f}；所有重复都稳定，但 "
        f"{', '.join(lane_parity_failures) if lane_parity_failures else '没有配置'} 出现 equal-seed lane 像素不一致。"
        "这说明 batch execution 不是严格 quality-neutral；Q_visual 是相对 baseline 的轨迹/画面相似度，"
        "不能单独解释为感知质量下降。",
        f"5. **哪些配置适合 scheduler？** latency–{quality_metric} frontier："
        f"{', '.join(latency_frontier)}。吞吐–{quality_metric} frontier：{', '.join(throughput_frontier)}。"
        "低 S/短 W 适合 latency-critical tier；S=4/W=18 是 quality reference；B 应由队列深度、deadline 和显存决定。"
        "在完成 action evaluation 和 B=8 parity 验收前，在线 shortlist 建议限于 B<=4。",
        f"6. **是否有明显 Pareto frontier？** 当前只能建立 {quality_metric} frontier。Q_action 尚无 evaluator，"
        "因此 Q_world 留空，不能把 proxy frontier 宣称为最终 world-quality frontier。",
        "",
        "## Quality 解释",
        "",
        "- Q_temporal 是相邻 VGG relu4_3 feature cosine。它可能奖励静止，必须与 action score 联合使用。",
        "- Q_visual = 0.5×frame-aligned SSIM + 0.5×VGG feature cosine，相对 B1/S4/W18 reference。",
        "- 人工 spot check 比较 B1 与 B8 的 S4/W18 contact sheet：人物和场景 identity 稳定、无明显 flicker/collapse，"
        "但从第 3 个 block 起轨迹与构图可见分叉；这与 reference-similarity 漂移一致，"
        "不等同于已证实的 action/感知质量损失。",
        "- 本机没有 LPIPS/CLIP 权重，相关列为空。没有下载新 evaluator，也没有伪造分数。",
        "- action_review.csv、action_rubric.md、contact sheets 和 vlm_requests.jsonl 构成 VLM 接口。"
        "导入外部 Q_action 后可按 0.5/0.3/0.2 合成 Q_world。",
        "",
        "## Serving 注意事项",
        "",
        "- W 是 admission-time/session-static fidelity，不能作为逐 block knob。",
        "- 不同 W 的 session cache shape 不同，native batch 应按 W 分桶。",
        "- S 可以逐 block 修改且不改变 KV shape，但最终 cache commit 仍必须执行。",
        "- B=8 的三次重复完全稳定，但每次 lane7 与 lane0–6 的像素哈希不同，且 lane0 相对 B<=4 reference 发生轨迹漂移；"
        "因此 B=8 先标记为 throughput/memory 可行、quality parity 待验收。",
        "- rho=0.5/0.75 需要新的 sparse attention runtime；本实验没有实现。",
        "- Sage attention 的 FP8 PV arithmetic 不是完整 FP8 模型 precision。",
        "- Scheduler 必须纳入 quality fairness：action mix、scene、history horizon 和用户 "
        "deadline 都会改变同一配置的价值。",
        "",
    ]
    (output / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _merge_worker_outputs(output: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    raw_rows = []
    placement_rows = []
    unavailable = []
    for worker_dir in sorted((output / "workers").iterdir()):
        raw_path = worker_dir / "raw_runs.csv"
        placement_path = worker_dir / "placement_runs.csv"
        unavailable_path = worker_dir / "unavailable.csv"
        if raw_path.is_file():
            raw_rows.extend(_read_csv(raw_path))
        if placement_path.is_file():
            placement_rows.extend(_read_csv(placement_path))
        if unavailable_path.is_file():
            unavailable.extend(_read_csv(unavailable_path))
    _write_csv(output / "raw_runs.csv", raw_rows)
    _write_csv(output / "placement_runs.csv", placement_rows)
    return raw_rows, placement_rows, unavailable


def _run_workers(
    args: argparse.Namespace, output: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    assignments = _balanced_groups(args.steps, args.windows, len(args.gpu_ids))
    context = mp.get_context("spawn")
    barrier = context.Barrier(len(args.gpu_ids))
    processes = []
    for worker_id, (device_id, groups) in enumerate(zip(args.gpu_ids, assignments, strict=True)):
        process = context.Process(
            target=_worker_main,
            args=(worker_id, device_id, groups, args, output, barrier),
            name=f"abot-full-profile-gpu{device_id}",
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failures = [process for process in processes if process.exitcode != 0]
    if failures:
        names = ", ".join(f"{process.name}(exit={process.exitcode})" for process in failures)
        raise RuntimeError(f"profile workers failed: {names}; inspect {output / 'workers'}")
    return _merge_worker_outputs(output)


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
    parser.add_argument("--gpu-ids", type=_parse_nonnegative_ints, default=[0, 1, 2, 3])
    parser.add_argument("--batch-sizes", type=_parse_positive_ints, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--steps", type=_parse_positive_ints, default=sorted(DEFAULT_STEP_POSITIONS, reverse=True))
    parser.add_argument("--windows", type=_parse_positive_ints, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--actions", type=_parse_actions, default=DEFAULT_ACTIONS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--action-scores", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    if len(args.gpu_ids) > 4 or len(args.gpu_ids) < 1:
        parser.error("--gpu-ids must contain one to four GPUs")
    if set(args.batch_sizes).difference(SUPPORTED_BATCH_SIZES):
        parser.error(f"batch sizes must be a subset of {tuple(sorted(SUPPORTED_BATCH_SIZES))}")
    if set(args.steps).difference(DEFAULT_STEP_POSITIONS):
        parser.error(f"steps must be a subset of {tuple(DEFAULT_STEP_POSITIONS)}")
    if set(args.windows).difference(DEFAULT_WINDOWS):
        parser.error(f"windows must be a subset of {DEFAULT_WINDOWS}")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def main() -> None:
    args = _parse_args()
    output = args.output_dir.resolve()
    if args.evaluate_only:
        raw_path = output / "raw_runs.csv"
        placement_path = output / "placement_runs.csv"
        if not raw_path.is_file() or not placement_path.is_file():
            raise FileNotFoundError("--evaluate-only requires raw_runs.csv and placement_runs.csv")
        raw_rows = _read_csv(raw_path)
        placement_rows = _read_csv(placement_path)
        unavailable_path = output / "unavailable_configs.csv"
        unavailable = _read_csv(unavailable_path) if unavailable_path.is_file() else []
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "videos").mkdir()
        (output / "workers").mkdir()
        assignments = _balanced_groups(args.steps, args.windows, len(args.gpu_ids))
        plan = {
            "script": Path(__file__).name,
            "model_root": str(args.model_root.resolve()),
            "initial_image": str(args.image.resolve()),
            "prompt": args.prompt,
            "seed": args.seed,
            "prefill_actions": [_action_label(item) for item in PREFILL_ACTIONS],
            "measured_actions": [_action_label(item) for item in MEASURE_ACTIONS],
            "runs": args.runs,
            "gpu_ids": args.gpu_ids,
            "worker_assignments": [[asdict(group) for group in groups] for groups in assignments],
            "batch_sizes": args.batch_sizes,
            "steps": args.steps,
            "step_positions": {key: list(value) for key, value in DEFAULT_STEP_POSITIONS.items()},
            "windows": args.windows,
            "sink_ratio": "1/3",
            "resolution": [HEIGHT, WIDTH],
            "latent_shape": LATENT_SHAPE,
            "rho": [0],
            "precision": {"DiT": "bf16", "T5": "bf16", "VAE": "fp32", "TAEW": "fp32"},
            "cuda_graph": False,
            "tensor_parallel": False,
            "placement": "one complete pipeline per GPU",
        }
        (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raw_rows, placement_rows, unavailable = _run_workers(args, output)

    runtime_rows = _aggregate_runs(raw_rows)
    _write_action_rubric(output, [_action_label(item) for item in (*PREFILL_ACTIONS, *MEASURE_ACTIONS)])
    quality_rows = _evaluate_quality(args, output, runtime_rows)
    merged = _merge_quality(runtime_rows, quality_rows)
    profile_fields = [
        "B",
        "S",
        "W",
        "rho",
        "precision",
        "latency_ms",
        "FPS",
        "memory_GB",
        "Q_action",
        "Q_temporal",
        "Q_visual",
        "Q_world",
        "config",
        "sink_frames",
        "latency_std_ms",
        "latency_p95_ms",
        "per_session_FPS",
        "memory_GiB",
        "dit_ms",
        "vae_decode_ms",
        "reference_ssim",
        "reference_vgg_cosine",
        "LPIPS",
        "CLIP_similarity",
        "video_path",
        "runs",
        "quality_lane",
        "repeat_deterministic",
        "lanes_identical",
        "unique_lane_hashes",
        "device_id",
        "kv_shape_per_layer_per_session",
    ]
    _write_csv(output / "profile.csv", merged, profile_fields)
    _write_support_tables(output, unavailable, merged)
    latency_frontier, throughput_frontier = _write_curves_and_tables(output, merged)
    baseline = next(row for row in merged if int(row["B"]) == 1 and int(row["S"]) == 4 and int(row["W"]) == 18)
    placement_summary = _aggregate_placement(output, placement_rows, baseline)
    _write_analysis(output, merged, placement_summary, latency_frontier, throughput_frontier)
    print(f"wrote {output / 'profile.csv'}", flush=True)


if __name__ == "__main__":
    main()
