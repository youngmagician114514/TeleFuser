#!/usr/bin/env python3
"""Attach profile-quality and quality-aware CPR metrics to a replay artifact.

The replay client computes a client decoded-frame playout proxy while it is
running.  The server dispatch trace supplies the fidelity and frame count for
each generated chunk.  This helper joins the two artifacts after a run and
adds a frame-weighted ``Q_world`` summary plus a dimensionless quality-aware
CPR score.  It never changes serving behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from telefuser.service.livekit.profile_quality import (
    normalize_profile_qualities,
    profile_batch_family,
)


class ProfileQualityValues(dict[str, float]):
    """Mapping of fidelity to semantic Q, with normalization audit metadata."""

    reference_config: str | None
    reference_raw: float | None
    normalized: bool
    batch_invariant: bool


def _quality_from_row(row: Mapping[str, str]) -> float:
    raw = row.get("Q_world", "")
    if raw not in {None, ""}:
        value = float(raw)
    else:
        values = [
            float(row[key])
            for key in ("Q_action", "Q_temporal", "Q_visual")
            if row.get(key, "") not in {None, ""}
        ]
        if not values:
            raise ValueError(f"profile row {row.get('config', '<unknown>')} has no quality value")
        value = sum(values) / len(values)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"profile row {row.get('config', '<unknown>')} has invalid quality {value!r}")
    return value


def load_profile_quality(path: Path) -> dict[str, float]:
    """Load semantic Q by fidelity, including conservative B3 interpolation.

    The returned mapping uses the same B-invariant family rule as the runtime
    scheduler.  Its attributes expose whether the explicit B1/S4/W18
    denominator was available without changing the mapping API used by older
    report scripts.
    """
    by_config: dict[str, float] = {}
    by_suffix: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            config = str(row.get("config") or "").strip()
            if not config:
                continue
            quality = _quality_from_row(row)
            by_config[config] = quality
            batch, suffix = profile_batch_family(config)
            if batch is not None:
                by_suffix[suffix][batch] = quality
    if not by_config:
        raise ValueError(f"profile contains no quality rows: {path}")
    normalization_probe = normalize_profile_qualities(by_config)
    # motivation_scheduler.load_profile_table() uses the conservative lower
    # quality of B2/B4 when a measured B3 row is absent in a legacy compact
    # table. Production tables use the family B1 semantic value.
    for suffix, rows in by_suffix.items():
        if 3 not in rows and 2 in rows and 4 in rows:
            by_config[f"b3_{suffix}"] = (
                rows.get(1, rows[2]) if normalization_probe.normalized else min(rows[2], rows[4])
            )
    normalized = normalize_profile_qualities(by_config)
    values = ProfileQualityValues(normalized.values)
    values.reference_config = normalized.reference_config
    values.reference_raw = normalized.reference_raw
    values.normalized = normalized.normalized
    values.batch_invariant = normalized.batch_invariant
    return values


def collect_dispatch_quality(
    dispatch_trace: Path, profile_quality: Mapping[str, float]
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Return frame-weighted quality facts keyed by server session ID."""
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"frames": 0.0, "quality_frames": 0.0, "chunks": 0.0})
    missing_fidelity = 0
    missing_quality = 0
    with dispatch_trace.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event_type") != "model_dispatch" or record.get("outcome") != "ok":
                continue
            batch_size = int(record.get("batch_size") or 0)
            for item in record.get("sessions", []):
                if not isinstance(item, Mapping):
                    continue
                session_id = item.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    continue
                frames = int(item.get("frames") or 0)
                if frames <= 0:
                    continue
                fidelity = item.get("fidelity")
                # Older traces omitted fidelity.  Do not silently assign a
                # quality in that case: a wrong quality is worse than a
                # visible missing-quality count.
                if not isinstance(fidelity, str) or not fidelity:
                    missing_fidelity += 1
                    continue
                quality = profile_quality.get(fidelity)
                if quality is None:
                    # A defensive fallback handles a trace that records a
                    # batch-size-independent suffix while the profile names
                    # include B explicitly.
                    suffix = fidelity.split("_", 1)[1] if "_" in fidelity else ""
                    quality = profile_quality.get(f"b{batch_size}_{suffix}") if suffix else None
                if quality is None:
                    missing_quality += 1
                    continue
                entry = totals[session_id]
                entry["frames"] += frames
                entry["quality_frames"] += frames * quality
                entry["chunks"] += 1
    return {
        session_id: {
            "frames_profiled": int(values["frames"]),
            "chunks_profiled": int(values["chunks"]),
            "q_world_frame_weighted": values["quality_frames"] / values["frames"]
            if values["frames"]
            else None,
        }
        for session_id, values in totals.items()
    }, {"missing_fidelity_records": missing_fidelity, "missing_quality_records": missing_quality}


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def count_released_action_jobs(action_trace: Path) -> int:
    """Count action jobs released by the trace producer contract.

    Trace replays carry the release decision explicitly: a non-empty control
    state releases work on a heartbeat or on the first input after
    arrival/resume. Immediate state changes update the latest controls but do
    not demand another generated chunk. This is the same input-side contract
    used by ``release_on_control_state`` and gives SLO an unbiased denominator
    that includes released actions later superseded before dispatch.
    """

    released = 0
    with action_trace.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("kind") != "action_update":
                continue
            data = record.get("data")
            if not isinstance(data, Mapping) or not data.get("controls"):
                continue
            reason = str(data.get("reason") or "")
            if bool(data.get("heartbeat")) or reason in {
                "first_nonempty_input",
                "resume_first_nonempty_input",
            }:
                released += 1
    return released


def collect_producer_metrics(
    dispatch_trace: Path,
    profile_quality: Mapping[str, float],
    *,
    released_action_jobs: int | None = None,
) -> dict[str, Any]:
    """Build transport-independent QoE from completed model chunks.

    Model completion adds the entire generated chunk to a logical per-session
    playout buffer.  The buffer drains at the chunk FPS between completions;
    after the last completion it is drained fully so every successfully
    generated frame is credited exactly once.  LiveKit queueing, capture and
    receiver decode therefore cannot reduce these scheduler/producer metrics.
    """

    events_by_session: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    job_latencies: list[float] = []
    first_action_latency: dict[str, float] = {}
    kinds: dict[str, int] = defaultdict(int)
    completed_job_deadlines_met = 0
    action_job_deadlines_met = 0
    total_jobs = 0
    total_frames = 0
    output_gate_modes: set[int] = set()
    missing_quality = 0
    reference = _quality_reference(profile_quality)

    with dispatch_trace.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event_type") != "model_dispatch" or record.get("outcome") != "ok":
                continue
            completed_at = record.get("model_completed_monotonic_seconds")
            model_seconds = record.get("model_duration_seconds")
            if not isinstance(completed_at, (int, float)) or not isinstance(model_seconds, (int, float)):
                continue
            batch_size = int(record.get("batch_size") or 0)
            for item in record.get("sessions", []):
                if not isinstance(item, Mapping):
                    continue
                session_id = item.get("session_id")
                frames = item.get("frames")
                fps = item.get("fps", 12)
                if (
                    not isinstance(session_id, str)
                    or not session_id
                    or not isinstance(frames, (int, float))
                    or int(frames) <= 0
                    or not isinstance(fps, (int, float))
                    or float(fps) <= 0
                ):
                    continue
                frame_count = int(frames)
                output_seconds = frame_count / float(fps)
                queue_wait = item.get("queue_wait_seconds", 0.0)
                queue_wait_seconds = float(queue_wait) if isinstance(queue_wait, (int, float)) else 0.0
                latency = max(0.0, queue_wait_seconds) + max(0.0, float(model_seconds))
                kind = str(item.get("motivation_kind") or "unknown")
                fidelity = item.get("fidelity")
                quality = profile_quality.get(str(fidelity)) if isinstance(fidelity, str) else None
                if quality is None and isinstance(fidelity, str) and "_" in fidelity:
                    quality = profile_quality.get(f"b{batch_size}_{fidelity.split('_', 1)[1]}")
                if quality is None:
                    missing_quality += 1

                event: dict[str, float | str] = {
                    "completed_at": float(completed_at),
                    "output_seconds": output_seconds,
                    "frames": float(frame_count),
                    "fps": float(fps),
                    "kind": kind,
                }
                if quality is not None:
                    event["quality"] = float(quality)
                events_by_session[session_id].append(event)
                gate_mode = item.get("output_gate_enabled")
                if isinstance(gate_mode, (int, float)):
                    output_gate_modes.add(int(bool(gate_mode)))
                total_jobs += 1
                total_frames += frame_count
                kinds[kind] += 1
                job_latencies.append(latency)
                if latency <= output_seconds:
                    completed_job_deadlines_met += 1
                    if kind == "action":
                        action_job_deadlines_met += 1
                if kind == "action" and session_id not in first_action_latency:
                    first_action_latency[session_id] = latency

    playable_seconds = 0.0
    stall_seconds = 0.0
    quality_playable_seconds = 0.0
    producer_session_count = 0
    for events in events_by_session.values():
        events.sort(key=lambda event: float(event["completed_at"]))
        if not events:
            continue
        producer_session_count += 1
        buffer_seconds = 0.0
        previous_at = float(events[0]["completed_at"])
        for event in events:
            completed_at = float(event["completed_at"])
            elapsed = max(0.0, completed_at - previous_at)
            consumed = min(buffer_seconds, elapsed)
            playable_seconds += consumed
            stall_seconds += elapsed - consumed
            buffer_seconds -= consumed
            output_seconds = float(event["output_seconds"])
            buffer_seconds += output_seconds
            quality = event.get("quality")
            if isinstance(quality, (int, float)):
                quality_playable_seconds += output_seconds * float(quality)
            previous_at = completed_at
        # Logical sink consumption: fully credit the final queued chunk(s).
        playable_seconds += buffer_seconds

    engaged_seconds = playable_seconds + stall_seconds
    producer_cpr = playable_seconds / engaged_seconds if engaged_seconds > 0 else None
    normalized_quality = (
        quality_playable_seconds / playable_seconds / reference
        if playable_seconds > 0 and missing_quality == 0
        else None
    )
    quality_adjusted_cpr = (
        quality_playable_seconds / engaged_seconds / reference
        if engaged_seconds > 0 and missing_quality == 0
        else None
    )
    producer_fps = total_frames / engaged_seconds if engaged_seconds > 0 else None
    p95_job = _percentile(job_latencies, 0.95)
    p95_first_action = _percentile(list(first_action_latency.values()), 0.95)
    completed_actions = kinds.get("action", 0)
    action_denominator = (
        max(0, int(released_action_jobs))
        if released_action_jobs is not None
        else completed_actions
    )
    return {
        "schema_version": "abot_producer_metrics_v2",
        "measurement_boundary": "model_completed_logical_playout",
        "jobs_completed": total_jobs,
        "released_action_jobs": action_denominator,
        "action_jobs_completed": completed_actions,
        "action_jobs_on_time": action_job_deadlines_met,
        "idle_jobs_completed": kinds.get("idle", 0),
        "generated_frames": total_frames,
        "producer_sessions": producer_session_count,
        "producer_cpr": round(producer_cpr, 6) if producer_cpr is not None else None,
        "producer_slo_attainment": (
            round(action_job_deadlines_met / action_denominator, 6)
            if action_denominator
            else None
        ),
        "action_job_completion_ratio": (
            round(completed_actions / action_denominator, 6) if action_denominator else None
        ),
        "completed_job_deadline_attainment": (
            round(completed_job_deadlines_met / total_jobs, 6) if total_jobs else None
        ),
        "producer_fps_per_engaged_session": round(producer_fps, 6) if producer_fps is not None else None,
        "normalized_quality": round(normalized_quality, 6) if normalized_quality is not None else None,
        "quality_adjusted_cpr": round(quality_adjusted_cpr, 6) if quality_adjusted_cpr is not None else None,
        "job_latency_p95_seconds": round(p95_job, 6) if p95_job is not None else None,
        "first_action_job_latency_p95_seconds": (
            round(p95_first_action, 6) if p95_first_action is not None else None
        ),
        "playable_seconds": round(playable_seconds, 6),
        "stall_seconds": round(stall_seconds, 6),
        "engaged_seconds": round(engaged_seconds, 6),
        "output_gate_enabled_values": sorted(output_gate_modes),
        "missing_quality_jobs": missing_quality,
        "definition": (
            "Producer CPR reconstructs continuous per-session playout from model-completed chunks; "
            "action and idle chunks both contribute all generated frames, the final buffer is fully drained, "
            "and LiveKit transport is excluded. Producer SLO is on-time completed action jobs divided by "
            "the action jobs released by the input trace; superseded or otherwise missing actions are misses. "
            "Completed-job deadline attainment is retained as a diagnostic."
        ),
    }


def _quality_reference(profile_quality: Mapping[str, float]) -> float:
    # Runtime/report Q is normalized against the explicit B1/S4/W18 point.
    # Compact legacy tables without that point retain a max-quality fallback,
    # which is surfaced as ``normalized=False`` on ProfileQualityValues.
    if bool(getattr(profile_quality, "normalized", False)):
        return 1.0
    return max(float(value) for value in profile_quality.values())


def augment_result(
    result: dict[str, Any],
    *,
    dispatch_quality: Mapping[str, Mapping[str, Any]],
    profile_quality: Mapping[str, float],
    dispatch_diagnostics: Mapping[str, int],
    profile_path: Path,
    producer_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add per-session and aggregate quality-aware CPR fields in place."""
    reference = _quality_reference(profile_quality)
    total_frames = 0
    total_quality_frames = 0.0
    missing_sessions = 0
    for session in result.get("sessions", []):
        server_session_id = session.get("server_session_id")
        quality = dispatch_quality.get(server_session_id) if isinstance(server_session_id, str) else None
        if quality is None:
            missing_sessions += 1
            session["quality"] = {
                "q_world_frame_weighted": None,
                "q_frame_weighted": None,
                "quality_factor_vs_profile_reference": None,
                "frames_profiled": 0,
                "chunks_profiled": 0,
            }
            continue
        q_world = float(quality["q_world_frame_weighted"])
        factor = max(0.0, min(1.0, q_world / reference))
        session["quality"] = {
            **dict(quality),
            "q_frame_weighted": round(q_world, 6),
            "quality_factor_vs_profile_reference": round(factor, 6),
        }
        total_frames += int(quality["frames_profiled"])
        total_quality_frames += int(quality["frames_profiled"]) * q_world

    quality_cpr_by_phase: list[dict[str, Any]] = []
    for phase in result.get("phase_results", []):
        summary = phase.get("summary")
        if not isinstance(summary, dict):
            continue
        phase_sessions = [
            session
            for session in result.get("sessions", [])
            if session.get("cpr", {}).get("playback_started")
        ]
        engaged = 0.0
        playable = 0.0
        quality_playable = 0.0
        q_values: list[float] = []
        for session in phase_sessions:
            cpr = session.get("cpr", {})
            quality = session.get("quality", {})
            session_engaged = float(cpr.get("engaged_seconds") or 0.0)
            session_playable = float(cpr.get("playable_seconds") or 0.0)
            factor = quality.get("quality_factor_vs_profile_reference")
            if session_engaged <= 0.0 or not isinstance(factor, (int, float)):
                continue
            engaged += session_engaged
            playable += session_playable
            quality_playable += session_playable * float(factor)
            q_world = quality.get("q_world_frame_weighted")
            if isinstance(q_world, (int, float)):
                q_values.append(float(q_world))
        cpr = playable / engaged if engaged > 0.0 else None
        quality_cpr = quality_playable / engaged if engaged > 0.0 else None
        phase_quality = {
            "profile_path": str(profile_path),
            "q_world_frame_weighted": total_quality_frames / total_frames if total_frames else None,
            "q_frame_weighted": total_quality_frames / total_frames if total_frames else None,
            "quality_reference_q_world": round(reference, 6),
            "quality_reference_q": round(reference, 6),
            "quality_reference_config": getattr(profile_quality, "reference_config", None),
            "quality_normalized": bool(getattr(profile_quality, "normalized", False)),
            "quality_factor_mean": round(statistics_mean(q_values) / reference, 6) if q_values else None,
            "cpr_proxy_time_weighted": round(cpr, 6) if cpr is not None else None,
            "quality_adjusted_cpr_proxy": round(quality_cpr, 6) if quality_cpr is not None else None,
            "definition": (
                "quality_adjusted_cpr_proxy = sum(playable_seconds * Q/Q_reference) / "
                "sum(engaged_seconds); Q is batch-invariant semantic quality and Q_reference is "
                "the explicit B1/S4/W18 point (legacy compact tables use a max-Q fallback)."
            ),
        }
        existing_cpr = summary.get("cpr_proxy")
        if isinstance(existing_cpr, dict):
            existing_cpr.update(
                {
                    "q_world_frame_weighted": phase_quality["q_world_frame_weighted"],
                    "quality_reference_q_world": phase_quality["quality_reference_q_world"],
                    "quality_adjusted_cpr": phase_quality["quality_adjusted_cpr_proxy"],
                }
            )
        summary["quality_cpr"] = phase_quality
        quality_cpr_by_phase.append({"phase": phase.get("phase"), **phase_quality})

    result["quality_cpr"] = {
        "schema_version": "abot_quality_cpr_v1",
        "profile_path": str(profile_path),
        "quality_reference_q_world": round(reference, 6),
        "quality_reference_q": round(reference, 6),
        "quality_reference_config": getattr(profile_quality, "reference_config", None),
        "quality_normalized": bool(getattr(profile_quality, "normalized", False)),
        "q_world_frame_weighted": round(total_quality_frames / total_frames, 6) if total_frames else None,
        "q_frame_weighted": round(total_quality_frames / total_frames, 6) if total_frames else None,
        "generated_frames_profiled": total_frames,
        "sessions_without_profile_quality": missing_sessions,
        "dispatch_diagnostics": dict(dispatch_diagnostics),
        "phases": quality_cpr_by_phase,
    }
    if producer_metrics is not None:
        result["paper_metrics"] = dict(producer_metrics)
        # One replay artifact currently contains one measured trace phase.
        # Mirror the compact paper block into that phase for existing report
        # tooling without replacing the client/transport diagnostic fields.
        phase_results = result.get("phase_results", [])
        if len(phase_results) == 1 and isinstance(phase_results[0], dict):
            summary = phase_results[0].get("summary")
            if isinstance(summary, dict):
                summary["paper_metrics"] = dict(producer_metrics)
    return result


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dispatch-trace", type=Path, required=True)
    parser.add_argument(
        "--action-trace",
        type=Path,
        help="Input lifecycle trace used to count released action requests for the SLO denominator.",
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("replay result must be a JSON object")
    profile_quality = load_profile_quality(args.profile)
    dispatch_quality, diagnostics = collect_dispatch_quality(args.dispatch_trace, profile_quality)
    trace_path_value = result.get("trace_path")
    action_trace = args.action_trace
    if action_trace is None and isinstance(trace_path_value, str):
        action_trace = Path(trace_path_value)
    released_action_jobs = (
        count_released_action_jobs(action_trace)
        if action_trace is not None and action_trace.is_file()
        else None
    )
    producer_metrics = collect_producer_metrics(
        args.dispatch_trace,
        profile_quality,
        released_action_jobs=released_action_jobs,
    )
    augmented = augment_result(
        result,
        dispatch_quality=dispatch_quality,
        profile_quality=profile_quality,
        dispatch_diagnostics=diagnostics,
        producer_metrics=producer_metrics,
        profile_path=args.profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(augmented, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(augmented["paper_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
