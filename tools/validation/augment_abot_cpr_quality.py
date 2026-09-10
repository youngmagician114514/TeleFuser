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
        values = [float(row[key]) for key in ("Q_action", "Q_temporal", "Q_visual") if row.get(key, "") not in {None, ""}]
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
    return result


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dispatch-trace", type=Path, required=True)
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
    augmented = augment_result(
        result,
        dispatch_quality=dispatch_quality,
        profile_quality=profile_quality,
        dispatch_diagnostics=diagnostics,
        profile_path=args.profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(augmented, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(augmented["quality_cpr"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
