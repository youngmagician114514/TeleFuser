"""Backend-neutral metric primitives for ABot serving comparisons.

The LiveKit runner, a future non-LiveKit baseline, and offline replay tools
should agree on the arithmetic while remaining free to collect observations
through different transports.  This module intentionally contains no HTTP,
LiveKit, CUDA, or TeleFuser runtime imports.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return the nearest-rank percentile used by the ABot reports."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return float(ordered[index])


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Return the canonical count/mean/quantile summary for one metric series."""
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6) if values else 0.0,
        "p50": round(percentile(values, 0.50), 6),
        "p95": round(percentile(values, 0.95), 6),
        "p99": round(percentile(values, 0.99), 6),
        "maximum": round(max(values), 6) if values else 0.0,
    }


def summarize_slo(
    observations: Sequence[float],
    *,
    target_fps: float,
    tolerance_fps: float,
) -> dict[str, float | int]:
    """Count observations meeting an FPS threshold.

    The returned field names deliberately match the existing phase artifact
    suffixes.  A baseline can therefore emit the same report without knowing
    anything about the LiveKit client implementation.
    """
    target = float(target_fps)
    tolerance = float(tolerance_fps)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target_fps must be positive and finite")
    if not math.isfinite(tolerance) or tolerance < 0 or tolerance >= target:
        raise ValueError("tolerance_fps must be finite, non-negative, and smaller than target_fps")
    threshold = target - tolerance
    values = [float(value) for value in observations]
    hits = sum(value >= threshold for value in values)
    return {
        "observation_samples": len(values),
        "satisfied_samples": hits,
        "sample_attainment": round(hits / len(values), 6) if values else 0.0,
    }


__all__ = ["percentile", "summarize", "summarize_slo"]
