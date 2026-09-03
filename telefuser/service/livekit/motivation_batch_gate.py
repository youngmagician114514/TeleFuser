"""Profile-driven batch formation wait calculations for Motivation scheduling.

The Motivation policy chooses a batch and fidelity, while an execution adapter
may briefly wait for another compatible action before dispatching a singleton.
This module contains only the profile geometry for that decision.  It does not
own timers, queues, or transport state, which keeps the policy easy to test and
allows a caller to apply its own deadline and version checks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .motivation_scheduler import MotivationProfileProvider


@dataclass(frozen=True)
class BatchFormationWindow:
    """A measured B1/B2 pair and its throughput-neutral waiting window.

    ``wait_seconds`` is the amount of time that can be spent collecting a
    second compatible job while preserving the serial B1 work budget:

    ``max(0, 2 * latency(B1) - latency(B2))``.

    ``fidelity`` is ``None`` when no common B1/B2 profile exists.  In that case
    the window is zero and callers should dispatch without profile-driven
    aggregation delay.
    """

    gpu_id: str
    fidelity: str | None
    b1_latency_seconds: float | None
    b2_latency_seconds: float | None
    wait_seconds: float

    def __post_init__(self) -> None:
        if not self.gpu_id:
            raise ValueError("gpu_id must be non-empty")
        if self.wait_seconds < 0 or not math.isfinite(self.wait_seconds):
            raise ValueError("wait_seconds must be finite and non-negative")
        for name in ("b1_latency_seconds", "b2_latency_seconds"):
            value = getattr(self, name)
            if value is not None and (value <= 0 or not math.isfinite(value)):
                raise ValueError(f"{name} must be positive and finite when supplied")
        if self.fidelity is None and self.wait_seconds != 0:
            raise ValueError("a missing fidelity cannot have a positive wait window")


def geometric_batch_wait_seconds(
    b1_latency_seconds: float,
    b2_latency_seconds: float,
    *,
    max_wait_seconds: float | None = None,
) -> float:
    """Return the throughput-neutral wait for a possible second job.

    The caller supplies latency in seconds.  A B2 invocation is useful only
    when its latency is lower than two serialized B1 invocations; otherwise
    the safe window is zero.  ``max_wait_seconds`` is an optional caller-side
    bound for deadline/slack policy and is applied after the geometric value.
    """
    _validate_latency("b1_latency_seconds", b1_latency_seconds)
    _validate_latency("b2_latency_seconds", b2_latency_seconds)
    cap = _validate_cap(max_wait_seconds)
    window = max(0.0, 2.0 * float(b1_latency_seconds) - float(b2_latency_seconds))
    return window if cap is None else min(window, cap)


class MotivationBatchGate:
    """Look up B1/B2 profile geometry without owning scheduling state.

    Profiles are matched by fidelity family so a caller never derives a window
    from two different quality configurations.  The ABot offline table names
    rows with a batch-size prefix (for example, ``b1_s4_w18`` and
    ``b2_s4_w18``); that prefix is ignored when pairing B1 and B2.  If
    ``fidelity`` is omitted, the largest window among common families is
    returned; this is useful before the policy has committed to a fidelity.
    Once a candidate fidelity is known, callers should pass it to avoid
    waiting longer than that candidate
    can justify.
    """

    def __init__(
        self,
        profile_provider: MotivationProfileProvider,
        *,
        max_wait_seconds: float | None = None,
    ) -> None:
        if not hasattr(profile_provider, "profiles_for"):
            raise TypeError("profile_provider must expose profiles_for")
        self._profile_provider = profile_provider
        self._max_wait_seconds = _validate_cap(max_wait_seconds)

    @property
    def max_wait_seconds(self) -> float | None:
        """Return the optional caller-supplied upper bound."""
        return self._max_wait_seconds

    def window(self, *, gpu_id: str, fidelity: str | None = None) -> BatchFormationWindow:
        """Return the B1/B2 window for ``gpu_id`` and an optional fidelity."""
        if not gpu_id:
            raise ValueError("gpu_id must be non-empty")
        b1_profiles = self._profiles(batch_size=1, gpu_id=gpu_id)
        b2_profiles = self._profiles(batch_size=2, gpu_id=gpu_id)
        b1_by_fidelity = _profiles_by_family(b1_profiles)
        b2_by_fidelity = _profiles_by_family(b2_profiles)

        if fidelity is not None:
            family = _fidelity_family(fidelity)
            if family not in b1_by_fidelity or family not in b2_by_fidelity:
                return BatchFormationWindow(gpu_id, None, None, None, 0.0)
            selected = (
                b1_by_fidelity[family][0],
                b1_by_fidelity[family][1],
                b2_by_fidelity[family][1],
            )
        else:
            common = sorted(set(b1_by_fidelity).intersection(b2_by_fidelity))
            if not common:
                return BatchFormationWindow(gpu_id, None, None, None, 0.0)
            selected = max(
                (
                    (
                        b1_by_fidelity[name][0],
                        b1_by_fidelity[name][1],
                        b2_by_fidelity[name][1],
                    )
                    for name in common
                ),
                key=lambda item: geometric_batch_wait_seconds(
                    item[1].latency_seconds,
                    item[2].latency_seconds,
                    max_wait_seconds=self._max_wait_seconds,
                ),
            )

        selected_fidelity, b1_profile, b2_profile = selected
        wait_seconds = geometric_batch_wait_seconds(
            b1_profile.latency_seconds,
            b2_profile.latency_seconds,
            max_wait_seconds=self._max_wait_seconds,
        )
        return BatchFormationWindow(
            gpu_id=gpu_id,
            fidelity=selected_fidelity,
            b1_latency_seconds=b1_profile.latency_seconds,
            b2_latency_seconds=b2_profile.latency_seconds,
            wait_seconds=wait_seconds,
        )

    def wait_seconds(self, *, gpu_id: str, fidelity: str | None = None) -> float:
        """Return only the profile-derived wait duration in seconds."""
        return self.window(gpu_id=gpu_id, fidelity=fidelity).wait_seconds

    def _profiles(self, *, batch_size: int, gpu_id: str):
        profiles = tuple(self._profile_provider.profiles_for(batch_size=batch_size, gpu_id=gpu_id))
        for profile in profiles:
            _validate_latency(f"{profile.fidelity} B{batch_size} latency", profile.latency_seconds)
        return profiles


def _fidelity_family(fidelity: str) -> str:
    """Return a batch-size-independent key for an offline fidelity name."""
    value = str(fidelity).strip()
    # Profile rows conventionally start with ``b<batch>_``. Keep arbitrary
    # names (for example ``high``) unchanged for compact hand-authored tables.
    return re.sub(r"^b\d+(?:[_-])?", "", value, count=1, flags=re.IGNORECASE)


def _profiles_by_family(
    profiles: tuple,
) -> dict[str, tuple[str, object]]:
    """Index profiles by family while retaining their original names."""
    indexed: dict[str, tuple[str, object]] = {}
    for profile in profiles:
        family = _fidelity_family(profile.fidelity)
        # Keep the first deterministic row if an external provider is malformed.
        indexed.setdefault(family, (profile.fidelity, profile))
    return indexed


def _validate_latency(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _validate_cap(value: float | None) -> float | None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError("max_wait_seconds must be finite and non-negative when supplied")
    return None if value is None else float(value)
