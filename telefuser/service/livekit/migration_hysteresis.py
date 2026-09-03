"""Cooldown and hysteresis admission for session-state migrations.

SlackServe avoids re-homing a session repeatedly by keeping a per-session
cooldown after a committed move.  This module isolates that policy from the
scheduler and transport backend so it can be enabled, disabled, or replaced
for an ablation without changing migration mechanics.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

_EPSILON = 1e-9
_DEFAULT_RECENT_LIMIT = 64


def _finite_nonnegative(value: object, *, field: str) -> float:
    """Validate one policy timestamp/duration."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


@dataclass(frozen=True)
class MigrationAdmission:
    """Result of checking whether one proposed migration may start."""

    allowed: bool
    reason: str
    retry_at: float | None = None
    cooldown_remaining_seconds: float = 0.0


@dataclass
class _SessionMigrationState:
    """Bounded state retained for one session's last committed move."""

    session_id: str
    source_worker_id: str
    target_worker_id: str
    committed_at: float
    cooldown_until: float
    commit_count: int = 1
    suppressed_count: int = 0
    emergency_bypass_count: int = 0
    last_reason: str = "committed"


class MigrationCooldownPolicy:
    """Suppress migration thrashing with an optional per-session cooldown.

    A successful move starts a residence window for that session.  Until the
    window expires, a subsequent move is denied.  ``cooldown_seconds=0`` (or
    ``enabled=False``) disables the temporal guard, which makes this class
    convenient for an ablation while preserving one call site.

    ``min_gain_seconds`` is an optional score hysteresis margin.  Callers that
    can estimate the benefit of a proposed move may pass
    ``expected_gain_seconds`` to :meth:`admit`; callers without that estimate
    leave it as ``None`` and rely only on the cooldown.  An explicit
    ``emergency=True`` admission bypasses both guards for a session whose
    playout slack is unsafe; the caller owns that safety decision.
    """

    def __init__(
        self,
        *,
        cooldown_seconds: float = 60.0,
        min_gain_seconds: float = 0.0,
        enabled: bool = True,
        recent_limit: int = _DEFAULT_RECENT_LIMIT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cooldown_seconds = _finite_nonnegative(cooldown_seconds, field="cooldown_seconds")
        self.min_gain_seconds = _finite_nonnegative(min_gain_seconds, field="min_gain_seconds")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not isinstance(recent_limit, int) or isinstance(recent_limit, bool) or recent_limit < 0:
            raise ValueError("recent_limit must be a non-negative integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.enabled = enabled
        self._clock = clock
        self._sessions: dict[str, _SessionMigrationState] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=recent_limit)
        self._commit_count = 0
        self._suppressed_count = 0
        self._emergency_bypass_count = 0
        self._lock = RLock()

    def admit(
        self,
        session_id: str,
        source_worker_id: str,
        target_worker_id: str,
        *,
        now: float | None = None,
        expected_gain_seconds: float | None = None,
        emergency: bool = False,
    ) -> MigrationAdmission:
        """Return whether a proposed move is allowed at ``now``.

        The method is read/write only for policy telemetry; it never mutates
        worker ownership.  A caller must invoke :meth:`record_commit` only
        after the backend has successfully committed the move.
        """

        session_key = str(session_id)
        source = str(source_worker_id)
        target = str(target_worker_id)
        observed_at = self._observed(now)
        if source == target:
            return MigrationAdmission(False, "same_owner", retry_at=None)
        if not isinstance(emergency, bool):
            raise ValueError("emergency must be a boolean")

        gain = None
        if expected_gain_seconds is not None:
            gain = _finite_nonnegative(expected_gain_seconds, field="expected_gain_seconds")

        with self._lock:
            state = self._sessions.get(session_key)
            cooldown_active = (
                self.enabled
                and self.cooldown_seconds > _EPSILON
                and state is not None
                and observed_at < state.cooldown_until - _EPSILON
            )
            gain_blocked = (
                self.enabled
                and self.min_gain_seconds > _EPSILON
                and gain is not None
                and gain + _EPSILON < self.min_gain_seconds
            )
            if (cooldown_active or gain_blocked) and not emergency:
                reason = "cooldown" if cooldown_active else "insufficient_gain"
                retry_at = state.cooldown_until if cooldown_active and state is not None else None
                remaining = max(0.0, retry_at - observed_at) if retry_at is not None else 0.0
                self._suppressed_count += 1
                if state is not None:
                    state.suppressed_count += 1
                    state.last_reason = reason
                self._record(
                    {
                        "event": "suppressed",
                        "session_id": session_key,
                        "source_worker_id": source,
                        "target_worker_id": target,
                        "reason": reason,
                        "observed_at": observed_at,
                        "retry_at": retry_at,
                        "expected_gain_seconds": gain,
                    }
                )
                return MigrationAdmission(False, reason, retry_at, remaining)

            if (cooldown_active or gain_blocked) and emergency:
                self._emergency_bypass_count += 1
                if state is not None:
                    state.emergency_bypass_count += 1
                    state.last_reason = "emergency"
                self._record(
                    {
                        "event": "emergency_bypass",
                        "session_id": session_key,
                        "source_worker_id": source,
                        "target_worker_id": target,
                        "reason": "emergency",
                        "observed_at": observed_at,
                        "retry_at": state.cooldown_until if cooldown_active and state is not None else None,
                        "expected_gain_seconds": gain,
                    }
                )
                return MigrationAdmission(True, "emergency")

            reason = "disabled" if not self.enabled or self.cooldown_seconds <= _EPSILON else "allowed"
            self._record(
                {
                    "event": "allowed",
                    "session_id": session_key,
                    "source_worker_id": source,
                    "target_worker_id": target,
                    "reason": reason,
                    "observed_at": observed_at,
                    "retry_at": None,
                    "expected_gain_seconds": gain,
                }
            )
            return MigrationAdmission(True, reason)

    def record_commit(
        self,
        session_id: str,
        source_worker_id: str,
        target_worker_id: str,
        *,
        committed_at: float | None = None,
    ) -> None:
        """Start (or restart) the residence window after a successful move."""

        session_key = str(session_id)
        source = str(source_worker_id)
        target = str(target_worker_id)
        if source == target:
            raise ValueError("source_worker_id and target_worker_id must differ")
        observed_at = self._observed(committed_at)
        cooldown_until = observed_at + self.cooldown_seconds if self.enabled else observed_at
        with self._lock:
            previous = self._sessions.get(session_key)
            commit_count = previous.commit_count + 1 if previous is not None else 1
            state = _SessionMigrationState(
                session_id=session_key,
                source_worker_id=source,
                target_worker_id=target,
                committed_at=observed_at,
                cooldown_until=cooldown_until,
                commit_count=commit_count,
                suppressed_count=previous.suppressed_count if previous is not None else 0,
                emergency_bypass_count=previous.emergency_bypass_count if previous is not None else 0,
            )
            self._sessions[session_key] = state
            self._commit_count += 1
            self._record(
                {
                    "event": "commit",
                    "session_id": session_key,
                    "source_worker_id": source,
                    "target_worker_id": target,
                    "reason": "committed",
                    "observed_at": observed_at,
                    "cooldown_until": cooldown_until,
                }
            )

    def forget(self, session_id: str) -> None:
        """Drop state when a session leaves the serving system."""

        with self._lock:
            self._sessions.pop(str(session_id), None)

    def cooldown_until(self, session_id: str) -> float | None:
        """Return the current cooldown deadline, if a commit was recorded."""

        with self._lock:
            state = self._sessions.get(str(session_id))
            return state.cooldown_until if state is not None else None

    def is_blocked(self, session_id: str, *, now: float | None = None) -> bool:
        """Return whether a session is currently inside its residence window.

        This read-only predicate lets a scheduler exclude only migration
        alternatives for a cooled-down session while still allowing work to
        execute on its current owner. It intentionally does not record a
        suppression event; admission telemetry is recorded by :meth:`admit`
        when a concrete source/target pair is considered.
        """

        observed_at = self._observed(now)
        with self._lock:
            state = self._sessions.get(str(session_id))
            return bool(
                self.enabled
                and self.cooldown_seconds > _EPSILON
                and state is not None
                and observed_at < state.cooldown_until - _EPSILON
            )

    def blocked_session_ids(
        self,
        session_ids: list[str] | tuple[str, ...] | None = None,
        *,
        now: float | None = None,
    ) -> tuple[str, ...]:
        """Return sorted IDs whose migration residence window is active."""

        observed_at = self._observed(now)
        with self._lock:
            candidates = tuple(self._sessions) if session_ids is None else tuple(map(str, session_ids))
            return tuple(
                sorted(
                    session_id
                    for session_id in candidates
                    if self.enabled
                    and self.cooldown_seconds > _EPSILON
                    and (state := self._sessions.get(session_id)) is not None
                    and observed_at < state.cooldown_until - _EPSILON
                )
            )

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        """Return JSON-compatible, bounded policy telemetry."""

        observed_at = self._observed(now)
        with self._lock:
            active = []
            for state in self._sessions.values():
                remaining = max(0.0, state.cooldown_until - observed_at)
                active.append(
                    {
                        "session_id": state.session_id,
                        "source_worker_id": state.source_worker_id,
                        "target_worker_id": state.target_worker_id,
                        "committed_at": state.committed_at,
                        "cooldown_until": state.cooldown_until,
                        "cooldown_remaining_seconds": remaining,
                        "cooldown_active": remaining > _EPSILON,
                        "commit_count": state.commit_count,
                        "suppressed_count": state.suppressed_count,
                        "emergency_bypass_count": state.emergency_bypass_count,
                        "last_reason": state.last_reason,
                    }
                )
            active.sort(key=lambda item: item["session_id"])
            return {
                "enabled": self.enabled and self.cooldown_seconds > _EPSILON,
                "cooldown_seconds": self.cooldown_seconds,
                "min_gain_seconds": self.min_gain_seconds,
                "tracked_sessions": len(self._sessions),
                "active_cooldowns": sum(bool(item["cooldown_active"]) for item in active),
                "commits_total": self._commit_count,
                "suppressed_total": self._suppressed_count,
                "emergency_bypasses_total": self._emergency_bypass_count,
                "sessions": active,
                "recent": [dict(item) for item in self._recent],
            }

    def _record(self, event: dict[str, Any]) -> None:
        self._recent.append(dict(event))

    def _observed(self, now: float | None) -> float:
        value = self._clock() if now is None else now
        return _finite_nonnegative(value, field="now")


__all__ = ["MigrationAdmission", "MigrationCooldownPolicy"]
