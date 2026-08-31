#!/usr/bin/env python3
"""Replay action-only ABot traces through the public LiveKit interfaces.

The input is the action-only trace emitted by ``abot_world_data_harness``.
Session/job admission and job formation remain server-side; this client only
replays user session lifecycle and control-state messages while consuming the
real WebRTC video output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.validation import benchmark_abot_livekit_burst as wave

_TRACE_KINDS = frozenset({"session_arrival", "user_active", "action_update", "user_idle", "session_departure"})


@dataclass(frozen=True)
class ActionTraceEvent:
    """One action-only user event."""

    time_seconds: float
    sequence: int
    kind: str
    session_id: str
    data: Mapping[str, Any]


@dataclass(frozen=True)
class ActionTrace:
    """Validated trace and its source metadata."""

    name: str
    duration_seconds: float
    events: tuple[ActionTraceEvent, ...]
    ignored_event_kinds: Mapping[str, int]


class ActionTraceError(ValueError):
    """Raised when a trace cannot be replayed safely."""


def _number(value: object, label: str, *, allow_zero: bool = True) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ActionTraceError(f"{label} must be numeric")
    parsed = float(value)
    if parsed < 0 or (not allow_zero and parsed == 0):
        raise ActionTraceError(f"{label} must be {'non-negative' if allow_zero else 'positive'}")
    return parsed


def _load_trace(path: Path, *, ignore_unsupported: bool = True) -> ActionTrace:
    """Load and validate an action-only JSONL trace."""
    events: list[ActionTraceEvent] = []
    ignored: dict[str, int] = {}
    previous_key = (-1.0, -1)
    arrivals: set[str] = set()
    active: set[str] = set()
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ActionTraceError(f"Invalid JSON at line {index + 1}") from exc
        if not isinstance(raw, Mapping):
            raise ActionTraceError(f"Trace line {index + 1} must be an object")
        kind = raw.get("kind")
        if not isinstance(kind, str):
            raise ActionTraceError(f"Trace line {index + 1} has no string kind")
        if kind not in _TRACE_KINDS:
            if ignore_unsupported:
                ignored[kind] = ignored.get(kind, 0) + 1
                continue
            raise ActionTraceError(f"Unsupported trace kind {kind!r}")
        event_time = _number(raw.get("time"), f"trace line {index + 1}.time")
        sequence = raw.get("sequence", len(events))
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise ActionTraceError(f"Trace line {index + 1}.sequence must be a non-negative integer")
        key = (event_time, sequence)
        if key <= previous_key:
            raise ActionTraceError("Trace events must be strictly ordered by (time, sequence)")
        previous_key = key
        session_id = raw.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ActionTraceError(f"Trace line {index + 1}.session_id must be non-empty")
        data = raw.get("data", {})
        if not isinstance(data, Mapping):
            raise ActionTraceError(f"Trace line {index + 1}.data must be an object")
        if kind == "session_arrival":
            if session_id in arrivals:
                raise ActionTraceError(f"Session {session_id!r} arrived twice")
            arrivals.add(session_id)
        elif kind == "session_departure":
            if session_id not in arrivals:
                raise ActionTraceError(f"Session {session_id!r} departed before arrival")
            active.discard(session_id)
        elif kind == "user_active":
            if session_id not in arrivals or session_id in active:
                raise ActionTraceError(f"Session {session_id!r} has invalid user_active transition")
            active.add(session_id)
        elif kind == "user_idle":
            if session_id not in arrivals or session_id not in active:
                raise ActionTraceError(f"Session {session_id!r} has invalid user_idle transition")
            active.remove(session_id)
        elif kind == "action_update":
            if session_id not in arrivals:
                raise ActionTraceError(f"Action for {session_id!r} arrived before session_arrival")
            wire_controls = data.get("wire_controls")
            if not isinstance(wire_controls, list) or not all(isinstance(item, str) for item in wire_controls):
                message = data.get("message")
                wire_controls = message.get("controls") if isinstance(message, Mapping) else None
            if not isinstance(wire_controls, list) or not all(isinstance(item, str) for item in wire_controls):
                raise ActionTraceError(f"Action at line {index + 1} has no string wire_controls")
        events.append(
            ActionTraceEvent(
                time_seconds=event_time,
                sequence=sequence,
                kind=kind,
                session_id=session_id,
                data=dict(data),
            )
        )
    if not events:
        raise ActionTraceError("Trace is empty")
    if active:
        raise ActionTraceError(f"Trace ended with active user input: {sorted(active)[:3]}")
    if arrivals != {event.session_id for event in events if event.kind == "session_departure"}:
        raise ActionTraceError("Every arrived session must have one session_departure")
    duration = max(event.time_seconds for event in events)
    return ActionTrace(
        name=path.parent.name,
        duration_seconds=duration,
        events=tuple(events),
        ignored_event_kinds=ignored,
    )


class ActionTraceSession(wave.LiveKitWaveSession):
    """Browser-equivalent session that sends trace controls instead of random controls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trace_actions_seen = 0
        self.trace_actions_published = 0
        self.trace_actions_deferred = 0
        self.trace_actions_dropped = 0
        self._deferred_controls: tuple[str, ...] | None = None

    def _start_control_task(self) -> None:
        """Disable the synthetic random-control task from the parent runner."""
        return

    async def _connect_room(self, livekit_url: str, token: str) -> None:
        await super()._connect_room(livekit_url, token)
        if self._deferred_controls is not None and not self.stop_requested:
            controls = self._deferred_controls
            self._deferred_controls = None
            await self._publish_trace_controls(controls, source="deferred_latest")

    def apply_trace_input_transition(self, enabled: bool, *, reason: str) -> None:
        """Mirror harness input transitions without synthesizing a control."""
        if self.stop_requested or self.input_enabled == enabled:
            return
        now = time.perf_counter()
        if enabled:
            self.input_resumes += 1
            paused_for = (
                max(0.0, now - self.input_pause_started_at)
                if self.input_pause_started_at is not None
                else None
            )
            self.input_pause_started_at = None
            self.input_enabled = True
            self.record_event(
                "trace_input_resumed",
                session=self.logical_id,
                reason=reason,
                paused_seconds=round(paused_for, 6) if paused_for is not None else None,
            )
        else:
            self.input_pauses += 1
            self.input_pause_started_at = now
            self.input_enabled = False
            self.record_event("trace_input_paused", session=self.logical_id, reason=reason)

    async def publish_trace_action(self, data: Mapping[str, Any]) -> None:
        """Publish one trace state, retaining only the latest pre-connect state."""
        raw_controls = data.get("wire_controls")
        if not isinstance(raw_controls, list):
            message = data.get("message")
            raw_controls = message.get("controls") if isinstance(message, Mapping) else None
        if not isinstance(raw_controls, list) or not all(isinstance(item, str) for item in raw_controls):
            raise ActionTraceError(f"{self.logical_id} action has invalid wire_controls")
        controls = tuple(raw_controls)
        self.trace_actions_seen += 1
        if self.stop_requested:
            self.trace_actions_dropped += 1
            return
        if not self.connected or self._room is None:
            self._deferred_controls = controls
            self.trace_actions_deferred += 1
            return
        await self._publish_trace_controls(controls, source="trace")

    async def _publish_trace_controls(self, controls: tuple[str, ...], *, source: str) -> None:
        await self._publish_control_state(controls)
        self.trace_actions_published += 1
        self.record_event(
            "trace_action_published",
            session=self.logical_id,
            controls=list(controls),
            source=source,
        )

    def snapshot(self, now: float) -> dict[str, Any]:
        snapshot = super().snapshot(now)
        snapshot.update(
            {
                "trace_actions_seen": self.trace_actions_seen,
                "trace_actions_published": self.trace_actions_published,
                "trace_actions_deferred": self.trace_actions_deferred,
                "trace_actions_dropped": self.trace_actions_dropped,
            }
        )
        return snapshot


class ActionTraceRunner(wave.LiveKitWaveRunner):
    """Replay exact user events while retaining the standard video monitor."""

    def __init__(self, scenario: wave.Scenario, trace: ActionTrace, *, drain_seconds: float) -> None:
        super().__init__(scenario)
        self.trace = trace
        self.drain_seconds = drain_seconds
        self._trace_sessions: dict[str, ActionTraceSession] = {}
        self._trace_tasks: set[asyncio.Task[None]] = set()
        self._trace_phase = wave.Phase(
            name=f"action_trace:{trace.name}",
            duration_seconds=trace.duration_seconds,
            target_users=0,
        )

    def _spawn_trace_task(self, coroutine: Any) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        self._trace_tasks.add(task)
        self._background_tasks.add(task)

        def _done(completed: asyncio.Task[None]) -> None:
            self._trace_tasks.discard(completed)
            self._background_tasks.discard(completed)

        task.add_done_callback(_done)
        return task

    async def _start_trace_session(self, session: ActionTraceSession) -> None:
        try:
            await asyncio.wait_for(session.start(), timeout=self.scenario.connect_timeout_seconds)
        except asyncio.TimeoutError:
            session.error = f"SessionStartTimeout after {self.scenario.connect_timeout_seconds:g}s"
            self.record_event("session_start_timeout", session=session.logical_id, error=session.error)

    async def _stop_trace_session(self, session: ActionTraceSession) -> None:
        await session.stop()

    async def _handle_event(self, event: ActionTraceEvent) -> None:
        if event.kind == "session_arrival":
            if event.session_id in self._trace_sessions:
                raise ActionTraceError(f"Session {event.session_id!r} arrived twice at replay")
            session = ActionTraceSession(
                index=len(self._sessions),
                scenario=self.scenario,
                http=self._http,
                rtc=self.rtc,
                record_event=self.record_event,
                started_at=self.started_at,
                trace_session_id=event.session_id,
            )
            session.scheduled_at = time.perf_counter()
            # Preserve the harness admission state in the client snapshot;
            # controls themselves remain an exact replay of the trace.
            input_enabled = event.data.get("input_enabled")
            if isinstance(input_enabled, bool):
                session.input_enabled = input_enabled
            self._sessions.append(session)
            self._trace_sessions[event.session_id] = session
            self.record_event("trace_session_arrival", session=event.session_id, trace_time=event.time_seconds)
            self._spawn_trace_task(self._start_trace_session(session))
            return
        session = self._trace_sessions.get(event.session_id)
        if session is None:
            raise ActionTraceError(f"Trace event references unknown session {event.session_id!r}")
        if event.kind == "action_update":
            await session.publish_trace_action(event.data)
            return
        if event.kind == "session_departure":
            del self._trace_sessions[event.session_id]
            session.departure_scheduled = True
            self.record_event("trace_session_departure", session=event.session_id, trace_time=event.time_seconds)
            self._spawn_trace_task(self._stop_trace_session(session))
            return
        session.apply_trace_input_transition(event.kind == "user_active", reason=event.kind)
        self.record_event(
            "trace_input_transition",
            session=event.session_id,
            source_event=event.kind,
            trace_time=event.time_seconds,
        )

    async def run(self) -> dict[str, Any]:
        """Run the exact trace and return the normal LiveKit result schema."""
        wave._disable_proxy_for_loopback(self.scenario.server_url)
        timeout = wave.httpx.Timeout(self.scenario.http_timeout_seconds)
        limits = wave.httpx.Limits(max_keepalive_connections=0)
        async with wave.httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as http:
            self._http = http
            self.started_at = time.perf_counter()
            self._trace_started_monotonic_seconds = time.monotonic()
            self._trace_started_unix_seconds = time.time()
            await self._capture_server_metadata("before_workload")
            self._phase_name = self._trace_phase.name
            self._phase_target_users = max(
                sum(1 for event in self.trace.events if event.kind == "session_arrival"),
                1,
            )
            self._phase_active_input_fraction = 1.0
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor(), name="abot-action-trace-monitor")
            try:
                self.record_event(
                    "action_trace_started",
                    trace_name=self.trace.name,
                    duration_seconds=self.trace.duration_seconds,
                    event_count=len(self.trace.events),
                    ignored_event_kinds=dict(self.trace.ignored_event_kinds),
                )
                for event in self.trace.events:
                    remaining = event.time_seconds - (time.perf_counter() - self.started_at)
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    await self._handle_event(event)
                if self.drain_seconds > 0:
                    await asyncio.sleep(self.drain_seconds)
                if self._trace_tasks:
                    await asyncio.gather(*tuple(self._trace_tasks), return_exceptions=True)
            finally:
                self._monitoring = False
                if self._monitor_task is not None:
                    self._monitor_task.cancel()
                    await asyncio.gather(self._monitor_task, return_exceptions=True)
                await self._stop_all_sessions()
                await self._capture_server_metadata("after_workload")
        completed_at = time.perf_counter()
        phase_completed = completed_at
        phase_result = self._summarize_phase(
            self._trace_phase,
            phase_started=self.started_at,
            phase_completed=phase_completed,
            samples=self._samples,
        )
        self._phase_results = [phase_result]
        self.record_event("action_trace_completed", event_count=len(self.trace.events))
        return {
            "schema_version": "abot_livekit_action_trace_replay_v1",
            "trace_path": str(self.trace.name),
            "trace": {
                "name": self.trace.name,
                "duration_seconds": self.trace.duration_seconds,
                "event_count": len(self.trace.events),
                "ignored_event_kinds": dict(self.trace.ignored_event_kinds),
            },
            "trace_clock": {
                "offset_clock": "time.perf_counter",
                "origin_performance_counter_seconds": round(self.started_at, 9),
                "origin_monotonic_seconds": round(self._trace_started_monotonic_seconds, 9),
                "origin_unix_seconds": round(self._trace_started_unix_seconds, 9),
            },
            "elapsed_seconds": round(completed_at - self.started_at, 6),
            "server_metadata": self._server_metadata,
            "warnings": self._warnings,
            "phase_results": self._phase_results,
            "sessions": [session.snapshot(completed_at) for session in self._sessions],
            "samples": self._samples,
            "events": self._events,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Action-only events.jsonl")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("tools/validation/workloads/abot_livekit_4gpu_lf3_12fps_wave.json"),
        help="Base LiveKit scenario for prompt, image and client timeouts",
    )
    parser.add_argument("--output", type=Path, help="Replay result JSON")
    parser.add_argument("--server-url", help="Override scenario.server_url")
    parser.add_argument("--drain-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    trace = _load_trace(args.trace.expanduser())
    scenario_path = args.scenario.expanduser()
    if not scenario_path.is_absolute():
        scenario_path = (_REPO_ROOT / scenario_path).resolve()
    scenario = wave.load_scenario(scenario_path, server_url_override=args.server_url)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "trace": trace.name,
                    "duration_seconds": trace.duration_seconds,
                    "event_count": len(trace.events),
                    "ignored_event_kinds": dict(trace.ignored_event_kinds),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output is None:
        raise SystemExit("--output is required unless --dry-run is used")
    if args.drain_seconds < 0:
        raise SystemExit("--drain-seconds must be non-negative")
    output = args.output.expanduser()
    if not output.is_absolute():
        output = (_REPO_ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(ActionTraceRunner(scenario, trace, drain_seconds=args.drain_seconds).run())
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wave._print_summary(result)
    print(f"Wrote complete artifact: {output}")


if __name__ == "__main__":
    main()
