# Motivation scheduler control plane

`telefuser.service.livekit.motivation_scheduler` is the policy-side scheduler
for the ABot-World motivation.  It is deliberately independent of CUDA and
LiveKit transport so that the same candidate search can be tested with the
offline profile table and then driven by a real worker adapter.

Measured ABot rows can be loaded directly with:

```python
from telefuser.service.livekit import load_motivation_profiles_csv

profiles = load_motivation_profiles_csv(
    "/path/to/profile_evaluated.csv",
    max_batch_size=4,
    output_seconds=1.0,
)
```

The loader reads `B`, `latency_ms`, `latency_p95_ms`, `memory_GB`, and
`Q_world` from the offline table; when `Q_world` is empty it averages the
available component quality columns. Rows with `B > 4` are ignored by the
default policy bound.

## State and job semantics

The control plane keeps one `SessionSchedulingState` per retained session:

- one pending action job (a newer released action replaces the older pending
  state);
- an in-flight job that is never cancelled by a later action;
- one pending idle sentinel, plus the duration of an idle video not yet
  consumed;
- playout slack, playback state, quality EMA, owner GPU, and migration state.

An action update with no release (for example, an intermediate state change in
the one-second heartbeat window) changes the latest controls but creates no
job.  Action jobs are considered before idle jobs globally.  An idle sentinel
cannot be regenerated until its generated video has been consumed.

A release that replaces an existing pending action only changes that session's
job version; it does not invalidate an unrelated global search.  If there is no
pending action, the new action creates a ready slot and invalidates the global
candidate, even when an older job is still in flight.  The in-flight job is
never cancelled.

`SessionSchedulingState.advance_to()` consumes slack independently of the
latest controls.  Once frames enter the consumer queue, releasing the controls
does not stop playback.  An explicitly paused consumer can set
`playback_active=False`.

## Candidate search

`MotivationScheduler.find_best()` enumerates, for every available GPU:

1. one ready job per session;
2. all batch sizes from one through `max_batch_size` (capped at four);
3. only batches with the same compatibility key;
4. every fidelity row in the offline profile provider;
5. migration readiness and target memory constraints;
6. a deliberate wait candidate.

The score uses the simulator's `U(P)=min(P, cap)` utility over every
non-departed session.  The candidate duration includes the predicted GPU free
time and migration-ready time.  Selected sessions receive the profile's
output duration and their quality EMA is updated with the profile's offline
quality.  The fairness check uses the predicted system mean quality and
`fairness_delta`.

The returned candidate captures the scheduler epoch, per-session state
versions, and GPU version.  `reserve()` validates all of them, so a search run
asynchronously while a GPU is computing cannot dispatch stale action state or
ownership.

## Asynchronous migration

`async_migration.AsyncMigrationManager` exposes a backend-neutral state machine:

```text
precopied -> ready -> committing -> completed
                         └────────> failed
precopied/ready ────────> aborted
```

A true backend can pre-copy KV state while the source computes and perform a
short final delta copy at a chunk boundary.  `RouterMigrationBackend` is a
compatibility adapter for the current blocking `TurboServePipelineRouter`:
the router operation runs in a private thread, so the scheduler is not
blocked, while the router still preserves its existing boundary-safe
quiesce/import/ownership-commit protocol.

The next runtime integration layer should feed migration estimates into
`MigrationEstimator`, reserve target memory, start the transfer during the
current batch, and call `commit()` only after the target reports `ready`.
## Runtime bridge

`MotivationRuntimeController` is an opt-in bridge around the policy core. It
forwards action and GPU events, starts at most one migration per scheduling
turn, validates the candidate again after ownership changes, and hands a
`DispatchLease` to a worker callback. The default ABot/SlackServe services are
unchanged until a runtime supplies this controller and an executor callback.

For overlap with an active GPU invocation, call `search_async()` at the
current completion boundary and later pass its result to
`dispatch_candidate()`. The background task only computes a snapshot;
reservation and dispatch remain synchronous and are rejected if the global
epoch, selected job version, owner, or GPU version changed. Call `close()`
when the owning runtime shuts down.
