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

When the LiveKit runtime reports a session as running, the execution bridge
materializes an idle sentinel if no action state is held. The bridge forwards
that sentinel as a one-shot `control_state` with `controls=[]` and
`motivation.kind="idle"`; the ABot service treats this as a valid no-new-action
continuation and then clears the one-shot marker. A subsequent idle sentinel is
created only after the generated chunk reaches the publisher, so an action never
replaces an unconsumed idle video.

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

`MotivationRuntimeController` is an opt-in bridge around the policy core. A
runtime can construct it directly, or load the measured table and register
worker GPUs in one step:

```python
from telefuser.service.livekit import (
    GpuSchedulingState,
    MotivationRuntimeController,
)

controller = MotivationRuntimeController.from_offline_table(
    "/path/to/profile_evaluated.csv",
    gpu_states=[GpuSchedulingState("gpu-0", memory_free_gb=80.0)],
    dispatch=execute_lease,
)
```

Use `on_session_registered()` and `on_session_departed()` at the room
lifecycle boundary, then forward action and GPU events with `on_action()` and
`on_gpu_update()`. For a worker-start event after construction,
`on_gpu_registered()` provides the equivalent registration hook. The
controller starts at most one migration per scheduling turn, validates the
candidate again after ownership changes, and hands a `DispatchLease` to a
worker callback. The default ABot/SlackServe services are unchanged until a
runtime supplies this controller and an executor callback.

`MotivationExecutionBridge` is the opt-in execution adapter for the LiveKit
runtime. Construct a controller from the offline table with GPU identifiers
matching the LiveKit worker identifiers (`worker-0`, `worker-1`, ...), then pass
it as `motivation_controller=` to `LiveKitServeRuntime`. The
`telefuser stream-serve` entrypoint exposes the same wiring with
`--motivation-profile /path/to/profile_evaluated.csv`; it initializes one
scheduler GPU per configured worker and accepts `--motivation-max-batch-size`
and `--motivation-memory-free-gb` for the policy-side limits. The bridge
intercepts
normalized action messages, keeps only the latest unreleased controls, releases
one action job per configured heartbeat, and forwards a reserved batch through
`WorkerPool.dispatch_batch()`. A batch is applied atomically at the ABot
service boundary and carries the selected profile metadata. ABot marks that
control as a one-shot lease, so its worker-local scheduler emits one chunk
instead of free-running until the normal control timeout. The lease is
completed when every selected session emits a `chunk` model-output event; an
empty control state still reaches ABot to stop new admission, and `stop`/`reset`
retain their normal lifecycle behavior. Plain `process` workers are rejected
for this mode because their child-side control callback cannot be synchronously
intercepted; use `in-process` or `process-nccl`.

For ABot-World, the bridge now applies the selected offline `fidelity` at the
model boundary. Standard dense BF16 profiles (`rho0_bf16`) map S=4/3/2 to
`(0,1,2,3)`/`(0,2,3)`/`(0,3)` official sampler positions and W=18/12/6 to
the corresponding causal KV capacity (with sink frames W/3). Sessions in one
batch must carry the same fidelity. Changing W resizes the retained K/V rows
while preserving the chronological prefix and newest rolling tail. Dynamic
fidelity intentionally uses eager denoising; the fixed-shape CUDA Graph path is
invalidated for that dispatch and is retained for the ordinary default path.
Unsupported sparse-attention or non-BF16 profile names are rejected explicitly.

At runtime startup, `in-process` and `process-nccl` pools automatically provide
the controller with an asynchronous migration backend. A remote-GPU candidate
starts `LiveKitServeRuntime.migrate_session()` without blocking the serving
loop; completion wakes the controller, which revalidates ownership and searches
again before dispatch.

For overlap with an active GPU invocation, call `search_async()` at the
current completion boundary and later pass its result to
`dispatch_candidate()`. The background task only computes a snapshot;
reservation and dispatch remain synchronous and are rejected if the global
epoch, selected job version, owner, or GPU version changed. Call `close()`
when the owning runtime shuts down.

## Modular diagnostics

Candidate observability is injected through the `MotivationDiagnosticsSink`
protocol and is independent of policy, CUDA, LiveKit, and model code. The
provided `MotivationDiagnosticsCollector` keeps process-wide aggregate counts
and optionally a bounded recent window:

```python
from telefuser.service.livekit import MotivationDiagnosticsCollector

diagnostics = MotivationDiagnosticsCollector(
    recent_search_limit=64,
    recent_dispatch_limit=64,
)
controller = MotivationRuntimeController.from_offline_table(
    profile_path, gpu_states=gpus, dispatch=execute_lease, diagnostics=diagnostics
)
```

Each search records ready action/idle counts, candidate enumeration by
`B=1..4`, compatibility/profile/memory/migration/fairness filtering, feasible
candidates not selected by score, and the selected `(B,c,g)`. Each dispatch
records whether the candidate was accepted, waited, rejected as stale, deferred
for migration, or failed in the worker callback. The bounded snapshot is
available from `MotivationRuntimeController.diagnostics_snapshot()` and is
embedded under `MotivationExecutionBridge.snapshot()["diagnostics"]`. A custom
sink can stream the same summaries to an experiment trace or replace the
collector entirely for an ablation; the sink is best-effort and never changes
policy availability.
