# Daemon Runtime State Map

## Purpose

The daemon runtime state surface bridges daemon protocol handlers to the kernel agent manager and persisted session stores. It owns runtime task submission, live session control, sidestep execution, session listing/search/detail/deletion, linked-family topology and archival, and branch operations.

This subsystem should preserve three guarantees:

- daemon APIs keep live runtime operations and persisted session reads clearly separated
- bare persisted session references resolve against the primary `state` store unless explicitly qualified
- live branch operations refuse ambiguous or busy runtime slots

## Files

- `src/daemon/state/runtime_tasks.rs`
  - Task submit/wait/cancel/promote, sidestep tasks, live session open/resume/cancel/kill, channel persistence/inference lookup, and live-session filtering.
- `src/daemon/state/runtime_sessions.rs`
  - Persisted session list/search/detail/title, on-demand turn-topology projection,
    branch listing/sibling lookup/create/checkout, durable deletion,
    persisted-session target resolution, and live mutation guards.
- `src/daemon/state/runtime_sessions/family.rs`
  - Linked-family topology/runtime projection and idle subtree archival.
- `src/daemon/state/harness_actions.rs`
  - Harness action runtime targeting, agent execution identity resolution, and action result collection.
- `src/daemon/state/harness_sources.rs`
  - Harness source listing/reading, candidate validation, safe relative-path resolution, hash-guarded batch saves, and file-atomic replacement.
- `src/daemon/server/dispatch/task.rs`
  - Daemon task request handlers.
- `src/daemon/server/dispatch/session.rs`
  - Daemon session request handlers.
- `src/daemon/server/events.rs`
  - Runtime event subscription loop, initial snapshot emission, session kernel-event forwarding, and task-update polling.
- `src/daemon/server/events/filter.rs`
  - Event subscription filters for agent/session/slot scoped streams.
- `src/daemon/server/events/scope.rs`
  - Scoped runtime snapshot projection and registry issue filtering.
- `src/perf_diagnostics.rs`
  - Compile-time optional operation stages, session scope propagation,
    daemon-process memory sampling, and enriched event emission.
- `src/kernel/agent_manager/operations.rs`
  - Live runtime/session/task operations called by this layer.
- `src/persistence/state/*`
  - Persisted session, branch, message, event, and tool execution storage.

## Data Flow

Task submission:

1. Daemon task handlers build `SubmitTaskParams` or `SidestepTaskParams`.
2. `runtime_tasks.rs` converts request params into `QueuedTask` and execution overrides.
3. Agent-manager methods enqueue the work by agent id or live session target.
4. `wait_for_task` polls the targeted task snapshot until completion or timeout; it must not materialize the full task list for each wait.
5. Runtime-wide task snapshots include a bounded human-readable prompt preview
   and optional task title alongside trace, slot, execution policy, and outcome.

Live session open/resume:

1. The daemon layer validates the target agent or resolves the persisted session channel id.
2. Channel-specific state/default-store selectors and inference overrides are resolved from registry data.
3. Agent-manager methods open or resume the runtime slot.
4. The caller receives the agent-manager `LiveSessionSnapshot`.

Harness action execution:

1. An explicit `harness_id` selects that harness's Lua runtime rather than deriving a runtime from the execution agent.
2. An explicit `agent_id` supplies execution and governance identity after binding validation.
3. Without an explicit agent, Turin uses the sole bound agent or the primary agent identity for an unbound shared harness.
4. Harnesses bound to multiple agents require the caller to choose an agent.

Harness source editing:

1. Source paths are normalized relative to the registered harness root and limited to Lua files.
2. Candidate validation layers every dirty source addition, replacement, or deletion over the current filesystem and builds the normal disposable harness VM.
3. Imports and `use` blocks consult that same overlay, so unsaved nested modules validate coherently.
4. Saving compares every current file hash before staging any replacement. New files require an absent expected hash; existing files require the hash returned by the read API.
5. Each replacement is staged beside its target and renamed into place. Saving does not validate, reload, or make the daemon the source of truth; the normal filesystem watcher owns activation.

Persisted session detail:

1. `runtime_sessions.rs` resolves the session reference into store selector and public UUID.
2. A full request loads session row, branches, events, messages, and tool executions.
3. An optional message limit projects the recent transcript, skips events when
   requested, and returns window offset/total metadata. Matching tool
   executions are limited to turns represented in that message window.
4. An optional absolute message offset selects an older bounded window. Window
   boundaries expand to complete turn groups so a tool cycle is not split.
5. Full and bounded detail requests independently load only
   `inference_request`, `message_end`, and `context_compaction` events to build
   a bounded request-efficiency projection. High-volume stream deltas are
   filtered in SQL rather than materialized for this projection.
6. Rows are converted into daemon-facing detail structs. Message projections
   retain their exact durable turn id as well as path-relative turn index so
   clients can perform exact contextual branch operations without loading the
   complete turn graph. Per-turn request
   estimates are paired with provider-reported input/output and optional
   prompt-cache usage when present, while older sessions remain valid without
   request telemetry or cache counters.
7. Session detail also projects the newest bounded task, plan, and turn
   lifecycle records into a typed execution summary. This projection retains
   execution context, terminal outcomes, branch outcomes, and errors without
   exposing the raw event log to clients.
8. With `perf-diagnostics` enabled, the projection emits nested operation
   timings, row/query counters, and correlated daemon RSS/PSS observations
   through the existing event broadcast. Normal builds compile those hooks
   away.
9. An optional exact-turn target applies the same bounded message/tool
   projection to the ancestral path ending at that turn. This is a read-only
   projection and does not alter the persisted active branch head.

Persisted session discovery:

1. Ordinary `session.list` returns top-level sessions only.
2. A `parent_session_id` request resolves that parent and returns its direct
   linked children from the same store.
3. Session summaries expose normalized linkage fields; relationship discovery
   does not parse metadata or query the semantic graph overlay.

Session graph:

1. `session.graph_get` resolves the persisted session and store in the same way
   as session detail.
2. One on-demand query loads every durable turn with its parent id, depth,
   compact user preview, and message/tool counts.
3. Branch heads are projected separately and carry exact internal head/source
   turn ids for typed lifecycle operations.
4. This projection is the conversation turn tree. It does not read or imply the
   optional harness-authored `runtime.graph.*` semantic overlay.

Branch activation/checkout:

1. The persisted session is resolved first.
2. Live attached runtime slots are checked for ambiguity and busy state.
3. The branch operation is applied in persistence.
4. If a live slot was targeted, the matching runtime is reloaded.

Exact-turn branch creation follows the same live-slot rules when activation is
requested. Non-activating forks may target a turn on any path; the store
validates that the turn belongs to the resolved session.

Persisted session deletion:

1. `runtime_sessions.rs` resolves the session reference and its linked descendants.
2. Any live runtime attached to the requested session or a descendant rejects
   deletion with `resource_busy`; clients must explicitly end those runtimes first.
3. Persistence deletes descendants deepest-first, then removes each session's
   transcript, branches, events, tool executions, semantic graph, and scoped data
   in a per-session transaction.
4. Work items survive deletion but any claim owned by the deleted session is
   released.

## Invariants

- Bare persisted session references use `StoreSelector::Alias("state")`.
- Cross-store session access must use a qualified session reference.
- Bounded session detail is a read projection only. It must not truncate
  persisted messages or change the runtime hot-history policy.
- Efficiency detail is also a read projection. It must preserve the distinction
  between provider-measured usage and Turin-estimated request composition.
- Cache-read and cache-creation counts remain optional provider measurements;
  absence must not be rendered as a measured zero or inferred from Turin's
  reusable-prefix estimate.
- Development diagnostic events must remain payload-sparse and observational.
  They may contain ids, counts, byte sizes, outcomes, and elapsed times, but not
  message, prompt, event-payload, or tool-output content.
- Execution detail is a durable read projection, not live client state. Its
  lifecycle-event query must remain independently capped and report when older
  execution records were omitted.
- Offset session windows are indexed from the oldest active-branch message and
  may contain slightly more than the requested limit to preserve complete turns.
- Exact-turn session projections validate turn ownership through the same
  `SessionReadTarget::TurnId` path used by persistence materialization and must
  remain observational.
- A projected message's `turn_id` is durable identity; `turn_index` remains a
  path-relative presentation depth and must not be used for exact branch writes.
- `slot_id` is invalid for task submission unless a `session_id` is also supplied.
- Channel-bound session open/resume should reuse channel persistence and inference overrides.
- Sidestep slots are temporary and should be killed after the task path completes or fails.
- Explicit harness action targets must run in the named harness runtime; agent identity must not silently redirect them to another harness.
- Harness source paths must not be absolute, traverse above the harness root, escape through symlinks, or target non-Lua files.
- Candidate validation is observational and must not mutate the active harness generation.
- Source saves must reject stale expected hashes before replacing files. A successful save means persisted, not validated or activated.
- Multi-file saves are conflict-checked as a batch, while replacement atomicity is per file; failed hot reload must leave the previous valid runtime active.
- Unbound shared harness actions use the primary agent identity, while multi-agent harness actions remain explicit.
- Branch activation/checkout must reject busy live sessions.
- Branch activation/checkout must reject slot-agnostic requests when multiple runtime slots are attached.
- Graph projection must remain on demand and must not make ordinary session
  detail load the complete turn tree.
- Exact-turn branch creation must validate session ownership and must not
  silently resolve the same numeric depth on the active path.
- Session deletion must reject any attached live runtime, regardless of slot.
- Root-session listing must not mix contextual or hidden child sessions into the
  operator's conversation list.
- Session deletion must be transactional and must not delete durable work
  items merely because the session claimed them.

## Common Changes

Change task submission behavior:

1. Update `runtime_tasks.rs`.
2. Preserve `agent_id` vs `session_id` targeting rules and task visibility after submit.
3. Run `cargo test -p turin --lib submit_task`.

Change sidestep behavior:

1. Update `runtime_tasks.rs` and sidestep preparation helpers.
2. Preserve ephemeral cleanup and durable fork/promotion behavior.
3. Run `cargo test -p turin --lib sidestep_task`.

Change persisted session/branch behavior:

1. Update `runtime_sessions.rs`.
2. Preserve store-qualified session references and live-slot ambiguity handling.
3. Run `cargo test -p turin --lib session_branch`.

Change persisted session deletion:

1. Update `runtime_sessions.rs` and the persistence session transaction.
2. Preserve live-runtime rejection and release rather than delete work claims.
3. Run `cargo test -p turin --lib deleting_session` and the control-client
   connectivity tests.

## Tests

Focused tests:

```sh
cargo test -p turin --lib submit_task
cargo test -p turin --lib wait_for_task
cargo test -p turin --lib sidestep_task
cargo test -p turin --lib session_branch
```

Basic compile/format checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current pass centralizes channel lookup, persisted session target resolution, and live-branch busy checks. It intentionally keeps daemon task and session files separate because task submission and persisted session inspection have different contracts even though both touch live runtime state.

Runtime event streaming is split so the async subscription loop stays in `events.rs`, while filter matching and scoped snapshot projection live in focused child modules.

Task lookup is intentionally direct. `task.get` and `task.wait` should look up one pending/completed task by request id, not build `task.list` and search it. This keeps large completed-task payloads from being cloned repeatedly during sequential task workloads.

The task event poller compares lightweight task fingerprints first, then fetches and serializes the full task snapshot only when the public `task.updated` payload actually needs to be emitted. The emitted event shape remains the full task snapshot.

Peer runtime idle shutdown supports an opt-in allocator diagnostic: when `TURIN_TRIM_ALLOCATOR_ON_PEER_IDLE` is truthy, Linux builds call `malloc_trim(0)` after the peer runtime has ended its session and shut down MCP clients. This is deliberately environment-gated so normal daemon behavior does not pay the trim cost unless a deployment or perf run asks for the lower retained-RSS profile.

Heap attribution is also opt-in. Build the daemon with `--profile profiling --features heap-profile` when a perf pass needs `dhat` heap data; normal release builds do not enable the feature, do not replace the allocator, and keep the stripped size-optimized release profile.

Live retrieval diagnostics are separately opt-in through `perf-diagnostics`.
The feature-enabled daemon publishes precise internal stages on its existing
event channel and samples its own Linux `/proc/self` memory at stage boundaries
and while root retrieval operations remain active. Nested stages retain timing
and counters without their own `/proc` reads. There is no diagnostics sidecar
or separate HTTP service. Normal builds compile the collector and call-site
macros away.

For low-memory deployments on Linux/glibc, allocator environment settings such as `MALLOC_TRIM_THRESHOLD_=0` and `MALLOC_ARENA_MAX=1` can materially reduce retained PSS after long channel/task runs. They are not enabled by Turin itself because they trade retained memory for allocator/syscall overhead and may reduce throughput under some concurrent workloads.
