# Agent Session Bindings Map

## Purpose

`agent.*` exposes local task queueing, peer-agent calls, sidesteps, promotion, and session branch helpers to Lua harnesses.

This surface sits between harness code and core runtime state. It should stay boring: validate policy, resolve the active session/store once, call the owning kernel or persistence API, and shape Lua return values consistently.

## Files

- `src/harness/stdlib/agent_bindings.rs`
  - Lua-facing `agent.*` and `agent.session.*` API.
  - Registration/orchestration only; helper domains live under `agent_bindings/`.
- `src/harness/stdlib/agent_bindings/queue.rs`
  - Local queue helpers for `agent.spawn`, `agent.sidestep`, and `agent.session.queue*`.
- `src/harness/stdlib/agent_bindings/session_store.rs`
  - Session-store lookup helpers for branch/list/load APIs.
- `src/harness/stdlib/agent_bindings/sidestep_graph.rs`
  - Sidestep graph relation parsing and persisted graph-edge attachment.
- `src/harness/stdlib/agent_bindings/branch_lua.rs`
  - Branch-head row to Lua-table conversion.
- `src/harness/stdlib/agent_bindings/options.rs`
  - Option parsing for spawn/sidestep/peer/session helper APIs.
- `src/harness/stdlib/runtime_agent.rs`
  - Lua-facing `runtime.agent.*` API for peer-agent submit, await, status, sidestep, and promotion.
- `src/kernel/agent_manager/*`
  - Peer-agent lifecycle, task submission, result await, and live-session reload.
  - `records.rs` owns physical runtime-slot identity plus queued task/session records.
  - `runtime_control.rs` owns each resident runtime's coherent session, execution,
    cancellation, and reset snapshot.
  - `caches.rs` owns bounded completed-result and trace delegation-budget retention.
  - `peer_runtime.rs` owns peer task execution against the currently resident session.
  - `peer_session.rs` owns peer session bootstrap, linked-session activation, switching,
    and shutdown.
  - `peer_signals.rs` owns durable runtime-signal polling, delivery, and acknowledgement.
  - `runtime_registry.rs` owns runtime-slot reuse, resume, creation, and registry publication.
  - `runtime_worker.rs` owns background worker startup, polling, idle policy, and shutdown.
  - `lane_scheduler.rs` owns fair cross-session selection while preserving per-session FIFO.
  - `task_results.rs` owns result waiting, task inspection, terminal bookkeeping, and
    completed-result promotion.
- `src/kernel/session.rs`
  - Queued task, execution context target, conflict policy, and branch outcome types.
- `src/kernel/task_promotion.rs`
  - Durable promotion of completed sidestep results.
- `src/persistence/state/branches.rs`
  - Branch-head creation, checkout, and sibling lookup.

## Data Flow

Local queueing:

1. Lua calls `agent.spawn`, `agent.sidestep`, or `agent.session.queue*`.
2. The binding checks governance and runtime policy.
3. It builds a `QueuedTask` with trace, conflict, execution, and branch metadata.
4. The task is pushed into the active in-process session queue.

Peer-agent calls:

1. Lua calls `agent.submit` or `agent.ask`.
2. The binding checks submit/await capabilities and child-agent governance.
3. Delegated capabilities are parsed, intersected with any inherited peer ceiling,
   and capped by active grants.
4. `AgentManager` owns peer submission and result waiting.
5. Linked-session admission checks persisted ancestry, direct fan-out, outstanding
   children, and any trace-scoped root delegation budget before queueing work.
6. `mode = "thread"` reuses a named logical child context; `mode = "fresh"` assigns a
   unique child key for one-shot durable delegation. The omitted mode remains `thread`.

Session branch helpers:

1. Lua calls `agent.session.load`, `branch_list`, `branch_create`, `branch_siblings`, or `branch_checkout`.
2. The requested or active session reference resolves to a store selector.
3. The store opens through `StoreManager`.
4. The session row is fetched by public id.
5. Branch operations call persistence APIs, then reload live sessions when needed.

Session metadata helpers:

1. Lua calls `agent.session.set_title(title, opts?)`.
2. The requested or active session resolves through the same store helper as branch operations.
3. Normal updates preserve unrelated metadata; `if_empty = true` uses a database-side conditional update so generated titles cannot overwrite an existing title.

Peer result promotion:

1. A linked peer task completed from a durable parent turn carries that parent/origin
   as its promotion target.
2. `agent.promote` or `runtime.agent.promote` creates a sibling branch from the
   origin turn and writes either the delegated request/result boundary or one explicitly
   selected completed child turn.
3. Branch provenance retains the linked source session and selected source turn. Internal child tool/event
   history remains in the child session and is not merged into the parent transcript.

Linked runtime residency:

1. The parent session and thread key identify a durable logical child session.
2. Their stable hash selects one of the configured physical runtime lanes for the target agent.
   Busy lanes belonging to same-agent ancestors are excluded to prevent await cycles.
3. Each task envelope carries its linked-session target; the lane creates or resumes
   that target before allocating the runtime task id and running inference.
4. Threads sharing a lane queue serially, bounding resident Lua VMs without combining
   their transcripts, harness state, or promotion provenance.
5. Pending and terminal task records retain the logical child session id. Session
   cancellation and queued-session kill target that id rather than draining the lane.
6. Runtime slots retain the immutable agent/harness catalog generation used at
   startup. Targeted registry reconciliation retires only changed idle agents;
   unaffected active slots continue and future slots use the new generation.

## Invariants

- `agent.spawn` requires `runtime.agent.spawn`.
- `agent.sidestep`, `agent.promote`, `agent.submit`, and `agent.ask` require submit-capable governance.
- `agent.ask` also requires `runtime.agent.await`.
- Child-agent access must pass `allowed_child_agents` checks before peer submission.
- Delegated capabilities must inherit active peer ceilings and be capped by active
  temporary grants; nested delegation may narrow authority but never widen it.
- Linked delegation depth is derived from persisted parent relationships rather than
  caller-supplied counters. Direct fan-out includes durable children plus child creation
  already reserved by queued work.
- Cooperative family cancellation reaches materialized and not-yet-materialized linked
  descendants. Force-killing a family is rejected because physical lanes may contain
  unrelated logical sessions.
- Context choices reuse existing primitives: ephemeral sidesteps are asides, durable
  fork-sibling sidesteps are branches, and peer submissions are reusable or fresh linked threads.
- Linked runtime lane reuse must switch sessions only between envelopes. It must never
  reset a lane globally while another logical session has queued work.
- Runtime handles are published only after the worker reports successful session bootstrap.
  Failed bootstrap must never leave a dead handle in the registry.
- Force-killing the active session in a shared linked lane must reject the operation
  while unrelated work is queued; cooperative cancellation remains session-selective.
- Resolve live linked-session affinity before hashing or probing a new lane.
- Same-agent delegation must never queue onto a busy ancestor's lane. If every linked
  lane is occupied by an awaiting ancestor, fail with bounded-capacity feedback.
- Per-agent catalog gates serialize task admission against that agent's rare
  configuration replacement without blocking submissions to unrelated agents.
- AgentManager's root module composes orchestration and public snapshots; mutable
  runtime-control state, queue records, and bounded caches retain focused internal owners.
- Non-executed terminal task results are constructed from their owning pending record or
  queued envelope so cancellation, shutdown, and lost-result paths retain one result shape.
- Queue mutations must honor `queue.max_depth`.
- Current-session branch checkout is deferred through `pending_branch_checkout`; it must not mutate the active branch immediately inside the harness callback.
- Non-current live sessions must be reloaded after branch activation or checkout.
- `agent.session.load` returns nil for missing sessions; branch APIs treat missing sessions as errors.
- Session titles are trimmed, non-empty, and bounded to 120 characters at the Lua boundary.
- Generated title policies should use `if_empty = true`; explicit retitles may overwrite.

## Common Changes

Change local queue semantics:

1. Update queue helpers in `agent_bindings/queue.rs`.
2. Preserve trace inheritance and queue depth checks.
3. Run local sidestep and queue tests.

Change session branch behavior:

1. Keep session reference/store/row lookup centralized.
2. Preserve the current-session deferred checkout rule.
3. Run:

```sh
cargo test -p turin --test harness_tests test_agent_persistence_store_overrides_default_scoped_data_store
cargo test -p turin --test harness_tests test_agent_sidestep_creates_hidden_sibling_branch_on_current_session
cargo test -p turin --test harness_tests test_agent_can_promote_detached_local_sidestep_result
cargo test -p turin --test harness_tests test_harness_conditionally_exposes_one_shot_session_title_tool
```

Change peer-agent delegation:

1. Keep child-agent and delegated-capability checks before submission.
2. Keep `agent.submit(prompt, opts?)` as the top-level request-id-returning counterpart to `agent.ask(prompt, opts?)`; the target agent comes from `opts.agent_id`.
3. Run:

```sh
cargo test -p turin --test harness_tests test_agent_allowed_child_agents_enforced_across_aliases
cargo test -p turin --test harness_tests test_agent_ask_applies_delegated_capability_ceiling
cargo test -p turin --test harness_tests test_runtime_agent_peer_submit_await_and_status
```

## Current Shape

The current pass keeps `agent_bindings.rs` as the Lua registration and policy-flow file, while private child modules own mechanical helper domains:

- `queue.rs` owns local queue depth checks, trace inheritance helpers, and queue push operations reused by runtime worklist bindings.
- `session_store.rs` owns session reference resolution, store opening, current-session matching, and completed-task-cache lookup.
- `options.rs` owns Lua option-table parsing for local tasks, sidesteps, peer-agent calls, and branch/session helpers.
- `sidestep_graph.rs` owns optional graph-edge attachment for persisted sidestep siblings.
- `branch_lua.rs` owns branch-head Lua table shaping.

This is still one client-facing Lua surface. The split is behavior-preserving and intended to make policy flow easier to audit without changing `agent.*` or `agent.session.*` authoring APIs.
