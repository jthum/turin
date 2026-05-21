# Agent Session Bindings Map

## Purpose

`agent.*` exposes local task queueing, peer-agent calls, sidesteps, promotion, and session branch helpers to Lua harnesses.

This surface sits between harness code and core runtime state. It should stay boring: validate policy, resolve the active session/store once, call the owning kernel or persistence API, and shape Lua return values consistently.

## Files

- `src/harness/stdlib/agent_bindings.rs`
  - Lua-facing `agent.*` and `agent.session.*` API.
  - Local queue helpers for `agent.spawn`, `agent.sidestep`, and `agent.session.queue*`.
  - Session-store lookup helpers for branch/list/load APIs.
  - Sidestep graph attachment and branch-row Lua conversion.
- `src/harness/stdlib/runtime_agent.rs`
  - Lua-facing `runtime.agent.*` API for peer-agent submit, await, status, sidestep, and promotion.
- `src/kernel/agent_manager/*`
  - Peer-agent lifecycle, task submission, result await, and live-session reload.
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
3. Delegated capabilities are parsed and capped by active grants.
4. `AgentManager` owns peer submission and result waiting.

Session branch helpers:

1. Lua calls `agent.session.load`, `branch_list`, `branch_create`, `branch_siblings`, or `branch_checkout`.
2. The requested or active session reference resolves to a store selector.
3. The store opens through `StoreManager`.
4. The session row is fetched by public id.
5. Branch operations call persistence APIs, then reload live sessions when needed.

## Invariants

- `agent.spawn` requires `runtime.agent.spawn`.
- `agent.sidestep`, `agent.promote`, `agent.submit`, and `agent.ask` require submit-capable governance.
- `agent.ask` also requires `runtime.agent.await`.
- Child-agent access must pass `allowed_child_agents` checks before peer submission.
- Delegated capabilities must be capped by active temporary grants.
- Queue mutations must honor `queue.max_depth`.
- Current-session branch checkout is deferred through `pending_branch_checkout`; it must not mutate the active branch immediately inside the harness callback.
- Non-current live sessions must be reloaded after branch activation or checkout.
- `agent.session.load` returns nil for missing sessions; branch APIs treat missing sessions as errors.

## Common Changes

Change local queue semantics:

1. Update queue helpers in `agent_bindings.rs`.
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
```

Change peer-agent delegation:

1. Keep child-agent and delegated-capability checks before submission.
2. Keep `agent.submit(prompt, opts?)` as the top-level request-id-returning counterpart to `agent.ask(prompt, opts?)`; the target agent comes from `opts.agent_id`.
3. Run:

```sh
cargo test -p turin --test harness_tests test_agent_allowed_child_agents_enforced_across_aliases
cargo test -p turin --test harness_tests test_agent_complete_applies_delegated_capability_ceiling
cargo test -p turin --test harness_tests test_runtime_agent_peer_submit_await_and_status
```

## Current Shape

The current pass keeps agent bindings in one Lua registration file but reduces repeated session-store lookup and Lua result shaping. That is deliberate: this surface is policy-heavy, and splitting it before the runtime agent/session boundary is clearer would add navigation cost without reducing much shipped code.
