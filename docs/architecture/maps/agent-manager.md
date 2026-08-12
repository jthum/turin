# Agent Manager Map

## Purpose

The agent manager owns live peer-agent runtimes, runtime slots, cross-agent task submission, task result tracking, live session control, cancellation, and runtime wakeups.

This subsystem should preserve three guarantees:

- session-targeted operations resolve to exactly one live runtime slot or fail clearly
- live session snapshots use one consistent shape across open, resume, reload, and list paths
- pending task state and result receivers stay paired until completion, timeout, cancellation, or kill

## Files

- `src/kernel/agent_manager.rs`
  - Core types, runtime handles, task/result records, runtime control state, and manager construction.
- `src/kernel/agent_manager/operations.rs`
  - Live session operation surface: open/resume/reload sessions, list statuses/live sessions, session event subscription, and runtime lookup helpers.
- `src/kernel/agent_manager/tasks.rs`
  - Peer task operation surface: submit/await tasks, task status snapshots, completed-result cache updates, promotion, pending-result bookkeeping, and runtime queue enqueueing.
- `src/kernel/agent_manager/cancellation.rs`
  - Runtime, session, and task cancellation/kill behavior.
- `src/kernel/agent_manager/runtime_registry.rs`
  - Runtime-slot creation, replacement, and registry lifecycle.
- `src/kernel/agent_manager/peer_runtime.rs`
  - Peer runtime loop, task execution, result construction, and runtime session hydration.

## Data Flow

Session open/resume/reload:

1. `operations.rs` resolves or creates a `RuntimeSlotKey`.
2. Registry helpers ensure the runtime exists or is resumed.
3. Runtime control publishes the current session id and execution state.
4. Operation methods return a `LiveSessionSnapshot` built from the runtime key and handle.

Task submission:

1. `tasks.rs` resolves the target runtime slot through direct agent id or session target.
2. It creates a request id and pending result receiver.
3. The task is enqueued into the runtime handle queue.
4. `peer_runtime.rs` executes the task and reports a `PeerAgentTaskResult`.
5. `tasks.rs` removes pending state and moves completed results into the bounded completed-result cache.
6. Pending and completed status retain an optional title plus a normalized,
   240-character prompt preview so operator surfaces can identify work without
   duplicating full prompts in the runtime task ledger.

Session targeting:

1. `find_runtimes_by_session` compares direct session ids and parsed public ids.
2. `runtime_by_session_target` applies optional slot filtering.
3. Ambiguous slot-agnostic operations fail instead of guessing.

## Invariants

- A slot-targeted session operation must only use the requested slot.
- Slot-agnostic session operations must reject multiple live matches.
- Busy runtime slots must not be reused for session resume or reload.
- `LiveSessionSnapshot` fields should be derived consistently from the runtime key, handle, and effective session id.
- Pending task records should be removed when a result is recorded or a submission fails.
- Timed-out `await_result` calls must put the receiver back so the result can still be awaited later.
- Task title and prompt preview must survive queued, running, terminal,
  cancellation, and kill snapshots. The preview stays bounded before entering
  pending or completed task state.

## Common Changes

Change live session operation behavior:

1. Update `operations.rs`.
2. Preserve ambiguity and busy-slot error behavior.
3. Run `cargo test -p turin --lib kernel::agent_manager`.

Change cancellation behavior:

1. Update `cancellation.rs`.
2. Verify queued, running, and session-level kill paths.
3. Run `cargo test -p turin --lib kernel::agent_manager`.

Change peer task result shape:

1. Update `peer_runtime.rs`, `tasks.rs`, and related snapshot structs in `agent_manager.rs`.
2. Keep `list_tasks`, `await_result`, and completed-result promotion aligned.
3. Run agent-manager and daemon runtime-task tests.

## Tests

Focused tests:

```sh
cargo test -p turin --lib kernel::agent_manager
cargo test -p turin --lib runtime_signals
cargo test -p turin --lib explicit_runtime_slots_allow_multiple_live_runtimes_for_one_session
```

Basic compile/format checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current operations pass centralizes live session snapshot construction and busy-slot checks. It intentionally leaves the resume/reload wait loops local because each loop has a different readiness condition and timeout message.
