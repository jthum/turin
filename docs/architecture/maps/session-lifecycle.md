# Session Lifecycle Map

## Purpose

`session_lifecycle.rs` owns how an `ExecutionHost` creates, resumes, refreshes, starts, and ends runtime sessions. It also owns local context selection and persisted sidestep preparation.

This file is central runtime plumbing. Prefer small, behavior-preserving cleanups here unless a semantic change has dedicated tests around persistence, branch heads, and resume behavior.

## Files

- `src/kernel/session_lifecycle.rs`
  - Session creation/resume/refresh/materialization.
  - Hot-history pruning after persisted refresh.
  - Local branch, turn, and external-reference selection.
  - Persistence attachment and background event flushing.
  - Persisted sidestep target normalization and sibling branch creation.
- `src/kernel/session.rs`
  - Session state, execution target, durability, visibility, write policy, and queued task types.
- `src/kernel/execution_host.rs`
  - Host construction, persistence locks, run-loop entry points, and task execution coordination.
- `src/persistence/state/*`
  - Session rows, messages, events, branches, turns, and worklist persistence.
- `src/kernel/hot_history.rs`
  - In-memory hot-history pruning policy.

## Data Flow

Create session:

1. Build a fresh `SessionState`.
2. Resolve agent state/default store selectors.
3. Persist a session row when possible.
4. Attach the background persistence lane.

Resume or refresh:

1. Resolve the session reference and state store.
2. Load the session row and active branch target.
3. Materialize messages/events for the execution target.
4. Rebuild history, counters, and context compaction checkpoint.
5. Reapply hot-history pruning.

Local context selection:

1. Validate the branch, turn, or external session reference.
2. Reject the switch when the local queue is not empty.
3. Update the session execution target.
4. Refresh from persistence to materialize the selected view.

End session:

1. Persist a session-end event.
2. Run harness `on_session_end`.
3. Close the durability channel and await the background persistence task.
4. Cancel the session token and clear the harness engine.

## Invariants

- A resumed session must belong to the requested agent.
- Refresh/materialization requires an internal persistence id.
- Local target switches must not run while tasks are queued.
- Branch-head targets preserve the active branch when no branch id is explicitly selected.
- External references must be normalized with an explicit store selector before being stored in the execution target.
- Hot-history pruning only applies to persisted branch-head sessions with `AdvanceBranchHead` write policy.
- Ending a session must drain the durability lane before marking the session inactive.
- Fork-sibling sidesteps must not mutate the persisted active head.

## Tests

Focused tests:

```sh
cargo test -p turin --test session_tests test_local_branch_selection_does_not_mutate_persisted_active_head
cargo test -p turin --test session_tests test_local_turn_selection_materializes_prefix_without_new_execution
cargo test -p turin --test session_tests test_local_external_reference_selection_materializes_remote_context_detached
cargo test -p turin --test session_tests test_tool_transcript_restores_and_continues_after_cold_resume
cargo test -p turin --test daemon_integration_tests daemon_session_resume_round_trip_over_restart
cargo test -p turin --test daemon_integration_tests daemon_task_sidestep_can_fork_a_sibling_branch
```

Basic checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current pass kept lifecycle logic in one file and made only a small lean cleanup: local branch/turn/external-reference selection now shares the queued-task guard. Larger extraction should wait for a dedicated pass over persisted session materialization and sidestep semantics.
