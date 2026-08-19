# Session Lifecycle Map

## Purpose

`session_lifecycle` owns how an `ExecutionHost` creates, resumes, refreshes, starts, and ends runtime sessions. It also owns local context selection, persisted execution-target materialization, and persisted sidestep preparation.

This module is central runtime plumbing. Prefer small, behavior-preserving cleanups here unless a semantic change has dedicated tests around persistence, branch heads, and resume behavior.

## Files

- `src/kernel/session_lifecycle.rs`
  - Session creation/resume/refresh orchestration.
  - Hot-history pruning after persisted refresh.
  - Local branch, turn, and external-reference selection.
  - Persistence attachment and background event flushing.
  - Session start/end lifecycle hooks.
- `src/kernel/session_lifecycle/materialization.rs`
  - Execution-target materialization for branch-head, turn, selected-path, summary-source, and external-reference targets.
  - Persisted history reconstruction.
  - Session counter and context compaction checkpoint reconstruction from persisted events.
- `src/kernel/session_lifecycle/sidestep.rs`
  - Persisted sidestep target normalization.
  - Ephemeral sidestep snapshots.
  - Fork-sibling branch source resolution and hidden sibling branch creation.
- `src/kernel/session.rs`
  - Session state, execution target, durability, visibility, write policy, plan progress, persistence record, and compaction checkpoint types.
- `src/kernel/session/completed_tasks.rs`
  - In-memory completed local task result cache and bounded promotion/result retention.
- `src/kernel/session/queued_tasks.rs`
  - Queued task DTO and constructor helpers for ad hoc, planned, inherited-trace, conflict-policy, execution, and branch-outcome task creation.
- `src/kernel/execution_host.rs`
  - Host construction, persistence locks, run-loop entry points, and task execution coordination.
- `src/persistence/state/*`
  - Session rows, messages, events, branches, atomic turn allocation, and worklist persistence.
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
4. Rebuild history, scalar counters, and the latest context compaction checkpoint.
5. Reapply hot-history pruning.

Durable turn writes:

1. Allocate a turn and advance its branch head in one transaction.
2. Persist the user message before adding it to resident history or invoking inference.
3. Stream events through the ordered background durability lane.
4. Persist the complete assistant message before emitting `TurnEnd` and adding it to resident history.
5. Persist finalized tool records and the tool-result message before exposing the result in resident history.
6. At task completion, use a barrier to report any background event-write failure to the caller.
7. Resume derives its next turn index from both materialized messages and the durable branch-head depth, so an allocated partial turn cannot move runtime progression backward.

Cancellation and timeout outcomes:

1. Inference streaming races the session cancellation token against the next provider event.
2. Parallel tool collection races the same token against all active tool futures; losing futures are dropped and must not publish tool results.
3. If cancellation and a provider error race, cancellation is the terminal task status.
4. Provider request and total-budget timeout errors become `timed_out`, distinct from cooperative `cancelled`, forceful `killed`, and generic `error`.
5. Individual tool timeouts remain tool errors that the agent or harness may recover from; they do not automatically terminate the whole task.

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

Delete persisted session:

1. Resolve the session reference and its state store.
2. Refuse deletion while any runtime slot still has the session open.
3. In one transaction, remove the turn graph, transcript, tools, events,
   semantic graph, and session-scoped KV/memory before removing the session row.
4. Leave unrelated worklists intact while clearing stale session claim references.

## Invariants

- A resumed session must belong to the requested agent.
- Runtime resume completion must compare session references semantically; a bare id and the
  canonical store-qualified reference for that id identify the same resumed session.
- Refresh/materialization requires an internal persistence id.
- Local target switches must not run while tasks are queued.
- Branch-head targets preserve the active branch when no branch id is explicitly selected.
- External references must be normalized with an explicit store selector before being stored in the execution target.
- Hot-history pruning only applies to persisted branch-head sessions with `AdvanceBranchHead` write policy.
- Ending a session must drain the durability lane before marking the session inactive.
- Turn insertion and branch-head advancement must commit or roll back together.
- Durable transcript and tool-record write failures must stop the active task; they must not be reduced to warnings.
- A durability barrier must report event-writer failures that occurred before it, then allow a recreated writer to serve later tasks.
- Resident history must not advance past a transcript write that failed.
- Resume must advance beyond the durable branch-head depth even when its newest turn has no messages.
- Cancellation must not append assistant output or tool results that did not complete before cancellation won.
- Runtime error classification must prefer cancellation over a concurrent provider failure.
- Provider timeout failures must remain distinguishable from cancellation and generic runtime errors.
- Deleting a session must never race an open runtime or leave a partial graph.
- Session deletion owns session-scoped KV and memory, including namespaced scope
  keys, but must not delete agent/user/global memory or worklist records.
- Fork-sibling sidesteps must not mutate the persisted active head.
- The background durability lane should reuse its event writer/connection for sequential event writes, but must recreate it after a write error so connection-local failures do not poison the lane.

## Tests

Focused tests:

```sh
cargo test -p turin --test session_tests test_local_branch_selection_does_not_mutate_persisted_active_head
cargo test -p turin --test session_tests test_local_turn_selection_materializes_prefix_without_new_execution
cargo test -p turin --test session_tests test_local_external_reference_selection_materializes_remote_context_detached
cargo test -p turin --test session_tests test_tool_transcript_restores_and_continues_after_cold_resume
cargo test -p turin --test session_tests test_run_stops_when_user_message_persistence_fails
cargo test -p turin --test session_tests test_run_stops_when_assistant_message_persistence_fails
cargo test -p turin --test session_tests test_run_reports_background_event_persistence_failure
cargo test -p turin --test session_tests test_resume_advances_past_allocated_turn_without_messages
cargo test -p turin --test session_tests test_cancelling_stalled_inference_does_not_append_assistant_output
cargo test -p turin --test session_tests test_cancelling_stalled_tool_does_not_append_tool_result
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

The current pass keeps lifecycle orchestration in `session_lifecycle.rs` and extracts two private helper boundaries:

- `materialization.rs` owns persisted target materialization and rebuild logic.
- `sidestep.rs` owns persisted sidestep preparation and branch-source normalization.

`session.rs` remains the public facade for session-domain types. Completed-task retention and queued-task construction live in child modules and are re-exported at the original `crate::kernel::session::*` paths.

Turn durability is incremental rather than one transaction held across inference or tool execution. This is deliberate: external provider and tool work must not hold a database transaction open. A stopped process may therefore leave a partial turn as durable evidence, but committed rows remain ordered, write failures are surfaced, branch allocation is atomic, and resume progresses from the durable branch head.
