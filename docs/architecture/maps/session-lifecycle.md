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
- `src/kernel/session_lifecycle/persistence.rs`
  - Session-row creation, initial branch cursor hydration, shared persistence-lock binding,
    and ordered background event durability attachment.
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
- `src/persistence/state/sessions.rs`
  - Session creation, normalized linkage, family lifecycle, title updates, and transactional deletion.
- `src/persistence/state/sessions/search.rs`
  - Ranked session, active-path message, tool-execution, and event search read model.
- `src/kernel/hot_history.rs`
  - In-memory hot-history pruning policy.

## Data Flow

Create session:

1. Build a fresh `SessionState`.
2. Resolve agent state/default store selectors.
3. Record an optional opaque origin in the normalized session row as creation
   provenance without treating it as ownership or configuration.
4. Persist a session row when possible.
5. Attach the background persistence lane.

Create linked peer session:

1. Resolve the originating session and its state store.
2. Enforce persisted family depth, direct fan-out, outstanding-child limits, and any
   trace-scoped root delegation budget.
3. Reuse the child identified by `(parent session, agent, thread key)` when it exists.
4. Otherwise create an agent-owned child session with explicit parent, root, relation,
   thread, visibility, and originating-turn columns. The child inherits its
   immutable client origin from the direct parent.
5. Route the child onto one of the configured deterministic linked-runtime lanes for its agent.
6. The lane creates or resumes the envelope's logical child session immediately before
   execution, then runs against that child's independent turn tree.

Resume or refresh:

1. Resolve the session reference and state store.
2. Load the session row and active branch target.
3. Materialize messages/events for the execution target.
4. Rebuild history, scalar counters, and the latest context compaction checkpoint.
5. Reapply hot-history pruning.

Durable turn writes:

1. Allocate a turn and advance its branch head in one transaction using the head turn
   that was observed while preparing the write as an optimistic precondition.
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

Kernel shutdown:

1. Stop accepting new peer runtimes and queued work.
2. Complete queued requests as `cancelled`, request cancellation for active work, and signal every peer event loop to stop.
3. Give peer runtimes a bounded grace period to end their sessions, flush durability, and close their MCP clients.
4. Record work still active after the grace period as `killed` and abort the stalled runtime task.
5. Stop the kernel watcher and close root MCP clients within their own bounded grace period.
6. Daemon shutdown broadcasts shutdown to background services, drains the kernel, and then removes its endpoint. Independent channels have their own lifecycle.

Unclean process restart:

1. Committed session, branch, turn, message, tool, event, linked-session, worklist,
   and scheduled-job rows remain the durable source of truth.
2. A partially written turn remains durable evidence. Resume advances from the durable
   branch-head depth and restores whatever transcript boundary committed successfully;
   it does not delete, complete, or replay that turn automatically.
3. Runtime queues, task result waiters, cancellation tokens, resident harness state, and
   temporary governance grants are process-local and disappear with the process.
4. Turin does not automatically replay interrupted inference or tool work. The runtime
   cannot prove whether an external side effect completed before the crash, so automatic
   replay would violate the safe at-most-once recovery default.
5. A linked session is recoverable once its session row is materialized. A queued child
   reservation that had not created that row is process-local work and is not recovered.
6. Lifecycle and audit events use the ordered background durability lane. A successful
   task barrier or clean shutdown flushes them; an abrupt process loss may omit the newest
   unacknowledged telemetry even when directly committed transcript rows survived.

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
2. Resolve linked descendants and refuse deletion while any runtime slot or pending
   task targets the requested session family. This includes queued child creation
   before the child session row has been materialized.
3. Delete descendants deepest-first, then remove the requested session. Each session
   deletion transaction removes its turn graph, transcript, tools, events, semantic
   graph, and session-scoped KV/memory before removing the session row.
4. Leave unrelated worklists intact while clearing stale session claim references.

## Invariants

- A resumed session must belong to the requested agent.
- A linked session has one direct parent, one stable root session, and an independent
  agent-owned transcript; linkage never changes turn ancestry or inference lookup.
- Harness delegation records the active durable turn as the child's origin when one
  exists. Reusing that child preserves its first origin rather than moving the thread.
- Repeated peer calls without an explicit `thread` reuse the `default` child thread.
  A named `thread` creates or reuses a separate child context under the same parent.
- Peer mode `fresh` generates a unique durable child thread for a one-shot independent
  context. It remains promotable and inspectable but is never selected by a later default call.
- Linked-session durability is independent of runtime residency. Any number of logical
  children may exist, but each agent owns at most its configured number of hot linked Lua
  runtimes; colliding threads execute serially, rotate fairly across logical sessions, and
  switch session context at envelope boundaries.
- Same-agent nested delegation excludes linked lanes occupied by busy ancestor sessions
  and probes the remaining lanes deterministically. Exhausting all configured ancestor lanes
  fails submission explicitly instead of queueing a child behind an ancestor awaiting it.
- A linked session that is already resident retains its physical lane so concurrent
  submissions cannot run one durable transcript through two Lua runtimes.
- Runtime session switches prepare and start the replacement before ending and publishing
  over the current session. A failed preparation must leave the current lane coherent.
- Runtime task ids must be allocated after a linked lane activates its target session so
  task counters advance in the session that actually executes the work.
- Successful linked-task results are promotable from their recorded origin into a new
  parent branch; promotion copies the task/result boundary or one selected completed child
  turn, not the child's internal transcript or tool lifecycle.
- A promoted branch, its promoted turn, and the user/assistant message boundary commit as
  one transaction. Repeating promotion for one completed task returns its recorded branch.
- Normal persisted-session listing returns top-level sessions only. Linked sessions are
  discovered through their indexed parent relationship rather than mixed into the
  conversation list.
- Explicit linked-session archival marks an idle subtree `archived` atomically. Archived
  children disappear from normal linked lists but remain in family topology and storage;
  deleting them remains a separate explicit operation. Reusing an archived named thread
  restores it to contextual visibility.
- Relationship indexes are partial and contain linked rows only; top-level sessions do
  not pay index-entry storage for nullable parent/root/thread relationships.
- Client-origin provenance has its own partial index. Sessions without an origin
  pay no origin index-entry cost, and origin filtering never parses JSON metadata.
- Family statistics read only session ids and parent ids. They must not materialize
  transcripts, events, graph rows, or complete session records.
- Relationship and visibility values remain validated text deliberately: they are sparse,
  operator-readable metadata, and compact numeric codes would add migration and DX cost
  without reducing the dominant transcript storage.
- Family depth and size queries traverse relationship-only rows in memory. They never load
  transcript content and remain independent of the number or size of turns in each session.
- Recursive cooperative cancellation covers the requested session and every linked
  descendant, including pending child reservations that have not created a session row.
- Runtime resume completion must compare session references semantically; a bare id and the
  canonical store-qualified reference for that id identify the same resumed session.
- Refresh/materialization requires an internal persistence id.
- Persisted origin identifies where a session family was created, not which client
  owns it or which store, inference configuration, or authority it uses. Linked
  descendants inherit the root provenance rather than accepting a new assertion.
- Local target switches must not run while tasks are queued.
- Local target switches materialize and validate the proposed projection before replacing
  the live execution target, history, counters, or branch cursor. A failed read must leave
  the resident session coherent on its previous target.
- Persisted sessions must name an active branch head before transcript or inference-context
  materialization. Missing branch rows, missing head turns, and invalid ancestry depths fail
  deterministically as persistence-integrity errors.
- Session-row creation, initial main-branch creation, and initial branch activation commit as
  one transaction. Turin must never expose a newly created session without its main branch.
- Creating a branch and optionally making it active commit as one transaction. Failed activation
  must not leave behind a branch that the caller was told could not be created.
- Context-compaction events are derived optimization records rather than transcript structure.
  Materialization skips malformed records with a warning and pages backward to the newest valid
  checkpoint instead of making an otherwise readable session unavailable.
- Persisted session metadata updates must preserve a valid JSON object. Title updates fail closed
  over malformed metadata rather than replacing unrelated or unreadable data; malformed optional
  metadata does not prevent listing or resuming the session itself.
- Branch-head targets preserve the active branch when no branch id is explicitly selected.
- External references must be normalized with an explicit store selector before being stored in the execution target.
- Hot-history pruning only applies to persisted branch-head sessions with `AdvanceBranchHead` write policy.
- Ending a session must drain the durability lane before marking the session inactive.
- Turn insertion and branch-head advancement must commit or roll back together. The head update
  must fail as a turn-write conflict if the branch moved after its parent was selected.
- Durable transcript and tool-record write failures must stop the active task; they must not be reduced to warnings.
- Task execution overrides, delegation budgets, and active-task state must be restored even when task hooks, execution, checkout, or durability finalization fails.
- A durability barrier must report event-writer failures that occurred before it, then allow a recreated writer to serve later tasks.
- Resident history must not advance past a transcript write that failed.
- Resume must advance beyond the durable branch-head depth even when its newest turn has no messages.
- Resume must preserve a committed user-only or otherwise partial turn and continue at the
  next durable branch depth without replaying the interrupted work.
- Unclean restart must fail closed for process-local authority and coordination state. In
  particular, temporary grants and queued/runtime tasks must not silently reappear.
- Recovery must never infer that an interrupted external tool call is safe to execute again.
- Cancellation must not append assistant output or tool results that did not complete before cancellation won.
- Runtime error classification must prefer cancellation over a concurrent provider failure.
- Provider timeout failures must remain distinguishable from cancellation and generic runtime errors.
- Kernel shutdown must reject new runtime creation and cannot wait indefinitely for stalled peer work or MCP clients.
- Cooperative shutdown records queued work as `cancelled`; only work that exceeds the grace period is `killed`.
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
cargo test -p turin kernel::agent_manager::tests::manager_shutdown --lib
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

The current shape keeps lifecycle orchestration in `session_lifecycle.rs` and extracts private helper boundaries:

- `materialization.rs` owns persisted target materialization and rebuild logic.
- `persistence.rs` owns persistence attachment and the background durability lane.
- `sidestep.rs` owns persisted sidestep preparation and branch-source normalization.

`session.rs` remains the public facade for session-domain types. Completed-task retention and queued-task construction live in child modules and are re-exported at the original `crate::kernel::session::*` paths.

Turn durability is incremental rather than one transaction held across inference or tool execution. This is deliberate: external provider and tool work must not hold a database transaction open. A stopped process may therefore leave a partial turn as durable evidence, but committed rows remain ordered, write failures are surfaced, branch allocation is atomic, and resume progresses from the durable branch head.
