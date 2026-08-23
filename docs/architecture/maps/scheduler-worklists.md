# Scheduler and Worklists Map

## Purpose

The scheduler/worklist subsystem owns durable scheduled jobs, scheduler execution, and worklist-backed task dispatch. It bridges daemon APIs, persistence rows, runtime agent sessions, harness actions, and shared work item domain helpers.

This subsystem should preserve two guarantees:

- scheduled jobs behave predictably across one-shot, recurring, overlapping, queued, and parallel runs
- worklist dispatch semantics stay consistent whether invoked from harness runtime APIs or scheduled daemon actions

## Files

- `src/daemon/state/scheduled_jobs.rs`
  - Scheduled job CRUD, listing, detail, enable/disable, deletion, run listing, input validation, and API-to-row mapping.
- `src/daemon/state/scheduled_execution.rs`
  - Scheduler tick loop, active-run reconciliation, due-job processing, recurrence advancement, prompt-job submission, scheduled action routing, and persistence selector resolution.
- `src/daemon/state/scheduled_worklist_actions.rs`
  - Scheduled `worklist.dispatch_next` and `worklist.release_stale` action execution.
- `src/daemon/state/worklists.rs`
  - Daemon-facing worklist list/detail/item query APIs.
- `src/work_items.rs`
  - Shared row-level work item domain helpers: public id formatting, pause/claimability/orphan checks, dependency checks, where filtering, and `WorkItemRow` to `QueuedTask` conversion.
- `crates/turin-harness-lua/src/harness/stdlib/runtime_worklist.rs`
  - Harness runtime worklist namespace registration, shared row conversion and dispatch helpers, and store opening/hydration.
- `crates/turin-harness-lua/src/harness/stdlib/runtime_worklist/item_proxy.rs`
  - Lua work-item fields and claim, heartbeat, dispatch, completion, failure, update, and child methods.
- `crates/turin-harness-lua/src/harness/stdlib/runtime_worklist/list_proxy.rs`
  - Lua worklist add, selection, claim-next, stale-release, progress, and dispatch-next methods.
- `crates/turin-harness-lua/src/harness/stdlib/runtime_worklist/params.rs`
  - Lua option and payload parsing for runtime worklist scope, add/update payloads, where/limit filters, stale-release options, and JSON field serialization.
- `crates/turin-harness-lua/src/harness/stdlib/runtime_worklist_selection.rs`
  - Shared runtime work item selection rules for pending, orphaned, paused, active, next, empty, progress, and child queries.
- `crates/turin-harness-lua/src/harness/stdlib/runtime_schedule.rs`
  - Lua-facing `runtime.schedule` API registration, capability gates, agent validation, result shaping, and scheduler access bridging.
- `crates/turin-harness-lua/src/harness/stdlib/runtime_schedule/params.rs`
  - Lua option decoding for schedule create/update, persistence target parsing, schedule action parsing, and next-run time parsing.
- `crates/turin-harness-lua/src/harness/stdlib/action_bindings.rs`
  - Built-in action bridge for worklist actions invoked from harness code.
- `src/persistence/state/scheduler.rs`
  - Scheduled job persistence operations: CRUD, due/running queries, overlap state, recurrence/failure status, enable/disable, and deletion.
- `src/persistence/state/scheduler/runs.rs`
  - Scheduled job run persistence operations: active/history queries, run start insertion, active-run counts, and run completion bookkeeping.
- `src/persistence/state/worklists.rs`
  - Worklist and work item persistence operations.

## Data Flow

Scheduled prompt job:

1. `scheduled_jobs.rs` creates or updates a durable scheduled job row.
2. `scheduled_execution.rs::scheduler_tick` loads due jobs.
3. `process_due_scheduled_job` applies overlap and work-key capacity rules.
4. `submit_scheduled_job` opens the target agent session and queues a `QueuedTask`.
5. Persistence records the active run and later reconciles it when the task reaches a terminal state.

Scheduled action job:

1. `scheduler_tick` loads the due action job.
2. `execute_scheduled_action` parses the scheduled action payload.
3. Built-in agent actions run through `execute_leaf_scheduled_action`.
4. Worklist actions delegate to `scheduled_worklist_actions.rs`.
5. Harness-defined actions fall through to the target harness runtime.

Scheduled worklist dispatch:

1. `scheduled_worklist_actions.rs` parses action params and opens the target worklist.
2. It scans eligible top-level rows using shared helpers from `work_items.rs`.
3. It claims one row.
4. Action rows execute as non-nested leaf actions.
5. Prompt rows become `QueuedTask`s through `work_item_prompt_task`.

Runtime worklist dispatch:

1. `runtime_worklist.rs` exposes `next`, `dispatch`, `dispatch_next`, `release_stale`, and related helpers to harness Lua.
2. Prompt rows use the same `work_item_prompt_task` helper as scheduled worklist dispatch.
3. Action rows invoke declared harness actions through `action_bindings`.

Runtime schedule API:

1. `runtime_schedule/params.rs` parses Lua create/update options into daemon protocol params.
2. `runtime_schedule.rs` checks `runtime.schedule.*` capabilities and validates scheduled agent ids against harness config.
3. It requires a daemon-managed `HarnessSchedulerAccess`; unmanaged runtimes return a Lua `(nil, err)` pair.
4. It delegates CRUD, runs, enable/disable, and delete operations to daemon scheduler access.

## Invariants

- A scheduled job must define exactly one payload kind: prompt or action.
- A scheduled job cannot define both `interval_seconds` and `recurring_pattern`.
- Active prompt runs are tracked through scheduled job run rows.
- Starting or finishing a scheduled run and refreshing the scheduled job's active-run summary
  commit as one transaction. A run row and `active_run_count`/`running_task_id` must not diverge.
- `skip` overlap advances the next recurring due time without setting `pending_rerun`.
- `queue` overlap advances recurrence and sets `pending_rerun` so the job runs again after the active run finishes.
- `parallel` overlap may start another active run if work-key capacity allows it.
- Work-key capacity applies before a due job is submitted or executed.
- Scheduled worklist dispatch only considers top-level work items.
- Scheduled worklist dispatch must not execute nested `worklist.*` actions from work item action payloads.
- Work item prompt rows should become `QueuedTask`s through `work_item_prompt_task`, not ad hoc row parsing.
- Worklist filtering should use shared `work_items.rs` helpers so scheduler, daemon, and runtime paths do not drift.
- Stale-claim release must recheck the persisted heartbeat and claim identity while updating.
  A heartbeat that lands after candidate selection must prevent release in both runtime and scheduled paths.
- Partial work-item updates mutate only fields present in the update request. They must not
  read and rewrite unrelated fields that another execution may have changed concurrently.
- `runtime_schedule.rs` should remain a binding/validation layer; scheduler semantics belong in daemon state code.

## Common Changes

Add a scheduled action:

1. Add routing in `scheduled_execution.rs::execute_named_scheduled_action` or `execute_leaf_scheduled_action`.
2. Keep worklist-specific actions in `scheduled_worklist_actions.rs`.
3. Add focused daemon state tests in `src/daemon/state/tests.rs`.
4. Run `cargo test -p turin schedule --lib`.

Change overlap or recurrence behavior:

1. Update `scheduled_execution.rs`.
2. Add or update semantic tests for skip, queue, parallel, one-shot, and recurring behavior.
3. Run `cargo test -p turin scheduled_ --lib`.

Change work item eligibility:

1. Prefer changing shared helpers in `src/work_items.rs`.
2. If the rule is only about runtime list/next/progress views, update `runtime_worklist_selection.rs`.
3. Check runtime worklist and scheduled worklist behavior together.
4. Run `cargo test -p turin worklist --lib` and `cargo test -p turin schedule --lib`.

Change `WorkItemRow` task mapping:

1. Update `work_item_prompt_task`.
2. Keep runtime and scheduled dispatch using the shared helper.
3. Run `cargo test -p turin work_item_prompt_task --lib` and `cargo test -p turin worklist --lib`.

## Tests

Focused tests:

```sh
cargo test -p turin scheduled_ --lib
cargo test -p turin schedule --lib
cargo test -p turin worklist --lib
cargo test -p turin work_item_prompt_task --lib
cargo test -p turin concurrent_scheduled_run_transitions_preserve_job_summary --lib
cargo test -p turin heartbeat_and_stale_release_cannot_both_win --lib
cargo test -p turin scheduler_reconciles_persisted_run_missing_from_runtime_after_restart --lib
```

Basic compile/format checks:

```sh
cargo check -p turin
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The recent refactor intentionally stopped short of pushing scheduler/worklist filtering into SQL. Current worklist dispatch paths still load rows and filter in Rust. That is acceptable for the present architecture, but it is the likely future performance/memory improvement area.

Runtime worklist list-style methods share option parsing for `where` and `limit` before applying the selection helpers. Keep new list/filter methods on that path so option validation and row selection do not drift.

Runtime worklist proxy methods also share small local helpers for row JSON decoding and harness-runtime async bridging. Keep new proxy fields/methods on those helpers rather than reintroducing repeated `parse_json_opt(...).ok().flatten()` or `block_on_current(...).map_err(...)` plumbing.

The current module split is deliberate:

- `scheduled_jobs.rs` answers "what jobs exist and how are they edited?"
- `scheduled_execution.rs` answers "what is due and how does a scheduled job run?"
- `scheduled_worklist_actions.rs` answers "how do scheduled jobs operate on worklists?"
- `work_items.rs` answers "what are the shared row-level work item rules?"
- `runtime_worklist_selection.rs` answers "which runtime-visible work items match this proxy method?"
- `runtime_worklist.rs` answers "how is the Lua worklist API registered and how are stores and shared values resolved?"
- `runtime_worklist/item_proxy.rs` and `list_proxy.rs` own the two proxy method surfaces.
- `runtime_worklist/params.rs` answers "how do Lua worklist options and payloads become typed runtime inputs?"
- `runtime_schedule/params.rs` answers "how do Lua schedule options become daemon protocol params?"
- `persistence/state/scheduler/runs.rs` answers "how are scheduled job run rows recorded and reconciled with active job state?"
