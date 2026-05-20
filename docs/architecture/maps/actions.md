# Actions Map

## Purpose

Actions are named Lua callbacks that can be invoked directly, scheduled, or attached to object proxies. They are also the execution surface for worklist action items.

Keep this subsystem small and explicit: action bindings should translate Lua values, build action context, invoke registered handlers, and persist work-item state transitions. Worklist selection and scheduling mechanics live elsewhere.

## Files

- `src/harness/stdlib/action_bindings.rs`
  - Lua `action.define`, `action.define_on`, and `action.run`.
  - Action context helpers: `ctx:complete`, `ctx:fail`, `ctx:cancel`, `ctx:pause`, `ctx:pause_for`, and `ctx.checkpoint`.
  - Work-item metadata patching and scheduled resume creation.
  - Built-in worklist actions: `worklist.dispatch_next`, `worklist.release_stale`.
- `src/harness/stdlib/runtime_worklist.rs`
  - Worklist proxies and dispatch integration that can invoke declared actions.
- `src/harness/scheduler.rs`
  - Harness-facing scheduler access used by action resume scheduling.
- `src/daemon/state/scheduled_worklist_actions.rs`
  - Daemon execution path for scheduled worklist actions.
- `src/harness/stdlib/object_refs.rs`
  - Object reference encoding and proxy-method registration used by `action.define_on`.

## Data Flow

Declared actions:

1. `action.define` registers a load-time callback in the harness action registry.
2. `action.run` encodes Lua params to JSON and calls `invoke_declared_action`.
3. The handler receives `ctx` and decoded params.
4. The return value is encoded back to JSON for callers outside Lua.

Object-scoped actions:

1. `action.define_on` resolves a target proxy and method.
2. It derives a method-specific action name.
3. The method is registered on matching object proxies.
4. Calls receive `ctx`, the subject object, and optional params.

Worklist actions:

1. Runtime worklist dispatch claims an item.
2. Action items call `invoke_declared_action` with `ActionWorkItemContext`.
3. `ctx:complete/fail/cancel/pause` updates work-item status and metadata.
4. Pause with a resume delay creates a scheduled job.

## Invariants

- `action.define`, `action.define_on`, `use`, and `watch` remain load-time declarations.
- Action names must be unique inside the declared action registry.
- Action params and results cross the Lua/Rust boundary as JSON-compatible values.
- Work-item metadata patches must merge with existing metadata instead of replacing unrelated keys.
- `ctx:complete` stores an `output` metadata patch.
- `ctx:fail` stores a `failure` metadata patch and extracts a reason when possible.
- `ctx:cancel` records cancellation metadata and releases the work item.
- `ctx:pause` records pause metadata and optionally schedules a resume.
- Built-in worklist actions should continue to call runtime worklist proxy methods rather than duplicating selection logic here.

## Tests

Focused tests:

```sh
cargo test -p turin --lib harness::engine::tests::test_reference_aware_action_round_trips_workitem_snapshot_and_ref_only
cargo test -p turin --lib harness::engine::tests::test_worklist_dispatches_prompt_and_action_payloads
cargo test -p turin --lib harness::engine::tests::test_worklist_action_pause_updates_checkpoint_and_schedules_resume
cargo test -p turin --lib harness::engine::tests::test_worklist_action_checkpoint_helpers_expose_saved_state
cargo test -p turin --lib harness::engine::tests::test_worklist_action_pause_for_sets_resume_delay
cargo test -p turin --lib harness::engine::tests::test_engine_invokes_declared_action_handler
cargo test -p turin --lib harness::engine::tests::test_engine_action_context_reports_cancellation
```

Basic checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current pass removed repeated Lua value conversion and action-status response construction from `action_bindings.rs`. It deliberately did not split scheduling or worklist branches into a new layer; those branches encode real behavior differences and are easier to audit inline for now.
