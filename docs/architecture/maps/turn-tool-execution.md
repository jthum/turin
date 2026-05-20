# Turn Tool Execution Map

## Purpose

Turn tool execution owns the path from provider-requested tool calls to durable tool results. It evaluates harness verdicts, applies built-in safety limits, runs native and virtual tools, persists audit/tool rows, and publishes tool results back into inference history.

This subsystem should preserve two guarantees:

- every requested tool call gets either a durable result or an explicit cancellation path
- harness and governance decisions are auditable and cannot be bypassed by virtual-tool nesting

## Files

- `src/kernel/turn/tool_execution.rs`
  - Main orchestration: verdict evaluation, rate limiting, native tool execution, virtual-plan dispatch, side effects, result persistence, and history publishing.
- `src/kernel/turn/tool_execution/virtual_tools.rs`
  - Virtual tool call expansion, recursion/depth checks, nested result aggregation, result-handler invocation, and hidden nested execution.
- `src/kernel/turn/tool_execution/plan_submission.rs`
  - `ToolEffect::EnqueuePlan` handling and `on_plan_submit` harness policy.
- `src/kernel/turn/tool_execution/result_hooks.rs`
  - Interactive escalation prompts and `on_tool_result` harness policy.
- `src/kernel/harness_hooks.rs`
  - Harness `on_tool_call` verdict evaluation used before execution.
- `src/harness/stdlib/tool_bindings.rs`
  - Lua declarations and invocation plumbing for harness-defined virtual tools.

## Data Flow

Normal tool call:

1. `execute_tool_calls` receives provider tool calls.
2. `evaluate_pending_tool_calls` applies `on_tool_call` verdicts.
3. `apply_tool_rate_limit` rejects excess calls inside the safety window.
4. `execute_validated_tool_calls` runs native tools in bounded parallelism.
5. `finalize_tool_records` applies `on_tool_result`, persists audit/tool rows, and appends tool results to history.

Virtual tool call:

1. The main executor asks the harness engine for a virtual tool plan.
2. `virtual_tools.rs` expands the plan into synthetic pending calls.
3. Nested calls run hidden from provider history until the outer virtual tool is finalized.
4. Result handlers may return final output or another virtual plan.
5. Recursion and nesting-depth checks apply before each virtual expansion.

Plan submission:

1. A native tool returns `ToolEffect::EnqueuePlan`.
2. `plan_submission.rs` applies `on_plan_submit`.
3. Accepted tasks are converted into queued tasks and attached to a plan id.

## Invariants

- `on_tool_call` verdicts apply before native or virtual execution.
- Governance capability checks apply to native registered tools before execution.
- Virtual tool expansion must reject direct or indirect recursion.
- Virtual tool nesting must stay under `MAX_VIRTUAL_TOOL_DEPTH`.
- Nested virtual calls should not publish intermediate tool messages to provider history.
- `on_tool_result` applies after execution and before persistence/history publication.
- Tool execution audit start/end and tool result rows should remain paired where possible.

## Common Changes

Change native tool execution behavior:

1. Update `tool_execution.rs`.
2. Preserve governance denial audit events and tool result persistence.
3. Run `cargo test -p turin --test harness_tests tool`.

Change virtual tool behavior:

1. Update `tool_execution/virtual_tools.rs`.
2. Test recursion, nesting depth, result handler output, and nested aggregation.
3. Run `cargo test -p turin --test harness_tests virtual_tool`.

Change plan submission:

1. Update `tool_execution/plan_submission.rs`.
2. Keep `on_plan_submit` verdict handling symmetric with other harness hooks.
3. Run relevant queue/plan tests or `cargo test -p turin plan --lib`.

## Tests

Focused tests:

```sh
cargo test -p turin --test harness_tests virtual_tool
cargo test -p turin --test harness_tests tool
cargo test -p turin --lib harness::engine::tests::test_engine_invokes_virtual_tool_result_handler
```

Basic compile/format checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current split keeps the high-level execution loop in `tool_execution.rs` and moves only the virtual-tool mechanics into a child module. That is deliberate: native and virtual tools still share finalization, persistence, audit events, result hooks, and history publication.
