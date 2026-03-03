# ADR 004: Keep Harness Engine Access Synchronous and Mutex-Guarded

- Status: Accepted
- Date: 2026-03-03

## Context

The harness engine is built on top of `mlua`/Luau and carries mutable execution context:

- active session identity
- active queue context
- active capability delegation
- active trace ID
- module/root attribution for governance

That state is execution-local and tightly coupled to the Lua VM.

## Decision

Keep harness engine access synchronous and guard it with a mutex rather than trying to expose it as a broadly shared async object.

This applies both to:

- `HarnessEngine` ownership
- execution-context mutation around hook evaluation and stdlib calls

## Consequences

Positive:

- simple correctness model around Lua VM access
- avoids reentrancy and shared-mutation footguns
- keeps governance attribution and execution context explicit

Negative:

- less parallelism inside one harness runtime
- mutex boundaries must be handled carefully in hot code paths

## Rejected alternatives

- fully async/shared harness VM access
  - rejected because the concurrency model is harder than the practical benefit
- per-hook cloned harness VMs
  - rejected because it would fragment state and make hot reload/governance attribution worse
