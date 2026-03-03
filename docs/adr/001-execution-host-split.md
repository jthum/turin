# ADR 001: Split ExecutionHost from Kernel

- Status: Accepted
- Date: 2026-03-03

## Context

`Kernel` originally owned both runtime composition and the hot execution path:

- session lifecycle
- task loop orchestration
- turn execution
- provider streaming
- tool execution
- persistence
- harness invocation

That shape became a liability once Turin grew:

- first-class multi-harness support
- peer-agent execution
- daemon mode

Peer runtimes were effectively forced to wrap a full `Kernel`, even when they only needed the execution behavior.

## Decision

Move execution-heavy behavior into `ExecutionHost` and keep `Kernel` as the thinner composition shell.

`ExecutionHost` owns:

- session lifecycle
- run loop / task execution
- turn pipeline
- event persistence
- harness hook invocation
- MCP runtime ownership for the active execution host
- harness resolution through `HarnessManager`

`Kernel` owns:

- top-level runtime composition
- watcher ownership
- CLI-facing orchestration
- access to the shared runtime managers

## Consequences

Positive:

- direct sessions and peer sessions now run on the same execution model
- multi-harness routing is explicit instead of being hidden behind kernel cloning
- daemon mode has a cleaner runtime substrate

Negative:

- there is one more architectural layer to understand
- execution behavior is no longer found in a single top-level type

## Rejected alternatives

- Keep all execution in `Kernel`
  - rejected because it keeps peer execution tied to full-kernel wrappers
- Fully global shared mutable execution host
  - rejected because the concurrency/state-isolation cost was worse than the practical benefit at this stage
