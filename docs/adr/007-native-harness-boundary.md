# ADR 007: Keep Native Harness Policy Separate From Runtime Operations

- Status: Accepted
- Date: 2026-08-20

## Context

Turin originally exposed harness capabilities through an embedded Lua VM. Supporting
compiled harnesses and builds without Lua requires an engine-neutral boundary, but the
Lua standard library combines several different concerns:

- synchronous lifecycle and request policy
- async agent delegation
- scheduler and worklist operations
- scoped memory, KV, database, and graph access
- load-time declarations and scripting conveniences

Exposing all underlying managers to a native harness callback would reproduce that
coupling as a public Rust API. It would also make authority ambiguous because lifecycle
hooks run inside an execution, while raw managers are process-wide handles.

## Decision

Treat compiled native harnesses as trusted, session-local policy objects. The public
contract directly supports typed lifecycle hooks, request preparation, signal delivery,
and named actions.

Keep agent-triggered I/O and async operations behind native `Tool` implementations,
`ToolRegistry`, and kernel-owned effects. Tools receive the resolved turn identity and
effective tool exposure through `ToolContext`; the kernel remains responsible for hook
policy, governance, rate limits, persistence, and effect application.

Do not introduce a generic native service object containing `AgentManager`,
`StoreManager`, scheduler access, governance managers, provider clients, or mutable
execution globals. Add a narrow typed operation or declarative kernel effect only when
a concrete native workflow cannot be expressed safely through the existing boundaries.

Keep adapter-only capabilities optional inside the internal harness contract. A native
or future scripting adapter implements only the capabilities it supports; it does not
pretend to provide Lua source loading, virtual-tool continuations, or Lua execution
globals.

## Consequences

Positive:

- native-only builds do not compile or initialize Lua
- normal hook dispatch stays synchronous and allocation-light
- compiled applications can combine native policy with native tools
- kernel authority and async side effects remain centralized
- another scripting adapter can target the neutral contract without wrapping Lua types
- the public API does not freeze internal manager ownership

Negative:

- native harnesses do not automatically receive every Lua standard-library helper
- a compiled workflow needing a new runtime operation may require a focused tool or
  kernel effect
- Lua virtual tools are not presented as native-harness parity; native tools are the
  compiled equivalent

## Rejected Alternatives

- pass a broad `NativeHarnessServices` manager bag to every callback
  - rejected because it exposes process-wide internals, obscures execution authority,
    and couples embedders to manager ownership
- make all harness callbacks async
  - rejected for now because lifecycle policy is synchronous, Lua uses a sync VM mutex,
    and changing every hook and lock boundary would add cost without a demonstrated need
- reproduce every Lua namespace as Rust methods
  - rejected because Lua namespaces are authoring conveniences, not the kernel contract
