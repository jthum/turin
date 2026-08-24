# Tool Authorization Map

## Purpose

Tool authorization pauses a harness-escalated tool call until an external actor approves or
denies it. It is deliberately separate from governance: governance establishes whether the
agent has authority to use a capability, while authorization requests a decision within that
authority.

Ordinary allowed tool calls do not enter this subsystem.

## Files

- `src/kernel/tool_authorization.rs`
  - Public request, decision, authorizer, and broker contracts.
  - Fail-closed default authorizer.
- `src/kernel/turn/tool_execution/validation.rs`
  - Converts `Verdict::Escalate` into asynchronous authorization.
- `src/kernel/turn/tool_execution/result_hooks.rs`
  - Emits request/resolution audit events and awaits the configured authorizer.
- `src/kernel/builder.rs`
  - Installs application-provided authorizers.
- `src/daemon/state/tool_authorizations.rs`
  - Projects pending broker requests and resolves daemon decisions.
- `src/daemon/server/dispatch/tool_authorization.rs`
  - Implements `tool_authorization.list` and `tool_authorization.resolve`.
- `crates/turin-control-client/src/authorizations.rs`
  - Typed local/remote client helpers.
- `src/commands/tool_authorization.rs`
  - Explicit interactive-terminal composition for direct CLI runs.

## Data Flow

1. A session-local harness returns `ESCALATE, reason` from `on_tool_call` or a tool-result hook.
2. Turin snapshots the exact tool name, call id, arguments, runtime identity, slot and reason.
3. The configured `ToolAuthorizer` receives the request and the tool execution awaits its future.
4. The daemon broker stores pending requests in memory and publishes a bounded notification.
5. Clients recover authoritative pending state through `tool_authorization.list` and resolve one
   request through `tool_authorization.resolve`.
6. Approval continues the exact in-memory call. Denial returns a tool error to the model; a user
   denial reason is optional.
7. Session or task cancellation resolves the wait as a denial and removes the pending request.

## Invariants

- Governance denial remains authoritative regardless of human approval.
- A request resolves at most once. Resolving an absent or completed id fails.
- Arguments are immutable snapshots; approval cannot substitute different arguments.
- Denial reasons are optional. Empty reasons normalize to absence.
- Pending state is process-local and authoritative only for the current daemon generation.
- Kernel rebuilds retain the daemon broker. Full daemon restart denies outstanding waits rather
  than claiming to reconstruct partially executed Rust futures.
- Notification delivery is best-effort and bounded. Clients must use the list operation after
  connecting or reconnecting.
- Waiting is asynchronous and does not block a Tokio worker thread, but the owning live execution
  and its session state remain resident until resolution or cancellation.
- Embeddings without an authorizer fail closed. Interactive stdin behavior is installed only by
  the CLI composition and is not a kernel default.

## Tests

```sh
cargo test -p turin --lib kernel::tool_authorization
cargo test -p turin --lib daemon::state::tests::daemon_lists_and_resolves_tool_authorization_without_denial_reason
cargo test -p turin --test rust_embedding escalated_tool_waits_for_external_authorization_before_execution
cargo test -p turin-daemon-protocol tool_authorization_requests_support_reasonless_denial
cargo test -p turin-control-client
```
