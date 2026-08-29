# Turin Client Map

## Purpose

`turin-client` is the shared Rust client facade for operator-facing code. It hides whether requests go through the local daemon socket or the remote HTTP bridge, exposes typed convenience methods, and re-exports DTOs used by manager and client hosts such as `turin-web`.

Keep this crate as a thin transport/domain facade. It should not own daemon semantics, UI presentation, config generation, or business rules that belong in the daemon/runtime.

## Files

- `crates/turin-client/src/lib.rs`
  - Public facade and crate-root re-exports.
- `crates/turin-client/src/client.rs`
  - `ConnectionSpec`, `ConnectionKind`, `Client`, local/remote connection setup, request helpers, managed event subscription, and daemon status/health entry points.
- `crates/turin-client/src/models.rs`
  - Public DTOs decoded from daemon responses and small private response wrapper structs.
- `crates/turin-client/src/health.rs`
  - `ControlHealth` and status-to-health summarization.
- `crates/turin-client/src/schedules.rs`
  - Schedule convenience methods.
- `crates/turin-client/src/sessions.rs`
  - Live and persisted session convenience methods, on-demand turn topology,
    direct linked-session discovery, family topology/archive, exact-turn branch
    creation, branch listing, checkout, and durable deletion.
- `crates/turin-client/src/tasks.rs`
  - Task submit/wait/cancel/promote convenience methods.
- `crates/turin-client/src/authorizations.rs`
  - Pending tool-authorization listing and approve/deny convenience methods.
- `crates/turin-client/src/harnesses.rs`
  - Harness detail, UI intent, source inspection/candidate validation/hash-guarded saves, and action invocation convenience methods.
- `crates/turin-client/src/worklists.rs`
  - Worklist and work-item convenience methods.
- `crates/turin-client/src/memories.rs`
  - Bounded memory inspection convenience method.
- `crates/turin-client/tests/connectivity.rs`
  - Local/remote connectivity and workflow coverage, including the Release
    Operator harness UI fixture and dynamic UI side effects from actions.

## Data Flow

1. Caller builds a `ConnectionSpec`.
2. `Client::connect` resolves either a local daemon endpoint or remote client.
3. Domain helpers build a `DaemonRequest`.
4. `request_ok` delegates to `turin-daemon-client` or `turin-remote-client`.
5. Responses decode into daemon protocol DTOs or client DTOs re-exported from the crate root.

## Invariants

- Public types should remain importable from `turin_client::TypeName`.
- Domain helper modules should stay thin: build protocol params, send the request, and unwrap list wrappers when helpful.
- Task status preserves the daemon's bounded title/prompt description so
  clients can identify runtime work without opening every owning session.
- Task lifecycle state is `TaskState`, re-exported by the client; presentation code converts it
  to text only at the rendering edge.
- Runtime-agent status includes base provider/model/harness identity and named
  effective inference contexts for client routing controls.
- `get_session` preserves the complete transcript and diagnostic projections but uses the
  daemon's bounded raw-event default. `get_session_with_all_events` is the explicit complete
  raw-event path, while `get_session_event_window` exposes paging and type filters.
- `get_session_window` requests a bounded recent transcript without persisted events for
  interactive clients.
- `get_session_graph` is an explicit on-demand topology read. Normal session
  detail must not absorb its complete turn-tree cost.
- `get_session_turn_window` is a bounded, read-only projection ending at an
  exact durable turn. It must not activate a branch or retarget a live session.
- Exact-turn branch creation uses an internal turn id because turn indexes are
  path-relative and can repeat across sibling branches.
- Session messages retain that exact turn id so clients can associate branch
  provenance and contextual actions without loading the on-demand graph.
- Session deletion remains a daemon-owned operation. The client only sends the
  persisted session reference and preserves live-session rejection semantics.
- Persisted session listing returns roots by default. `list_linked_sessions`
  requests only direct children of an explicit parent; it must not turn the
  client into a generic session-relationship query layer.
- `open_session_with_origin` and `list_sessions_for_origin` expose opaque
  client provenance without defining client identity, authentication, or
  ownership in this crate.
- Local and remote behavior should stay symmetric unless a transport limitation is explicit.
- Tool authorization denial reasons are optional; clients must not add friction by requiring one.
- Harness source editing must go through daemon protocol operations so local and remote clients share path, conflict, validation, and persistence semantics.
- `ControlHealth` is a derived summary; daemon status remains the source of truth.
- `ControlHealth::agent_count` counts effective configured runtime agents, including
  the bootstrap agent; the filesystem registry alone is not a complete inventory.
- Operator-facing applications use this generic facade. Local-only channel
  binaries may use `turin-daemon-client` directly; channel configuration,
  access policy, bindings, and process health must not become client domains.
- Web/manager presentation formatting does not belong in this crate.
- Daemon wire-shape changes should be made in `turin-daemon-protocol` first, then reflected here.

## Common Changes

Add a daemon request helper:

1. Add the request/response shape in `turin-daemon-protocol`.
2. Add the thin convenience method to the matching domain file in this crate.
3. Keep the method return type typed and avoid leaking `serde_json::Value` unless the daemon surface is intentionally dynamic.
4. Run the focused client checks.

Change connection behavior:

1. Update `client.rs`.
2. Check local and remote connectivity tests.
3. Avoid making manager/UI call sites know about transport-specific behavior.

## Tests

Focused checks:

```sh
cargo test -p turin-client
cargo check -p turin-manager
cargo check -p turin-web
```

Basic checks:

```sh
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The old single-file crate has been split by ownership while preserving the public crate-root API. This makes the client easier to extend during the UI/platform chapter without turning the facade into a second daemon implementation.
