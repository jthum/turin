# Control Client Map

## Purpose

`turin-control-client` is the shared Rust client facade for operator-facing code. It hides whether requests go through the local daemon socket or the remote HTTP bridge, exposes typed convenience methods, and re-exports DTOs used by manager, TUI, app, and UI-core code.

Keep this crate as a thin transport/domain facade. It should not own daemon semantics, UI presentation, config generation, or business rules that belong in the daemon/runtime.

## Files

- `crates/turin-control-client/src/lib.rs`
  - Public facade and crate-root re-exports.
- `crates/turin-control-client/src/client.rs`
  - `ConnectionSpec`, `ConnectionKind`, `ControlClient`, local/remote connection setup, request helpers, managed event subscription, and daemon status/health entry points.
- `crates/turin-control-client/src/models.rs`
  - Public DTOs decoded from daemon responses and small private response wrapper structs.
- `crates/turin-control-client/src/health.rs`
  - `ControlHealth` and status-to-health summarization.
- `crates/turin-control-client/src/schedules.rs`
  - Schedule convenience methods.
- `crates/turin-control-client/src/sessions.rs`
  - Live and persisted session convenience methods.
- `crates/turin-control-client/src/tasks.rs`
  - Task submit/wait/cancel/promote convenience methods.
- `crates/turin-control-client/src/harnesses.rs`
  - Harness detail, UI intent, and action invocation convenience methods.
- `crates/turin-control-client/src/worklists.rs`
  - Worklist and work-item convenience methods.
- `crates/turin-control-client/src/channels.rs`
  - Agent/channel detail, channel runtime status, settings update, and access-room convenience methods.
- `crates/turin-control-client/tests/connectivity.rs`
  - Local/remote connectivity and workflow coverage.

## Data Flow

1. Caller builds a `ConnectionSpec`.
2. `ControlClient::connect` resolves either a local daemon endpoint or remote client.
3. Domain helpers build a `DaemonRequest`.
4. `request_ok` delegates to `turin-daemon-client` or `turin-remote-client`.
5. Responses decode into daemon protocol DTOs or client DTOs re-exported from the crate root.

## Invariants

- Public types should remain importable from `turin_control_client::TypeName`.
- Domain helper modules should stay thin: build protocol params, send the request, and unwrap list wrappers when helpful.
- Local and remote behavior should stay symmetric unless a transport limitation is explicit.
- `ControlHealth` is a derived summary; daemon status remains the source of truth.
- UI/manager presentation formatting does not belong in this crate.
- Daemon wire-shape changes should be made in `turin-daemon-protocol` first, then reflected here.

## Common Changes

Add a daemon request helper:

1. Add the request/response shape in `turin-daemon-protocol`.
2. Add the thin convenience method to the matching domain file in this crate.
3. Keep the method return type typed and avoid leaking `serde_json::Value` unless the daemon surface is intentionally dynamic.
4. Run the focused control-client checks.

Change connection behavior:

1. Update `client.rs`.
2. Check local and remote connectivity tests.
3. Avoid making manager/UI call sites know about transport-specific behavior.

## Tests

Focused checks:

```sh
cargo test -p turin-control-client
cargo check -p turin-manager
cargo check -p turin-ui-core
```

Basic checks:

```sh
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The old single-file crate has been split by ownership while preserving the public crate-root API. This makes the client easier to extend during the UI/platform chapter without turning the facade into a second daemon implementation.
