# Daemon Protocol Map

## Purpose

`turin-daemon-protocol` owns the typed JSON request, response, event, and handshake DTOs used between clients and the daemon.

This crate is wire-shape sensitive. Internal organization can change, but serialized operation names, parameter names, enum casing, and crate-root public exports should remain stable unless a deliberate protocol change is being made.

## Files

- `crates/turin-daemon-protocol/src/lib.rs`
  - Crate-root facade and public re-exports.
- `crates/turin-daemon-protocol/src/handshake.rs`
  - Protocol version constants, transport/wire-format constants, daemon capabilities, and handshake DTO.
- `crates/turin-daemon-protocol/src/common.rs`
  - Shared no-params, id, store target, and persistence selector DTOs.
- `crates/turin-daemon-protocol/src/agents.rs`
  - Agent create/update and harness binding request DTOs.
- `crates/turin-daemon-protocol/src/harnesses.rs`
  - Harness action invocation DTOs.
- `crates/turin-daemon-protocol/src/tasks.rs`
  - Task submit, sidestep, wait, and promote DTOs.
- `crates/turin-daemon-protocol/src/schedule.rs`
  - Schedule create/update/runs params plus schedule job/run detail DTOs.
- `crates/turin-daemon-protocol/src/worklists.rs`
  - Worklist and work item query/detail DTOs.
- `crates/turin-daemon-protocol/src/sessions.rs`
  - Session open/resume/list/search/title/delete/branch/live-target DTOs.
- `crates/turin-daemon-protocol/src/request.rs`
  - `DaemonRequest` operation enum and op-name serde mapping.
- `crates/turin-daemon-protocol/src/envelopes.rs`
  - Request, response, error, and event envelopes.
- `crates/turin-daemon-protocol/src/tests.rs`
  - Wire-shape round-trip tests.

## Invariants

- Public DTOs remain re-exported from the crate root.
- `DaemonRequest` variant serde names are the daemon wire contract.
- `session.delete` uses the shared session-id shape and remains transport
  independent; deletion semantics belong to daemon state and persistence.
- `session.family_get` is an on-demand relationship/runtime projection;
  `session.archive` marks an idle linked subtree without deleting it.
- `session.open.origin_id` is opaque creation provenance. `session.list.origin_id`
  filters root sessions by that value; neither field represents authenticated
  client identity or delegated authority.
- Response error codes serialize as snake_case.
- Default values must stay explicit where they affect wire behavior.
- Domain DTO modules should not depend on daemon server, manager, or control-client code.
- Channels use generic session/task/event operations; the daemon protocol must
  not grow channel configuration, access, binding, presence, or
  lifecycle operations.
- Local IPC is a trusted-operator transport rather than a client ACL system.
  Unix listener creation sets the endpoint to `0600`, and stale/shutdown
  cleanup must reject non-socket filesystem entries.

## Common Changes

Add a daemon operation:

1. Add or reuse a params DTO in the relevant domain module.
2. Add a `DaemonRequest` variant in `request.rs` with an explicit `serde(rename = "...")`.
3. Add a round-trip test in `tests.rs`.
4. Update daemon dispatch and control-client call sites separately.

Change an existing DTO:

1. Treat it as a protocol change unless the serialized shape is provably unchanged.
2. Add or update a wire-shape test.
3. Check manager, control-client, and independent channel call sites.

## Tests

Focused tests:

```sh
cargo test -p turin-daemon-protocol
```

Downstream compile checks:

```sh
cargo check -p turin-control-client
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The crate root is intentionally a facade. DTOs are grouped by daemon domain, while the public root API remains stable through `pub use` re-exports.
