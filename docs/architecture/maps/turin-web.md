# Turin Web Map

## Purpose

`turin-web` is Turin's web-facing client adapter. It exposes a small JSON API
over the existing control layer so browser clients can render the same semantic
UI intent as `turin-app` and `turin-tui`.

Keep this crate thin. It should not host a second runtime, own durable UI
session state, invent renderer-specific harness APIs, or bypass
`turin-control-client` for daemon operations that already have typed helpers.

## Files

- `crates/turin-web/src/main.rs`
  - CLI argument parsing and `ConnectionSpec` selection.
- `crates/turin-web/src/server.rs`
  - Hyper HTTP/1 server startup, shutdown, bind policy, and shared state setup.
- `crates/turin-web/src/routes.rs`
  - JSON route handling for status, app registry, UI list loading, action runs,
    liveness, SSE event streaming, and first-party static shell routes.
- `crates/turin-web/static/`
  - Dependency-free browser shell assets for rendering semantic UI intent
    against the HTTP API.
- `crates/turin-web/tests/release_operator.rs`
  - End-to-end smoke against a temporary daemon and the Release Operator
    harness, both through local IPC and through `turin-remote`.

## Data Flow

1. CLI or caller builds a `WebServeOptions`.
2. `turin-web` connects through `turin-control-client`.
3. HTTP routes call typed control-client helpers.
4. UI app responses derive a `UiRegistry` from harness UI intent in daemon
   status.
5. UI list responses resolve semantic sources such as `worklists.release` into
   bounded worklist item queries.
6. SSE event streams proxy managed runtime subscriptions for invalidation and
   refresh hints.
7. Static browser assets load from the same origin and call `/api/*` routes
   directly.
8. The browser remains responsible for selected app, active screen, form drafts,
   panes, modals, filters, loading, and local error state.
9. Browser `ui.refresh` handling invalidates cached list requests whose
   semantic source matches the refresh binding, then reloads visible data.
10. Browser action responses can include returned dynamic UI intents; the shell
    applies them locally before refreshing visible data.
11. Browser-local dynamic badges are re-applied to freshly fetched app records
    after status refreshes, because action-returned badges are presentation
    hints rather than durable daemon state.

## Invariants

- Runtime state remains in the daemon; web session state remains in the browser.
- Browser form drafts, confirmation modals, and action-running state are
  memory-local and should not be persisted by `turin-web`.
- Dynamic UI intents returned by action responses are local presentation hints,
  not durable browser session state.
- Browser-local dynamic badge overlays may survive status refreshes, but they
  still remain client memory state and must not be persisted by `turin-web`.
- Local and remote control connections should stay behaviorally symmetric.
- `GET /api/apps` and `GET /api/apps/{app_id}` should expose semantic UI
  surfaces, not renderer-specific widget state.
- `POST /api/ui/list` should accept semantic UI list requests first. Add raw
  daemon-query escape hatches only after the UI model proves it needs them.
- Unsupported sources and planned endpoints should return explicit JSON errors,
  not silent empty responses.
- `GET /api/events` is an invalidation/event feed. Browser `ui.refresh`
  handling should stay cache invalidation plus visible-data reload, not grow
  into a live-query result cache.
- Static assets are a bootstrap shell, not the final web framework decision.
  Keep them small unless the project deliberately adopts a frontend build step.
- The browser shell should keep semantic list constraints such as
  filter/sort/limit visible when rendering list nodes, but those constraints are
  request metadata, not browser session state.
- Non-loopback binds require explicit opt-in.

## Common Changes

Add a new UI data endpoint:

1. Check whether the control client already has a typed helper.
2. Add the route in `routes.rs` and keep request/response structs serializable.
3. Prefer semantic UI requests over raw daemon protocol shapes for harness UI
   client routes.
4. Extend the Release Operator smoke when the behavior can be exercised there.

Add browser support:

1. Keep the HTTP API stable and small.
2. Let the browser own ephemeral navigation and selection state.
3. Use `GET /api/events` for event-driven invalidation rather than adding live
   query semantics prematurely.

## Tests

Focused checks:

```sh
cargo test -p turin-web
cargo check -p turin-web
```

Basic checks:

```sh
cargo fmt --all -- --check
git diff --check
```
