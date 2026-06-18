# Turin UI Core Map

## Purpose

`turin-ui-core` is the shared Rust client substrate for Turin's operator-facing
clients. It owns connection/profile handling, the controller command/update
loop, bounded dashboard snapshots, semantic UI intent indexing, and small
worklist display helpers used by multiple clients.

Keep this crate lean. It should help `turin-app`, `turin-tui`, and `turin-web`
share transport and semantic-contract discipline, not become a renderer, a
runtime-owned UI session store, or a second daemon implementation.

## Files

- `crates/turin-ui-core/src/lib.rs`
  - Public facade and crate-root re-exports.
- `crates/turin-ui-core/src/controller.rs`
  - Connection options, profile drafts, preflight helpers, controller spawning,
    `OperatorCommand`, `UiUpdate`, UI list loading, and command execution.
- `crates/turin-ui-core/src/dashboard.rs`
  - `DashboardState`, `DashboardSnapshot`, health/freshness summaries, bounded
    recent events/notices, and update application.
- `crates/turin-ui-core/src/intents.rs`
  - `UiRegistry` and `UiAppRecord` indexing for harness-declared app surfaces
    plus dynamic UI intent queues.
- `crates/turin-ui-core/src/worklist_view.rs`
  - Stateless worklist display derivation helpers for counts, grouping, and
    field labels.
- `crates/turin-ui-core/src/tests/controller.rs`
  - Controller/profile helper tests.

## Data Flow

1. A client builds `ConnectionOptions` or a `ConnectionSpec`.
2. `connect_dashboard` connects through `turin-control-client` and loads a
   bounded `DashboardState`.
3. `spawn_controller` starts refresh, event, focused-session-event, and command
   tasks.
4. The controller emits `UiUpdate` values from snapshots, runtime events, and
   operator commands.
5. Clients apply updates to their own `DashboardState` copy, then project that
   state into app-specific navigation, focus, forms, panes, modals, and caches.
6. Harness UI declarations from daemon status are indexed into `UiRegistry`.
7. Dynamic `ui.open`, `ui.show`, `ui.focus`, and `ui.refresh` intents are queued
   as one-shot suggestions for each client to drain or ignore locally. Runtime
   events and completed harness action results can both carry these dynamic
   intents.
8. `OperatorCommand::LoadUiList` resolves semantic `UiListRequest` values. Today
   only `worklists.<name>` sources load, through typed control-client worklist
   helpers.

## Invariants

- Client navigation state does not belong here. Active app, active screen,
  selected row, focused field, open modal, pane stack, scroll position, and form
  drafts remain client-owned.
- `UiRegistry` records semantic facts and requests, not subscriptions or open
  windows. Multiple clients or multiple windows may consume the same registry
  independently.
- Dynamic UI intents are hints. A client may honor, defer, degrade, or ignore
  them based on its renderer and local state.
- Dynamic UI intents returned by harness action results must be applied to the
  local `UiRegistry` before clients drain open/show/focus/refresh queues.
- Event streams are invalidation and feedback channels, not live-query result
  caches.
- `DashboardState` is a bounded client snapshot/cache. Durable runtime state
  still lives behind daemon primitives such as sessions, tasks, channels,
  worklists, events, memory, and KV.
- Keep local and remote transport behavior symmetric by going through
  `turin-control-client`.
- UI list requests should stay semantic. Do not expose raw daemon queries from
  this crate unless the UI contract explicitly grows that escape hatch.
- Worklist display helpers must stay stateless and renderer-neutral.
- Do not add renderer-specific concepts such as egui widgets, Ratatui layout,
  browser route state, or CSS classes to this crate.
- Connection/profile helpers may be shared here because they affect all Rust
  clients, but they should not become a general settings framework.

## Common Changes

Add a new operator command:

1. Add a typed `OperatorCommand` variant.
2. Use existing `turin-control-client` helpers or add a thin helper there first.
3. Emit a focused `UiUpdate` when the result should be handled specially.
4. Emit a snapshot refresh only when the command changes overview state.
5. Add focused tests for pure validation/mapping logic when possible.

Add a new semantic UI list source:

1. Extend `UiListRequest` handling in `controller.rs`.
2. Prefer typed control-client helpers over raw protocol values.
3. Return explicit unsupported-source errors until the loader exists.
4. Update at least one client smoke path if the new source is user-visible.

Add shared display derivation:

1. Put only stateless, renderer-neutral helpers in `worklist_view.rs` or a new
   narrow module.
2. Keep formatting suitable for multiple clients; renderer-specific truncation,
   colors, focus, and layout stay in the client crate.

## Tests

Focused checks:

```sh
cargo test -p turin-ui-core
cargo check -p turin-app
cargo check -p turin-tui
cargo check -p turin-web
```

Basic checks:

```sh
cargo fmt --all -- --check
git diff --check
```

## Current Shape

`turin-ui-core` currently shares the pieces that have proven common across the
new UI clients: connection/profile UX, dashboard refresh and event plumbing,
semantic harness UI indexing, bounded notices/events, UI list loading for
worklists, harness action command dispatch with returned UI intent application,
and small worklist summaries. It intentionally does not provide a common
active-screen model or shared UI session state; those seams should be extracted
later only if the clients independently converge on the same shape.
