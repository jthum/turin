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
- `crates/turin-ui-core/src/ui_actions.rs`
  - Stateless harness action result/failure app-scoping helpers shared by Rust
    clients.
- `crates/turin-ui-core/src/ui_copy.rs`
  - Stateless shared copy for semantic UI fallback and not-yet-loaded states.
- `crates/turin-ui-core/src/ui_badges.rs`
  - Stateless semantic UI badge text derivation shared by Rust clients.
- `crates/turin-ui-core/src/ui_data.rs`
  - Stateless semantic UI data-source helpers, including default worklist-backed
    surface limits and `UiListRequest` discovery from node trees.
- `crates/turin-ui-core/src/ui_navigation.rs`
  - Stateless semantic UI navigation helpers for declared screen target lookup
    and default screen selection from `opens_with`.
  - Stateless node target matching for semantic focus/open helpers; clients
    still decide local focus behavior.
- `crates/turin-ui-core/src/form_values.rs`
  - Stateless semantic UI form value helpers for default display values, field
    kind aliases, typed option preservation, password-field classification, and
    scalar coercion.
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
   helpers. A named worklist that has not been created yet reads as an empty
   collection; transport and query failures emit request-scoped `UiListFailed`
   updates so clients can clear local loading state and render retryable copy.
9. Clients can reuse stateless helpers to discover visible worklist-backed data
   requests from their own active screen/pane nodes without sharing active view
   state.

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
- Harness action result/failure app-scoping may be shared here, but latest
  result storage and rendering remain client-local operator feedback.
- Daemon status refreshes replace declared app surfaces in `UiRegistry` instead
  of merging them indefinitely. Local action/event badge hints remain client
  overlays, but removed apps, screens, panes, and menus must disappear after the
  next snapshot.
- Event streams are invalidation and feedback channels, not live-query result
  caches.
- `DashboardState` is a bounded client snapshot/cache. Durable runtime state
  still lives behind daemon primitives such as sessions, tasks,
  worklists, events, memory, and KV.
- Keep local and remote transport behavior symmetric by going through
  `turin-control-client`.
- UI list requests should stay semantic. Do not expose raw daemon queries from
  this crate unless the UI contract explicitly grows that escape hatch.
- UI list load failures should stay request-scoped. The shared controller can
  identify the failed semantic request, but retry state, visible error copy, and
  cache invalidation remain client-owned.
- A missing named worklist is an empty dynamic collection, not a load failure.
  This lets harness screens render before the first action creates their data.
- Worklist source validation, request-discovery, selected item key matching,
  list filter/sort field derivation, field label/sort marker display,
  default-screen target lookup, node target matching, show-target
  classification, refresh request selection, badge text derivation, and display
  helpers must stay stateless and renderer-neutral.
- Form value helpers may parse/default individual field values, but form drafts,
  field focus, validation display, submission timing, and modal state remain in
  each client.
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

Add shared form value behavior:

1. Keep helpers stateless and field-level.
2. Preserve typed option values before scalar parsing.
3. Keep form drafts, focus, validation messages, and submit lifecycle in the
   client crate.

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
semantic harness UI indexing, declared-surface replacement on status refresh,
bounded notices/events, UI list loading for named `worklists.<name>` sources,
stateless default-screen lookup, stateless node target matching, stateless
badge text derivation, stateless show-target classification, stateless action
feedback app-scoping, stateless visible-node request derivation,
stateless refresh request selection for matching semantic list bindings,
shared fallback/not-loaded copy, harness action command dispatch with returned
UI intent application, stateless form value coercion and password-field
classification, and small
worklist summaries, including stable work-item key matching for client-local
row selection, list filter/sort field summaries, and advisory sort-direction
labels for Rust-client table headers. It intentionally does not provide a
common active-screen model or shared UI session state; those seams should be
extracted later only if the clients independently converge on the same shape.
