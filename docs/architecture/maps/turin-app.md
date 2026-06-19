# Turin App Map

## Purpose

`turin-app` is the native graphical operator client for Turin. It owns desktop
layout, Cast/egui rendering, connection profile editing, local harness UI
navigation, modal state, selected worklist rows, form drafts, and graphical
degradation of semantic harness UI intent.

Keep this crate as a client over `turin-ui-core` and `turin-control-client`.
Runtime semantics, daemon transport, harness UI indexing, and shared
worklist/default-screen/node-target derivation belong outside this crate.

## Files

- `crates/turin-app/src/main.rs`
  - CLI flags, connection setup, controller wiring, app-local state, command
    dispatch, tab layout, modals, profile editor, and runtime inspectors.
- `crates/turin-app/src/harness_ui.rs`
  - Cast/egui renderer and pure navigation helpers for semantic harness UI
    apps, screens, menus, panes, nodes, forms, badges, and worklist-backed
    surfaces.
- `crates/turin-app/src/presentation.rs`
  - Small display helpers for status labels, truncation, summary cards, and
    reusable presentation formatting.

## Data Flow

1. `main.rs` builds `ConnectionOptions`, connects through `turin-ui-core`, and
   spawns a `UiController`.
2. `TurinDesktopApp` drains `UiUpdate` values into a local `DashboardState`.
3. App tabs, selected harness app/screen, selected list rows, open panes,
   confirmation modals, and form drafts remain app-local state.
4. Visible harness screens and panes are projected into `UiListRequest` values.
5. `OperatorCommand::LoadUiList` loads semantic worklist-backed data through the
   shared controller path.
6. Harness actions and forms emit local `HarnessUiEvent` values, then run
   through `OperatorCommand::RunHarnessAction`.
7. Dynamic `ui.open`, `ui.show`, `ui.focus`, and `ui.refresh` intents from
   runtime events and completed harness action results are drained as local
   navigation, pane, focus, and cache-invalidation requests.
8. `harness.action_ran` and explicit `ui.refresh(...)` invalidate matching
   visible list caches; they are not live queries.

## Invariants

- The app owns presentation state. Active tab, active screen, open pane,
  selected row, form draft, and confirmation state must not become runtime
  state.
- Cast widgets are renderer details. Do not leak Cast or egui concepts into the
  harness/Lua/protocol APIs.
- Harness UI intent is semantic. Render, degrade, or ignore by capability, but
  do not mutate the contract to fit desktop layout.
- Unsupported list/activity/detail/report/chart sources should remain visible
  with explicit fallback copy rather than becoming blank panels.
- Worklist item actions are ordinary harness actions from the client point of
  view and should stay behind confirmation when launched from item detail.
- Form drafts are local until submit. Submitted values become action params
  only through the action command path.
- Latest action results are local operator feedback. Returned UI intents from
  those action results are local presentation hints. Durable workflow state
  belongs in runtime primitives such as worklists, events, memory, or KV.
- Keep `turin-ui-core` extraction conservative: only move stateless,
  renderer-neutral helpers there after app/TUI/web behavior independently
  converges.

## Common Changes

Add or change a harness UI renderer:

1. Update `harness_ui.rs`.
2. Preserve explicit fallback behavior for unsupported sources or widgets.
3. Keep app-local state in `main.rs`; do not add shared active-screen/session
   state.
4. Check the Release Operator fixture still loads.

Add a command or modal:

1. Route daemon work through `OperatorCommand`.
2. Keep confirmation/error/action-result feedback local to the app.
3. Avoid blocking the egui frame loop; use the existing controller/runtime path.

## Tests

Focused checks:

```sh
cargo check -p turin-app
cargo check -p turin-ui-core
cargo test -p turin --lib harness::engine::tests::test_ui_release_operator_example_loads
```

Basic checks:

```sh
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current app is the richest local client. It includes connection profile
editing, runtime overview tabs, semantic harness app rendering, nested menu
navigation, app-local row detail with filter/sort/limit metadata, row-count and
selected-row feedback, sorted-column markers, pause/claim/failure item context,
filtered empty-list copy, editable forms, confirmation modals, latest action
result feedback, shown panes, dynamic UI navigation/focus/refresh
handling, dynamic badges, and lightweight worklist-backed activity, detail,
report, and chart surfaces with explicit no-data copy and grouping hints. Form
field defaults and typed scalar coercion come from stateless `turin-ui-core`
helpers, default-screen and node-target lookup also come from `turin-ui-core`,
and drafts and controls remain app-local. It should remain replaceable at the
presentation layer while preserving the semantic client contract.
