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
3. Without a harness-declared app, the client projects agents, live sessions,
   session detail, and focused session events into a default conversation
   workspace. Runtime inspection remains available through an explicit tools
   sheet rather than becoming the landing page.
4. App tabs, selected conversation, pending first prompt, selected harness
   app/screen, selected list rows, open panes, confirmation modals, and form
   drafts remain app-local state.
5. Visible harness screens and panes are projected into `UiListRequest` values.
6. `OperatorCommand::LoadUiList` loads semantic worklist-backed data through the
   shared controller path. Request-scoped list failures clear local loading
   state and render explicit retryable fallback copy.
7. Harness actions and forms emit local `HarnessUiEvent` values, then run
   through `OperatorCommand::RunHarnessAction`.
8. Dynamic `ui.open`, `ui.show`, `ui.focus`, and `ui.refresh` intents from
   runtime events and completed harness action results are drained as local
   navigation, pane, focus, and cache-invalidation requests.
9. `harness.action_ran` and explicit `ui.refresh(...)` invalidate matching
   visible list caches; they are not live queries.

## Invariants

- The app owns presentation state. Active tab, active screen, open pane,
  selected row, form draft, and confirmation state must not become runtime
  state.
- The no-harness path is an opinionated agent workspace, not a runtime
  diagnostics dashboard. Diagnostics belong behind deliberate secondary
  navigation and ordinary surfaces should not expose socket paths, internal
  session ids, or raw protocol payloads.
- Opening a conversation and attaching its first prompt is client orchestration
  over typed session commands. The pending selection is transient client state;
  durable conversation state remains in the runtime.
- Cast widgets are renderer details. Do not leak Cast or egui concepts into the
  harness/Lua/protocol APIs.
- Harness UI intent is semantic. Render, degrade, or ignore by capability, but
  do not mutate the contract to fit desktop layout.
- Semantic sections establish hierarchy but do not automatically become nested
  panels. Use bounded Cast surfaces for data, forms, reports, and detail while
  keeping prose and action groups in the page flow.
- Internal source names, report prompts, public ids, and query mechanics are
  not ordinary application copy. Keep them out of default surfaces and expose
  them only through deliberate technical detail or Runtime Tools.
- The harness app shell must remain usable below desktop-sidebar width. Compact
  mode replaces the permanent sidebar with app selection and scrollable screen
  tabs while preserving client-local navigation state.
- Unsupported list/activity/detail/report/chart sources should remain visible
  with explicit fallback copy rather than becoming blank panels.
- Failed list/activity/detail/report/chart loads should remain visible with
  explicit local error copy rather than staying in a loading state.
- Worklist item actions are ordinary harness actions from the client point of
  view and should stay behind confirmation when launched from item detail.
- Form drafts are local until submit. Submitted values become action params
  only through the action command path.
- Password-like form fields render with Cast password inputs when declared, but
  this remains a client presentation behavior; submitted values are still
  ordinary form params.
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
navigation, app-local row detail with named filter/sort/limit metadata,
row-count and selected-row feedback, direction-aware sorted-column markers,
timeline/pause/claim/failure item context, filtered empty-list copy, editable
forms with masked password-like fields, confirmation modals, latest action
result feedback, shown panes,
dynamic UI navigation/focus/refresh
handling, dynamic badges, and lightweight worklist-backed activity, detail,
report, and chart surfaces with explicit no-data copy, grouping hints, and
percentage labels. Form field defaults and typed scalar coercion come from
stateless `turin-ui-core` helpers, default-screen, node-target, badge-text, and
action-feedback
app-scoping lookup also come from `turin-ui-core`, request-scoped list load
failures are rendered through app-local cache/error state, and drafts and
controls remain app-local. It should remain replaceable at the presentation
layer while preserving the semantic client contract.

The harness renderer uses a responsive shell: wide windows receive a compact
sidebar, while narrower windows receive app selection and screen tabs above the
content stage. Top-level semantic sections remain lightweight; lists and detail
use Cast cards, forms and reports use their dedicated Cast compositions, and
loading uses skeletons. Narrow lists degrade from tables to selectable Cast
list rows with the declared fields summarized beneath each title. Default list
columns favor application content (`title`, `status`, `kind`, `priority`) rather
than exposing public ids, and report prompts remain execution metadata rather
than rendered prose.

When no harness declares an app surface, the desktop client renders a focused
conversation workspace built from Cast. It provides agent selection when
needed, active and recent conversation navigation, durable-session resume,
structured message and tool-call presentation, a persistent composer, and a
first-prompt flow that opens and titles a session before dispatching the
prompt. Connection details, sessions, tasks, channels, events, and custom-app
inspection remain available through Runtime Tools without competing with the
primary workflow. The shell keeps its own selection and pending-command state
and uses focused session events to invalidate session detail; it does not
create runtime-owned view state.
