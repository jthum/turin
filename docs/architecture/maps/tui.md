# TUI Map

## Purpose

`turin-tui` is the terminal client for Turin. It owns terminal layout, keyboard
interaction, focus, work-item selection, confirmation modals, and
terminal-specific degradation of harness UI intent. Interactive form drafts,
field focus, and shown pane overlays are also TUI-local state.

Keep this crate as a lean Ratatui client. Runtime semantics, daemon transport,
and shared operator commands belong in `turin-ui-core`, `turin-control-client`,
and `turin-daemon-protocol`.

## Files

- `crates/turin-tui/src/main.rs`
  - CLI flags, connection setup, controller spawn, and app bootstrap.
- `crates/turin-tui/src/terminal.rs`
  - Ratatui lifecycle, event polling, update draining, and render loop.
- `crates/turin-tui/src/app.rs`
  - TUI-owned state, keyboard handling, command dispatch, list cache
    invalidation, and top-level screen rendering.
- `crates/turin-tui/src/harness_ui.rs`
  - Terminal renderer and pure navigation helpers for harness UI app/screen/menu/node
    contracts.
- `crates/turin-tui/src/theme.rs`
  - Terminal color/style tokens.

## Data Flow

1. `main.rs` builds `ConnectionOptions` and connects with `turin-ui-core`.
2. `UiController` emits `UiUpdate` values from snapshots, events, and commands.
3. `TuiApp` applies updates to `DashboardState` and TUI-local state.
4. Keyboard input mutates TUI-local state or sends `OperatorCommand`.
5. Harness UI list rows can be selected locally and rendered in the inspector.
6. Selected harness UI work items can queue their declared item action for
   confirmation when an action payload exists.
7. Harness UI actions run through `OperatorCommand::RunHarnessAction`.
8. Harness UI forms open a terminal-local editor; submit merges typed field
   values over form params and runs the declared harness action.
9. Completed harness action results are retained as local operator feedback,
   applied for returned dynamic UI intents, and rendered in the inspector.
10. One-shot `ui.open`, `ui.show`, and `ui.focus` requests are drained into
   local TUI navigation or overlay state.
11. Shown panes render as terminal-local overlays, reuse the visible screen/list
   request path for any pane nodes, and can select pane-local work items or run
   pane-local actions/forms through the same harness action path as screens.
12. `ui.refresh(...)` and `harness.action_ran` invalidate visible list caches.

## Invariants

- TUI state is client-owned: selected tab, screen, work item, action, modal,
  and cache state must not become runtime state.
- Form drafts, field focus, and validation errors are client-local. Submitted
  values become action params only when the operator submits the form.
- Latest harness action results are client-local feedback. Returned UI intents
  from those action results are local presentation hints. Durable workflow state
  should still be stored through runtime primitives such as worklists, events,
  memory, or KV.
- Harness menu items are semantic navigation targets. The TUI may flatten nested
  menus into terminal navigation, but it must not mutate the harness contract to
  fit terminal layout.
- Dynamic UI requests are suggestions to this client. Applying `ui.open` or
  `ui.focus` changes only local TUI selection/focus state, and applying
  `ui.show` to a pane changes only local overlay state.
- Rendering functions should not perform daemon requests directly.
- Harness UI rendering must degrade semantically instead of assuming desktop
  widgets exist.
- Worklist-backed `list` nodes render as compact terminal tables with
  terminal-local row-position and action-available cues. When selection moves
  beyond the first visible page, the renderer windows rows around the selected
  item instead of making selection disappear off-screen. Other list sources
  remain visible with metadata and an explicit unsupported-adapter message
  until the client has a loader for that source.
- Worklist item actions are ordinary harness action runs from the client's
  point of view, but the TUI must queue them for confirmation because work item
  payloads do not carry a UI-specific confirm flag.
- Worklist-backed `activity` and `detail` nodes render from bounded cached
  worklist data. They are not live queries, and they must remain client-local
  views over existing control-plane data.
- Worklist-backed `report` and `chart` nodes render lightweight terminal
  summaries from bounded cached worklist data using stateless worklist
  derivation helpers from `turin-ui-core`. This is still a client adapter, not a
  shared analytics/query semantics layer.
- Form nodes render as editable terminal modals. Textarea/markdown fields can
  accept `Ctrl+J` newlines but still display as compact previews. Unsupported
  rich form controls should degrade to text/option/boolean editing rather than
  forcing renderer concepts into the protocol.
- Pane targets render as terminal overlays with their own local item/action
  focus. Pane item/action selection must not overload screen item/action focus.
- Keep keyboard behavior discoverable through the help overlay/footer.
- Page and boundary navigation (`PageUp`, `PageDown`, `Home`, `End`) should stay
  local to the currently focused region and must not become runtime state.
- Harness focus cycling should skip empty item/action regions and fall back to
  navigation rather than trapping the operator on blank panes.
- Prefer small modules over rebuilding a monolithic terminal app.

## Common Changes

Add a harness UI node renderer:

1. Update `harness_ui.rs`.
2. Preserve fallback text for unsupported terminal renderings.
3. Check the `examples/harnesses/ui_release_operator` fixture still loads and
   remains usable.

Add a command or keybinding:

1. Add state/effect handling in `app.rs`.
2. Keep the footer/help overlay in sync.
3. Route daemon work through `OperatorCommand` rather than direct client calls.

## Tests

Focused checks:

```sh
cargo check -p turin-tui
cargo check -p turin-ui-core
cargo test -p turin --lib harness::engine::tests::test_ui_release_operator_example_loads
```

Basic checks:

```sh
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current TUI foundation is intentionally smaller than the previous terminal
client. It starts with an operator overview, harness app rendering, nested menu
navigation, focus cycling that skips empty regions, local work-item selection
with inspector detail, selected-row table windowing, page/boundary navigation in
focused regions, dynamic open/focus handling, shown pane overlays with
pane-local item/action selection, editable forms, worklist-backed
activity/detail/report/chart surfaces, latest action result feedback, task and
event inspectors, confirmation flow, UI notices, and list invalidation. Chat,
search, connection profile editing, and deeper inspectors should be reintroduced
only as they fit the new terminal UX model.
