# TUI Map

## Purpose

`turin-tui` is the terminal client for Turin. It owns terminal layout, keyboard
interaction, focus, selection, confirmation modals, and terminal-specific
degradation of harness UI intent.

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
5. Harness UI actions run through `OperatorCommand::RunHarnessAction`.
6. One-shot `ui.open`, `ui.show`, and `ui.focus` requests are drained into
   local TUI navigation state.
7. `ui.refresh(...)` and `harness.action_ran` invalidate visible list caches.

## Invariants

- TUI state is client-owned: selected tab, screen, action, modal, and cache
  state must not become runtime state.
- Harness menu items are semantic navigation targets. The TUI may flatten nested
  menus into terminal navigation, but it must not mutate the harness contract to
  fit terminal layout.
- Dynamic UI requests are suggestions to this client. Applying `ui.open` or
  `ui.focus` changes only local TUI selection/focus state.
- Rendering functions should not perform daemon requests directly.
- Harness UI rendering must degrade semantically instead of assuming desktop
  widgets exist.
- Keep keyboard behavior discoverable through the help overlay/footer.
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
navigation, dynamic open/focus handling, task list, event list, confirmation
flow, UI notices, and list invalidation. Chat, search, connection profile
editing, and deeper inspectors should be reintroduced only as they fit the new
terminal UX model.
