# Turin UI Clients

`turin-tui`, `turin-app`, and `turin-web` are separate operator clients over the
same control layer. They share daemon/remote transport, dashboard updates, event
streaming, and harness UI intents. They do not share presentation state: each
client owns its own selected tab, screen, focus, modal, cache, and
scroll/selection state.

For the current per-client semantic UI support matrix, see
`docs/operations/ui-client-parity.md`.

Current layering:

- daemon / remote protocol: shared wire surface
- `turin-control-client`: typed local/remote control client
- `turin-ui-core`: shared connection options, profile loading, controller loop,
  dashboard state, and UI intent registry
- `turin-tui`: lean Ratatui terminal client
- `turin-app`: native graphical client built on egui/Cast
- `turin-web`: thin HTTP/SSE API plus minimal browser shell

All clients can talk to:

- a local Turin daemon over the existing local IPC transport
- a remote Turin daemon through `turin-remote`

## Current Client Scope

`turin-tui` is currently the keyboard-first low-capability client. It focuses on:

- runtime overview and notices
- harness UI apps, screens, nested menus, lists, forms, and actions
- confirmation for destructive or explicit-confirm harness actions
- latest harness action result visibility
- interactive terminal forms with local drafts, basic validation, and typed
  action params
- task and event tables with detail inspectors
- event-driven harness list refresh through `ui.refresh(...)` and
  `harness.action_ran`
- dynamic `ui.open`, `ui.show`, and `ui.focus` requests as local navigation
  suggestions
- dynamic navigation badges and worklist-backed report/chart summaries

`turin-app` is the broader graphical operator console. It currently has more
desktop-specific surface area, including the connection profile editor, editable
harness forms, dynamic UI navigation, latest action result panels, and wider
runtime tabs.

`turin-web` is an API-first web adapter with a small browser shell. It exposes
status, app registry, semantic list loading, action execution, and SSE
invalidation routes, and it renders the current semantic harness UI vocabulary
without a frontend build step. See `docs/operations/turin-web.md`.

The old chat-first TUI, TUI settings file, in-TUI connection editor, and session
transcript panes were removed during the clean TUI rebuild. Reintroduce those
capabilities only if they fit the new terminal UX model.

## Prerequisites

For local use, make sure the daemon is running:

```bash
turin daemon ensure --config .turin/config.toml
```

For remote use, make sure the daemon and bridge are running:

```bash
turin daemon ensure --config .turin/config.toml
turin-remote --config .turin/config.toml
```

See `docs/operations/daemon.md` for the daemon model and
`docs/operations/remote.md` for the network bridge.

## Build

```bash
cargo build --release -p turin-tui -p turin-app -p turin-web
```

## Local Usage

TUI against the local daemon:

```bash
target/release/turin-tui --config .turin/config.toml
```

Desktop app against the local daemon:

```bash
target/release/turin-app --config .turin/config.toml
```

Web client against the local daemon:

```bash
target/release/turin-web --config .turin/config.toml --bind 127.0.0.1:8787
```

If you want to bypass config-based endpoint resolution and point directly at a
daemon endpoint:

```bash
target/release/turin-tui --endpoint .turin/daemon.sock
target/release/turin-app --endpoint .turin/daemon.sock
target/release/turin-web --endpoint .turin/daemon.sock --bind 127.0.0.1:8787
```

## Remote Usage

Both clients can connect through `turin-remote`.

Using an explicit token:

```bash
target/release/turin-tui \
  --remote-url http://127.0.0.1:9324 \
  --auth-token "replace-me"

target/release/turin-app \
  --remote-url http://127.0.0.1:9324 \
  --auth-token "replace-me"

target/release/turin-web \
  --remote-url http://127.0.0.1:9324 \
  --auth-token "replace-me" \
  --bind 127.0.0.1:8787
```

Using an env var:

```bash
export TURIN_REMOTE_TOKEN="replace-with-a-long-random-token"

target/release/turin-tui \
  --remote-url http://127.0.0.1:9324 \
  --auth-token-env TURIN_REMOTE_TOKEN

target/release/turin-app \
  --remote-url http://127.0.0.1:9324 \
  --auth-token-env TURIN_REMOTE_TOKEN

target/release/turin-web \
  --remote-url http://127.0.0.1:9324 \
  --auth-token-env TURIN_REMOTE_TOKEN \
  --bind 127.0.0.1:8787
```

## Connection Profiles

Both clients can load shared connection profiles at startup.

By default, `--profile` reads from `.turin/ui-profiles.toml`:

```bash
target/release/turin-tui --profile local
target/release/turin-app --profile lab
```

You can also point at a different file:

```bash
target/release/turin-tui --profiles-file path/to/ui-profiles.toml --profile local
target/release/turin-app --profiles-file path/to/ui-profiles.toml --profile lab
```

Example:

```toml
default_profile = "local"

[profiles.local]
config = ".turin/config.toml"

[profiles.lab]
remote_url = "http://192.168.1.50:9324"
auth_token_env = "TURIN_REMOTE_TOKEN"
```

Profile rules:

- CLI flags win over profile-file values.
- `--profile <name>` selects `[profiles.<name>]`.
- `--profiles-file <path>` changes the source file.
- If `default_profile = "..."` exists and you pass `--profiles-file` without
  `--profile`, that default profile is used.
- Without `--profile` or `--profiles-file`, the clients do not auto-load a
  profile file.

There is a copyable example at `examples/config/ui-profiles.toml.example`.

## TUI Keyboard Model

Global keys:

- `Tab` / left / right: switch top-level tabs
- `j` / `k`: move the active selection
- `r`: refresh the current view
- `?`: toggle help
- `q` / Ctrl-C: quit

Harness tab keys:

- `[` / `]`: switch harness app
- `f`: cycle focus between navigation, visible work items, and actions
- `h` / `l`: switch screens directly
- `Enter`: open the selected navigation target or run the selected action
- `y` / `Enter`: confirm a pending action
- `n` / Esc: cancel a pending action

The TUI renders harness UI contracts semantically:

- screens and menus become terminal navigation
- nested menu items are flattened with indentation
- worklist-backed lists become compact terminal tables
- worklist-backed activity and detail nodes become compact recent-activity or
  snapshot/detail views
- worklist-backed report and chart nodes become lightweight summaries and bar
  breakdowns
- forms open a terminal editor that keeps draft values local to the client,
  validates required/numeric fields, coerces common scalar types, and submits
  merged action params
- unsupported list/detail/activity sources remain visible with source metadata
- desktop-only surfaces such as panes degrade to notices until the TUI has a
  native terminal representation for them

## Lightweight Footprint Check

For a no-build footprint baseline during UI work, run:

```bash
tools/footprint-report --top-files 30
```

The report is written under `.workspace/perf-reports/`, which is intentionally
ignored. It scans shipped Rust source roots, excludes obvious tests, examples,
target artifacts, and scratch data, and records release binary sizes only when
artifacts already exist. It does not build Turin.

The latest local UI-chapter sample on June 18, 2026 reported:

- `87333` Rust code lines under `src` and `crates`
- `4431` code lines in `crates/turin-app`
- `3552` code lines in `crates/turin-tui`
- `720` code lines in `crates/turin-web`

Use this as a trend signal, not a hard budget. The goal is to keep UI clients
lean and to notice accidental source or binary growth before it becomes normal.

## Current Limitations

- `turin-tui` does not currently include the old chat transcript view.
- `turin-tui` does not currently include an interactive connection profile
  editor.
- `turin-tui` does not read a separate `turin-tui.toml` settings file.
- `turin-tui` form editing is line-oriented; textarea/markdown fields degrade
  to single-value text editing for now.
- `turin-tui` work-item selection is local to visible compact list rows.
- `turin-tui` does not yet render panes beyond semantic notices.
- `turin-app` remains the richer graphical surface while the TUI proves the
  lean terminal abstraction.
