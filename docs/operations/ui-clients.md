# Turin UI Clients

`turin-tui`, `turin-app`, and `turin-web` are separate operator clients over the
same control layer. They share daemon/remote transport, dashboard updates, event
streaming, and harness UI intents. They do not share presentation state: each
client owns its own selected tab, screen, focus, modal, cache, and
scroll/selection state.

For the current per-client semantic UI support matrix, see
`docs/operations/ui-client-parity.md`. For the product framing behind default
consoles and app-like harness UI, see
`docs/concepts/harness-apps-and-ui-clients.md`.

Current layering:

- daemon / remote protocol: shared wire surface
- `turin-control-client`: typed local/remote control client
- `turin-ui-core`: shared connection options, profile loading, controller loop,
  dashboard state, UI intent registry, and stateless semantic UI data helpers
- `turin-tui`: lean Ratatui terminal client
- `turin-app`: native graphical client built on egui/Cast
- `turin-web`: thin HTTP/SSE API plus minimal browser shell

All clients can talk to:

- a local Turin daemon over the existing local IPC transport
- a remote Turin daemon through `turin-remote`

If no harness declares `ui.app(...)`, the clients still behave as default Turin
operator consoles. Runtime overview, tasks, events, status, and control actions
remain available; harness-specific screens appear when semantic UI intent is
declared. `turin-app` and `turin-web` show default console summaries with
runtime/work/UI signal counts in the UI area, while `turin-tui` shows the same
default-console guidance and counts in its harness tab.

## Current Client Scope

`turin-tui` is currently the keyboard-first low-capability client. It focuses on:

- runtime overview and notices
- default no-harness console guidance and summary counts in the harness tab
- harness UI apps, screens, nested menus, lists, forms, and actions
- confirmation for destructive or explicit-confirm harness actions
- latest harness action result/failure visibility
- interactive terminal forms with local drafts, basic validation, and typed
  action params
- local worklist row selection that preserves selected item identity across
  refresh/reorder where possible
- selected work-item inspectors with timeline, pause, claim, parent, failure,
  metadata, and action context
- task and event tables with detail inspectors
- page and boundary navigation inside the currently focused terminal region
- event-driven harness list refresh through `ui.refresh(...)` and
  `harness.action_ran`
- dynamic `ui.open`, `ui.show`, and `ui.focus` requests as local navigation
  suggestions
- returned harness action UI intents as local navigation/notice/refresh hints
- terminal-local shown pane overlays that reuse the same semantic node/list
  adapters as screens
- dynamic navigation/node badges and worklist-backed report/chart summaries
  derived through shared stateless UI/worklist helpers

`turin-app` is the broader graphical operator console. It currently has more
desktop-specific surface area, including the connection profile editor, a
default no-harness console summary, editable harness forms, app-local worklist
row detail with timeline and operational context, row-level work-item action
cues, dynamic UI navigation, latest action result/failure panels, returned
action UI intent handling, app-local shown panes, and wider runtime tabs.

`turin-web` is an API-first web adapter with a small browser shell. It exposes
status, app registry, semantic list loading, action execution, and SSE
invalidation routes, and it renders the current semantic harness UI vocabulary
without a frontend build step. When no harness declares `ui.app(...)`, the shell
renders a default runtime console from `/api/status` instead of a blank custom-UI
placeholder. Browser list rows include action-availability cues, selected-row
timeline and operational context, and browser-local `ui.show` pane targets
render as overlays and reuse the same node/list adapters as screens. See
`docs/operations/turin-web.md`.

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

## Footprint Watchpoint

The UI clients should stay lean enough for older machines. Use the no-build
footprint report for a quick Rust source and first-party web asset snapshot:

```bash
tools/footprint-report --top-files 30
```

If debug or release binaries already exist and you want them recorded without
triggering a build, pass them explicitly:

```bash
tools/footprint-report \
  --top-files 30 \
  --binary target/debug/turin-tui \
  --binary target/debug/turin-app \
  --binary target/debug/turin-web
```

Recent local release-backed sample from
`.workspace/perf-reports/footprint-1781856843-1915216.md`:

| area | code lines |
| --- | ---: |
| `crates/turin-app` | 4,822 |
| `crates/turin-tui` | 4,136 |
| `crates/turin-ui-core` | 3,596 |
| `crates/turin-web` | 687 |

| static asset | bytes | lines |
| --- | ---: | ---: |
| `crates/turin-web/static/app.css` | 14,169 | 800 |
| `crates/turin-web/static/app.js` | 65,785 | 2,062 |
| `crates/turin-web/static/index.html` | 1,600 | 52 |

| release binary | bytes |
| --- | ---: |
| `target/release/turin-tui` | 5,565,984 |
| `target/release/turin-app` | 14,098,520 |
| `target/release/turin-web` | 5,134,664 |

Use the release and idle-memory baseline below when a UI change may affect
startup or resident memory.

## Release And Idle-Memory Baseline

Use this lightweight procedure when a UI change might affect footprint,
startup, or idle memory. Record the commit, OS, display environment, and whether
the daemon was local or remote beside the numbers.

For existing binaries, prefer the no-build UI client collector:

```bash
tools/ui-client-baseline
```

It records release-client binary sizes and `--help` startup elapsed/RSS values
when `target/release/turin-tui`, `target/release/turin-app`, or
`target/release/turin-web` already exist. It writes JSON and Markdown reports to
`.workspace/perf-reports/` and never builds Turin. Use `--skip-help` when you
only want binary sizes.

Recent no-build release checkpoint from
`.workspace/perf-reports/ui-client-baseline-1781856844-1915341.md`:

| client | path | bytes | help max RSS KB |
| --- | --- | ---: | ---: |
| `turin-tui` | `target/release/turin-tui` | 5,565,984 | 5,104 |
| `turin-app` | `target/release/turin-app` | 14,098,520 | 6,476 |
| `turin-web` | `target/release/turin-web` | 5,134,664 | 5,304 |

This is a local release-artifact checkpoint for `--help` startup only. For true
idle memory, run the client against a daemon and sample the live process with
the procedure below.

Build release clients:

```bash
cargo build --release -p turin-tui -p turin-app -p turin-web
```

Record source and release-binary footprint:

```bash
tools/footprint-report \
  --top-files 30 \
  --binary target/release/turin-tui \
  --binary target/release/turin-app \
  --binary target/release/turin-web
```

Record release-client binary size and `--help` startup RSS/elapsed:

```bash
tools/ui-client-baseline
```

For idle memory, start the client against a local daemon, wait a few seconds,
then sample the process. On Linux, use the idle snapshot collector; it prefers
`smaps_rollup` and falls back to `/proc/<pid>/status` when needed:

```bash
tools/ui-client-idle-snapshot --settle 5
```

To sample a known process explicitly:

```bash
tools/ui-client-idle-snapshot --pid "$pid"
```

For `turin-tui` and `turin-app`, run the client in a normal terminal or desktop
session before sampling. Close the client normally after sampling. Do not
compare numbers across machines as hard budgets; compare them against previous
samples from the same machine and release profile.

Recent local live release-client checkpoints:

| client | path | rss KB | pss KB | source |
| --- | --- | ---: | ---: | --- |
| `turin-tui` | `target/release/turin-tui` | 6,172 | 3,646 | `smaps_rollup` |
| `turin-app` | `target/release/turin-app` | 92,656 | 83,198 | `smaps_rollup` |
| `turin-web` | `target/release/turin-web` | 5,564 | 3,042 | `smaps_rollup` |

Reports:

- `.workspace/perf-reports/ui-client-idle-snapshot-1781831522-1764905.md`
- `.workspace/perf-reports/ui-client-idle-snapshot-1781831569-1765289.md`
- `.workspace/perf-reports/ui-client-idle-snapshot-1781831251-1761607.md`

These samples launched release clients against `.turin/config.toml`, waited a
few seconds, sampled the process, and shut it down. Treat the values as
same-machine trend baselines, not universal budgets.

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

All three clients can connect through `turin-remote`.

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

All three clients can load shared connection profiles at startup.

By default, `--profile` reads from `.turin/ui-profiles.toml`:

```bash
target/release/turin-tui --profile local
target/release/turin-app --profile lab
target/release/turin-web --profile lab --bind 127.0.0.1:8787
```

You can also point at a different file:

```bash
target/release/turin-tui --profiles-file path/to/ui-profiles.toml --profile local
target/release/turin-app --profiles-file path/to/ui-profiles.toml --profile lab
target/release/turin-web --profiles-file path/to/ui-profiles.toml --profile lab --bind 127.0.0.1:8787
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
- `g` / `G`: jump to the first or last selectable row
- `r`: refresh the current view
- `?`: toggle help
- `q` / Ctrl-C: quit

Harness tab keys:

- `[` / `]`: switch harness app
- `f`: cycle focus through navigation and non-empty item/action regions
- `h` / `l`: switch screens directly
- `Enter`: open the selected navigation target, queue the selected work-item
  action for confirmation, or run the selected action
- `y` / `Enter`: confirm a pending action
- `n` / Esc: cancel a pending action

The TUI renders harness UI contracts semantically:

- screens and menus become terminal navigation
- nested menu items are flattened with indentation
- worklist-backed lists become compact terminal tables with an `action` cue
  when a row has an item action that will be queued for confirmation, plus
  visible row-range and selected-row feedback
- selected worklist rows appear in the inspector; row identity is preserved
  across refresh/reorder where possible, and rows with item actions can be
  queued for confirmation from the item focus; inspector detail includes
  worklist, pause, claim, parent, failure, metadata, and action context
- worklist-backed activity and detail nodes become compact recent-activity or
  snapshot/detail views
- worklist-backed report and chart nodes become lightweight summaries, next
  pending item highlights, and bar breakdowns
- forms open a terminal editor that keeps draft values local to the client,
  validates required/numeric fields, coerces common scalar types, and submits
  merged action params
- unsupported and failed list/activity/detail/report/chart loads remain visible
  with source metadata or retryable error copy instead of blank panels
- shown panes render as terminal overlays, load any visible worklist-backed
  nodes through the same cache/invalidation path as screens, and support
  pane-local item/action/form selection with confirmation when needed

## Lightweight Footprint Check

For a no-build footprint baseline during UI work, run:

```bash
tools/footprint-report --top-files 30
```

The report is written under `.workspace/perf-reports/`, which is intentionally
ignored. It scans shipped Rust source roots, excludes obvious tests, examples,
target artifacts, scratch data, and inline `#[cfg(test)] mod tests` blocks,
records first-party web static asset bytes/lines separately, and records release
binary sizes only when artifacts already exist. It does not build Turin.

The latest local UI-chapter sample
(`.workspace/perf-reports/footprint-1781853200-1869339.md`) reported:

- `86320` Rust code lines under `src` and `crates`
- `4723` code lines in `crates/turin-app`
- `4004` code lines in `crates/turin-tui`
- `3563` code lines in `crates/turin-ui-core`
- `676` code lines in `crates/turin-web`
- `80380` bytes across first-party `turin-web` static assets

Use this as a trend signal, not a hard budget. The goal is to keep UI clients
lean and to notice accidental source or binary growth before it becomes normal.
The repo `.ignore` also excludes local build artifacts such as `target/`,
`tools/perf-suite/target/`, and `.workspace/` from ripgrep scans so footprint
audits and agent context gathering do not accidentally walk generated output.

## Current Limitations

- `turin-tui` does not currently include the old chat transcript view.
- `turin-tui` does not currently include an interactive connection profile
  editor.
- `turin-tui` does not read a separate `turin-tui.toml` settings file.
- `turin-tui` form editing is compact; textarea/markdown fields support
  `Ctrl+J` newlines but still render as a preview rather than a rich editor.
- `turin-tui` work-item selection is local to compact list rows; it is
  remembered by item identity where possible and falls back safely when a row
  disappears.
- `turin-tui` pane overlays use pane-local item/action focus separate from the
  main screen focus.
- `turin-app` remains the richer graphical surface while the TUI proves the
  lean terminal abstraction.
