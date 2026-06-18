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

If no harness declares `ui.app(...)`, the clients still behave as default Turin
operator consoles. Runtime overview, tasks, events, status, and control actions
remain available; harness-specific screens appear when semantic UI intent is
declared.

## Current Client Scope

`turin-tui` is currently the keyboard-first low-capability client. It focuses on:

- runtime overview and notices
- harness UI apps, screens, nested menus, lists, forms, and actions
- confirmation for destructive or explicit-confirm harness actions
- latest harness action result visibility
- interactive terminal forms with local drafts, basic validation, and typed
  action params
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
  derived through shared stateless worklist helpers

`turin-app` is the broader graphical operator console. It currently has more
desktop-specific surface area, including the connection profile editor, editable
harness forms, app-local worklist row detail, dynamic UI navigation, latest
action result panels, returned action UI intent handling, app-local shown panes,
and wider runtime tabs.

`turin-web` is an API-first web adapter with a small browser shell. It exposes
status, app registry, semantic list loading, action execution, and SSE
invalidation routes, and it renders the current semantic harness UI vocabulary
without a frontend build step. When no harness declares `ui.app(...)`, the shell
renders a default runtime console from `/api/status` instead of a blank custom-UI
placeholder. Browser-local `ui.show` pane targets render as overlays and reuse
the same node/list adapters as screens. See `docs/operations/turin-web.md`.

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

Recent local sample from `.workspace/perf-reports/footprint-1781792349.md`:

| area | code lines |
| --- | ---: |
| `crates/turin-app` | 4,665 |
| `crates/turin-tui` | 3,843 |
| `crates/turin-ui-core` | 3,108 |
| `crates/turin-web` | 646 |

| static asset | bytes | lines |
| --- | ---: | ---: |
| `crates/turin-web/static/app.css` | 13,332 | 760 |
| `crates/turin-web/static/app.js` | 51,820 | 1,635 |
| `crates/turin-web/static/index.html` | 1,547 | 50 |

No binary sizes were recorded in that run. Use the release and idle-memory
baseline below when a UI change may affect startup or resident memory.

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

Recent no-build debug checkpoint from
`.workspace/perf-reports/ui-client-baseline-1781793121.md`:

| client | path | bytes | help max RSS KB |
| --- | --- | ---: | ---: |
| `turin-tui` | `target/debug/turin-tui` | 25,452,288 | 8,016 |
| `turin-app` | `target/debug/turin-app` | 69,948,320 | 10,836 |
| `turin-web` | `target/debug/turin-web` | 23,312,144 | 7,652 |

This is a local debug-artifact checkpoint only. Use release binaries for
meaningful size comparisons, but the debug sample is useful when disk space
rules out a release rebuild during UI iteration.

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
then sample the process. On Linux, prefer `smaps_rollup` when available:

```bash
pid="$(pgrep -n turin-web)"
awk '
  /^Rss:/ { rss_kb = $2 }
  /^Pss:/ { pss_kb = $2 }
  END { printf "rss_kb=%s pss_kb=%s\n", rss_kb, pss_kb }
' "/proc/$pid/smaps_rollup"
```

For `turin-tui` and `turin-app`, run the client in a normal terminal or desktop
session and sample the newest matching process with `pgrep -n turin-tui` or
`pgrep -n turin-app`. Close the client normally after sampling. Do not compare
numbers across machines as hard budgets; compare them against previous samples
from the same machine and release profile.

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
- worklist-backed lists become compact terminal tables
- selected worklist rows appear in the inspector; rows with item actions can be
  queued for confirmation from the item focus
- worklist-backed activity and detail nodes become compact recent-activity or
  snapshot/detail views
- worklist-backed report and chart nodes become lightweight summaries, next
  pending item highlights, and bar breakdowns
- forms open a terminal editor that keeps draft values local to the client,
  validates required/numeric fields, coerces common scalar types, and submits
  merged action params
- unsupported list/activity/detail/report/chart sources remain visible with
  source metadata
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

The latest local UI-chapter sample on June 18, 2026 reported:

- `85611` Rust code lines under `src` and `crates`
- `4665` code lines in `crates/turin-app`
- `3843` code lines in `crates/turin-tui`
- `646` code lines in `crates/turin-web`
- `66699` bytes across first-party `turin-web` static assets

Use this as a trend signal, not a hard budget. The goal is to keep UI clients
lean and to notice accidental source or binary growth before it becomes normal.

## Current Limitations

- `turin-tui` does not currently include the old chat transcript view.
- `turin-tui` does not currently include an interactive connection profile
  editor.
- `turin-tui` does not read a separate `turin-tui.toml` settings file.
- `turin-tui` form editing is compact; textarea/markdown fields support
  `Ctrl+J` newlines but still render as a preview rather than a rich editor.
- `turin-tui` work-item selection is local to visible compact list rows.
- `turin-tui` pane overlays use pane-local item/action focus separate from the
  main screen focus.
- `turin-app` remains the richer graphical surface while the TUI proves the
  lean terminal abstraction.
