# Turin UI Clients

`turin-tui` and `turin-app` are built on the same transport-agnostic control layer, but they are no longer shaped the same way.

- `turin-tui` is now chat-first, with optional side panes for session navigation and live inspection.
- `turin-app` is still the broader operator console.

They can both talk to:

- a local Turin daemon over the existing local IPC transport
- a remote Turin daemon through `turin-remote`

That means the same core workflows work in both modes:

- inspect agents, sessions, tasks, channels, and events
- open live sessions
- resume stored sessions
- submit prompts to live sessions
- cancel tasks or sessions
- inspect recent session transcript and tool history

## Turin TUI: Chat-First Mode

`turin-tui` now opens on a dedicated Chat view instead of dropping you into an operator dashboard first.

The default shape is:

- left pane: sessions
- center: transcript
- right pane: thinking

The chat view is backed by the same daemon/remote event stream as the rest of Turin, so it can render:

- persisted transcript from session detail
- pending user prompts before the next session refresh lands
- streamed assistant previews from `message_delta`
- streamed thinking in a separate inspector pane when the model/provider emits `thinking_delta`

The current chat hotkeys are:

- `Enter`: prompt the current live chat session, or open/resume the selected agent/session from the left pane
- `p`: prompt the current live chat session
- `,`: cycle the left pane between `sessions`, `agents`, `channels`, `events`, and `none`
- `.`: cycle the right pane between `thinking`, `tools`, `events`, `session`, and `none`
- `h`: show/hide the thinking pane
- `v`: show/hide streamed preview text
- `f`: toggle chat follow-latest
- `PageUp` / `PageDown`: scroll through the loaded transcript window
- `Home` / `End`: jump toward the oldest loaded lines or back to the latest output

The transcript view uses a bounded in-memory window instead of keeping the entire conversation rendered at once. Older persisted history can still be reloaded on demand from Turin storage.

## Prerequisites

For local use, make sure the daemon is running:

```bash
turin daemon ensure --config turin.toml
```

For remote use, make sure both are running:

```bash
turin daemon ensure --config turin.toml
turin-remote --config turin.toml
```

See `docs/operations/daemon.md` for the daemon model and `docs/operations/remote.md` for the network bridge.

## Build

```bash
cargo build --release -p turin-tui -p turin-app
```

## Local Usage

TUI against the local daemon:

```bash
target/release/turin-tui --config turin.toml
```

Desktop app against the local daemon:

```bash
target/release/turin-app --config turin.toml
```

If you want to bypass config-based endpoint resolution and point directly at a daemon endpoint:

```bash
target/release/turin-tui --endpoint .turin/daemon.sock
target/release/turin-app --endpoint .turin/daemon.sock
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
```

## Connection Profiles

The UI clients share one connection-profile format.

By default, `--profile` reads from `ui-profiles.toml` in the current working directory:

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
config = "turin.toml"

[profiles.lab]
remote_url = "http://192.168.1.50:9324"
auth_token_env = "TURIN_REMOTE_TOKEN"
```

There is also a copyable example at `ui-profiles.toml.example`.

## Turin TUI Settings

`turin-tui` also has its own UI settings file, separate from connection profiles.

By default it looks for `turin-tui.toml` in the current working directory. You can override that with:

```bash
target/release/turin-tui --tui-config path/to/turin-tui.toml
```

A copyable example is included at `turin-tui.toml.example`.

Current settings:

- `[layout].left_pane`
- `[layout].right_pane`
- `[chat].transcript_memory_budget_bytes`
- `[chat].show_streaming_preview`
- `[chat].show_thinking`
- `[chat].follow_latest`
- `[chat].user_label`

The Settings tab inside `turin-tui` can change those values interactively, and `w` writes them back to the configured `turin-tui.toml`.

Profile rules:

- CLI flags win over profile-file values
- `--profile <name>` selects `[profiles.<name>]`
- `--profiles-file <path>` changes the source file
- if `default_profile = "..."` exists and you pass `--profiles-file` without `--profile`, that default profile is used
- without `--profile` or `--profiles-file`, the clients do not auto-load a profile file

## In-UI Profile Switching

Both operator shells now expose a dedicated Connections view on top of the shared profile catalog.

Current behavior:

- both clients can load and display profiles from `ui-profiles.toml` or an explicit `--profiles-file`
- both clients can switch the active backend to a selected profile without restarting the UI
- both clients can edit a profile draft with connection kind, target, and auth settings
- both clients track whether the editor draft has diverged from the last loaded/saved baseline, so it is obvious when you are holding unsaved connection changes
- both clients validate the edited draft inline before saving, including remote URL and auth checks
- both clients can diff the editor draft against the selected saved profile before overwriting it
- both clients keep the latest preflight result in the Connections view, including target, auth source, readiness, and latency
- both clients keep lightweight in-memory profile activity metadata like connect attempts, successful connects, and last preflight result for the selected saved profile
- both clients can connect directly from the current unsaved draft, so you can test edits before writing them to disk
- both clients keep a small recent-successful-drafts history so you can reload draft connections after switching away
- both clients can save the edited draft back into the profile file
- both clients can duplicate or rename an existing profile from inside the UI
- both clients can delete a selected profile from inside the UI
- the current connection target and active profile are shown in the shell chrome
- reloading the profile file happens inside the UI, so you can edit the file and refresh the profile list
- both clients can run a draft or selected-profile preflight before reconnecting
- both clients can ask Turin to `daemon ensure` for local-config drafts, which is useful when you are switching from a remote profile back to a local workspace

The desktop app exposes this through the Connections tab controls:

- `Load Current` copies the active connection into the editor draft
- `Load Selected` copies the highlighted stored profile into the editor draft
- `Load Latest Recent` reloads the most recent successful draft connection into the editor
- `New Draft` resets the editor to a fresh draft
- a Recent Drafts list shows recent successful draft connections and lets you load one back into the editor
- edit the draft kind, target, and remote auth mode/value inline in the Connections tab
- invalid draft fields are highlighted inline, `Update Selected` stays disabled until the draft is valid, and `Save As Name` also requires a typed target name
- when the editor is dirty, the Connections tab shows which baseline it differs from and which fields changed
- `Update Selected` overwrites the highlighted saved profile in place using the current draft
- `Save As Name` writes the edited draft into the profile file under the typed profile name
- `Test Draft` preflights the current unsaved draft without reconnecting
- `Test Selected` preflights the highlighted saved profile
- `Ensure Draft Local` runs `turin daemon ensure --config ...` for local-config drafts
- `Connect Draft` switches the UI to the currently edited draft without saving it first
- `Duplicate Selected` copies the highlighted profile to the typed name
- `Rename Selected` renames the highlighted profile to the typed name
- `Set as default` marks the saved profile as `default_profile`
- if you try to load another profile while the editor is dirty, the Connections tab makes you explicitly discard or cancel the pending action
- delete is a two-step flow: `Arm Delete`, then `Confirm Delete` or `Cancel Delete`

The TUI still exposes the profile system through the Connections tab plus keyboard actions:

- `Enter` or `s` connects to the selected profile
- `v` loads the current connection into the profile draft
- `b` loads the selected stored profile into the draft
- if the draft is dirty, `v`, `b`, and `R` switch into explicit discard confirmation before replacing the editor
- `m` cycles the draft kind between local-config, local-endpoint, and remote
- `o` cycles the draft auth mode for remote drafts
- `t` edits the draft target
- `g` edits the draft auth value
- the detail pane and footer show draft validation issues before save, and `g` only edits auth when the draft is using env or inline auth
- `T` preflights the current draft without reconnecting
- `P` preflights the selected saved profile
- `E` runs `turin daemon ensure --config ...` for local-config drafts
- `C` connects to the current draft without saving it first
- `S` overwrites the selected saved profile in place using the current draft
- `R` loads the selected recent draft back into the editor
- `[` and `]` move through the recent draft history shown in the detail pane
- `a` saves the current draft to a typed profile name
- `A` saves the current draft to a typed profile name and marks it as the default profile
- `y` duplicates the selected profile to a typed name
- `Y` duplicates the selected profile to a typed name and marks it default
- `u` renames the selected profile to a typed name
- `U` renames the selected profile to a typed name and marks it default
- `d` enters delete confirmation for the selected profile
- `l` reloads the profile file

If you delete the profile that the current UI session was launched from, the clients detach that running connection from the deleted profile entry so `Reconnect Current` continues to work. The TUI requires `y` or `Enter` to confirm a delete once it is armed, and `n` or `Esc` cancels it.

## Session Detail Loading

Session transcripts and tool history are loaded lazily when you focus a live session or stored session.

That keeps the initial dashboard refresh lightweight while still giving you richer operator detail when you drill into a session.

Current behavior:

- the chat-first TUI transcript pane loads session detail lazily and then overlays live streamed output on top
- the TUI detail pane in non-chat tabs expands from session summary to full session detail once it is fetched, including recent transcript turns, events, and tool calls
- the desktop app shows recent messages, events, and tool calls in the session detail panel
- session detail is cached in the UI state until that session disappears from the current dashboard snapshot

## Filtering And Event Flow

The UI clients now include lightweight operator filtering so the runtime views stay usable once the daemon is busy:

- tasks can be filtered by request ID, agent ID, or state
- channels can be filtered by channel ID, kind, or agent ID
- events can be filtered by event name or payload text

The desktop app exposes those as inline filter fields inside the Tasks, Channels, and Events tabs.

The TUI exposes them as keyboard actions:

- Tasks: `/` edits the task filter, `F` clears it
- Channels: `/` edits the channel filter, `F` clears it, `[` / `]` move across discovered access entries, `m` cycles `pairing_mode`, `p` edits `pairing_users`, `u` edits `allowed_users`, `b` edits `banned_users`, `o` cycles Telegram `respond_mode`, `a` approves pending rooms, `x` rejects pending rooms, and `v` revokes approved rooms
- Events: `/` edits the event filter, `F` clears it

For high-volume event streams, both clients also support:

- pause/resume against the current event snapshot
- follow-latest behavior so selection can keep snapping to the newest event while you monitor
- explicit “jump to latest” behavior in the event view

## Current Scope

The UI clients are still not full config management surfaces.

Today they are best for:

- chatting with a live session while keeping Turin runtime context nearby
- monitoring runtime health
- opening and resuming sessions
- inspecting task/session state
- sending prompts to active sessions
- switching between local and remote profile targets from inside the UI
- validating local-versus-remote control parity

They still do not replace the richer filesystem/CLI management surfaces for daemon-owned agents, harnesses, or channels.
