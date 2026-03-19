# Turin UI Clients

`turin-tui` and `turin-app` are operator-facing clients built on the same transport-agnostic control layer.

They can both talk to:

- a local Turin daemon over the existing local IPC transport
- a remote Turin daemon through `turin-remote`

That means the same operator workflows work in both modes:

- inspect agents, sessions, tasks, channels, and events
- open live sessions
- resume stored sessions
- submit prompts to live sessions
- cancel tasks or sessions
- inspect recent session transcript and tool history

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
- both clients validate the edited draft inline before saving, including remote URL and auth checks
- both clients can connect directly from the current unsaved draft, so you can test edits before writing them to disk
- both clients can save the edited draft back into the profile file
- both clients can duplicate or rename an existing profile from inside the UI
- both clients can delete a selected profile from inside the UI
- the current connection target and active profile are shown in the shell chrome
- reloading the profile file happens inside the UI, so you can edit the file and refresh the profile list

The desktop app exposes this through the Connections tab controls:

- `Load Current` copies the active connection into the editor draft
- `Load Selected` copies the highlighted stored profile into the editor draft
- `New Draft` resets the editor to a fresh draft
- edit the draft kind, target, and remote auth mode/value inline in the Connections tab
- invalid draft fields are highlighted inline, and `Save Draft` stays disabled until the draft is valid
- `Save Draft` writes the edited draft into the profile file under the typed profile name
- `Connect Draft` switches the UI to the currently edited draft without saving it first
- `Duplicate Selected` copies the highlighted profile to the typed name
- `Rename Selected` renames the highlighted profile to the typed name
- `Set as default` marks the saved profile as `default_profile`
- delete is a two-step flow: `Arm Delete`, then `Confirm Delete` or `Cancel Delete`

The TUI exposes it through the Connections tab plus keyboard actions:

- `Enter` or `s` connects to the selected profile
- `v` loads the current connection into the profile draft
- `b` loads the selected stored profile into the draft
- `m` cycles the draft kind between local-config, local-endpoint, and remote
- `o` cycles the draft auth mode for remote drafts
- `t` edits the draft target
- `g` edits the draft auth value
- the detail pane and footer show draft validation issues before save, and `g` only edits auth when the draft is using env or inline auth
- `C` connects to the current draft without saving it first
- `a` saves the current draft to a typed profile name
- `A` saves the current draft and marks it as the default profile
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

- the TUI detail pane expands from session summary to full session detail once it is fetched
- the desktop app shows recent messages and tool calls in the session detail panel
- session detail is cached in the UI state until that session disappears from the current dashboard snapshot

## Current Scope

The UI clients are operator shells, not full config management surfaces.

Today they are best for:

- monitoring runtime health
- opening and resuming sessions
- inspecting task/session state
- sending prompts to active sessions
- switching between local and remote profile targets from inside the UI
- validating local-versus-remote control parity

They still do not replace the richer filesystem/CLI management surfaces for daemon-owned agents, harnesses, or channels.
