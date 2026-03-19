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
- the current connection target and active profile are shown in the shell chrome
- reloading the profile file happens inside the UI, so you can edit the file and refresh the profile list

The desktop app exposes this through the Connections tab buttons.
The TUI exposes it through the Connections tab plus keyboard actions:

- `Enter` or `s` connects to the selected profile
- `l` reloads the profile file

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
