# Turin Daemon

Turin now has a local-first daemon mode for dynamic agent and harness management.

The daemon is built around three rules:

1. The filesystem is the persisted source of truth.
2. `turin.toml` remains bootstrap/global config, not a live mutable registry.
3. Bad agents or harnesses fail in isolation instead of poisoning the whole runtime.

## Authoritative Filesystem Layout

The daemon treats these paths as authoritative:

```text
turin.toml
agents/
  docs-reviewer/
    agent.toml
    harness/
      main.lua
harnesses/
  reviewer/
    main.lua
channels/
  discord/
    channel.toml
```

Default dynamic shape:

- `agents/<id>/agent.toml` defines the agent.
- `agents/<id>/harness/` is that agent's local harness.

Optional shared harness shape:

- `harnesses/<id>/` defines a reusable shared harness program.
- `agents/<id>/agent.toml` can bind to it with `harness = "<id>"`.

If an agent directory exists with a valid `agent.toml`, that agent exists.
If the directory is removed, the agent is removed.

## Bootstrap Config

Daemon-related bootstrap settings live under `[daemon]` in `turin.toml`:

```toml
[daemon]
agents_dir = "agents"
harnesses_dir = "harnesses"
socket_path = ".turin/daemon.sock"
```

These values define where the daemon reads and watches filesystem-backed state.

Channel-related bootstrap settings also live under `[daemon]`:

```toml
[daemon]
channels_dir = "channels"
```

Each channel directory is authoritative the same way an agent directory is.
If `channels/<id>/channel.toml` exists and is valid, that channel exists.

## Runtime Model

The daemon owns a live `Kernel` plus a filesystem-backed registry scan.

It:

- scans `agents/` and `harnesses/`
- scans `channels/`
- synthesizes effective runtime config
- rebuilds the live kernel on daemon-level rescan
- watches the daemon registry roots for changes
- keeps harness-script hot reload delegated to the kernel's harness watcher

Important distinction:

- editing `agents/<id>/agent.toml` or creating/removing agent directories is a **daemon registry** change
- editing `channels/<id>/channel.toml` or creating/removing channel directories is a **daemon registry** change
- editing `agents/<id>/harness/*.lua` or `harnesses/<id>/*.lua` is a **harness runtime** change

## Fault Isolation

One broken agent or harness should not stop the daemon.

Current behavior:

- invalid `agent.toml` becomes a daemon runtime issue
- invalid harness config/load only affects that harness
- invalid `channel.toml` only affects that channel
- unrelated agents and harnesses keep running

Use:

```bash
turin daemon errors
```

to inspect isolated load/config problems.

## Transport and Protocol

Current daemon transport:

- Unix domain socket
- default path: `.turin/daemon.sock`

Current protocol:

- NDJSON request/response
- NDJSON event stream for subscriptions

Example request:

```json
{"id":"req_1","op":"agent.list","params":{}}
```

Example response:

```json
{"id":"req_1","ok":true,"result":{"agents":[]}}
```

Event stream example:

```json
{"event":"task.updated","data":{"request_id":"...","state":"completed"}}
```

## Current Command Surface

### Daemon lifecycle

```bash
turin daemon start
turin daemon ping
turin daemon status
turin daemon reload
turin daemon rescan
turin daemon errors
turin daemon stop
turin daemon events
```

### Agents

```bash
turin daemon agent list
turin daemon agent get <id>
turin daemon agent status <id>
turin daemon agent issues <id>
turin daemon agent reload <id>
turin daemon agent create <id> --provider mock --model mock-model
turin daemon agent update <id> --model new-model
turin daemon agent enable <id>
turin daemon agent disable <id>
turin daemon agent bind-harness <id> <harness_id>
turin daemon agent use-local-harness <id>
turin daemon agent delete <id>
```

### Tasks

```bash
turin daemon task submit <agent_id> "prompt"
turin daemon task submit <agent_id> "prompt" --wait
turin daemon task submit --session <session_id> "prompt"
turin daemon task wait <request_id> --timeout-ms 30000
turin daemon task cancel <request_id>
turin daemon task get <request_id>
turin daemon task list
```

### Harnesses

```bash
turin daemon harness list
turin daemon harness create <id>
turin daemon harness get <id>
turin daemon harness issues <id>
turin daemon harness reload <id>
turin daemon harness validate <id>
turin daemon harness delete <id>
```

### Channels

```bash
turin daemon channel list
turin daemon channel create fs-local --kind fs --agent default --setting inbox_dir=inbox --setting outbox_dir=outbox
turin daemon channel create discord-dev --kind discord --agent default --setting token_env=DISCORD_TOKEN --setting channel_id=1234567890
turin daemon channel get fs-local
turin daemon channel status fs-local
turin daemon channel issues fs-local
turin daemon channel enable fs-local
turin daemon channel disable fs-local
turin daemon channel update fs-local --idle-ttl-secs 900 --setting poll_interval_ms=50
turin daemon channel delete fs-local
```

Channel settings are intentionally adapter-specific. The daemon accepts repeated
`--setting key=value` entries and persists them into `channel.toml`. Values are
parsed as JSON when possible, otherwise they are stored as strings.

`kind = "fs"` is currently available as the first built-in adapter:

- inbound messages are read from `<channel-dir>/inbox/*.json`
- outbound messages are written to `<channel-dir>/outbox/*.json`
- processed inbound files are moved to `<channel-dir>/processed/`
- invalid inbound files are moved to `<channel-dir>/failed/`

`kind = "discord"` is also available as a daemon-owned adapter:

- polls Discord channel messages over HTTP for inbound events
- posts outbound responses back to Discord messages API
- requires:
  - `token_env` (environment variable containing a Discord bot token)
  - `channel_id` (Discord channel/thread ID to poll and respond in)
- optional settings:
  - `poll_interval_ms`
  - `max_messages_per_poll`
  - `workspace_id`
  - `room_id`
  - `start_from_latest`
  - `ignore_bot_messages`
  - `base_url`

When a channel is `enabled`, the daemon owns the runtime lifecycle:
- `channel.status <id>` reports live runtime status (`starting`, `running`, `stopped`, `failed`, `unsupported`)
- `daemon.status` includes a `channel_runtimes` snapshot for control-plane visibility
- channel runtime state updates automatically after channel/agent/harness/runtime changes and watcher rescans

### Sessions

```bash
turin daemon session list
turin daemon session live
turin daemon session open <agent_id>
turin daemon session open <agent_id> --slot-id thread-123
turin daemon session resume <session_id>
turin daemon session resume <session_id> --slot-id thread-123
turin daemon session get <session_id>
turin daemon session cancel <session_id>
turin daemon session kill <session_id>
```

## Live Runtime Visibility

The daemon now exposes:

- top-level daemon status with:
  - registry snapshot
  - harness runtime snapshots
  - live agent runtime snapshots
- per-agent live runtime status via `agent.status`
- per-agent isolated registry/load issues via `agent.issues`
- per-harness isolated registry/load issues via `harness.issues`
- channel registry inspection and isolated issues via `channel.get` / `channel.issues`
- channel live runtime inspection via `channel.status` and `daemon.status.channel_runtimes`
- live session opening and listing for multi-threaded clients
- persisted-session resume into a live runtime slot after daemon restart
- persisted session inspection
- task submission/list/get/wait
- task submission into an explicit live session
- queued-task cancellation
- runtime event subscription

The CLI defaults are also intentionally human-readable:

- `turin daemon status`, `agent list`, and `harness list` render tables
- `turin daemon task *` renders compact task summaries by default
- `turin daemon session *` renders readable session summaries and persisted detail tables by default
- `--json` remains available everywhere when a machine-readable response is needed

Task state semantics are intentionally explicit:

- `queued`: accepted by the daemon and waiting in an agent runtime queue
- `running`: currently executing inside an agent runtime
- `cancelling`: cancellation has been requested for a running task and the runtime is draining toward a terminal result
- `completed`: reached a terminal status (`success`, `rejected`, `max_turns`, `error`, `cancelled`, `killed`)

`task.cancel` is truthful by design:

- queued tasks cancel immediately and become terminal `cancelled`
- running tasks transition to `cancelling` and stop cooperatively at real execution boundaries

`session.cancel` is cooperative:

- queued work for that runtime session is cancelled
- the active task is asked to stop
- the peer runtime rotates to a fresh session once the stop completes

`session.kill` is forceful:

- queued and running work for that runtime is marked `killed`
- the peer runtime is aborted and recreated on demand later

This makes the daemon usable as the control surface for future channels, desktop, and web clients without forcing those clients to scrape files directly.

## Design Boundaries

What the daemon does:

- validate and apply filesystem-backed runtime state
- provide a safer control API over that state
- allow direct user edits without trying to prohibit them

What the daemon does not do:

- hide the filesystem from advanced users
- introduce a second registry file
- make direct edits impossible

That is intentional. Turin remains filesystem-native and operator-visible.
