# Turin Daemon

Turin now has a local-first daemon mode for dynamic agent and harness management.

For authenticated network access on top of the daemon, see `docs/operations/remote.md`.
For operator shells on top of the same control surface, see `docs/operations/ui-clients.md`.

The daemon is built around three rules:

1. The filesystem is the persisted source of truth.
2. `.turin/config.toml` remains bootstrap/global config, not a live mutable registry.
3. Bad agents or harnesses fail in isolation instead of poisoning the whole runtime.

## Authoritative Filesystem Layout

The daemon treats these paths as authoritative:

```text
.turin/
  config.toml
  harnesses/
    reviewer/
      main.lua
  runtime/
    agents/
      docs-reviewer/
        config.toml
        harness/
          main.lua
    channels/
      discord/
        config.toml
```

Default dynamic shape:

- `.turin/runtime/agents/<id>/config.toml` defines the agent.
- `.turin/runtime/agents/<id>/harness/` is that agent's local harness.

Optional shared harness shape:

- `.turin/harnesses/<id>/` defines a reusable shared harness program.
- `.turin/runtime/agents/<id>/config.toml` can bind to it with `harness = "<id>"`.

If an agent directory exists with a valid `config.toml`, that agent exists.
If the directory is removed, the agent is removed.

## Bootstrap Config

Daemon-related bootstrap settings live under `[daemon]` in `.turin/config.toml`:

```toml
[daemon]
agents_dir = "runtime/agents"
harnesses_dir = "harnesses"
endpoint = "daemon.sock"
```

These values define where the daemon reads and watches filesystem-backed state.
On Windows, Turin derives a stable named pipe endpoint from the configured endpoint seed.

Channel-related bootstrap settings also live under `[daemon]`:

```toml
[daemon]
channels_dir = "runtime/channels"
```

Each channel directory is authoritative the same way an agent directory is.
If `.turin/runtime/channels/<id>/config.toml` exists and is valid, that channel exists.

## Context-Local Persistence Overrides

Turin now distinguishes between:

- `persistence.state`: the owning session/runtime database for that context
- `persistence.store`: the default scoped-data store for that context

Top-level defaults live in `.turin/config.toml`:

```toml
[persistence.state]
path = ".turin/data/state.db"

# Optional; defaults to the same target as `state`
# [persistence.store]
# path = ".turin/data/store.db"
```

Agent and channel configs can override those targets locally:

```toml
# .turin/runtime/agents/<id>/config.toml
[persistence.state]
path = ".turin/data/states/reviewer.db"

[persistence.store]
path = ".turin/data/stores/reviewer-store.db"
```

```toml
# .turin/runtime/channels/<id>/config.toml
[persistence.state]
path = ".turin/data/states/telegram.db"

# Optional; if omitted, scoped data also uses the channel state DB
# [persistence.store]
# path = ".turin/data/stores/telegram-store.db"
```

Current implemented local override surfaces are:

- `.turin/config.toml`
- `.turin/runtime/agents/<id>/config.toml`
- `.turin/runtime/channels/<id>/config.toml`

Generic per-scope config files such as `.turin/scopes/<kind>/<id>/...` are still planned, not implemented.

## Runtime Model

The daemon owns a live `Kernel` plus a filesystem-backed registry scan.

It:

- scans `.turin/runtime/agents/` and `.turin/harnesses/`
- scans `.turin/runtime/channels/`
- synthesizes effective runtime config
- rebuilds the live kernel on daemon-level rescan
- watches the daemon registry roots for changes
- keeps harness-script hot reload delegated to the kernel's harness watcher

Important distinction:

- editing `.turin/runtime/agents/<id>/config.toml` or creating/removing agent directories is a **daemon registry** change
- editing `.turin/runtime/channels/<id>/config.toml` or creating/removing channel directories is a **daemon registry** change
- editing `.turin/runtime/agents/<id>/harness/*.lua` or `.turin/harnesses/<id>/*.lua` is a **harness runtime** change

## Session Scope Across Multiple State DBs

Current daemon behavior is intentionally conservative:

- persisted session listing is primary-state scoped by default
- persisted session history search is primary-state scoped by default
- bare session ids are interpreted against the primary `state` store unless a caller supplies a store-qualified reference

This means the daemon does not automatically aggregate sessions across every configured state DB.
That aggregation is a client concern when a UI wants it.

Cross-state references remain explicit:

- bare session: `018f...`
- aliased session: `018f...@telegram`
- path-qualified session: `018f...@.turin/data/states/telegram.db`

Persisted session queries can now also target an explicit state DB:

```bash
turin daemon session list --store telegram
turin daemon session search "borrow checker" --store telegram
turin daemon session list --path /srv/turin/project-alpha.db
```

## Session Branches

Sessions now support first-class branching with one active branch head per session.

- persisted transcript, tool history, and turn-bound events follow the active branch path
- branch creation can fork from the current head or an earlier `turn_index`
- live branch checkout is supported for idle sessions, and harness-initiated self-checkout is deferred until the current turn completes

CLI examples:

```bash
turin daemon session branch-list 018f...
turin daemon session branch-create 018f... alt --from-turn 12 --activate
turin daemon session branch-checkout 018f... alt
```

Current scope:

- branch create / list / checkout: supported
- branch creation from focused turns: supported in the TUI
- branch rename / delete / archive: not implemented yet
- merge/rebase semantics: intentionally not implemented

## Fault Isolation

One broken agent or harness should not stop the daemon.

Current behavior:

- invalid agent `config.toml` becomes a daemon runtime issue
- invalid harness config/load only affects that harness
- invalid channel `config.toml` only affects that channel
- unrelated agents and harnesses keep running

Use:

```bash
turin daemon errors
```

to inspect isolated load/config problems.

## Transport and Protocol

Current daemon transport:

- local IPC endpoint
- Unix domain socket on macOS/Linux
- Windows named pipe on Windows
- default endpoint seed: `.turin/daemon.sock`

Current protocol:

- NDJSON request/response
- NDJSON event stream for subscriptions
- `daemon.ping` is the handshake endpoint for protocol compatibility checks

Example request:

```json
{"id":"req_1","op":"agent.list","params":{}}
```

Example response:

```json
{"id":"req_1","ok":true,"result":{"agents":[]}}
```

`daemon.ping` returns a typed handshake payload with:

- `version`
- `protocol_version`
- `transport`
- `wire_format`
- `capabilities`

Current handshake values:

- `transport = "unix"` on macOS/Linux
- `transport = "named_pipe"` on Windows
- `wire_format = "ndjson"`
- `protocol_version = 1`

External channel sidecars also expose a separate manifest protocol through `describe`; see [channel-sidecars.md](../reference/channel-sidecars.md).

Event stream example:

```json
{"event":"task.updated","data":{"request_id":"...","state":"completed"}}
```

Subscription semantics:

- `runtime.events.subscribe` starts with a `runtime.snapshot` event
- `runtime.snapshot` uses the same control-plane shape as `daemon.status`, including `channel_runtimes`
- when `agent_id` and/or `session_id` filters are supplied, the snapshot is scoped to that view instead of leaking unrelated runtime state
- if the event stream lags, the daemon emits `runtime.events_lagged` and then immediately sends a fresh `runtime.snapshot`
- if a watcher-triggered registry rescan fails, the daemon emits `runtime.rescan_failed` with the error message and changed paths

External channel runtime semantics:

- sidecar-backed channels start in `starting`
- the daemon marks them `running` after the sidecar sends `channel.runner.hello`
- `channel.status` now includes runner handshake metadata such as sidecar binary name, version, pid, and manifest protocol version when available

For local GUI wrappers, prefer the managed client subscription in `turin-daemon-client`:

- `DaemonClient::subscribe(...)` is the low-level raw stream
- `DaemonClient::subscribe_managed(...)` reconnects and resubscribes after daemon restarts
- after a managed reconnect, the first event is a fresh `runtime.snapshot`

## Current Command Surface

### Daemon lifecycle

```bash
turin daemon start
turin daemon start --background
turin daemon ensure
turin daemon ping
turin daemon health
turin daemon status
turin daemon wait
turin daemon reload
turin daemon rescan
turin daemon errors
turin daemon logs
turin daemon stop
turin daemon events
```

Wrapper-oriented lifecycle notes:

- `turin daemon start --background` spawns the daemon and waits for readiness
- `turin daemon ensure` is single-instance friendly and only starts a new daemon if one is not already reachable at the configured endpoint
- `turin daemon health --json` returns a compact readiness/degradation/offline view for local wrappers
- `turin daemon logs` uses the default background log path at `<workspace>/.turin/daemon.log` unless `--log-file` is explicitly supplied

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
turin daemon channel create telegram-ops --kind telegram --agent default --setting token_env=TELEGRAM_BOT_TOKEN --setting chat_id=-1001234567890
turin daemon channel create whatsapp-agent --kind whatsapp --agent default --setting account_mode=personal --setting pairing_mode=pending --setting trigger_prefix=/turin
turin daemon channel get fs-local
turin daemon channel status fs-local
turin daemon channel issues fs-local
turin daemon channel enable fs-local
turin daemon channel disable fs-local
turin daemon channel update fs-local --idle-ttl-secs 900 --setting poll_interval_ms=50
turin daemon channel access telegram-ops
turin daemon channel approve telegram-ops --workspace-id telegram --room-id -1001234567890 --thread-id -1001234567890
turin daemon channel reject telegram-ops --workspace-id telegram --room-id -1001234567890 --thread-id -1001234567890
turin daemon channel revoke telegram-ops --workspace-id telegram --room-id -1001234567890 --thread-id -1001234567890
turin daemon channel delete fs-local
```

Channel settings are intentionally adapter-specific. The daemon accepts repeated
`--setting key=value` entries and persists them into the channel `config.toml`. Values are
parsed as JSON when possible, otherwise they are stored as strings.

For known channel kinds (`fs`, `discord`, `telegram`, `whatsapp`), settings are validated on
`channel.create` and `channel.update` before write/rescan.

`channel.update --setting ...` performs a partial merge into existing settings
rather than replacing the whole settings table.

Channels can also use generic runner-level access control settings such as:

- `pairing_mode = off | pending | auto`
- `pairing_users = [...]`
- `allowed_users = [...]`
- `banned_users = [...]`
- `[tools].allow = [...]`
- `[tools].exclude = [...]`
- `task_timeout_ms = 0 | <positive integer milliseconds>`

Some channel adapters also support `session_scope`, for example:

- Telegram: `user | thread | room`
- Discord: `user | thread`
- WhatsApp: `user | room`

Those settings are enforced before a message is routed into a Turin session, which is why they live in the shared channel runner rather than in a harness script.

Channel tool selection is downward-only: channel `[tools].allow` and
`[tools].exclude` can only subset the native tool surface already granted by
`.turin/config.toml` and the bound `.turin/runtime/agents/<id>/config.toml`.

Channel tool behavior can also override inherited defaults through nested
`[tools.<name>]` tables in the channel `config.toml`, for example `[tools.web_fetch]` or
`[tools.web_search]`.

Supported native tool groups are:

- `group:all`
- `group:fs`
- `group:shell`
- `group:web`
- `group:memory`
- `group:planning`
- `group:integration`

Example:

```bash
turin daemon channel update telegram-ops \
  --setting tools='{"allow":["group:web","read_file"],"exclude":["web_search"]}'
```

`kind = "fs"` is currently available as a built-in adapter:

- inbound messages are read from `<channel-dir>/inbox/*.json`
- outbound messages are written to `<channel-dir>/outbox/*.json`
- processed inbound files are moved to `<channel-dir>/processed/`
- invalid inbound files are moved to `<channel-dir>/failed/`

`kind = "discord"` is also available through a daemon-managed sidecar runner:

- uses Discord Gateway (WebSocket) by default for low-latency inbound events
- posts outbound responses back to Discord messages API
- requires:
  - `token_env` (environment variable containing a Discord bot token)
  - `channel_id` (Discord channel/thread ID to poll and respond in)
- optional settings:
  - `transport` (`gateway` default, or `polling` fallback mode)
  - `poll_interval_ms`
  - `max_messages_per_poll`
  - `workspace_id`
  - `room_id`
  - `start_from_latest`
  - `ignore_bot_messages`
  - `gateway_url`
  - `gateway_intents`
  - `base_url`

`kind = "telegram"` is also available through a daemon-managed sidecar runner:

- uses Telegram Bot API long polling (`getUpdates`) for inbound events
- accepts inbound text messages and posts outbound replies with `sendMessage`
- routes forum-topic messages to stable Turin slots using Telegram `message_thread_id` when present
- requires:
  - `token_env` (environment variable containing a Telegram bot token)
  - `chat_id` (Telegram numeric chat id to listen on and reply to)
- optional settings:
  - `poll_timeout_secs` (default `30`, maximum `50`)
  - `poll_interval_ms` (default `250`)
  - `max_updates_per_poll`
  - `stream_mode` (`off`, `typing`, `draft`, `block`)
  - `stream_thinking` (`true` / `false`)
  - `persist_thinking` (`true` / `false`)
  - `workspace_id`
  - `start_from_latest`
  - `ignore_bot_messages`
  - `base_url`

`kind = "whatsapp"` is also available through a daemon-managed sidecar runner:

- uses a WhatsApp linked-device session for inbound and outbound traffic
- supports QR pairing by default and pairing-code auth for headless servers
- accepts inbound text and media messages and sends outbound text replies plus local file uploads
- supports direct messages and group chats
- common settings:
  - `account_mode` (`personal` default, or `dedicated`)
  - `pairing_mode` (`auto`, `pending`, `off`)
  - `session_scope` (`user`, `room`)
  - `workspace_id`
  - `trigger_prefix`
  - `allowed_chats`
  - `banned_chats`
  - `session_store_path`
  - `pair_code_phone_number`
  - `pair_code_custom_code`
- current behavior notes:
  - self-originated messages are ignored
  - personal mode defaults `trigger_prefix` to `/turin` when unset
  - dedicated mode does not force a prefix
  - inbound media is downloaded into managed local storage and forwarded as attachment refs
  - outbound media currently requires `local_path`; remote URLs are not uploaded directly
  - streaming previews are not implemented yet

When a channel is `enabled`, the daemon owns the runtime lifecycle:
- `channel.status <id>` reports live runtime status (`starting`, `running`, `stopped`, `failed`, `unsupported`)
- `daemon.status` includes a `channel_runtimes` snapshot for control-plane visibility
- channel runtime state updates automatically after channel/agent/harness/runtime changes and watcher rescans

For sidecar-backed kinds (`discord`, `telegram`, `whatsapp`), the daemon resolves and starts the runner automatically. Resolution order is:

1. explicit env override
   - `TURIN_CHANNEL_DISCORD_BIN`
   - `TURIN_CHANNEL_TELEGRAM_BIN`
   - `TURIN_CHANNEL_WHATSAPP_BIN`
2. a sibling binary next to the running `turin` executable
3. the binary name on `PATH`
4. during source-checkout development, `cargo run -p <runner> -- ...` as a fallback

Channel runtime events are also streamed via `runtime.events.subscribe`:
- `channel.runtime.updated` for state transitions and error updates
- `channel.runtime.removed` when a runtime disappears from the active set
- `runtime.rescanned` for successful registry rescans, using the same snapshot shape as `daemon.status`
- `runtime.rescan_failed` when filesystem-triggered rescans are rejected or fail

`channel.status` and `daemon.status.channel_runtimes` now include lifecycle metrics:
- `start_count`
- `restart_count`
- `failure_count`
- `last_transition_unix_ms`
- `last_started_unix_ms`
- `last_stopped_unix_ms`
- `last_error_code` (normalized runtime failure code)

Discord runtime behavior notes:
- Gateway reconnect now uses bounded exponential backoff.
- Gateway session resume is attempted automatically when session/sequence state is available.
- Duplicate inbound message IDs are suppressed across reconnect/replay windows.
- Outbound responses support rich payloads (`content`, `embeds`, `components`, and local file attachments) with Discord-safe content chunking.

Telegram runtime behavior notes:

- The first pass is long-polling only; Turin does not auto-manage Telegram webhooks.
- If the bot still has an active webhook, runtime startup fails with a polling/webhook error until the webhook is removed.
- Transient Telegram polling/send failures now use bounded retry/backoff instead of immediately failing the runtime.
- Telegram replies default to `reply_to_message_id=<inbound message id>` when the inbound event came from Telegram and no explicit override is set.
- Outbound text is chunked to Telegram-safe message sizes.
- Code blocks render with Telegram HTML `<pre>` formatting by default.
- `stream_mode = typing` sends Telegram typing actions while a task is running.
- `stream_mode = draft` streams partial previews and then sends the final formatted reply.
- `stream_mode = block` streams less frequently than `draft`, using chunkier preview updates.
- `stream_thinking = true` lets `draft`/`block` previews include streamed model thinking when the provider emits thinking deltas.
- `persist_thinking = true` includes captured thinking in the final Telegram reply as a separate preformatted block.
- inbound Telegram media is downloaded into managed local storage and forwarded as attachment refs
- outbound image/document attachments are uploaded through `sendPhoto` / `sendDocument`

Telegram outbound metadata keys:

- `telegram_reply_to_message_id`: override or clear the reply target for a specific outbound message
- `telegram_disable_web_page_preview`: defaults to `true`
- `telegram_disable_notification`: defaults to `false`
- `telegram_format`: `plain`/`text` to force plain text, or `html` to force Telegram HTML parse mode
- `telegram_parse_mode`: currently supports `html`

For a step-by-step operator walkthrough, see `docs/guides/channels/telegram.md`.
For WhatsApp-specific setup and account-mode guidance, see `docs/guides/channels/whatsapp.md`.

To emit rich outbound payloads from task output, return a JSON envelope with
`_turin_channel_outbound = true`, for example:

```json
{
  "_turin_channel_outbound": true,
  "content": "Build summary",
  "embeds": [{ "title": "CI", "description": "All checks passed" }],
  "components": [{ "type": 1, "components": [] }],
  "attachments": [
    { "name": "report.txt", "local_path": "/abs/path/report.txt", "content_type": "text/plain" }
  ]
}
```

If no structured envelope is present, the channel runner also maps assistant content parts
(`text`, `image`, `file`) into outbound channel payloads automatically on adapters that
implement attachment delivery.

For Phase 1 multimodal task input, attachment persistence, and the current provider/channel support matrix, see `docs/guides/multimodal.md`.

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
  - local IPC endpoint
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
- `turin daemon health` and `turin daemon ensure` render compact lifecycle/readiness summaries
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
