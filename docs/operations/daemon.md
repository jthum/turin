# Turin Daemon

Turin now has a local-first daemon mode for dynamic agent and harness management.

For authenticated network access on top of the daemon, see `docs/operations/remote.md`.

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
runtime_db = "data/runtime.db"
endpoint = "daemon.sock"
```

These values define where the daemon reads and watches filesystem-backed state.
On Windows, Turin derives a stable named pipe endpoint from the configured endpoint seed.

Channel configuration is deliberately outside daemon bootstrap. Independent
channel runners conventionally use `.turin/channels`; the daemon does not scan
or watch that directory.

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

Agent configs can override those targets locally:

```toml
# .turin/runtime/agents/<id>/config.toml
[persistence.state]
path = ".turin/data/states/reviewer.db"

[persistence.store]
path = ".turin/data/stores/reviewer-store.db"
```

Current implemented local override surfaces are:

- `.turin/config.toml`
- `.turin/runtime/agents/<id>/config.toml`

Generic per-scope config files such as `.turin/scopes/<kind>/<id>/...` are still planned, not implemented.

## Runtime Model

The daemon owns a live `Kernel` plus a filesystem-backed registry scan.

It:

- scans `.turin/runtime/agents/` and `.turin/harnesses/`
- synthesizes effective runtime config
- rebuilds the live kernel on daemon-level rescan
- watches the daemon registry roots for changes
- keeps harness-script hot reload delegated to the kernel's harness watcher

Important distinction:

- editing `.turin/runtime/agents/<id>/config.toml` or creating/removing agent directories is a **daemon registry** change
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
- branch creation from focused turns: supported by the daemon protocol
- branch rename / delete / archive: not implemented yet
- merge/rebase semantics: intentionally not implemented

## Fault Isolation

One broken agent or harness should not stop the daemon.

Current behavior:

- invalid agent `config.toml` becomes a daemon runtime issue
- invalid harness config/load only affects that harness
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

The local endpoint is a trusted-operator boundary, not a per-client ACL
surface. On Unix, Turin explicitly sets the socket to owner read/write (`0600`)
after binding. Stale-endpoint cleanup removes only Unix socket entries and
refuses regular files and symlinks at the configured path.

Local trusted clients such as the CLI and independently operated channel
runners may connect directly. Network-facing or multi-user deployments should
place authentication and user authorization in `turin-remote` or another
trusted boundary service rather than exposing local IPC.

Current protocol:

- NDJSON request/response
- NDJSON event stream for subscriptions
- `daemon.ping` is the handshake endpoint for protocol compatibility checks

Registry-only agent changes are reconciled independently. Adding an agent does
not interrupt existing sessions or wait for unrelated tasks. Updating, disabling,
deleting, or explicitly reloading an agent retires only that agent's idle runtime
slots; if the affected agent is busy, reconciliation fails and the file change can
be applied with a later rescan. Changes to the bootstrap config and explicit full
runtime reloads still replace the kernel and require all work to be idle.

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

Independent channel runners expose a separate setup manifest through
`describe`; see [Channel Runners](../reference/channel-sidecars.md). The daemon
does not consume that manifest.

Daemon/control-plane surfaces now also include store-targeted worklist inspection:

- `worklist.list`
- `worklist.get`
- `worklist.items`
- `workitem.get`

Important boundary:

- scheduled jobs and runtime coordination records are globally indexed in daemon-owned `runtime.db`
- worklists are not; they live in whichever state/store backend a harness chose
- so control-plane worklist queries must carry or assume an explicit persistence target instead of pretending there is one global worklist namespace

Example `worklist.items` request:

```json
{
  "id": "req_work_items",
  "op": "worklist.items",
  "params": {
    "id": "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d47",
    "persistence": {
      "state": { "alias": "project_alpha" }
    },
    "status": "active",
    "paused_only": false,
    "due_only": false,
    "where": {
      "role": "browser"
    },
    "claimed_only": true,
    "limit": 10
  }
}
```

This returns active claimed items in the targeted worklist whose metadata matches `role = "browser"`.

For paused work inspection, `worklist.items` also accepts:

- `status = "paused"`
- `paused_only`
- `due_only`

That lets control-plane tooling ask for:

- all paused items
- only paused items whose resume window is already due

Event stream example:

```json
{"event":"task.updated","data":{"request_id":"...","state":"completed"}}
```

Subscription semantics:

- `runtime.events.subscribe` starts with a `runtime.snapshot` event
- `runtime.snapshot` uses the same control-plane shape as `daemon.status`
- when `agent_id` and/or `session_id` filters are supplied, the snapshot is scoped to that view instead of leaking unrelated runtime state
- if the event stream lags, the daemon emits `runtime.events_lagged` and then immediately sends a fresh `runtime.snapshot`
- if a watcher-triggered registry rescan fails, the daemon emits `runtime.rescan_failed` with the error message and changed paths

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
turin daemon stop --timeout-ms 10000 --poll-interval-ms 100
turin daemon events
```

Wrapper-oriented lifecycle notes:

- `turin daemon start --background` spawns the daemon and waits for readiness
- `turin daemon ensure` is single-instance friendly and only starts a new daemon if one is not already reachable at the configured endpoint
- concurrent `turin daemon ensure` calls converge on the daemon that acquires the configured endpoint; callers wait for that daemon to become ready
- `turin daemon stop` waits for the endpoint to disappear before succeeding, so a subsequent start cannot race graceful shutdown; its timeout and poll interval are bounded and configurable
- `turin daemon health --json` returns a compact readiness/degradation/offline view for local wrappers
- `turin daemon logs` uses the default background log path at `<workspace>/.turin/daemon.log` unless `--log-file` is explicitly supplied
- daemon commands emit a nonzero process status for unsuccessful protocol responses; with `--json`, the error response envelope remains available as parseable stdout

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
turin daemon task submit --agent <agent_id> "prompt"
turin daemon task submit --agent <agent_id> "prompt" --wait
turin daemon task submit --session-id <session_id> "prompt"
turin daemon task wait <request_id> --timeout-ms 30000
turin daemon task cancel <request_id>
turin daemon task get <request_id>
turin daemon task list
```

Task responses now include an execution snapshot, not just queue/terminal state.

Current execution fields on `task.submit`, `task.wait`, `task.get`, `task.list`, and `task.sidestep`:

- `execution.execution_id`
- `execution.context_target`
- `execution.write_policy`
- `execution.durability`
- `execution.visibility`

This matters because a completed task can now be understood operationally without guessing from branch state alone. For example, an operator can see whether a task:

- advanced a branch head normally
- ran detached against a fixed turn or selected path
- ran ephemerally on a hidden sidestep
- wrote durably onto a hidden sibling branch

`branch_outcome` remains separate from `execution.*`:

- `execution.*` describes how the task ran
- `branch_outcome` describes what durable branch mutation, if any, it produced

Typical examples:

- an ephemeral sidestep reports `write_policy = "detached"` with `durability = "ephemeral"`
- a durable sibling sidestep reports `write_policy = "advance_branch_head"` with `visibility = "hidden"`

Example `task.get` / `task.wait` payload shape:

```json
{
  "request_id": "01968f6d5fa87e5f93d7f4e1a9d31f49",
  "agent_id": "default",
  "slot_id": "sd_01968f6d5f8f7ef8b6f23c4cf4b516d7",
  "trace_id": "tr_01968f6d5fa07f0fb28d8c3aa51b5f4c",
  "state": "completed",
  "runtime_task_id": "t_2",
  "execution": {
    "execution_id": "ex_01968f6d5f947c45a5d66e2f618b5cb3",
    "context_target": {
      "kind": "turn_id",
      "turn_id": 42
    },
    "visibility": "hidden",
    "durability": "ephemeral",
    "write_policy": "detached"
  },
  "status": "success",
  "task_turn_count": 1,
  "branch_outcome": null,
  "promotion_candidate": {
    "session_id": "01968f6d5e8a72d5859c4c0dbe9d44b1",
    "source_turn_id": 42
  },
  "promoted_branch": null,
  "output": "Side answer",
  "error": null
}
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

Channels are independent clients and are not configured, launched, watched, or
reported by the Turin daemon. Use `turin-manager channels configure <kind>` to
create channel-owned configuration, then launch the printed
`turin-channel-<kind> run --config ... --turin-config ...` command separately.

The runner uses generic daemon session, task, and event operations. Channel access
policy, credentials, conversation bindings, retries, and process health remain
outside Turin core. See [Channel Runners](../reference/channel-sidecars.md).

Live session/runtime observability now also exposes execution-scoped state.

Current live-session fields on `session.open`, `session.resume`, `session.list_live`, `daemon.status.live_sessions`, and `runtime.snapshot.live_sessions`:

- `execution.execution_id`
- `execution.context_target`
- `execution.write_policy`
- `execution.durability`
- `execution.visibility`
- `conflict_policy`

This is the operator-facing view of the active execution head for that live slot. It answers:

- what persisted path this runtime is materializing right now
- whether that path is visible or hidden
- whether it is expected to persist durable turns
- how stale branch-head conflicts will be resolved if they occur

Example `session.list_live` item:

```json
{
  "agent_id": "default",
  "slot_id": "main",
  "session_id": "01968f6d5e8a72d5859c4c0dbe9d44b1",
  "running": true,
  "active_tasks": 1,
  "queued_tasks": 0,
  "current_request_id": "01968f6d5fa87e5f93d7f4e1a9d31f49",
  "execution": {
    "execution_id": "ex_01968f6d5f947c45a5d66e2f618b5cb3",
    "context_target": {
      "kind": "branch_head",
      "branch_head_id": null
    },
    "visibility": "visible",
    "durability": "durable",
    "write_policy": "advance_branch_head"
  },
  "conflict_policy": "reject"
}
```

Channel adapter behavior, structured outbound payloads, and multimodal support are
documented with the independent runners in [Channel Runners](../reference/channel-sidecars.md)
and the channel-specific guides.

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

This makes the daemon usable as the control surface for channels and UI clients
without forcing those clients to scrape files directly.

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
