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

## Runtime Model

The daemon owns a live `Kernel` plus a filesystem-backed registry scan.

It:

- scans `agents/` and `harnesses/`
- synthesizes effective runtime config
- rebuilds the live kernel on daemon-level rescan
- watches the daemon registry roots for changes
- keeps harness-script hot reload delegated to the kernel's harness watcher

Important distinction:

- editing `agents/<id>/agent.toml` or creating/removing agent directories is a **daemon registry** change
- editing `agents/<id>/harness/*.lua` or `harnesses/<id>/*.lua` is a **harness runtime** change

## Fault Isolation

One broken agent or harness should not stop the daemon.

Current behavior:

- invalid `agent.toml` becomes a daemon runtime issue
- invalid harness config/load only affects that harness
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
turin daemon task wait <request_id> --timeout-ms 30000
turin daemon task get <request_id>
turin daemon task list
```

### Harnesses

```bash
turin daemon harness list
turin daemon harness create <id>
turin daemon harness get <id>
turin daemon harness reload <id>
turin daemon harness validate <id>
turin daemon harness delete <id>
```

### Sessions

```bash
turin daemon session list
turin daemon session get <session_id>
```

## Live Runtime Visibility

The daemon now exposes:

- top-level daemon status with:
  - registry snapshot
  - harness runtime snapshots
  - live agent runtime snapshots
- per-agent live runtime status via `agent.status`
- persisted session inspection
- task submission/list/get/wait
- runtime event subscription

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
