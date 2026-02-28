# Turin

**A programmable, single-binary runtime for AI agents.**

Turin is a Rust runtime that separates inference from execution policy.
The model proposes actions. The harness (Luau scripts) decides what is allowed. The kernel executes and persists everything.

Turin is intentionally unopinionated about workflow and personality. It provides execution physics, persistence, tools, events, and a programmable harness surface so you can build radically different agents with the same binary.

## What Turin Is (Current Baseline)

Turin now ships a coherent, canonical runtime with:

- **Canonical Harness API** (`runtime.*`) plus ergonomic aliases (`memory.*`, `kv.*`, `agent.*`, `session.*`, `user.*`)
- **Multi-DB runtime** with dynamic DB handles (`runtime.db.open/query/exec/list/close`)
- **Multi-agent runtime** with peer agent submit/await/status orchestration (`runtime.agent.*`)
- **Stable hook model** with explicit lifecycle hooks and typed event payloads
- **Opt-in governance model** with profiles, capabilities, import scoping, agent ceilings, and temporary grants
- **Provider-agnostic core** (provider quirks belong in `inference-sdk-rust`, not Turin)
- **Hot-reloadable harness scripts** with `import(...)` and `import_scoped(...)`
- **Durable event persistence** and optional immutable audit behavior

## Philosophy

Turin follows a strict separation:

- **Inference proposes**
- **Harness decides**
- **Kernel enforces**

This gives you a runtime that can be:

- minimalist and wide open (`open` governance profile)
- tightly governed and auditable (`governed` profile)
- anything in between, with explicit capability knobs

Simple things should be simple. Powerful things should be possible.

## Core Features

- **Single binary** (`turin`) with no service dependencies beyond your configured provider and local SQLite/libSQL database
- **Harness scripting in Luau** for governance, workflows, context engineering, memory policies, and orchestration
- **Canonical stdlib API**:
  - `runtime.context`, `runtime.memory`, `runtime.kv`, `runtime.db`, `runtime.agent`, `runtime.policy`, `runtime.governance`
- **Top-level ergonomic aliases**:
  - `fs`, `json`, `time`, `log`, `import`, `import_scoped`
  - `memory`, `kv`, `session`, `user`, `agent`
- **Multi-provider support** through normalized `InferenceProvider` clients (`anthropic`, `openai`, `mock`, compatible proxies)
- **Persistent state** for sessions, messages, events, tool executions, KV, and memory records
- **Hybrid memory search** with vector + FTS5 fallback/degradation paths
- **Peer-agent orchestration** with status inspection and async submit/await result handling
- **Opt-in governance** with profiles/capabilities/import scoping/agent ceilings/grants
- **Live provider smoke tooling** (manual/opt-in) for real endpoint validation

## Quickstart

### 1. Build

```bash
cargo build --release
```

### 2. Create `turin.toml`

Start from the example:

```bash
cp turin.toml.example turin.toml
```

Minimal example:

```toml
[agent]
system_prompt = "You are a helpful coding assistant."
model = "claude-sonnet-4-20250514"
provider = "anthropic"

[kernel]
workspace_root = "."
max_turns = 50

[persistence]
database_path = ".turin/state.db"

[harness]
directory = ".turin/harnesses"

[providers.anthropic]
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
# For Anthropic-compatible proxies, include the version segment (usually /v1)
# base_url = "https://api.minimax.io/anthropic/v1"
```

### 3. Add a harness script

```bash
mkdir -p .turin/harnesses
cat > .turin/harnesses/01_safety.lua <<'LUA'
function on_tool_call(call)
  if call.name == "shell_exec" then
    local cmd = call.args.command or ""
    if cmd:find("rm %-rf") then
      return REJECT, "Destructive command blocked"
    end
  end
  return ALLOW
end
LUA
```

### 4. Run Turin

```bash
target/release/turin run --prompt "List the files in this project and summarize the layout."
```

## CLI Commands

- `turin run --prompt ...` — one-shot execution
- `turin repl` — interactive session
- `turin script PATH` — run a harness script directly for testing
- `turin init` — scaffold a Turin project
- `turin check` — validate config + harness scripts

Global options:
- `--log-level error|warn|info|debug|trace`
- `--log-file PATH`

## Canonical Harness API (Overview)

Turin’s harness surface is split between **canonical runtime APIs** and **ergonomic aliases**.

### Canonical (`runtime.*`) — preferred for new harnesses

- `runtime.context`
  - callable selector builder (`runtime.context(...)`)
  - alias discovery (`runtime.context.glob(pattern)`)
- `runtime.memory`
  - `search(query, ctx, opts?)`
  - `store(content, ctx, metadata?, opts?)`
- `runtime.kv`
  - `get(key, ctx)` / `set(key, value, ctx)` / `delete(key, ctx)`
- `runtime.db`
  - `open`, `close`, `list`, `query`, `exec`
- `runtime.agent`
  - `list`, `get_status`, `submit`, `await`
- `runtime.policy`
  - `get`, `set`
- `runtime.governance`
  - profile/snapshot/check
  - temporary grants (`grant_issue`, `grant_get`, `with_grant`, `grant_revoke`)

### Ergonomic aliases (still supported and documented)

- `memory.*` / `kv.*` for default agent-scoped data
- `memory.as(ctx)` / `kv.as(ctx)` for scoped proxies
- `session.memory/kv.*`, `user.memory/kv.*`
- `agent.spawn`, `agent.complete`, `agent.send`, `agent.session.*`, `agent.mode.*`
- `fs`, `json`, `time`, `log`, `import`, `import_scoped`

See `docs/PRIMITIVES.md` for the full surface.

## Hooks (Stable Lifecycle)

Turin’s hook lifecycle is explicit and stable in the current baseline.
Core hooks include:

- `on_session_start`
- `on_task_start`
- `on_turn_start`
- `on_turn_prepare` (mutable context checkpoint)
- `on_tool_call`
- `on_tool_result` (supports `MODIFY`)
- `on_kernel_event`
- `on_token_usage`
- `on_inference_error`
- `on_task_complete`
- `on_plan_submit`
- `on_plan_complete`
- `on_all_tasks_complete`
- `on_session_end`

See `docs/HOOKS.md` for payloads, verdict semantics, and examples.

## Multi-DB and Multi-Agent (What’s New)

### Multi-DB

Harnesses can open and operate on multiple state stores dynamically:

```lua
local handle, err = runtime.db.open({ path = "scratch/analysis.db" })
if not handle then error(err) end

local rows, qerr = runtime.db.query(
  "select name from sqlite_master where type = 'table' order by name",
  nil,
  { handle = handle.handle }
)
```

### Multi-Agent

Harnesses can submit work to peer runtimes and await results:

```lua
local task_id, err = runtime.agent.submit("reviewer", {
  prompt = "Review the last patch for regressions",
  title = "peer review"
})
if not task_id then error(err) end

local result, aerr = runtime.agent.await(task_id, { timeout_ms = 30_000 })
```

Peer-agent dispatch can be governed by capabilities, per-agent ceilings, allowlists, and temporary grants.

## Governance (Opt-In, Flexibility-First)

Governance is **not** hardcoded restriction. It is an opt-in capability system layered over Turin’s programmable runtime.

### Profiles

- `open` — maximum flexibility, minimal restrictions
- `balanced` — safer defaults, still overrideable
- `governed` — stricter capability enforcement and optional immutable audit semantics
- `custom` — build your own policy shape

### Core governance features

- Capability enforcement for `runtime.*` APIs and built-in tools
- Import scoping (`import_scoped`) with governance roots
- Per-agent ceilings and child-agent allowlists
- Temporary grants (TTL / max-uses)
- Optional immutable audit persistence semantics

See `docs/GOVERNANCE.md` for configuration and runtime behavior.

## Live Provider Testing (Manual / Opt-In)

Turin does **not** call live providers during normal `cargo test` / `cargo build`.

Use the opt-in live suite script when you want to validate a real endpoint (e.g. MiniMax Anthropic-compatible or OpenAI-compatible):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite smoke
```

MiniMax OpenAI-compatible endpoint:

```bash
scripts/live_minimax_smoke.sh \
  --env-file ~/Documents/minimax.env \
  --api-format openai \
  --suite smoke
```

For broader real-world confidence (governance, multi-db, grants, audit, peer delegation):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite core
```

For flakiness/lifecycle confidence, run the soak suite (repeats the core case set; default `--repeat 3`):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite soak
```

Current live cases include:
- `basic`
- `tool_read`
- `tool_error`
- `tool_write_read`
- `governed_denial`
- `peer_agent`
- `queue_steer`
- `runtime_db`
- `grant_flow`
- `token_reject_task`
- `immutable_audit`
- `peer_grant`

Run a custom case set:

```bash
scripts/live_minimax_smoke.sh \
  --env-file ~/Documents/minimax.env \
  --cases basic,tool_read,tool_error,tool_write_read,governed_denial,peer_agent,queue_steer,runtime_db,grant_flow
```

OpenAI-compatible endpoint examples (MiniMax):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --api-format openai --suite core --log-level error --report-json -
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --api-format openai --suite soak --log-level error --report-json -
```

See `docs/LIVE_PROVIDER_TESTING.md` for setup and troubleshooting.

## Copyable Example Library

Turin now ships a copyable example library under `examples/`.

Current packs include:

- `examples/harnesses/openclaw_style_workspace/` — markdown-driven `SOUL.md` + `AGENTS.md` harness
- `examples/harnesses/governed_peer_review/` — peer-agent review under temporary grants
- `examples/harnesses/durable_journal/` — durable note/journal pattern over `runtime.db`

Examples are exercised by `cargo test --test example_harness_examples`, so they stay runnable instead of drifting into stale documentation.

## Documentation Map

- `docs/INDEX.md` — docs landing page and recommended reading paths
- `docs/TURIN.md` — philosophy and product framing
- `docs/ARCHITECTURE.md` — current runtime architecture and module layout
- `docs/HOOKS.md` — stable hook lifecycle, payloads, verdict semantics
- `docs/PRIMITIVES.md` — canonical stdlib API + aliases
- `docs/HARNESS_GUIDE.md` — writing production harness scripts
- `docs/EXAMPLES.md` — copyable, tested example packs
- `docs/GOVERNANCE.md` — capability model, profiles, import scoping, grants
- `docs/TESTING.md` — local validation, test suite, and smoke workflows
- `docs/LIVE_PROVIDER_TESTING.md` — live endpoint testing procedures

## Thanks to the Turso Team

Turin's persistence layer is built on Turso — a native-Rust SQLite engine that happens to be exactly what an agent runtime needs: embedded, fast, no separate process, and vector search built in. We could have made SQLite work, but Turso's direction — AgentFS, agentic workflow primitives, a genuine focus on what AI systems need from a database — made it feel less like a dependency and more like a collaborator. If you're building anything in this space, it's worth paying attention to what they're doing.

## Versioning Note

Turin remains pre-1.0, but the current line formalizes the canonical harness API (`runtime.*`) and governance model as the forward-looking baseline. Internal refactors may continue aggressively; public harness surfaces should now change more deliberately.
