# Turin

**A programmable, single-binary runtime for AI agents.**

Turin is a Rust runtime that separates inference from execution policy.
The model proposes actions. The harness (Luau scripts) decides what is allowed. The kernel executes and persists everything.

Turin is intentionally unopinionated about workflow and personality. It provides execution physics, persistence, tools, events, and a programmable harness surface so you can build radically different agents with the same binary.

## What Turin Is (Current Baseline)

Turin now ships a coherent, canonical runtime with:

- **Canonical Harness API** (`runtime.*`) plus ergonomic aliases (`memory.*`, `kv.*`, `agent.*`, `session.*`, `user.*`)
- **First-party Harness DX layer** (`verdict.*`, `allowed`, `needs`, callable `runtime.db(...)`, callable `runtime.agent(...)`, `remember`, `recall`, `cache.file`, `code.find`, grant/time/json helpers)
- **Multi-DB runtime** with dynamic DB handles (`runtime.db.open/query/exec/list/close`)
- **Multi-agent runtime** with peer agent submit/await/status orchestration (`runtime.agent.*`)
- **Memory v2 primitives** with lifecycle controls (`feedback`, `correct`, `purge`) and lexical/semantic/hybrid recall
- **Content cache primitives** for session-aware file reuse and token savings (`runtime.cache.*`)
- **Code search primitives** backed by an optional `turin-map` indexing companion and direct runtime reads (`runtime.code.search.*`)
- **Stable hook model** with explicit lifecycle hooks and typed event payloads
- **Opt-in governance model** with profiles, capabilities, import scoping, agent ceilings, and temporary grants
- **Harness Library** with reusable `blocks/` and ready-to-run `workflows/`
- **Provider-agnostic inference and embeddings path** (provider quirks belong in `inference-sdk-rust`, not Turin)
- **Composable harness scripts** with `import(...)`, `import_scoped(...)`, `use(...)`, and explicit `watch(...)`
- **Named harness programs** so different configured agents can bind to different harness directories in one runtime
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

- **Self-contained runtime binary** (`turin`) with no service dependencies beyond your configured provider and local SQLite/libSQL database
- **Optional indexing companion** (`turin-map`) for code-search indexing without bloating the runtime execution path
- **Harness scripting in Luau** for governance, workflows, context engineering, memory policies, and orchestration
- **Canonical stdlib API**:
  - `runtime.context`, `runtime.memory`, `runtime.cache`, `runtime.code.search`, `runtime.kv`, `runtime.db`, `runtime.agent`, `runtime.policy`, `runtime.governance`
- **First-party DX layer**:
  - `verdict`, `allowed`, `needs`, `session`, `user`, `remember`, `recall`, `cache.file`, `code.find`, callable `runtime.db(...)`, callable `runtime.agent(...)`
- **Top-level ergonomic aliases**:
  - `fs`, `json`, `time`, `log`, `import`, `import_scoped`, `use`, `use_scoped`, `watch`
  - `memory`, `kv`, `session`, `user`, `agent`
- **Multi-provider support** through normalized `InferenceProvider` clients (`anthropic`, `openai`, `mock`, compatible proxies)
- **Provider-agnostic embeddings** with OpenAI-compatible local endpoint support
- **Persistent state** for sessions, messages, events, tool executions, KV, and memory records
- **Hybrid memory search** with native lexical/vector/hybrid retrieval
- **Hybrid code search** with lexical, semantic, hybrid, and traceable fallback behavior
- **Peer-agent orchestration** with status inspection and async submit/await result handling
- **Opt-in governance** with profiles/capabilities/import scoping/agent ceilings/grants
- **Live provider smoke tooling** (manual/opt-in) for real endpoint validation

## Harness Library

Turin ships a Harness Library under `library/`:

- `library/blocks/` — reusable harness units for focused jobs
- `library/workflows/` — complete end-to-end harness systems

Current workflows include:

- `openclaw_style_personal_assistant`
- `full_coding_harness`
- `bug_triage_desk`
- `release_manager`
- `docs_team_assistant`

See `docs/HARNESS_LIBRARY.md` for the current catalog and validation approach.

## Quickstart

### 1. Build

```bash
cargo build --release

# Optional: build the code-indexing companion if you want runtime code search setup
cargo build --release -p turin-map
```

### 2. Try Turin with no API key

The fastest local-first smoke path is:

```bash
target/release/turin quickstart --prompt "Summarize this workspace."
```

That will:

- scaffold `turin.toml`
- create `.turin/harnesses/`
- add `.turin/` to `.gitignore`
- run a real Turin session with the mock provider

### 3. Scaffold a real project

If you want a real provider-backed config from the start:

```bash
target/release/turin init \
  --provider anthropic \
  --harness-template coding-assistant \
  --governance balanced
```

Useful starter commands:

```bash
target/release/turin init --yes --provider openai --harness-template starter
target/release/turin harness new reviewer --dir .turin/harnesses-reviewer
target/release/turin harness test --response "HARNESS_TEST_OK"
```

`turin init` is interactive when run in a terminal without `--yes`.

### 4. Manual `turin.toml` path

If you prefer hand-edited config, start from the example:

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

# Optional named harnesses for other configured agents
# [harnesses.reviewer]
# directory = ".turin/harnesses-reviewer"

# [agents.reviewer]
# id = "reviewer"
# system_prompt = "You are a strict code reviewer."
# model = "claude-sonnet-4-20250514"
# provider = "anthropic"
# harness = "reviewer"

[providers.anthropic]
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
# For Anthropic-compatible proxies, include the version segment (usually /v1)
# base_url = "https://api.minimax.io/anthropic/v1"
```

Current persistence note: the Turso 0.5 memory/search baseline is a breaking reset. Delete and recreate existing Turin DBs when moving to this baseline; no schema migration path is provided.

Optional semantic memory and code search:

```toml
[providers.local_embeddings]
type = "openai"
base_url = "http://127.0.0.1:11434/v1"

[embeddings]
provider = "local_embeddings"   # or "openai" to reuse providers.openai
model = "your-small-embedding-model"
dimensions = 384                # set this to the model's actual output size
```

Quick local check from the project root:

```bash
target/release/turin-map index
target/release/turin-map status
```

You should see `Semantic: enabled (...)` in the status output once the local endpoint, model, and dimensions are wired correctly.

Use `--config path/to/turin.toml` if the config file lives elsewhere, and use explicit `--embedding-*` flags only when you want to override the configured profile for one run.

Turin matches semantic/hybrid queries against the index embedding profile. If the runtime provider and index disagree on driver, base URL, model, or dimensions, `strict = false` falls back to lexical search and `strict = true` returns an error. If you do not configure embeddings at all, Turin still works with lexical-only recall and code search.

### 5. Add a harness script manually

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

### 6. Run Turin

```bash
target/release/turin run --prompt "List the files in this project and summarize the layout."
```

## CLI Commands

- `turin run --prompt ...` — one-shot execution
- `turin run --agent reviewer --prompt ...` — run against a specific configured agent/harness binding
- `turin repl` — interactive session
- `turin repl --agent reviewer` — interactive session for a specific configured agent
- `turin script PATH` — run a harness script directly for testing
- `turin init` — scaffold a Turin project, interactively or from flags
- `turin quickstart` — scaffold a mock-backed project if needed and run a first prompt immediately
- `turin harness new ...` — generate a starter harness template
- `turin harness test ...` — run a harness against the mock provider
- `turin check` — validate config + harness scripts

Global options:
- `--log-level error|warn|info|debug|trace`
- `--log-file PATH`

See [docs/HARNESS_COOKBOOK.md](docs/HARNESS_COOKBOOK.md) for the progressive starter flow.

## Daemon Mode

Turin now has a local-first daemon mode for dynamic agent and harness management.

The daemon uses:

- `turin.toml` for bootstrap/global config
- `agents/<id>/agent.toml` for daemon-managed agents
- `agents/<id>/harness/` for local per-agent harnesses
- optional `harnesses/<id>/` for shared harness programs

Core commands:

```bash
turin daemon ensure
turin daemon health --json
turin daemon status
turin daemon logs
turin daemon agent list
turin daemon agent create docs-reviewer --provider mock --model mock-model
turin daemon session open docs-reviewer
turin daemon session resume <session-id>
turin daemon task submit docs-reviewer "Review the docs" --wait
turin daemon events
```

For local wrappers and desktop apps:

- `turin daemon ensure` starts the daemon in the background if needed
- `turin daemon wait` blocks until the daemon is ready
- `turin daemon health --json` returns a compact readiness snapshot
- `turin daemon logs` resolves the background daemon log path and shows recent lines

See `docs/DAEMON.md` for the daemon filesystem model, runtime behavior, and command surface.

## Architecture Notes

For the key design decisions behind the current runtime and daemon shape, see `docs/adr/README.md`.

## Canonical Harness API (Overview)

Turin’s harness surface is split between **canonical runtime APIs** and **ergonomic aliases**.

### Canonical (`runtime.*`) — preferred for new harnesses

- `runtime.context`
  - callable selector builder (`runtime.context(...)`)
  - alias discovery (`runtime.context.glob(pattern)`)
- `runtime.memory`
  - `search`, `store`, `feedback`, `correct`, `purge`
- `runtime.cache`
  - `read`, `invalidate`, `stats`, `reset`
- `runtime.code.search`
  - `status`, `lexical`, `semantic`, `hybrid`
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
- `remember`, `recall`, `cache.file`, `code.find`
- `session.memory/kv.*`, `user.memory/kv.*`
- `agent.spawn`, `agent.complete`, `agent.send`, `agent.session.*`, `agent.mode.*`
- `fs`, `json`, `time`, `log`, `import`, `import_scoped`, `use`, `use_scoped`, `watch`

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

Configured agents can also bind to different named harnesses in the same Turin runtime. The default harness comes from `[harness]`; additional harnesses live under `[harnesses.*]`, and each agent can opt into one with `harness = "<id>"`.

## Governance (Opt-In, Flexibility-First)

Governance is **not** hardcoded restriction. It is an opt-in capability system layered over Turin’s programmable runtime.

### Profiles

- `open` — maximum flexibility, minimal restrictions
- `balanced` — safer defaults, still overrideable
- `governed` — stricter capability enforcement and optional immutable audit semantics
- `custom` — build your own policy shape

### Core governance features

- Capability enforcement for `runtime.*` APIs and built-in tools
- Import / behavior-block scoping (`import_scoped`, `use_scoped`) with governance roots
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
- `peer_complete_caps`
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
  --cases basic,tool_read,tool_error,tool_write_read,governed_denial,peer_agent,peer_complete_caps,queue_steer,runtime_db,grant_flow
```

OpenAI-compatible endpoint examples (MiniMax):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --api-format openai --suite core --log-level error --report-json -
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --api-format openai --suite soak --log-level error --report-json -
```

See `docs/LIVE_PROVIDER_TESTING.md` for setup and troubleshooting.

## Harness Library

Turin now ships a Harness Library under `library/`.

Current entries include:

- `library/workflows/openclaw_style_personal_assistant/` — markdown-driven personal assistant with planner/reviewer routing and durable artifacts
- `library/workflows/full_coding_harness/` — spec/task-driven coding workflow with planner + reviewer specialists
- `library/workflows/bug_triage_desk/` — issue-intake workflow with triager + responder specialists
- `library/workflows/release_manager/` — release-readiness workflow with review + changelog drafting
- `library/workflows/docs_team_assistant/` — docs-maintenance workflow with review + draft writing
- `library/blocks/code_reviewer/` — focused review contract for correctness/regression review
- `library/blocks/task_planner/` — focused planning contract for sequenced task breakdowns
- `library/blocks/spec_writer/` — focused contract for turning rough ideas into executable specs
- `library/blocks/test_gap_finder/` — focused contract for identifying likely missing tests
- `library/blocks/repo_librarian/` — focused contract for repository-aware routing and guidance
- `library/blocks/release_readiness_checker/` — focused contract for blocker/risk-based release assessment
- `library/blocks/docs_maintainer/` — focused contract for documentation drift analysis
- `library/blocks/changelog_writer/` — focused contract for concise release-note drafting
- `library/blocks/governed_peer_review/` — peer-agent review under temporary grants
- `library/blocks/delegated_peer_capabilities/` — peer completion with an explicit delegated capability ceiling
- `library/blocks/durable_journal/` — durable note/journal pattern over `runtime.db`

The library is exercised by `cargo test --test example_harness_examples`, so it stays runnable instead of drifting into stale documentation.

## Documentation Map

- `docs/INDEX.md` — docs landing page and recommended reading paths
- `docs/TURIN.md` — philosophy and product framing
- `docs/ARCHITECTURE.md` — current runtime architecture and module layout
- `docs/HOOKS.md` — stable hook lifecycle, payloads, verdict semantics
- `docs/PRIMITIVES.md` — canonical stdlib API + aliases
- `docs/HARNESS_GUIDE.md` — writing production harness scripts
- `docs/HARNESS_LIBRARY.md` — ready-to-use harness library entries
- `docs/GOVERNANCE.md` — capability model, profiles, import scoping, grants
- `docs/TESTING.md` — local validation, test suite, and smoke workflows
- `docs/LIVE_PROVIDER_TESTING.md` — live endpoint testing procedures

## Thanks to the Turso Team

Turin's persistence layer is built on Turso — a native-Rust SQLite engine that happens to be exactly what an agent runtime needs: embedded, fast, no separate process, and vector search built in. We could have made SQLite work, but Turso's direction — AgentFS, agentic workflow primitives, a genuine focus on what AI systems need from a database — made it feel less like a dependency and more like a collaborator. If you're building anything in this space, it's worth paying attention to what they're doing.

## Versioning Note

Turin remains pre-1.0, but the current line formalizes the canonical harness API (`runtime.*`) and governance model as the forward-looking baseline. Internal refactors may continue aggressively; public harness surfaces should now change more deliberately.
