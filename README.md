# Turin

**A programmable runtime for AI agents.**

Turin is a Rust runtime that separates inference from execution policy.
The model proposes actions. The harness (Luau scripts) decides what is allowed. The kernel executes and persists everything.

Turin is intentionally unopinionated about workflow and personality. It provides execution physics, persistence, tools, events, and a programmable harness surface so you can build radically different agents with the same runtime.

In practical terms, Turin is for turning an AI assistant into an
application-shaped workflow: a private operator, coding desk, release console,
triage queue, documentation helper, channel bot, or governed team tool.

## What Can You Build With It?

Turin is useful when a generic chat assistant is not enough and the workflow
needs durable state, explicit rules, tools, approvals, or a purpose-built
operator surface.

Good Turin-shaped systems include:

- private assistants with memory, schedules, local context, and controlled tool
  access
- coding workspaces that follow repository rules, run tests, inspect diffs, and
  delegate review
- release consoles with approval lists, QA gates, action buttons, reports, and
  human-in-the-loop decisions
- bug triage desks that classify issues, create durable work items, and keep an
  audit trail
- documentation or research assistants that split long-running work into
  inspectable tasks
- channel-connected agents for Telegram, Discord, Rocket.Chat, or WhatsApp
- governed team tooling where different agents have different capability
  ceilings

The common thread is that the model can propose work, but your harness code
decides what is allowed and Turin keeps the workflow durable and inspectable.

If you are evaluating Turin rather than implementing a harness yet, start with:

- [What is Turin?](docs/concepts/what-is-turin.md)
- [Who is Turin for?](docs/concepts/who-is-turin-for.md)
- [What can you do with Turin?](docs/concepts/what-can-you-do.md)
- [Scenario starter cards](docs/concepts/scenario-starter-cards.md)
- [Scenario blueprints](docs/concepts/scenarios.md)
- [Choose a first workflow](docs/getting-started/choose-first-workflow.md)

## How It Feels To Use

Without a custom harness, Turin can behave like a default agentic application:
start a session, ask for work, use configured tools, and keep state durable.

With a harness, the same runtime can become a purpose-built workflow. A release
harness can expose approval lists, readiness reports, and action buttons. A
coding harness can expose tasks, review status, and test actions. A personal
assistant can expose follow-ups, reminders, memory controls, and activity.

The important boundary is deliberate:

- Turin stores durable workflow state.
- Harness code defines rules, actions, memory, and semantic UI intent.
- Clients decide how to present that state while keeping navigation, selection,
  and other temporary view choices local.

## What Turin Is (Current Baseline)

Turin now ships a coherent, canonical runtime with:

- **Canonical Harness API** (`runtime.*`) plus ergonomic aliases (`memory.*`, `kv.*`, `agent.*`, `session.*`, `user.*`)
- **First-party Harness DX layer** (`verdict.*`, `allowed`, `needs`, `scope(...)`, `graph.*`, `schedule.*`, `action.define(...)`, callable `runtime.db(...)`, callable `runtime.agent(...)`, `remember`, `recall`, `fs.summary`, `code.find`, grant/time/json helpers)
- **Multi-DB runtime** with dynamic DB handles (`runtime.db.open/query/exec/list/close`)
- **Multi-agent runtime** with peer agent submit/await/status orchestration (`runtime.agent.*`)
- **Daemon-backed durable scheduler** with harness-facing helpers, built-in or harness-defined action jobs, and daemon-owned `jobs.db` (`runtime.schedule.*`, `schedule.*`, including in-place job updates)
- **Durable worklists** with hierarchical claimable items, prompt/action payloads, stale-claim recovery, store/scope routing, and daemon query APIs for external inspection and filtered item lookup (`runtime.worklist.*`, `worklist(...)`, `worklist.list/get/items`, `workitem.get`)
- **Memory v2 primitives** with lifecycle controls (`feedback`, `correct`, `purge`) and lexical/semantic/hybrid recall
- **Code search primitives** backed by an optional `turin-map` indexing companion and direct runtime reads (`runtime.code.search.*`)
- **Stable hook model** with explicit lifecycle hooks and typed event payloads
- **Opt-in governance model** with explicit capabilities, import scoping, agent ceilings, and temporary grants
- **Harness Library** with reusable `blocks/` and ready-to-run `workflows/`
- **Provider-agnostic inference and embeddings path** (provider quirks belong in `inference-sdk-rust`, not Turin)
- **Composable harness scripts** with `import(...)`, `import_scoped(...)`, `use(...)`, and explicit `watch(...)`
- **Named harness programs** so different configured agents can bind to different harness directories in one runtime
- **Experimental semantic UI intent** so harnesses can describe app-like
  screens, menus, lists, forms, actions, panes, reports, charts, notices, and
  badges without dictating a renderer
- **Durable event persistence** and optional immutable audit behavior

## Philosophy

Turin follows a strict separation:

- **Inference proposes**
- **Harness decides**
- **Kernel enforces**

This gives you a runtime that can be:

- wide open by default (enforcement off, unmatched capabilities allowed)
- tightly governed and auditable (enforcement on, explicit capability maps)
- anything in between, with explicit capability knobs

Simple things should be simple. Powerful things should be possible.

## Core Features

- **Lean core runtime** (`turin`) with no required services beyond your configured provider and local SQLite/libSQL database
- **Independent channel clients** for Telegram, Discord, Rocket.Chat, WhatsApp, and filesystem messaging without coupling adapters or credentials to the core runtime
- **Optional indexing companion** (`turin-map`) for code-search indexing without bloating the runtime execution path
- **Harness scripting in Luau** for governance, workflows, context engineering, memory policies, and orchestration
- **Canonical stdlib API**:
  - `runtime.context`, `runtime.memory`, `runtime.code.search`, `runtime.kv`, `runtime.db`, `runtime.agent`, `runtime.policy`, `runtime.governance`
  - `runtime.schedule`
  - `runtime.worklist`
- **First-party DX layer**:
  - `verdict`, `allowed`, `needs`, `scope`, `graph`, `schedule`, `action`, `session`, `user`, `remember`, `recall`, `fs.summary`, `code.find`, callable `runtime.db(...)`, callable `runtime.agent(...)`
- **Top-level ergonomic aliases**:
  - `fs`, `json`, `time`, `log`, `import`, `import_scoped`, `use`, `use_scoped`, `watch`
  - `memory`, `kv`, `session`, `user`, `agent`
- **Built-in model tool surface**:
  - default exposed set: `read_file`, `write_file`, `edit_file`, `shell_exec`, `web_fetch`, `web_search`, `remember`, `recall`, `submit_plan`, `bridge_mcp`
  - additional opt-in native tool: `apply_patch`
- **Native tool delegation**:
  - optional `[tools].allow` / `[tools].exclude` delegation at the runtime, agent, and channel layers with built-in shorthands such as `group:fs` and `group:web`
- **Tool behavior settings**:
  - optional global `[tools.<name>]` config for request headers and `web_search` provider order
- **Multi-provider support** through normalized `InferenceProvider` clients (`anthropic`, `openai`, `mock`, compatible proxies)
- **Provider-agnostic embeddings** with OpenAI-compatible local endpoint support
- **Persistent state** for sessions, messages, events, tool executions, KV, and memory records
- **Hybrid memory search** with native lexical/vector/hybrid retrieval
- **Hybrid code search** with lexical, semantic, hybrid, and traceable fallback behavior
- **Peer-agent orchestration** with status inspection and async submit/await result handling
- **Scheduled task orchestration** with one-shot/recurring jobs, anchored `daily` / `weekly` recurrence, prompt or named action payloads, overlap policy, and cross-store execution targeting
- **Durable work coordination** with reusable worklists, optional hierarchy, claim heartbeats, stale-claim recovery, and prompt/action payload parity
- **Opt-in governance** with capabilities/import scoping/agent ceilings/grants
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

See `docs/guides/harness-library.md` for the current catalog and validation approach.

## Quickstart

### 1. Build

```bash
cargo build --release

# Optional: build the setup/installation helper
cargo build --release -p turin-manager

# Optional: build the code-indexing companion if you want runtime code search setup
cargo build --release -p turin-map

# Optional: build independent channel runners for Telegram, Discord, or WhatsApp
cargo build --release -p turin-channel-telegram -p turin-channel-discord -p turin-channel-whatsapp --bins
```

### 2. Try Turin with no API key

The fastest local-first smoke path is:

```bash
target/release/turin quickstart --prompt "Summarize this workspace."
```

That will:

- scaffold `.turin/config.toml`
- create `.turin/harnesses/`
- add `.turin/` to `.gitignore`
- run a real Turin session with the mock provider

### 3. Scaffold a real project

If you want a real provider-backed config from the start:

```bash
target/release/turin init \
  --provider anthropic \
  --harness-template coding-assistant
```

Useful starter commands:

```bash
target/release/turin init --yes --provider openai --harness-template starter
target/release/turin harness new reviewer --dir .turin/harnesses-reviewer
target/release/turin harness test --response "HARNESS_TEST_OK"
```

`turin init` is interactive when run in a terminal without `--yes` and writes
an explicit, enforcement-disabled governance block for local use.

If you want the newer manager-driven setup path instead:

```bash
target/release/turin-manager init
target/release/turin-manager channels list
target/release/turin-manager channels configure telegram
target/release/turin-manager channels status
target/release/turin-manager doctor
```

`turin-manager init` can expand an open, balanced, or governed template into
explicit policy fields; Turin core does not interpret those preset names.

`turin-manager` stages diffs before writing, validates assembled channel settings through the channel runner, stores optional secrets in a `.env` file next to `.turin/config.toml`, and prints the exact foreground launch command. The channel runner loads that environment file and connects to the independently running Turin daemon.

Independent channel runners are described in [docs/reference/channel-sidecars.md](docs/reference/channel-sidecars.md).

Tool delegation notes:

- `[tools].allow = [...]` means "from what I inherited, expose only these native tools"
- `[tools].exclude = [...]` means "from what I inherited, remove these native tools"
- child configs cannot escalate past what the parent granted
- current groups are `group:all`, `group:fs`, `group:shell`, `group:web`, `group:memory`, `group:planning`, and `group:integration`
- `apply_patch` is available through explicit opt-in, for example `[tools] allow = ["group:fs"]`

Tool behavior notes:

- `[tools.web_fetch]` controls browser-like fetch headers such as `user_agent`
- `[tools.web_search].providers` controls ordered fallback across `duckduckgo_html`, `tavily`, `brave`, and `searxng`
- API-backed search providers are configured with environment-variable references such as `api_key_env = "TAVILY_API_KEY"`
- the same `[tools.<name>]` tables can also be used under `[agent.tools]`, `[agents.<id>.tools]`, and channel `settings.tools` to override inherited behavior

### 4. Manual `.turin/config.toml` path

If you prefer hand-edited config, start from the example:

```bash
mkdir -p .turin
cp examples/config/config.toml.example .turin/config.toml
```

Minimal example:

```toml
# Optional global native tool ceiling for every agent and channel.
# Defaults to Turin's standard built-in surface.
# [tools]
# allow = ["group:web", "read_file"]
# exclude = ["shell_exec"]

# Optional global tool behavior settings.
# [tools.web_fetch]
# max_response_bytes = 16777216 # bounds the decoded response body; raise deliberately for large files
# user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
#
# [tools.web_search]
# providers = ["tavily", "duckduckgo_html"]
# user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
#
# [tools.web_search.tavily]
# api_key_env = "TAVILY_API_KEY"
#
# [tools.web_search.brave]
# api_key_env = "BRAVE_SEARCH_API_KEY"
#
# [tools.web_search.searxng]
# base_url = "http://localhost:8080"

[agent]
system_prompt = "You are a helpful coding assistant."
model = "claude-sonnet-4-20250514"
provider = "anthropic"

# [agent.tools]
# allow = ["group:web", "read_file"] # optional per-agent subset from the inherited parent set
# exclude = ["write_file"]
#
# [agent.tools.web_fetch]
# user_agent = "Mozilla/5.0 ..."

[kernel]
workspace_root = "."
max_turns = 50

[persistence.state]
path = ".turin/data/state.db"

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
#
# [agents.reviewer.tools]
# allow = ["group:fs", "group:web"]
# exclude = ["shell_exec"]

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

Use `--config path/to/.turin/config.toml` if the config file lives elsewhere, and use explicit `--embedding-*` flags only when you want to override the configured profile for one run.

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
- `turin script PATH` — run a harness script directly for testing
- `turin init` — scaffold a Turin project, interactively or from flags
- `turin quickstart` — scaffold a mock-backed project if needed and run a first prompt immediately
- `turin harness new ...` — generate a starter harness template
- `turin harness test ...` — run a harness against the mock provider
- `turin check` — validate config + harness scripts
- `turin check --json` — emit the same validation as a machine-readable report
- `turin doctor` — validate the project and inspect local daemon readiness

Global options:
- `--log-level error|warn|info|debug|trace`
- `--log-file PATH`

See [docs/getting-started/harness-cookbook.md](docs/getting-started/harness-cookbook.md) for the progressive starter flow.

## Daemon Mode

Turin now has a local-first daemon mode for dynamic agent and harness management.

The daemon uses:

- `.turin/config.toml` for bootstrap/global config
- `.turin/runtime/agents/<id>/config.toml` for daemon-managed agents
- `.turin/runtime/agents/<id>/harness/` for local per-agent harnesses
- optional `.turin/harnesses/<id>/` for shared harness programs

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
turin daemon session branch-list <session-id>
turin daemon session branch-create <session-id> alt --from-turn 12 --activate
turin daemon session branch-checkout <session-id> alt
turin daemon task submit --agent docs-reviewer "Review the docs" --wait
turin daemon stop
turin daemon events
```

For local wrappers and desktop apps:

- `turin daemon ensure` starts the daemon in the background if needed
- `turin daemon wait` blocks until the daemon is ready
- `turin daemon stop` waits for graceful shutdown to complete, avoiding a stop/start race
- `turin daemon health --json` returns a compact readiness snapshot
- `turin daemon logs` resolves the background daemon log path and shows recent lines

Use `turin doctor` for one consolidated local diagnostic. It validates the
configuration, active provider credentials, harness directories and scripts,
state database location, and daemon health. An offline daemon is reported as a
warning because `turin run` can execute directly.

See `docs/operations/daemon.md` for the daemon filesystem model, runtime behavior, and command surface.

## Architecture Notes

For the key design decisions behind the current runtime and daemon shape, see `docs/adr/index.md`.

## Canonical Harness API (Overview)

Turin’s harness surface is split between **canonical runtime APIs** and **ergonomic aliases**.

### Canonical (`runtime.*`) — explicit substrate

- `runtime.context`
  - callable selector builder (`runtime.context(...)`)
  - alias discovery (`runtime.context.glob(pattern)`)
- `runtime.memory`
  - `search`, `store`, `feedback`, `correct`, `purge`
- `runtime.code.search`
  - `status`, `lexical`, `semantic`, `hybrid`
- `runtime.kv`
  - `get(key, ctx)` / `set(key, value, ctx)` / `delete(key, ctx)`
- `runtime.db`
  - `open`, `close`, `list`, `query`, `exec`
- `runtime.agent`
  - `list`, `get_status`, `submit`, `await`, `ask`
- `runtime.schedule`
  - `create`, `get`, `list`, `runs`, `update`, `enable`, `disable`, `delete`
- `runtime.worklist`
  - `open`
- `runtime.policy`
  - `get`, `set`
- `runtime.governance`
  - profile/snapshot/check
  - temporary grants (`grant_issue`, `grant_get`, `with_grant`, `grant_revoke`)

### Promoted helper/DX layer

- `memory.*` / `kv.*` for default agent-scoped data
- `memory.as(ctx)` / `kv.as(ctx)` for scoped proxies
- `remember`, `recall`, `scope(...)`, `graph.*`, `schedule.*`, `worklist(...)`, `fs.summary`, `code.find`
- `session.memory/kv.*`, `user.memory/kv.*`
- `agent.spawn`, `agent.submit`, `agent.ask`, `agent.session.*`
- `fs`, `json`, `time`, `log`, `import`, `import_scoped`, `use`, `use_scoped`, `watch`

See `docs/reference/primitives.md` for the full surface.

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

See `docs/reference/hooks.md` for payloads, verdict semantics, and examples.

## Multi-DB and Multi-Agent (What’s New)

### Multi-DB

Harnesses can open and operate on multiple state stores dynamically:

```lua
local handle = runtime.db.open({ path = "scratch/analysis.db" })

local rows = runtime.db.query(
  "select name from sqlite_master where type = 'table' order by name",
  nil,
  { handle = handle.handle }
)
```

### Multi-Agent

Harnesses can submit work to peer runtimes and await results:

```lua
local task_id = runtime.agent.submit("reviewer", {
  prompt = "Review the last patch for regressions",
  title = "peer review"
})

local result = runtime.agent.await(task_id, { timeout_ms = 30_000 })
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

See `docs/concepts/governance.md` for configuration and runtime behavior.

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
- `peer_ask_caps`
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
  --cases basic,tool_read,tool_error,tool_write_read,governed_denial,peer_agent,peer_ask_caps,queue_steer,runtime_db,grant_flow
```

OpenAI-compatible endpoint examples (MiniMax):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --api-format openai --suite core --log-level error --report-json -
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --api-format openai --suite soak --log-level error --report-json -
```

See `docs/operations/live-provider-testing.md` for setup and troubleshooting.

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

- `docs/index.md` — docs landing page and recommended reading paths
- `docs/getting-started/choose-first-workflow.md` — choose a first Turin workflow and decide when custom UI is worth adding
- `docs/concepts/turin.md` — philosophy and product framing
- `docs/concepts/architecture.md` — current runtime architecture and module layout
- `docs/reference/hooks.md` — stable hook lifecycle, payloads, verdict semantics
- `docs/reference/primitives.md` — canonical stdlib API + aliases
- `docs/guides/harness-guide.md` — writing production harness scripts
- `docs/guides/harness-library.md` — ready-to-use harness library entries
- `docs/guides/channels/telegram.md` — step-by-step Telegram channel setup
- `docs/guides/channels/whatsapp.md` — WhatsApp personal vs dedicated account guidance and linked-device setup
- `docs/concepts/governance.md` — capability model, import scoping, grants
- `docs/operations/remote.md` — authenticated remote bridge for HTTP + SSE/WebSocket daemon access
- `docs/operations/testing.md` — local validation, test suite, and smoke workflows
- `docs/operations/live-provider-testing.md` — live endpoint testing procedures

## A note on how Turin is built

Turin is developed with AI agents as collaborators, not just tools. I don't write Rust myself — the implementation is carried out by agents working under my direction. That's a conscious choice.

This is not a "one-shot" project. The architecture behind Turin is the product of hundreds of hours of deliberation—debating API shapes, stress-testing implementation plans, and iterating through many rejected proposals before landing on what exists today. Every significant decision has been argued, not just generated.

The ideas are mine. The code is theirs. And quality is something I take seriously — agents are directed to cross-review code, flag issues, and fix problems as they surface. But I won't pretend I can audit every line of Rust myself. If you find something wrong, please open an issue. I can direct an agent to investigate and fix it even if I can't read the stack trace myself.

Turin is, in a small way, a proof of its own concept — an agentic workflow where the human defines what gets built, and agents figure out how to build it.

## Standing on the Shoulders of Giants

The agentic space is moving very quickly. Every day brings new ideas, experiments, and breakthroughs from a community that is collectively defining the future of autonomy. Turin does not exist in a vacuum; it stands on the shoulders of giants. I want to express my deep gratitude to the researchers, developers, and experimenters who share their work openly. We are all learning from each other, iterating on each other's failures, and taking the industry forward together.

This project is rooted in a belief I've held since the late nineties, when I first started building for the web — that open sharing drives whole industries forward faster than any one company ever could. That the person who builds on your idea and takes it somewhere you never imagined is not a threat, but the whole point.

Turin is my contribution to that shared evolution. By keeping the runtime sovereign, the state durable, and the harness logic open, I hope to give the next builder a foundation worth building on — rather than a black box they have to work around.

## Thanks to the Turso Team

Turin's persistence layer is built on Turso — a native-Rust SQLite engine that happens to be exactly what an agent runtime needs: embedded, fast, no separate process, and vector search built in. We could have made SQLite work, but Turso's direction — AgentFS, agentic workflow primitives, a genuine focus on what AI systems need from a database — made it feel less like a dependency and more like a collaborator. If you're building anything in this space, it's worth paying attention to what they're doing.

## Versioning Note

Turin remains pre-1.0, but the current line formalizes the canonical harness API (`runtime.*`) and governance model as the forward-looking baseline. Internal refactors may continue aggressively; public harness surfaces should now change more deliberately.
