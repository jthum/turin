# Turin

**A programmable runtime for AI agents, delivered as a single binary.**

LLMs are probabilistic by nature. Turin lets you shape the environment in which inference becomes action.

Through hot-swappable harness scripts, Turin can:

- Intercept and modify model inputs and outputs
- Inject and enrich context before every inference call
- Enforce hard execution boundaries on tool usage
- Control tool access and arguments deterministically
- Shape memory, task flow, and multi-step plans
- Route between providers mid-session

The model remains probabilistic. The execution layer becomes programmable and enforceable.

Turin sits between inference and action — shaping how AI outputs become structured, reliable execution. It does not replace prompts. It engineers the conditions under which prompts operate.

---

## Why Turin

Most agent frameworks bake behavior into their core: how the agent plans, what it's allowed to do, how it manages context. If you want something different, you fork or fight the framework.

Turin takes the opposite approach. The Kernel handles inference, streaming, tool execution, and persistence. Everything else lives in harness scripts — governance, workflows, context engineering, memory strategies, even personality. Same binary, different harness, completely different agent.

- A **coding assistant** that injects project instructions, compacts context at 80% capacity, and blocks destructive shell commands
- A **research agent** that routes queries to different LLM providers based on task type
- A **planning-first agent** that must submit a task plan before taking any action
- A **conversation-only coach** with no tool access at all

All the same binary. Different `.lua` files in the harness directory.

---

## Features

- **Harness Scripts** — Define agent behavior in hot-reloadable Lua (Luau). Governance, workflows, context engineering — all in scripts you can read, modify, and share.
- **Deterministic Governance** — When a harness returns `REJECT`, the kernel physically cannot execute the action. This is code, not a suggestion.
- **Single Binary** — Rust. ~13MB. No runtime dependencies. `cargo build --release` and deploy.
- **Scaffolding & Validation** — Built-in `turin init` to bootstrap projects and `turin check` for static config/harness verification.
- **Multi-Provider** — Anthropic, OpenAI, or any OpenAI-compatible API. Multiple named providers in the same session. Switch mid-turn from a harness script.
- **Persistent State** — Every event, message, and tool execution logged to a portable SQLite database (Turso). Modular architecture with per-connection busy timeouts ensures reliability under contention.
- **Cognitive Memory** — Semantic memory with hybrid search (vector + FTS5 + Reciprocal Rank Fusion). Agents remember across sessions.
- **Automated Quality Controls** — GitHub Actions CI for automated testing and builds, with `cargo-deny` for security and license auditing.
- **Context Engineering** — The `on_turn_prepare` hook gives harness scripts full control over what the LLM sees: inject instructions, compact history, swap providers, adjust thinking budgets.
- **Task Decomposition** — Built-in `submit_plan` tool with harness hooks for plan review, modification, and steering.
- **Subagents** — Spawn isolated nested kernel instances for recursive task delegation, with independent provider and harness configurations.
- **MCP Bridge** — Dynamic tool discovery via Model Context Protocol. Connect to any MCP server at runtime.
- **Hot Reload** — Edit harness scripts while the agent is running. Changes take effect immediately with atomic swap (bad scripts don't crash the running harness).
- **Extended Thinking** — Streaming thinking blocks with configurable budget, controllable from harness scripts.

---

## Quickstart

### Build

```bash
cargo build --release
```

### Configure

Create a `turin.toml`:

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
api_key_env = "ANTHROPIC_API_KEY"

[providers.openai]
api_key_env = "OPENAI_API_KEY"
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="sk-..."
```

### Run

```bash
# One-shot execution
turin run --prompt "Read main.rs and explain what it does"

# Interactive REPL
turin repl

# With verbose event output
turin run --verbose --prompt "Fix the bug in utils.rs"

# Override provider from CLI
turin run --provider openai --model gpt-4o --prompt "Explain this codebase"
```

---

## Harness Scripts

Harness scripts are `.lua` files in your harness directory. They hook into the kernel's event lifecycle to control agent behavior. No recompilation needed.

### Governance: Block Dangerous Commands

```lua
-- .turin/harnesses/safety.lua

function on_tool_call(call)
    if call.name == "shell_exec" then
        local cmd = call.args.command
        if cmd:find("rm %-rf") or cmd:find("sudo") then
            return REJECT, "Destructive/privileged commands are not allowed"
        end
    end
    return ALLOW
end
```

### Workflow: Budget Enforcement

```lua
-- .turin/harnesses/budget.lua

local BUDGET_LIMIT = 50000

function on_token_usage(usage)
    local used = usage.total_tokens
    if used > (BUDGET_LIMIT * 0.8) then
        log(string.format("Warning: %d%% of budget used", (used / BUDGET_LIMIT) * 100))
    end
    db.kv_set("session_tokens", tostring(used))
    return ALLOW
end

function on_tool_call(call)
    local used = tonumber(db.kv_get("session_tokens")) or 0
    if used >= BUDGET_LIMIT then
        return REJECT, "Token budget exceeded"
    end
    return ALLOW
end
```

### Context Engineering: Project Instructions + Memory

```lua
-- .turin/harnesses/coding_agent.lua

function on_turn_prepare(ctx)
    -- Inject project instructions
    if fs.exists("TURIN.md") then
        local instructions = fs.read("TURIN.md")
        ctx.system_prompt = ctx.system_prompt .. "\n\n=== Project Instructions ===\n" .. instructions
    end

    -- Recall relevant memories
    if turin.memory and turin.memory.search then
        local messages = ctx.messages
        if messages and #messages > 0 then
            for i = #messages, 1, -1 do
                if messages[i].role == "user" then
                    local results = turin.memory.search(messages[i].content, 3)
                    if results and #results > 0 then
                        local block = "\n\n=== Relevant Memories ===\n"
                        for _, mem in ipairs(results) do
                            block = block .. "- " .. mem.content .. "\n"
                        end
                        ctx.system_prompt = ctx.system_prompt .. block
                    end
                    break
                end
            end
        end
    end

    return ALLOW
end
```

### Workflow: Force Planning Before Action

```lua
-- .turin/harnesses/planning.lua

function on_turn_prepare(ctx)
    local msgs = ctx.messages
    if msgs and #msgs > 0 then
        local latest = msgs[#msgs]
        if latest.role == "user" then
            local text = latest.content
            if type(text) == "table" and text[1] then text = text[1].text or "" end
            if text:lower():find("plan") or text:lower():find("complex") then
                ctx.system_prompt = ctx.system_prompt ..
                    "\n\nYour first step MUST be to use 'submit_plan' to break this down."
            end
        end
    end
    return ALLOW
end

function on_plan_submit(payload)
    log("Plan submitted: " .. payload.title)
    -- You could MODIFY the plan here, or REJECT it
    return ALLOW
end
```

### Composition

Multiple harness scripts compose automatically. Place them in the harness directory and they load in alphabetical order. For each event:

- If **any** harness returns `REJECT` — the action is blocked
- If **any** harness returns `ESCALATE` — the action pauses for human approval
- If **all** harnesses return `ALLOW` — the action proceeds

This lets you layer concerns: `01_safety.lua` for hard constraints, `02_budget.lua` for cost control, `03_workflow.lua` for context engineering.

---

## Architecture

Turin has three layers:

```
┌─────────────────────────────────────────────────┐
│           Layer 3: Inference (LLM)              │
│           The agent proposes actions             │
├─────────────────────────────────────────────────┤
│           Layer 2: Harness (Lua)                │
│           Your scripts decide what's allowed     │
│                                                  │
│  on_tool_call  on_turn_prepare  on_turn_end  │
│       │               │                │         │
│       ▼               ▼                ▼         │
│  ALLOW / REJECT / ESCALATE / MODIFY              │
├─────────────────────────────────────────────────┤
│           Layer 1: Kernel (Rust)                │
│           Executes, persists, streams            │
│                                                  │
│  Event Loop → Streaming → Tool Exec → Persist   │
│                                                  │
│  ┌──────────┐  ┌────────┐  ┌────────────────┐   │
│  │  Tools   │  │ Events │  │  Turso (State)  │   │
│  │ Registry │  │  Bus   │  │                 │   │
│  └──────────┘  └────────┘  └────────────────┘   │
└─────────────────────────────────────────────────┘
```

The LLM proposes. The harness decides. The kernel enforces.

For a deeper technical walkthrough, see [Architecture](docs/ARCHITECTURE.md).

---

## Harness Hook Reference

| Hook | Trigger | Can Modify | Use Cases |
|------|---------|-----------|-----------|
| `on_session_start` | Session begins | Queue tasks | Session setup, queue initial tasks |
| `on_turn_prepare` | Before each LLM call | System prompt, messages, provider, thinking budget | Context engineering, instruction injection, compaction |
| `on_tool_call` | LLM requests a tool | Tool args (via MODIFY) | Governance, safety, allowlisting |
| `on_tool_result` | Tool execution completes | Tool output / error flag | Post-processing, result redaction, normalization |
| `on_plan_submit` | Agent proposes a plan | Task list (via MODIFY) | Plan review, steering, modification |
| `on_task_complete` | A task reaches terminal status | Additional tasks (via MODIFY) | Per-task validation, retry/branch flows |
| `on_all_tasks_complete` | Global queue is empty | Additional tasks (via MODIFY) | End-of-run validation and continuation |
| `on_token_usage` | Token accounting update | — | Budget enforcement, cost tracking |
| `on_turn_start` | New LLM turn begins | — | Logging, turn-level logic |
| `on_turn_end` | LLM turn completes | — | Post-turn analysis |
| `on_session_end` | Session completes | — | Cleanup, final reporting |

For the full harness scripting guide, see [Writing Harnesses](docs/HARNESS_GUIDE.md), [Harness Hooks](docs/HOOKS.md), and [Harness Primitives](docs/PRIMITIVES.md).

---

## Kernel Primitives

These are available to harness scripts via the Turin Standard Library:

| Module | Functions | Description |
|--------|-----------|-------------|
| **Verdicts** | `ALLOW`, `REJECT`, `ESCALATE`, `MODIFY` | Return values from hooks |
| **fs** | `read`, `write`, `exists`, `list`, `is_safe_path` | Sandboxed filesystem access |
| **db** | `kv_get`, `kv_set` | Persistent key-value store (backed by Turso) |
| **json** | `encode`, `decode` | JSON serialization |
| **time** | `now_utc` | Timestamps |
| **log** | `log(message)` | Write to kernel event log |
| **session** | `id`, `list`, `load`, `queue`, `queue_next` | Session management and task queuing |
| **turin.memory** | `store`, `search` | Semantic memory (vector + FTS5) |
| **turin.agent** | `spawn` | Nested subagent execution |
| **turin.context** | `glob` | Safe workspace file search |
| **turin.import** | `import(name)` | Import harness modules |

---

## Built-in Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with line numbers |
| `write_file` | Create or overwrite a file |
| `edit_file` | Apply targeted string replacements |
| `shell_exec` | Execute shell commands |
| `submit_plan` | Propose a multi-step plan |
| `bridge_mcp` | Connect to an MCP server for dynamic tool discovery |

All tool calls pass through the harness before execution. The kernel provides the capability; your harness decides whether to allow it.

---

## Configuration Reference

```toml
[agent]
system_prompt = "You are a helpful assistant."  # Base system prompt
model = "claude-sonnet-4-20250514"              # Model identifier
provider = "anthropic"                           # Default provider name

[agent.thinking]
enabled = true          # Enable extended thinking
budget_tokens = 4096    # Thinking token budget

[kernel]
workspace_root = "."             # Root for relative paths
max_turns = 50                   # Max agent loop iterations
heartbeat_interval_secs = 30     # Liveness check interval

[persistence]
database_path = ".turin/state.db"  # SQLite database location

[harness]
directory = ".turin/harnesses"     # Harness script directory

[providers.anthropic]
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"    # Env var containing API key
# base_url = "https://api.anthropic.com/v1"  # Optional override
# max_retries = 2
# request_timeout_secs = 60
# total_timeout_secs = 120
# [providers.anthropic.headers]
# anthropic-beta = "output-128k-2025-02-19"

[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"
# base_url = "https://api.openai.com/v1"
# max_retries = 2
# request_timeout_secs = 60
# total_timeout_secs = 120
# [providers.openai.headers]
# x-foo = "bar"

# Named providers for multi-provider setups
[providers.fast]
type = "openai"
api_key_env = "OPENAI_API_KEY"

[embeddings]
type = "openai"  # or "no_op" for environments without embedding support
```

---

## Project Status

Turin is at **v0.14.0**. The core runtime is functional, production-hardened, and verified. What's implemented:

- Multi-provider inference (Anthropic, OpenAI) with streaming
- Full tool execution loop (read, write, edit, shell, submit_plan, bridge_mcp)
- Harness engine with all hooks, verdict composition, hot-reload, and module system
- Persistent state (events, messages, tool log, KV store) via Turso
- Cognitive memory with hybrid search (vector + FTS5 + RRF)
- Subagent spawning with isolated kernel instances
- Extended thinking with harness-controlled budgets
- REPL mode with live harness reloading

See [Architecture](docs/ARCHITECTURE.md) for technical details and the [Harness Reference](docs/HOOKS.md) for hooks and [primitives](docs/PRIMITIVES.md).

---

## License

MIT. See LICENSE for details.
