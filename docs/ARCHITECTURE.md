# Turin Architecture

This document describes the technical architecture of Turin (v0.10.0) for contributors, evaluators, and anyone who wants to understand how the system works under the hood.

---

## Design Principles

1. **Kernel is physics, not opinion.** The kernel executes; it does not decide.
2. **Behavior is deterministic code.** Lua harness scripts — not prompts — control what the agent can and cannot do.
3. **Inference proposes, Harness decides, Kernel enforces.**
4. **Composability over convenience.** No baked-in workflows. Primitives over features.
5. **Boring is a feature.** Stability over novelty.

---

## The Three Layers

### Layer 1: Kernel (Rust)

The kernel is the execution substrate. It knows *how* to do things but has no opinions about *whether* to do them. It manages:

| Responsibility | Description |
|----------------|-------------|
| **Event Loop** | Turn-by-turn agent execution with streaming |
| **Session Lifecycle** | Managed `AgentStart` and `AgentEnd` boundaries for clean state transitions |
| **Inference Transport** | HTTP connections to LLM providers via the `InferenceProvider` trait |
| **Stream Parsing** | MSE/SSE events into structured `KernelEvent`s |
| **Tool Execution** | Runs tools via the `ToolRegistry`, handles side-effects through `ToolEffect` |
| **State Persistence** | Atomic writes to Turso (SQLite) with WAL mode enabled for concurrent performance |
| **Event Bus** | Every kernel action produces a categorized event (Lifecycle, Stream, Audit) |
| **Cognitive Memory** | Semantic storage with hybrid vector + FTS5 search |
| **Subagent Primitive** | Isolated nested kernel instances for recursive task delegation |
| **MCP Bridge** | Dynamic tool discovery via Model Context Protocol |

### Layer 2: Harness Engine (Embedded Luau)

All kernel events pass through the harness engine before the kernel acts on them. The harness engine:

- Runs in a **sandboxed Luau VM** via `mlua`.
- Has **no direct OS access** — only capabilities explicitly injected by the kernel (fs, db, etc.).
- Evaluates **verdicts** sequentially using a blocking `std::sync::Mutex` to guarantee event observation.
- Returns **verdicts**: `ALLOW`, `REJECT`, `ESCALATE`, or `MODIFY`.
- Maintains persistent state via `db.kv_get/kv_set` (backed by Turso).
- Supports **hot-reload** with atomic swap (invalid scripts don't crash the running harness).
- Provides a **module system** (`turin.import`) for reusable harness libraries.

The harness is not just a governance layer. It's the entire behavioral layer. Context engineering, workflow orchestration, memory strategies, task steering, provider routing — all implemented as harness scripts using kernel primitives.

### Layer 3: Inference (LLM)

The LLM proposes actions. It has no ability to enforce them. It is a brain in a room with a microphone — the kernel only uses its "hands" (tools) if the harness says the request is allowed.

Turin supports multiple LLM providers through a standardized `InferenceProvider` trait. Providers are configured by name and can be switched at runtime from harness scripts via the `on_before_inference` hook.

---

## The Agent Lifecycle

```
turin run --prompt "Fix the bug in main.rs"
│
├─ LOAD config (turin.toml)
├─ LOAD harness scripts (*.lua)
├─ INIT Turso state store (WAL mode + busy timeout)
│
├─ start_session()
│   └─ EMIT Lifecycle::AgentStart → on_agent_start hook
│
├─ run() loop (turns)
│   ├─ EMIT Lifecycle::TurnStart → on_turn_start hook
│   ├─ ASSEMBLE context (system prompt + messages + tool results)
│   ├─ CALL on_before_inference(ctx) → harness
│   │   └─ Harness may: modify system prompt, inject/remove messages,
│   │      swap provider, adjust thinking budget
│   ├─ CALL LLM (stream)
│   │   ├─ EMIT Stream::MessageStart
│   │   ├─ EMIT Stream::MessageDelta (per chunk)
│   │   ├─ EMIT Stream::ThinkingDelta (if enabled)
│   │   └─ EMIT Stream::MessageEnd
│   │
│   ├─ IF tool_calls in response:
│   │   ├─ FOR each tool_call:
│   │   │   ├─ EMIT audit::ToolCall → on_tool_call hook
│   │   │   │   ├─ REJECT/ESCALATE? → inject rejection, continue
│   │   │   │   ├─ MODIFY? → use modified arguments
│   │   │   │   └─ ALLOW? ▼
│   │   │   ├─ EXECUTE tool
│   │   │   ├─ CAPTURE ToolEffect (e.g., submit_task side-effects)
│   │   │   ├─ EMIT audit::ToolResult → on_tool_result hook
│   │   │   └─ PERSIST event
│   │   └─ CONTINUE to next turn
│   │
│   ├─ ELSE (no tool calls):
│   │   └─ BREAK (agent is done)
│   │
│   ├─ EMIT audit::TokenUsage → on_token_usage hook
│   └─ EMIT Lifecycle::TurnEnd → on_turn_end hook
│
├─ IF queue empty:
│   └─ EMIT on_task_complete → harness (can MODIFY queue to continue)
│
├─ end_session()
│   ├─ EMIT Lifecycle::AgentEnd → on_agent_end hook
│   └─ CLEAR session_active_queue
└─ EXIT
```

### The Queue

The kernel maintains a task queue. When `run()` is called with a prompt, the prompt is the first task. Harness scripts and the `submit_task` tool can add tasks to the queue. The kernel processes tasks sequentially until the queue is empty.

In REPL mode, each user input becomes a new task. Harness scripts can use `session.queue()` and `session.queue_next()` to inject follow-up work.

---

## Event System

Every action in Turin produces a categorized `KernelEvent`. This categorization improves performance and clarity for downstream consumers (like the harness or external observers).

```rust
pub enum KernelEvent {
    Lifecycle(LifecycleEvent), // AgentStart, AgentEnd, TurnStart, TurnEnd
    Stream(StreamEvent),       // MessageStart/Delta/End, ThinkingDelta
    Audit(AuditEvent),         // ToolCall, ToolResult, TokenUsage, HarnessRejection
}
```

Events serve three purposes:
1. **Streaming** — `Stream::MessageDelta` drives the real-time CLI/UI output.
2. **Governance** — All events flow through the `on_kernel_event` hook for observation/rejection.
3. **Persistence** — Every event is written to the SQLite indexed event log.

Events are serialized with tagged JSON (`#[serde(tag = "type")]`), which means the event log is both human-readable and machine-parseable.

---

## Verdict System

Harness hooks return verdicts that the kernel must obey:

| Verdict | Effect |
|---------|--------|
| `ALLOW` | Proceed with the action |
| `REJECT, "reason"` | Block the action. The reason is injected into the LLM's context so it can adapt. |
| `ESCALATE, "reason"` | Pause execution and wait for human input |
| `MODIFY, data` | Proceed with modified data (e.g., modified tool arguments or task list) |

When multiple harness scripts are loaded, verdicts compose with **first-REJECT-wins** semantics:

1. Scripts are evaluated in alphabetical order
2. If any script returns `REJECT`, the action is blocked immediately
3. If any script returns `ESCALATE`, the action is paused
4. Only if all scripts return `ALLOW` (or `MODIFY`) does the action proceed

This makes governance additive: you can layer safety constraints, budget limits, and workflow rules in separate files without them conflicting.

---

## Tool System

Tools are the only mechanism for an agent to interact with its environment. Each tool implements the `Tool` trait, which returns a `ToolOutput` containing both the `content` (visible to the LLM) and structured `metadata` (processed by the kernel).

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> Value;
    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError>;
}
```

Tools receive a `ToolContext` with workspace root and session ID. They have **no direct access** to the harness engine or LLM — they are pure I/O operations. The kernel mediates between tools and everything else.

`ToolOutput` has two fields:
- `content` — returned to the LLM
- `metadata` — structured data for logging and kernel-level side effects (e.g., `submit_task` uses metadata to signal queue operations)

### Tool Side-Effects (`ToolEffect`)

Rather than letting tools modify kernel state directly, the kernel processes "hints" in the tool result metadata. 
- **Task Submission**: `submit_task` returns metadata that the kernel maps to a `ToolEffect::SubmitTask`, which appends new prompts to the session queue.
- **Workflow Control**: Metadata can signal the kernel to terminate a session or switch modes.

### Built-in Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with line numbers and metadata |
| `write_file` | Create or overwrite a file |
| `edit_file` | Apply targeted string replacements |
| `shell_exec` | Execute shell commands with stdout/stderr/exit code capture |
| `submit_task` | Propose a multi-step plan (triggers `on_task_submit` hook) |
| `bridge_mcp` | Connect to an MCP server and register its tools |

### MCP Integration

The `bridge_mcp` tool spawns an MCP server subprocess and registers its tools into the kernel's tool registry at runtime. This allows agents to dynamically discover and use tools from external MCP-compatible servers without recompilation.

---

## Persistence

Turin uses **Turso** (SQLite) for all persistence. The database (default: `.turin/state.db`) is configured with **Write-Ahead Logging (WAL)** and a 5-second busy timeout to handle the high-frequency event streaming produced by modern LLMs.

### Cognitive Memory

Turin implements hybrid semantic search using three strategies, combined via **Reciprocal Rank Fusion (RRF)**:
1. **Vector search** — Cosine similarity (weighted at 0.6).
2. **FTS5 search** — Keyword ranking (weighted at 0.4).
3. **LIKE fallback** — Robust tokenized pattern matching for zero-configuration search.

Harness scripts access memory through `turin.memory.store(text, metadata)` and `turin.memory.search(query, limit)`.

---

## Harness Engine Internals

### Sandboxing

The harness engine uses Luau's built-in sandbox mode. The initialization sequence is:

1. Create a fresh Lua VM
2. Register all Turin globals (`fs`, `db`, `json`, `time`, `session`, `turin`, `log`, verdict constants)
3. Enable sandbox mode — all globals become read-only, dangerous standard library functions are removed

This means harness scripts can read the injected globals but cannot modify them or access the underlying OS. The only way a harness script touches the filesystem, database, or network is through the explicit capability handles the kernel provides.

### Hot Reload

When a harness file changes (detected by filesystem watcher or `/reload` command):

1. A new Lua VM is created
2. New harness scripts are loaded and syntax-checked
3. If valid, the old VM is swapped atomically (behind an `Arc<Mutex<>>`)
4. If invalid, the old harness continues running and an error is logged

This means you can iterate on harness scripts while the agent is running without risking a crash.

### Module System

Harness scripts can return tables to act as reusable modules:

```lua
-- harnesses/utils.lua
local M = {}
function M.is_dangerous(cmd)
    return cmd:find("rm %-rf") or cmd:find("sudo")
end
return M
```

Other scripts import modules via `turin.import`:

```lua
-- harnesses/safety.lua
local utils = turin.import("utils")

function on_tool_call(call)
    if call.name == "shell_exec" and utils.is_dangerous(call.args.command) then
        return REJECT, "Dangerous command blocked"
    end
    return ALLOW
end
```

---

## Provider Architecture

Turin supports multiple LLM providers through a normalized abstraction:

```rust
pub trait InferenceProvider: Send + Sync {
    fn complete(&self, request: InferenceRequest) -> BoxFuture<Result<InferenceResult, SdkError>>;
    fn stream(&self, request: InferenceRequest) -> BoxFuture<Result<InferenceStream, SdkError>>;
}
```

Providers are configured by name in `turin.toml`:

```toml
[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

[providers.openai]
api_key_env = "OPENAI_API_KEY"

[providers.fast]
type = "openai"
api_key_env = "OPENAI_API_KEY"
base_url = "https://custom-endpoint.example.com/v1"
```

The default provider is set in `[agent].provider`. Harness scripts can switch providers mid-turn via `ctx.provider = "fast"` in `on_before_inference`.

Provider-specific SDK events are mapped to `KernelEvent`s at the boundary, so the rest of the system is provider-agnostic.

---

## Configuration

Configuration is loaded from `turin.toml` and parsed into a typed `TurinConfig` struct. The config has five sections:

| Section | Purpose |
|---------|---------|
| `[agent]` | System prompt, model, provider, thinking config |
| `[kernel]` | Workspace root, max turns, heartbeat interval |
| `[persistence]` | Database path |
| `[harness]` | Harness script directory |
| `[providers.*]` | Named provider configurations |
| `[embeddings]` | Embedding provider for cognitive memory |

CLI flags (`--model`, `--provider`, `--verbose`, `--json`) override config values.

---

## Project Structure (Core)

| Component | Path | Responsibility |
|-----------|------|----------------|
| **Kernel** | `src/kernel/` | The core agent loop, session manager, and event dispatcher. |
| **Harness**| `src/harness/`| The Luau evaluator, global API injections, and verdict logic. |
| **Tools**  | `src/tools/`  | The registry and built-in implementations (fs, mcp, shell). |
| **Inference**| `src/inference/`| Provider clients and the normalized event streaming adapter. |
| **Persistence**| `src/persistence/`| The Turso event log, memory storage, and schema management. |

---

## Dependencies

| Dependency | Purpose | Rationale |
|-----------|---------|-----------|
| `inference-sdk-core` | Provider trait | Shared abstraction for LLM providers |
| `anthropic-sdk` | Anthropic client | Claude API support |
| `openai-sdk` | OpenAI client | GPT/compatible API support |
| `mcp-sdk` | MCP client | Model Context Protocol integration |
| `mlua` (Luau) | Lua runtime | Harness engine with built-in sandboxing |
| `turso` | Database | Pure-Rust SQLite for persistence and memory |
| `tokio` | Async runtime | Required by inference SDKs |
| `clap` | CLI parsing | Argument parsing and subcommands |
| `serde` / `serde_json` | Serialization | Config parsing and event serialization |
| `notify` | File watching | Harness hot-reload trigger |

### Why Turso Over rusqlite

Pure Rust (no C linking), native async, and a path to Turso Cloud replication if needed. Slightly larger than rusqlite (~665KB vs ~500KB overhead) but eliminates cross-compilation issues and provides a paved road to cloud persistence.

### Why Luau Over Lua 5.4 / LuaJIT / Rhai

Built-in sandboxing is the deciding factor. Luau's sandbox mode removes dangerous functions and makes globals read-only without manual intervention. It also has a gradual type system (helpful for harness authors), is actively developed by Roblox, and is syntactically close enough to standard Lua that LLMs generate valid code for it.

### Build Profile

Release builds are optimized for distribution (~11MB binary):
- **LTO**: Enabled for full-binary optimization.
- **Panic**: Set to `abort` to reduce size and improve performance.
- **Strip**: Symbols are removed to minimize binary footprint.
- **SQLite**: Statically linked via the Turso crate with no external C dependencies.
