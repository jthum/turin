# Turin: A Substrate for Programmatic Autonomy

**Turin** is not an agent. It is a single-binary, event-driven runtime designed to be the "rock" that agents stand on.

Most AI frameworks bake **Personality** (how the agent talks) and **Workflow** (how the agent sequences tasks) into their core. Turin takes a different approach: it provides the **Physics of Execution** — how an agent interacts with the real world, how state is persisted, how events flow — and then exposes primitives that let **Lua harness scripts** define personality, workflow, governance, and any other behavior. The Kernel has no opinions. Your harness has all of them.

---

## 1. The Core Philosophy: Physics vs. Harness

The defining characteristic of Turin is the strict separation of concerns into three distinct layers:

### Layer 1: The Kernel (Rust) — "The Physics"
The Kernel is written in Rust for performance and safety. It doesn't know *why* an agent is doing something; it only knows *how* to do it. It manages:
- **Transport**: HTTP connections to LLMs (Anthropic, OpenAI, etc.) via the normalized `InferenceProvider` trait.
- **Named Providers**: Support for multiple configured instances of the same or different providers, resolved by name.
- **Streaming**: Parsing raw SSE events into structured `KernelEvent`s.
- **Tool Execution**: Running shell commands, reading/writing files — with deterministic result capture.
- **Persistence**: Atomic transactions to a libSQL (Turso-compatible) database.
- **Heartbeat**: Ensuring the process doesn't just "hang" or disappear.
- **MCP Bridge**: Dynamic tool discovery via Model Context Protocol.
    > **Security Note**: The `bridge_mcp` tool allows the agent to spawn arbitrary subprocesses (e.g., `npx`, `python`). In production environments, this should be restricted via the Harness.
- **Adaptive Thinking**: Extended reasoning support with streaming thinking blocks and budget control.
- **Cognitive Memory**: State-of-the-art hybrid search (Vector + FTS5) re-ranked via Reciprocal Rank Fusion (RRF), with graceful fallback to tokenized LIKE scans in degraded environments.
- **Subagent Primitive**: Recursive task delegation through isolated nested kernels.
- **Event Bus**: Every action in the system produces a typed event that flows through the Harness Engine before execution.

### Layer 2: The Harness Engine (Embedded Lua) — "The Law"
This is Turin's unique differentiator. All events in the Kernel must pass through a sandboxed Lua engine.
- **Deterministic**: Unlike "System Prompts" which the LLM might ignore, a Lua script is code. If it returns `REJECT`, the Kernel **physically cannot** execute the action.
- **Pluggable**: Users customize behavior by writing Lua harness scripts — no recompilation needed.
- **First-Class Modules**: Scripts can `return` tables of functions to act as reusable libraries.
- **Atomic Hot-Reload**: Scripts can be reloaded on-the-fly via `/reload` or automatic file watching, with "fail-safe" swap logic that preserves the working harness if the new one has errors.
- **Composable**: Multiple harness scripts can be loaded and composed. Each subscribes to the events it cares about.

### Layer 3: The Inference Layer (LLM) — "The Intent"
The LLM proposes actions (e.g., "I want to delete this file").
- **The LLM can propose, but it never enforces.** It is the "brain" sitting in a room with a microphone; it has no hands. The Kernel only uses its "hands" if the Harness Engine says the LLM's request is legal.

> **Inference can propose. Harness decides. Kernel enforces.**

---

## 2. Competitive Landscape

| Framework | Philosophy | Governance | Stack |
|-----------|-----------|-----------|-------|
| **Turin** | Substrate-first. Zero opinions on agent behavior. | **Hard.** Lua code. The agent *must* comply (or the Kernel stops it). | Rust (Single Binary) |
| **AIOS** | Research-kernel. Treats LLM as a CPU for scheduling. | Policy-oriented. Focuses on scheduling efficiency. | Python (Research) |
| **LangGraph / AutoGen** | Workflow-orchestrators. Focused on multi-agent graphs. | Implicit. Hardcoded in the graph logic. | Python |
| **Guardrails AI** | Filter-first. Sits between LLM and app to clean output. | Reactive. Tells you *after* the LLM said something bad. | Python |
| **pi-agent-core** | Stateful agent with tool execution and event streaming. | None. All trust is in the LLM. | TypeScript |

**The Gap Turin Fills**: Most tools treat AI governance as a "prompting challenge" or a "scheduling challenge." Turin treats it as an **Operating System challenge** focused on deterministic enforcement. It is for people who want to build agents that are as reliable as a cron job, but as smart as a human.

---

## 3. The Kernel Event System

Every action in Turin produces a typed `KernelEvent`. These events flow through the Harness Engine before the Kernel executes them. This is the core mechanism that makes governance deterministic.

### Event Dictionary

| Event | Payload | Harness Hook | Description |
|-------|---------|-------------|-------------|
| `session_start` | `{ session_id }` | `on_session_start` | Agent session begins |
| `session_end` | `{ session_id, turn_count, total_input_tokens, total_output_tokens }` | `on_session_end` | Agent session completes |
| `task_start` | `{ session_id, task_id, plan_id?, title?, prompt, queue_depth }` | `on_task_start` | Queued task begins |
| `task_complete` | `{ session_id, task_id, plan_id?, status, task_turn_count, turn_count }` | `on_task_complete` | Per-task terminal callback |
| `plan_complete` | `{ session_id, plan_id, title, total_tasks, completed_tasks }` | `on_plan_complete` | Plan terminal callback |
| `all_tasks_complete` | `{ session_id, turn_count }` | `on_all_tasks_complete` | Global queue exhausted |
| `turn_start` | `{ session_id, task_id, plan_id?, turn_index, task_turn_index }` | `on_turn_start` | New LLM call begins |
| `turn_prepare` | `{ turn_index, task_id, task_turn_index }` | `on_turn_prepare` | Context assembled, about to call LLM |
| `turn_end` | `{ session_id, task_id, plan_id?, turn_index, task_turn_index, has_tool_calls }` | `on_turn_end` | LLM call completes |
| `message_start` | `{ role, model }` | — | Streaming message begins |
| `message_delta` | `{ content_delta }` | — | Streaming chunk received |
| `thinking_delta` | `{ thinking }` | — | Streaming thinking chunk received |
| `message_end` | `{ message, usage }` | — | Complete message assembled |
| `tool_call` | `{ id, name, args }` | `on_tool_call` | LLM requests a tool execution |
| `tool_result` | `{ id, output, is_error }` | `on_tool_result` | Tool execution completed |
| `plan_submit` | `{ title, tasks, clear_existing }` | `on_plan_submit` | Agent proposes a multi-step plan |
| `token_usage` | `{ input, output, cost }` | `on_token_usage` | Token/cost accounting update |

### Harness Verdicts

Every harness hook returns one of three verdicts:

```lua
return ALLOW                          -- Proceed normally
return REJECT, "reason"               -- Block the action, feed reason to LLM
return ESCALATE, "reason"             -- Pause and ask a human
return MODIFY, { ... }                -- Carry modified data (args or tasks)
```

### The `on_turn_prepare` Hook — Context Engineering

The `on_turn_prepare` hook is uniquely powerful: it can **modify the context** before it reaches the LLM. This is how all "context engineering" workflows are implemented — not as Kernel features, but as harness scripts.

The Kernel provides primitives (`context.summarize()`, `context.slice()`, `session.load()`, `fs.read()`), and the harness script decides _how_ to use them:

- **Project instructions**: Read a `turin.md` file and inject it into the system prompt
- **Context compaction**: Summarize old messages when approaching the context window limit
- **Session resumption**: Load previous session messages from the database on startup
- **Dynamic injection**: Add recent git diffs, test results, or any other context before each LLM call
- **Adaptive Thinking**: Dynamically boost or cap the `thinking_budget` based on task difficulty
- **Named Providers**: Switch between multiple configured provider instances (e.g., `ctx.provider = "gpt-4o-primary"`) mid-turn.
- **Anchorage Strategy**: Recursively spawn subagents to summarize sessions and store facts in the memory store.

> **The Kernel provides the physics. Your harness defines the universe.**

---

## 4. The Power of Pluggable Governance

Because governance is a Lua script, you can build harnesses that change how the agent functions without re-compiling the binary.

### Example A: The "Limited Autonomy" Budget
An agent is given a $10.00 daily budget.
```lua
local current_spend = 0

function on_token_usage(event)
  current_spend = current_spend + event.cost
  if current_spend > 10.00 then
    return REJECT, "Daily budget exceeded ($" .. current_spend .. ")"
  end
  return ALLOW
end
```

### Example B: The "Escalation" Boundary
An agent is allowed to read any file, but must ask a human before editing source code.
```lua
function on_tool_call(call)
  if call.name == "write_file" and call.args.path:find("/src") then
    return ESCALATE, "Human approval required for source edits"
  end
  return ALLOW
end
```

### Example C: The "Loop Detector"
Prevent the LLM from retrying the same rejected action repeatedly.
```lua
local rejection_counts = {}

function on_tool_call(call)
  local key = call.name .. ":" .. tostring(call.args)
  rejection_counts[key] = (rejection_counts[key] or 0)

  -- If the same action was rejected 3+ times, hard-stop
  if rejection_counts[key] >= 3 then
    return REJECT, "Action attempted too many times. Try a different approach."
  end
  return ALLOW
end
```

---

## 5. Self-Evolving Governance

One of the most intriguing possibilities Turin enables is allowing an agent to **write its own governance scripts.**

### The Workflow
1. A "Senior Agent" is tasked with managing a "Junior Agent."
2. The Senior Agent observes the Junior Agent's mistakes via the event log.
3. The Senior Agent **drafts a new Lua harness script** to prevent those mistakes.
4. The draft is validated by the Kernel (syntax check, sandbox test).
5. If the harness passes validation, it is loaded into the Junior Agent's Harness Engine.
6. The Junior Agent is now physically constrained by the new "Law."

### Constitutional Invariants
Not everything should be modifiable. Turin distinguishes between:

| Category | Modifiable by Agent? | Example |
|----------|---------------------|---------|
| **Kernel Invariants** | ❌ Never | Sandbox boundaries, max file size, capability grants |
| **Boot Harnesses** | ❌ Never | Harness scripts loaded at startup via config |
| **Runtime Harnesses** | ✅ With validation | Harness scripts drafted/loaded by the Agent during execution |

This means an agent can propose new governance rules, but it can **never** remove the foundational safety constraints set by the operator at boot time.

### The "Self-Evolving" Runtime
For those running Turin in isolated, sandboxed environments, you could give an agent the power to draft new harness scripts that refine its own behavior:
- **The Promise**: An agent that learns from its mistakes and codifies those lessons as deterministic rules.
- **The Danger**: Recursive self-modification. Turin mitigates this by making Kernel Invariants and Boot Harnesses immutable — the "constitutional turin" that no runtime harness can override.

---

## 6. Theory vs. Practice: Real-World Challenges

Turin sounds like a perfect "Agentic Kernel" in theory, but it faces significant practical hurdles:

1. **The "Reasoning" Gap**: LLMs are fluent in Lua, but not perfect. If an agent drafts a harness script with a syntax error, it could brick itself. The Kernel needs robust validation and a "Safe Mode" recovery path.

2. **State Bloat**: Storing every event, heartbeat, and retry is great for auditability, but for long-running agents this grows massive. The Harness Engine itself should be able to define pruning strategies — governance over its own history.

3. **Context Drift**: Even with hard governance, the LLM needs to *understand* why it was rejected. When the Harness Engine rejects an action, Turin injects the rejection reason as structured context into the LLM's next turn — not as a vague error, but as an explicit `harness_rejection` event with the reason and the offending action.

4. **Lua Limitations**: Lua is fast and small, but not as expressive as Python for complex data processing. Turin addresses this through **Metatables** (bridging Rust strict types to Lua), **Userdata** (zero-copy access to large contexts), and the **Turin Standard Library** (curated helpers).

---

## 7. The Case for Lua + Luau

While we initially explored **Rhai** (a native Rust scripting language), **Lua** is the definitive governance substrate for Turin, specifically via the **Luau** dialect.

### Why Lua Wins (The "Game Engine" Rationale)
The limitations of Lua's expressiveness are well-known, but they have been solved by the gaming industry over the last 30 years:

1. **Metatables (Physics-Level Syntax)**: By using metatables, we "teach" Lua how to interact with native Rust types. The Lua script interacts with a proxy to a real Rust `Vec`, preserving type-safety when data returns to the Kernel.

2. **Userdata (Zero-Copy Performance)**: Large LLM contexts (e.g., 1MB strings) are passed to Lua via Userdata handles. The script can inspect specific byte ranges without copying the entire string into the Lua VM.

3. **LLM Familiarity**: Every modern LLM (Claude, GPT-4o) is fluent in Lua. An agent can draft its own policies accurately without needing bespoke specification for a newer DSL like Rhai.

### Why Luau Specifically

| Feature | Lua 5.4 | LuaJIT | Luau |
|---------|---------|--------|------|
| Gradual type system | ❌ | ❌ | ✅ |
| Sandboxing (built-in) | ❌ | ❌ | ✅ |
| JIT compilation | ❌ | ✅ (best) | ✅ (x64/arm64) |
| `mlua` support | ✅ | ✅ | ✅ |
| LLM fluency | ✅ | ✅ | ✅ (Lua-compatible) |
| Active development | Maintenance | Maintenance | Active (Roblox) |

**Luau** gives us built-in sandboxing (critical for self-evolving governance), a gradual type system (better tooling for harness script authors), and competitive performance — all while remaining close enough to standard Lua that LLMs can generate it fluently.

### Capability-Oriented Architecture
Regardless of the Lua dialect, Turin uses a **Capability-Oriented Architecture**. Instead of exposing the entire OS to Lua, the Kernel injects specific "Resource Handles" (e.g., `fs`, `network`, `db`) only when the harness script explicitly requests them. Even a self-evolving harness cannot access subsystems it wasn't granted at boot time.

---

## 8. The Turin Standard Library (Turin-SL)

A curated set of helper functions available to all harness scripts:

### Must-Have: Core Survival Essentials

| Function | Purpose |
|----------|---------|
| `json.decode(str) / json.encode(tbl)` | Bridge between Rust types and Lua tables |
| `fs.is_safe_path(path)` | Normalize paths, prevent traversal attacks |
| `str.match_pattern(text, pattern)` | Unified pattern matching for keyword/PII detection |
| `llm.token_count(text, model)` | Accurate token counting for cost-gating |
| `turin.context.glob(pattern)` | Search for files relative to workspace root (safe) |
| `harness.reject(reason)` | Early exit with deterministic failure |
| `ctx.thinking_budget` | Read/write property for adaptive thinking budget |
| `turin.import(name)` | Import a harness module by its filename |

### Should-Have: Quality of Autonomy

| Function | Purpose |
|----------|---------|
| `db.kv_get(key) / db.kv_set(key, val, ttl)` | Ephemeral harness memory (e.g., retry counts) |
| `shell.validate_command(cmd)` | Check commands against whitelist |
| `str.redact_pii(text)` | Auto-mask emails, keys, phone numbers |
| `time.now_utc()` | Deterministic time for time-based governance |
| `http.head_status(url)` | Check URL reachability without full download |

### Could-Have: High-Stakes Experiments

| Function | Purpose |
|----------|---------|
| `crypto.sha256(data)` | Integrity checks for self-modifying code |
| `llm.estimate_cost(tokens, model)` | Real-time budget enforcement |
| `event.get_history(limit)` | Detect looped or obsessive behavior |
| `net.dns_lookup(host)` | Verify destinations before allowing requests |
| `fs.get_size(path)` | Prevent disk-bombing |

---

## 9. Future Roadmap: The "Shadow" Filesystem

Taking inspiration from projects like **Turso AgentFS**, we envision a future where Turin's filesystem governance moves from "Permission Checks" to **"Transparent Isolation."**

### Key Inspirations
- **Copy-on-Write (CoW) Isolation**: Instead of just allowing/denying a write, the Kernel intercepts the write and commits it to a "Shadow" layer in the database. The source tree remains untouched until a human "merges" the changes.
- **Durable Auditing**: Every file operation (even rejected ones) becomes a first-class event in the log. The filesystem becomes a time-machine for agent debugging.
- **Single-File Portability**: By using a SQLite-backed virtual filesystem, the entire state of an agent — its code, its memory, its experimental file changes — can be zipped into a single `.db` file and moved between machines.

### The "Boring" Path Forward
Turin will continue to prioritize the "boring" substrate. We won't implement a virtual filesystem today, but the choice of **Rust + libSQL + Lua** ensures we can evolve toward CoW isolation without breaking the "Physics vs. Harness" contract.

---

## 11. Task Decomposition & Steering (Opt-in Planning)

Turin does not force agents to plan. However, it provides the **primitives** for a harness to enforce a planning-first discipline. 

### The Steering Circuit
1.  **Harness Instruction**: Use `on_turn_prepare` to inject a requirement: *"You MUST use 'submit_plan' before taking action."*
2.  **Agent Proposal**: The agent calls the `submit_plan` tool.
3.  **Kernel Interception**: The Kernel emits `on_plan_submit` to the harness.
4.  **Harness Revision**: The harness can `ALLOW` the plan, `REJECT` it, or `MODIFY` it to correct the agent's path.
5.  **Execution**: The Kernel populates the session queue with the (potentially modified) tasks.

By keeping the queue in the Kernel and the logic in the Harness, Turin enables complex steering workflows while maintaining a lean, unopinionated core.

---

## 12. Conclusion

Turin is an attempt to move away from the **"Hype-Driven Architecture"** of the agentic world.

It assumes that in two years, the prompts we use today will be obsolete, but the need for a **reliable, single-binary, stateful runtime** will still exist. By stripping away the "Magic" and focusing on the "Physics," Turin provides a durable substrate.

It's not the smartest agent in the room; it's the room itself — built of reinforced concrete and governed by code you can actually read and audit.
