# Writing Harness Scripts

This guide covers everything you need to write Turin harness scripts — from basic governance to advanced workflow orchestration.

---

## What Is a Harness?

A harness is a Lua script that hooks into the kernel's event lifecycle. When the kernel is about to do something — call an LLM, execute a tool, start a session — it fires an event through the harness engine. Your script intercepts that event and returns a verdict: allow it, reject it, escalate it to a human, or modify it.

But harnesses go beyond governance. The `on_turn_prepare` hook gives you full control over what the LLM sees. Combined with the kernel's primitives (filesystem, database, memory, session management, subagents), harness scripts define the agent's entire behavior: workflow, personality, context strategy, memory policy.

The kernel provides the physics. Your harness defines the universe.

---

## Getting Started

### Setup

1. Create a harness directory (default: `.turin/harnesses/`)
2. Add `.lua` files to the directory
3. Ensure your `turin.toml` points to it:

```toml
[harness]
directory = ".turin/harnesses"
```

Scripts load in **alphabetical order**. Name them with numeric prefixes to control evaluation order:

```
.turin/harnesses/
├── 01_safety.lua       # Hard constraints (evaluated first)
├── 02_budget.lua       # Cost controls
└── 03_workflow.lua     # Context engineering and workflows
```

### Your First Harness

```lua
-- .turin/harnesses/01_safety.lua

function on_tool_call(call)
    if call.name == "shell_exec" then
        local cmd = call.args.command
        if cmd:find("rm %-rf") then
            return REJECT, "Destructive command blocked by safety harness"
        end
    end
    return ALLOW
end
```

That's it. Save the file, and the kernel will physically prevent `rm -rf` from executing. The rejection reason is fed back to the LLM so it can try a different approach.

### Hot Reload

You can modify harness scripts while the agent is running:

- **Automatic**: If file watching is active, changes are picked up automatically
- **Manual**: Type `/reload` in the REPL

If your new script has a syntax error, the old harness keeps running and an error is logged. You won't crash the agent.

---

## Hooks Reference

For lifecycle hooks, payloads now include `event.identity` with:
`session_id` and optional `agent_id`, `user_id`, `channel_id`, `tenant_id`, `run_id`.

### `on_session_start(event)`

Fires once when a session begins.

```lua
function on_session_start(event)
    -- event.session_id: string
    log("Session started: " .. event.session_id)

    -- Queue follow-up tasks
    session.queue("Review the test results")

    return ALLOW
end
```

### `on_turn_prepare(ctx)`

Fires before every LLM call. This is the most powerful hook — it gives you access to the full context and lets you modify it.

```lua
function on_turn_prepare(ctx)
    -- Readable properties:
    -- ctx.model: string (current model)
    -- ctx.system_prompt: string (current system prompt)
    -- ctx.messages: table (message history)
    -- ctx.token_count: number (current context size)
    -- ctx.provider: string (current provider name)
    -- ctx.thinking_budget: number (thinking token budget)
    -- ctx.request_options: table (request overrides for this turn)
    --   request_options.headers: table<string, string>
    --   request_options.max_retries: number | nil
    --   request_options.request_timeout_secs: number | nil
    --   request_options.total_timeout_secs: number | nil

    -- Writable properties:
    -- ctx.system_prompt = "new prompt"
    -- ctx.provider = "different-provider"
    -- ctx.thinking_budget = 8192
    -- ctx.request_options = {
    --   headers = { ["anthropic-beta"] = "output-128k-2025-02-19" },
    --   max_retries = 1,
    --   request_timeout_secs = 45,
    -- }

    -- Methods:
    -- ctx:add_message({ role = "user", content = {{type="text", text="..."}} })
    -- ctx:summarize() → string (calls LLM to summarize current messages)

    return ALLOW
end
```

Inject request headers dynamically:
```lua
function on_turn_prepare(ctx)
    if ctx.provider == "anthropic" and ctx.thinking_budget > 0 then
        local opts = ctx.request_options or {}
        opts.headers = opts.headers or {}
        opts.headers["anthropic-beta"] = "output-128k-2025-02-19"
        ctx.request_options = opts
    end
    return ALLOW
end
```

**Common patterns:**

Inject project instructions:
```lua
function on_turn_prepare(ctx)
    if fs.exists("TURIN.md") then
        ctx.system_prompt = ctx.system_prompt .. "\n\n" .. fs.read("TURIN.md")
    end
    return ALLOW
end
```

Switch provider based on task:
```lua
function on_turn_prepare(ctx)
    local msgs = ctx.messages
    if msgs and #msgs > 0 then
        local last = msgs[#msgs]
        -- Use a cheaper model for simple questions
        if type(last.content) == "string" and #last.content < 100 then
            ctx.provider = "fast"
        end
    end
    return ALLOW
end
```

### `on_tool_call(call)`

Fires when the LLM requests a tool execution. This is where governance lives.

```lua
function on_tool_call(call)
    -- call.id: string (tool call ID)
    -- call.name: string (tool name)
    -- call.args: table (tool arguments)

    -- Block specific tools
    if call.name == "shell_exec" then
        return REJECT, "Shell access is disabled"
    end

    -- Modify arguments
    if call.name == "write_file" then
        -- Force all writes to a sandbox directory
        call.args.path = "sandbox/" .. call.args.path
        return MODIFY, call.args
    end

    -- Escalate to human
    if call.name == "write_file" and call.args.path:find("/src/") then
        return ESCALATE, "Human approval required for source file edits"
    end

    return ALLOW
end
```

### `on_tool_result(result)`

Fires after a tool executes. Useful for logging and post-processing.

```lua
function on_tool_result(result)
    -- result.id: string
    -- result.output: string
    -- result.is_error: boolean

    if result.is_error then
        log("Tool error: " .. result.output)
    end
    return ALLOW
end
```

### `on_plan_submit(payload)`

Fires when the agent calls `submit_plan` to propose a plan.

```lua
function on_plan_submit(payload)
    -- payload.title: string
    -- payload.tasks: table (list of task strings)

    -- Review the plan
    log("Agent proposed: " .. payload.title)
    for i, task in ipairs(payload.tasks) do
        log("  " .. i .. ". " .. task)
    end

    -- Modify the plan
    if payload.title == "Refactor" then
        return MODIFY, { "Write tests first", "Then refactor", "Run tests again" }
    end

    -- Reject the plan
    if #payload.tasks > 10 then
        return REJECT, "Plan is too complex. Break it into smaller chunks."
    end

    return ALLOW
end
```

### `on_task_complete(event)`

Fires once per task when it reaches a terminal status (`success`, `rejected`, `max_turns`, `error`, `cancelled`).

```lua
function on_task_complete(event)
    -- event.session_id: string
    -- event.task_id: string
    -- event.plan_id: string | nil
    -- event.status: string

    -- Retry policy example
    if event.status == "error" then
        return MODIFY, {
            "Investigate failure for task " .. event.task_id,
            "Re-attempt task " .. event.task_id .. " with a narrower scope"
        }
    end

    return ALLOW
end
```

### `on_inference_error(event)`

Fires when a task fails with a runtime inference/provider error. This hook can enqueue fallback tasks before the run exits.

```lua
function on_inference_error(event)
    -- event.session_id: string
    -- event.task_id: string
    -- event.plan_id: string | nil
    -- event.turn_count: number
    -- event.error: string

    return MODIFY, {
        {
            title = "Fallback with backup provider",
            prompt = "Retry the failed task using provider 'openai-backup'."
        }
    }
end
```

### `on_all_tasks_complete(event)`

Fires when the global queue is empty.

```lua
function on_all_tasks_complete(event)
    -- event.session_id: string
    -- event.turn_count: number

    -- Optional end-of-run anchoring
    if turin.memory and turin.agent then
        local history = turin.session.load(event.session_id)
        if history and #history > 2 then
            local summary = turin.agent.spawn(
                "Summarize this session in one sentence: " .. json.encode(history),
                { system_prompt = "You are a concise summarizer.", max_turns = 1 }
            )
            if summary then
                turin.memory.store(summary, { session_id = event.session_id })
            end
        end
    end

    return ALLOW
end
```

### `on_token_usage(usage)`

Fires after each LLM call with token accounting.

```lua
function on_token_usage(usage)
    -- usage.input_tokens: number
    -- usage.output_tokens: number
    -- usage.total_tokens: number
    -- usage.cost_usd: number (estimated)
    return ALLOW
end
```

### `on_turn_start(event)` / `on_turn_end(event)`

Fire at the beginning and end of each LLM turn.

```lua
function on_turn_start(event)
    -- event.turn_index: number
    return ALLOW
end

function on_turn_end(event)
    -- event.turn_index: number
    -- event.has_tool_calls: boolean
    return ALLOW
end
```

### `on_session_end(event)`

Fires when the session completes.

```lua
function on_session_end(event)
    -- event.message_count: number
    -- event.total_input_tokens: number
    -- event.total_output_tokens: number
    return ALLOW
end
```

---

## Kernel Primitives

These globals are injected by the kernel and available to all harness scripts.

### Verdicts

```lua
ALLOW     -- Proceed with the action
REJECT    -- Block the action (return with reason: return REJECT, "why")
ESCALATE  -- Pause for human input (return with reason: return ESCALATE, "why")
MODIFY    -- Proceed with modified data (return with data: return MODIFY, new_data)
```

### Filesystem (`fs`)

All paths are sandboxed to the configured `fs_root`.

```lua
fs.read(path)           -- Read file contents. Returns string or nil.
fs.write(path, content) -- Write content to a file.
fs.exists(path)         -- Returns boolean.
fs.list(path)           -- List directory contents. Returns table of filenames.
fs.is_safe_path(path)   -- Returns boolean. Checks path is within sandbox.
```

### Database (`db`)

Persistent key-value store backed by Turso. Survives restarts.

```lua
db.kv_get(key)              -- Returns string or nil.
db.kv_set(key, value)       -- Set a value.
db.kv_set(key, value, ttl)  -- Set with TTL (seconds).
db.kv_set(key, nil)         -- Delete a key.
```

### JSON

```lua
json.encode(table)  -- Serialize a Lua table to a JSON string.
json.decode(str)    -- Deserialize a JSON string to a Lua table.
```

### Time

```lua
time.now_utc()  -- Returns current UTC timestamp as ISO 8601 string.
```

### Logging

```lua
log(message)  -- Write a message to the kernel event log (visible in verbose mode).
```

### Session

```lua
session.identity.session_id -- Current session ID (string).
session.list()            -- List all previous session IDs.
session.load(id)          -- Load message history from a session. Returns table.
session.queue(prompt)     -- Add a task to the end of the queue.
session.queue_next(prompt) -- Add a priority task to the front of the queue.
```

### Memory (`turin.memory`)

Semantic memory with hybrid search.

```lua
turin.memory.store(text, metadata)   -- Store a memory. metadata is an optional table.
turin.memory.search(query, limit)    -- Search memories. Returns table of {content, score}.
```

### Subagents (`turin.agent`)

Spawn isolated nested kernel instances.

```lua
turin.agent.spawn(prompt, options)
-- options (optional table):
--   system_prompt: string
--   max_turns: number
--   provider: string (named provider from config)
-- Returns: string (agent response) or nil, error
```

### File Search (`turin.context`)

```lua
turin.context.glob(pattern)  -- Search for files matching a glob pattern.
                                -- Returns table of file paths.
```

### Module System (`turin.import`)

```lua
local module = turin.import("module_name")  -- Import a harness module by filename (without .lua)
```

---

## Composition

Multiple harness scripts compose automatically. For each event, every loaded script's hook is called in alphabetical order.

**Verdict composition rules:**
1. If any script returns `REJECT` — the action is blocked (first reject wins)
2. If any script returns `ESCALATE` — the action is paused for human approval
3. If all scripts return `ALLOW` — the action proceeds
4. `MODIFY` is treated as `ALLOW` but carries modified data

This lets you separate concerns into focused scripts:

```
01_safety.lua     -- Hard constraints. Never allow rm -rf, sudo, etc.
02_budget.lua     -- Token budget enforcement.
03_workflow.lua   -- Context engineering, instruction injection.
04_memory.lua     -- Memory recall and anchoring.
```

Each script only handles its own concern. A rejection from `01_safety.lua` short-circuits — `02_budget.lua` and beyond don't need to evaluate.

---

## Modules

Harness scripts can return a table to act as an importable module:

```lua
-- harnesses/patterns.lua
local M = {}

function M.is_destructive_command(cmd)
    local patterns = { "rm %-rf", "sudo", "mkfs", "dd if=", "> /dev/" }
    for _, p in ipairs(patterns) do
        if cmd:find(p) then return true end
    end
    return false
end

function M.is_source_file(path)
    return path:find("%.rs$") or path:find("%.py$") or path:find("%.ts$")
end

return M
```

```lua
-- harnesses/safety.lua
local patterns = turin.import("patterns")

function on_tool_call(call)
    if call.name == "shell_exec" and patterns.is_destructive_command(call.args.command) then
        return REJECT, "Destructive command blocked"
    end
    if call.name == "write_file" and patterns.is_source_file(call.args.path) then
        return ESCALATE, "Human approval required for source file edits"
    end
    return ALLOW
end
```

---

## Recipes

### Coding Assistant with Project Awareness

```lua
-- 03_coding.lua

function on_turn_prepare(ctx)
    -- Inject project instructions
    if fs.exists("TURIN.md") then
        ctx.system_prompt = ctx.system_prompt ..
            "\n\n=== Project Instructions ===\n" .. fs.read("TURIN.md")
    end

    -- Recall relevant memories
    if turin.memory and turin.memory.search then
        local msgs = ctx.messages
        for i = #msgs, 1, -1 do
            if msgs[i].role == "user" then
                local query = msgs[i].content
                if type(query) == "table" and query[1] then query = query[1].text or "" end
                if type(query) == "string" and #query > 10 then
                    local memories = turin.memory.search(query, 3)
                    if memories and #memories > 0 then
                        local block = "\n\n=== Relevant Memories ===\n"
                        for _, m in ipairs(memories) do
                            block = block .. "- " .. m.content .. "\n"
                        end
                        ctx.system_prompt = ctx.system_prompt .. block
                    end
                end
                break
            end
        end
    end

    return ALLOW
end
```

### Session Continuity

```lua
-- 04_resume.lua

function on_session_start(event)
    local sessions = session.list()
    if #sessions > 0 then
        local prev = session.load(sessions[#sessions])
        if prev and #prev > 0 then
            local summary = turin.agent.spawn(
                "Summarize this conversation in 2-3 sentences:\n" .. json.encode(prev),
                { system_prompt = "Be concise.", max_turns = 1 }
            )
            if summary then
                db.kv_set("prev_session_summary", summary)
            end
        end
    end
    return ALLOW
end

function on_turn_prepare(ctx)
    local summary = db.kv_get("prev_session_summary")
    if summary and #ctx.messages <= 1 then
        ctx.system_prompt = ctx.system_prompt ..
            "\n\n=== Previous Session ===\n" .. summary
        db.kv_set("prev_session_summary", nil)
    end
    return ALLOW
end
```

### Loop Detection

```lua
-- 02_loops.lua

local rejection_counts = {}

function on_tool_call(call)
    local key = call.name .. ":" .. json.encode(call.args)
    rejection_counts[key] = (rejection_counts[key] or 0)

    if rejection_counts[key] >= 3 then
        return REJECT, "This action has been attempted too many times. Try a different approach."
    end
    return ALLOW
end
```

### Conversation-Only Agent (No Tools)

```lua
-- 01_no_tools.lua

function on_turn_prepare(ctx)
    ctx.system_prompt = [[
You are a thoughtful advisor. You provide guidance through conversation only.
You do not write code, modify files, or run commands.
]]
    return ALLOW
end

function on_tool_call(call)
    return REJECT, "This agent operates through conversation only"
end
```

### Planning-First Workflow

```lua
-- 03_planning.lua

function on_turn_prepare(ctx)
    local msgs = ctx.messages
    if msgs and #msgs > 0 then
        local last = msgs[#msgs]
        if last.role == "user" then
            local text = last.content
            if type(text) == "table" and text[1] then text = text[1].text or "" end
            if type(text) == "string" then
                text = text:lower()
                if text:find("build") or text:find("implement") or text:find("create") then
                    ctx.system_prompt = ctx.system_prompt ..
                        "\n\nIMPORTANT: Before taking any action, use 'submit_plan' to propose your plan."
                end
            end
        end
    end
    return ALLOW
end

function on_plan_submit(payload)
    if #payload.tasks > 8 then
        return REJECT, "Too many tasks. Break this into phases of 5 or fewer steps."
    end
    log("Plan approved: " .. payload.title .. " (" .. #payload.tasks .. " steps)")
    return ALLOW
end
```

---

## Tips

- **Start simple.** A 10-line safety harness is more useful than an ambitious 200-line workflow that's hard to debug.
- **Use `log()` liberally.** Run with `--verbose` to see harness output. It's the easiest way to understand what your hooks are doing.
- **Use `pcall` for fallible operations.** If `turin.memory.search` or `turin.agent.spawn` might fail, wrap them in `pcall` to avoid crashing the harness.
- **Separate concerns.** One script per concern (safety, budget, workflow, memory) is easier to reason about than one monolithic script.
- **Test with mock provider.** Configure a `mock` provider in `turin.toml` to test harness logic without spending API tokens.
- **Use the KV store for state.** `db.kv_set/kv_get` persists across restarts. Use it for budgets, counters, session summaries — anything the harness needs to remember.
