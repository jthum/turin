# Writing Harness Scripts

This guide covers how to build production-quality Turin harnesses using the canonical `runtime.*` API, stable hook lifecycle, and opt-in governance features.

## What a Harness Does

A Turin harness is a set of Luau scripts that define agent behavior.

Harness scripts can:

- govern tool execution (`on_tool_call`, `on_tool_result`)
- shape context before every inference (`on_turn_prepare`)
- route providers/models and transport options per turn
- orchestrate plans and tasks (`on_plan_submit`, queue steering hooks)
- use memory/KV/DB primitives for stateful behavior
- orchestrate peer agents
- observe all kernel events (`on_kernel_event`)

The kernel provides execution primitives. The harness provides policy and workflow.

## Harness Layout and Loading

Default directory:

```toml
[harness]
directory = ".turin/harnesses"
```

Recommended structure:

```text
.turin/harnesses/
├── 01_safety.lua
├── 02_routing.lua
├── 03_memory.lua
├── 10_workflow.lua
└── lib/
    ├── selectors.lua
    └── sanitizers.lua
```

Scripts are loaded as modules and can export functions/tables. Hooks can exist in globals or returned module tables (engine discovers them).

## Your First Harness

```lua
function on_tool_call(call)
  if call.name == "shell_exec" then
    local cmd = call.args.command or ""
    if cmd:find("rm %-rf") then
      return REJECT, "destructive command blocked"
    end
  end
  return ALLOW
end
```

## Writing With the DX Layer

Turin now ships a first-party DX layer for harness authors.

Use it when you want harnesses to read more like intent and less like plumbing.

Important:

- `runtime.*` is still the canonical primitive layer
- DX helpers are wrappers over the same runtime/governance semantics
- DX helpers are best for readability, especially in real harness code

Concrete DX fixture harnesses live in:

- `tests/fixtures/dx/`

These are useful as small, executable reference harnesses rather than just inline snippets.

Ready-to-use harness library entries live in:

- `library/`

Those harnesses are intended as starting points you can lift into a project with minimal editing.

### Verdict helpers

```lua
function on_tool_call(call)
  return verdict.reject_if(call.name == "shell_exec", "shell disabled")
    or verdict.escalate_if(call.name == "write_file", "approve file write?")
    or verdict.allow()
end
```

### Access helpers

```lua
function on_turn_prepare(ctx)
  if not allowed("runtime.db.exec") then
    return verdict.reject("No DB exec capability")
  end

  needs("runtime.agent.submit")

  local d = access.check("runtime.policy.set")
  if not d.allowed then
    log("policy mutation denied: " .. tostring(d.reason))
  end

  return ALLOW
end
```

`needs(...)` raises a Lua error on denial. Use it when denial should abort the current path immediately.

### Session and user helpers

```lua
function on_session_start(ev)
  session.remember("User prefers concise answers")
  user.set("timezone", "UTC")
  session.incr("session_starts")
  return ALLOW
end
```

### Fluent DB access

```lua
function on_turn_prepare(ctx)
  runtime.db.with("state", function(db)
    db:exec("CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, text TEXT)")
    db:exec("INSERT INTO notes(text) VALUES (?)", { "hello" })

    local row = db:one("SELECT text FROM notes ORDER BY id DESC LIMIT 1")
    if row and row.text == "hello" then
      session.set("last_note", row.text)
    end
  end)

  return ALLOW
end
```

Notes:

- `db:one(...)` returns the first row or `nil`
- `runtime.db.with(...)` prioritizes callback errors over close errors

### Fluent peer-agent access

```lua
function on_turn_prepare(ctx)
  local reviewer = runtime.agent("reviewer")
  local summary = reviewer:complete("Summarize the diff in 3 bullets")
  session.set("review_summary", summary)
  return ALLOW
end
```

Canonical equivalent:

```lua
function on_turn_prepare(ctx)
  local summary, err = runtime.agent.complete("reviewer", "Summarize the diff in 3 bullets")
  if not summary then error(err) end
  session.set("review_summary", summary)
  return ALLOW
end
```

### Grant wrapper

```lua
function on_turn_prepare(ctx)
  local result = runtime.governance.grant({
    ttl_ms = 5000,
    capabilities = {
      ["runtime.agent.submit"] = true,
      ["runtime.agent.await"] = true,
    }
  }, function()
    return runtime.agent("reviewer"):complete("Review this patch")
  end)

  session.set("grant_review", result)
  return ALLOW
end
```

### Time and JSON helpers

```lua
function on_turn_prepare(ctx)
  local started = session.get("started_at")
  if started and time.after(started, 300) then
    return verdict.escalate("Session has been running for more than 5 minutes")
  end

  local cfg = fs.read_json("config/agent.json")
  cfg.last_seen = time.now_utc()
  fs.write_json("config/agent.json", cfg, { pretty = true })

  return ALLOW
end
```

## Module Imports (`import` / `import_scoped`)

### `import(name)`

Basic module import by harness module name.

```lua
local helpers = import("helpers")
```

### `import_scoped(name, opts)`

Import with governance root and optional delegated capability ceiling.

```lua
local plugin = import_scoped("plugins/reformatter", {
  root = "plugins_writable",
  capabilities = {
    ["runtime.db.query"] = true,
    ["runtime.db.exec"] = false,
    ["fs.read"] = true,
    ["fs.write"] = true,
  }
})
```

Use this when you want self-evolving or semi-trusted harness modules with constrained authority.

## Hook Patterns

## 1. Tool Governance (`on_tool_call`)

```lua
function on_tool_call(call)
  if call.name == "shell_exec" then
    local cmd = call.args.command or ""
    if cmd:find("sudo") then
      return REJECT, "sudo not allowed"
    end
    if cmd:find("git push") then
      return ESCALATE, "git push requires approval"
    end
  end
  return ALLOW
end
```

## 2. Tool Result Sanitization (`on_tool_result`)

```lua
function on_tool_result(result)
  if result.name == "read_file" and not result.is_error then
    local out = result.output or ""
    out = out:gsub("API_KEY=%S+", "API_KEY=[REDACTED]")
    return MODIFY, { output = out, is_error = false }
  end
  return ALLOW
end
```

## 3. Context Engineering (`on_turn_prepare`)

```lua
function on_turn_prepare(ctx)
  if ctx.is_first_turn_in_task then
    ctx.system_prompt = ctx.system_prompt .. "\n\nAlways explain your plan before editing files."
  end

  -- Dynamic provider routing (example)
  if ctx.prompt and ctx.prompt:find("fast") then
    ctx.provider = "openai_fast"
  end

  -- Request transport tuning for this turn
  ctx.request_options = {
    headers = { ["x-run-purpose"] = "interactive" },
    request_timeout_secs = 45,
    total_timeout_secs = 90,
  }

  return ALLOW
end
```

Note: `ctx.provider` and other mutable fields are part of the `ContextWrapper` contract. `ctx.model` is currently readable but not writable. See `docs/HOOKS.md` for exact semantics.

## 4. Plan Review (`on_plan_submit`)

```lua
function on_plan_submit(plan)
  if #plan.tasks > 20 then
    return REJECT, "plan too large"
  end

  -- Ensure plan does not wipe current queue by default
  return MODIFY, {
    title = plan.title,
    tasks = plan.tasks,
    clear_existing = false,
  }
end
```

## 5. Recovery (`on_inference_error`)

```lua
function on_inference_error(event)
  log("inference error on task " .. event.task_id .. ": " .. event.error)

  -- Queue a fallback task
  return MODIFY, {
    { prompt = "Retry the task with a shorter plan and no shell commands." }
  }
end
```

## Using the Canonical Runtime API (`runtime.*`)

Prefer `runtime.*` in new harnesses, even though aliases remain available.
When in doubt:

- use DX helpers for readability
- use `runtime.*` directly when you need exact primitive control or want the tuple-style API explicitly

### Context selectors

```lua
local ctx = runtime.context("agent", "coder", {
  namespace = "project_memory",
  visibility = "private",
})
```

### Memory and KV

```lua
local hits, err = runtime.memory.search("compiler error parsing", ctx, { limit = 5 })
if hits then
  for _, row in ipairs(hits) do
    log((row.score or 0) .. " " .. (row.content or ""))
  end
end

local ok, kerr = runtime.kv.set("last_error", "E0425", ctx)
```

### Multi-DB access

```lua
local handle, err = runtime.db.open({ path = "scratch/analysis.db" })
if not handle then
  return REJECT, err
end

local changed, e = runtime.db.exec(
  "create table if not exists notes (id integer primary key, text text)",
  nil,
  { handle = handle.handle }
)

local rows, qerr = runtime.db.query(
  "select * from notes where id > :min_id",
  { min_id = 0 },
  { handle = handle.handle }
)
```

### Peer-agent orchestration

```lua
local task_id, err = runtime.agent.submit("reviewer", {
  prompt = "Review the proposed patch and list regressions",
  title = "regression review",
}, {
  capabilities = {
    ["runtime.db.query"] = true,
    ["runtime.db.exec"] = false,
    ["fs.read"] = true,
    ["fs.write"] = false,
  }
})

if task_id then
  local result, aerr = runtime.agent.await(task_id, { timeout_ms = 30000 })
  if result then
    log(json.encode(result))
  end
end
```

## Using Top-Level Aliases (Ergonomic)

Turin keeps ergonomic aliases for common workflows.

### Agent-scoped defaults

```lua
local ok, err = kv.set("task_state", "working")
local rows, merr = memory.search("build failure")
```

### Session/User scoped aliases

```lua
local ok, err = session.kv.set("step", "planning")
local profile, perr = user.kv.get("profile")
```

These aliases rely on the active runtime identity; `user.*` requires `identity.user_id`.

## Governance-Aware Harness Design

Turin is flexibility-first, but you can design harnesses to cooperate with governance cleanly.

### 1. Prefer capability checks for feature toggles

```lua
local decision = runtime.governance.check("runtime.db.exec")
if decision and decision.allowed == false then
  return REJECT, "db writes not allowed in this profile"
end
```

### 2. Use temporary grants for explicit elevation

```lua
local grant, err = runtime.governance.grant_issue({
  capabilities = { ["runtime.db.exec"] = true },
  ttl_ms = 15000,
  max_uses = 1,
  reason = "one-shot migration",
})

if grant then
  runtime.governance.with_grant(grant.grant_id, function()
    local changed, e = runtime.db.exec("delete from temp_rows where stale = 1")
    if not changed then error(e) end
  end)
end
```

### 3. Partition harnesses by roots

A strong pattern is to keep:

- `core` harness scripts in a read-only root
- `plugins` in a writable root
- `import_scoped(..., { root = "plugins", capabilities = ... })` for self-evolving modules

This preserves user control while allowing controlled autonomy.

## Patterns for Maintainable Harnesses

## Split by concern

- `01_safety.lua` — hard constraints
- `02_governance.lua` — capability-aware behavior
- `10_context.lua` — `on_turn_prepare` context shaping
- `20_workflow.lua` — queue steering, plan shaping
- `30_memory.lua` — retrieval/anchoring

## Use modules for shared logic

```lua
-- helpers.lua
local M = {}
function M.redact_secrets(s)
  return (s or ""):gsub("API_KEY=%S+", "API_KEY=[REDACTED]")
end
return M
```

```lua
-- 01_safety.lua
local helpers = import("helpers")
function on_tool_result(r)
  if r.name == "read_file" and not r.is_error then
    return MODIFY, { output = helpers.redact_secrets(r.output), is_error = r.is_error }
  end
  return ALLOW
end
```

## Be explicit about tuple returns

Many primitives return `(value, err)` / `(ok, err)`.
Always branch on the first return value.

## Prefer canonical APIs in new code

Aliases are great for ergonomics, but `runtime.*` is clearer and more future-proof for shared harness libraries.

## Debugging Harnesses

- Use `log("...")` for harness-level diagnostics
- Use `turin repl` and `/reload` to iterate quickly
- Use `on_kernel_event` for temporary deep observability
- Run `turin check` to validate config + harness syntax
- Use `scripts/live_minimax_smoke.sh` (manual/opt-in) for real endpoint testing

## Compatibility Note

Older Turin docs/examples used `turin.*` and `db.*` namespaces.
The canonical API is now `runtime.*` plus the aliases documented in `docs/PRIMITIVES.md`.
