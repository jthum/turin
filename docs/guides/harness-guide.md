# Writing Harness Scripts

This guide covers how to build production-quality Turin harnesses using Turin's promoted helper-first DX layer, the canonical `runtime.*` substrate, the stable hook lifecycle, and opt-in governance features.

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

Additional named harnesses can be configured under `[harnesses.*]`, and agents can bind to them with `harness = "<id>"`. That lets one Turin runtime host multiple distinct harness programs without forcing them into one shared script tree.

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

Turin auto-loads top-level `.lua` files in the harness directory.

Nested files are inert unless you explicitly bring them in with:

- `import(...)` / `import_scoped(...)` for code reuse
- `use(...)` / `use_scoped(...)` for behavior blocks that contribute hooks
- `watch(...)` for extra hot-reload roots

Hot reload remains whole-harness, not per-file. In a multi-harness runtime, Turin now reloads only the affected harness runtime(s) for a file change instead of reloading every configured harness. If a harness reload changes its own `watch(...)` declarations, watcher roots are rebuilt from the reloaded harness state.

That means you can keep the flat multi-file style for simple harnesses, or move to an entrypoint-style structure by keeping one top-level `main.lua` and placing reusable blocks/modules under subdirectories.

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

## Declared Virtual Tools

Harnesses can now declare model-visible tools without adding new native Rust tools.

Use this when the agent should see a domain-specific tool name such as `play_song`, `lookup_ticket`, or `summarize_log_bundle`, while Turin still executes the underlying essential tools.

Example:

```lua
tool.declare("play_song", {
  description = "Play an audio file with mpg123",
  params = {
    filename = { type = "string", required = true }
  },
  handler = function(args)
    return tool.call("shell_exec", {
      command = "mpg123 " .. shell.quote(args.filename)
    })
  end
})
```

Multiple native calls are supported:

```lua
tool.declare("read_pair", {
  description = "Read two files in order",
  params = {
    first = { type = "string", required = true },
    second = { type = "string", required = true }
  },
  handler = function(args)
    return tool.sequence({
      tool.call("read_file", { path = args.first }),
      tool.call("read_file", { path = args.second })
    })
  end
})
```

Nested tool results can now be post-processed in Lua:

```lua
tool.declare("summarize_pair", {
  description = "Read two files and summarize them",
  params = {
    first = { type = "string", required = true },
    second = { type = "string", required = true }
  },
  handler = function(args)
    return tool.sequence({
      tool.call("read_file", { path = args.first }),
      tool.call("read_file", { path = args.second })
    }, function(results)
      return {
        content = "Combined: " .. results[1].content .. " | " .. results[2].content,
        is_error = results[1].is_error or results[2].is_error
      }
    end)
  end
})
```

Virtual tools can also call other virtual tools:

```lua
tool.declare("read_note", {
  description = "Read a note from disk",
  params = {
    path = { type = "string", required = true }
  },
  handler = function(args)
    return tool.call("read_file", { path = args.path })
  end
})

tool.declare("read_note_wrapped", {
  description = "Read a note through another virtual tool",
  params = {
    path = { type = "string", required = true }
  },
  handler = function(args)
    return tool.call("read_note", { path = args.path }, function(result)
      return "wrapped: " .. result.content
    end)
  end
})
```

Result callbacks can also return a follow-up plan:

```lua
tool.declare("resolve_pointer", {
  description = "Resolve a pointer file and read the final note",
  params = {
    pointer = { type = "string", required = true }
  },
  handler = function(args)
    return tool.call("read_file", { path = args.pointer }, function(result)
      return tool.call("read_note", { path = result.content })
    end)
  end
})
```

Notes:

- `tool.declare(...)` is load-time only, just like `use(...)` and `watch(...)`
- `params` is sugar that Turin normalizes into JSON Schema before sending the tool to the provider
- `input_schema = {...}` is still available when you need the full JSON Schema shape directly
- handlers run in the normal harness environment, so they can use `runtime.*`, DX helpers, memory, KV, DB, session helpers, and policy checks
- handlers may return a direct string or `{ content = "...", is_error = bool? }` after performing Lua-side work, or return nested calls for Turin to execute
- `follow_up = "if_no_response"` avoids another provider request when the assistant already streamed a response alongside the tool call; tool-only responses still continue
- `tool.call(...)` and `tool.sequence(...)` accept an optional callback that receives structured nested results after execution
- callbacks may return final content or another `tool.call(...)` / `tool.sequence(...)` follow-up plan
- handlers still do not await nested tool results inline; result callbacks run after Turin finishes the nested native tool execution
- `on_tool_call` / `on_tool_result` governance still applies to the outer virtual tool and to the nested native calls it dispatches
- declaration order does not matter; virtual tool names are resolved after harness load completes
- virtual tools can call other virtual tools
- Turin rejects recursive virtual-tool chains and enforces a max virtual nesting depth of `8`

Tools can be selected for each inference without changing their load-time declarations:

```lua
tool.declare("set_conversation_title", {
  description = "Set a concise title for the current conversation",
  follow_up = "if_no_response",
  params = {
    title = { type = "string", required = true, maxLength = 80 },
  },
  handler = function(args)
    agent.session.set_title(args.title, { if_empty = true })
    return "Conversation title recorded"
  end,
})

function on_turn_prepare(turn)
  turn.tools:exclude("set_conversation_title")

  if turn.session.is_first_user_message
    and turn.session.title == nil
  then
    turn.tools:include("set_conversation_title")
    turn.system_prompt = turn.system_prompt .. [[

Choose a concise title for this conversation. While answering the user
normally, also call set_conversation_title exactly once.]]
  end

  return ALLOW
end
```

`turn.tools:only(...)` starts from a narrow set, `include(...)` adds tools,
`exclude(...)` removes tools, and `all()` restores every tool available to the
turn. This reduces provider context and tool-selection ambiguity; it does not
replace tool policy, governance, or `on_tool_call` validation.

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
function on_turn_prepare(turn)
  if not allowed("db.exec") then
    return verdict.reject("No DB exec capability")
  end

  needs("agent.submit")

  local d = access.check("policy.set")
  if not d.allowed then
    log("policy mutation denied: " .. tostring(d.reason))
  end

  return ALLOW
end
```

`needs(...)` raises a Lua error on denial. Use it when denial should abort the current path immediately.

Helper-layer capability names may omit the `runtime.` prefix. Examples:

- `allowed("db.exec")`
- `needs("agent.submit")`
- `access.check("policy.set")`

File capabilities remain `fs.read` / `fs.write`.

### Session and user helpers

```lua
function on_session_start(session)
  session.remember("User prefers concise answers")
  user.set("timezone", "UTC")
  session.incr("session_starts")
  return ALLOW
end
```

### Top-level shortcuts

```lua
function on_turn_prepare(turn)
  remember("Build failures should stay concise")

  local spec = try(fs.read, "SPEC.md")
  local rows = code.find("capability decision")

  if spec and rows and #rows > 0 then
    session.set("last_code_hit", rows[1].path)
  end

  return ALLOW
end
```

Notes:

- `remember(...)` / `recall(...)` are intent wrappers over default agent memory
- `code.find(...)` defaults to the workspace root and delegates to hybrid code search

### Fluent DB access

```lua
function on_turn_prepare(turn)
  runtime.db.with(".turin/runtime/harness.db", function(db)
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
- Kernel alias `state` rejects DDL. Put harness tables in a path database.

### Fluent peer-agent access

```lua
function on_turn_prepare(turn)
  local reviewer = runtime.agent("reviewer")
  local summary = reviewer:ask("Summarize the diff in 3 bullets")
  session.set("review_summary", summary)
  return ALLOW
end
```

Canonical equivalent:

```lua
function on_turn_prepare(turn)
  local summary = runtime.agent.ask("reviewer", "Summarize the diff in 3 bullets")
  session.set("review_summary", summary)
  return ALLOW
end
```

### Durable scheduling

```lua
function on_turn_prepare(turn)
  local nightly = schedule.every(3600, "Review the workspace and continue useful work", {
    overlap = "skip",
    state = "ops",
    work_key = "project:alpha:heartbeat",
    max_concurrency = 1,
  })

  schedule.after(300, "Follow up on the last failed build")
  schedule.at("2030-01-02T03:04:05Z", "Run the weekly release checklist", {
    recurring = "weekly",
  })
  schedule.at("08:00", "Send the morning digest", {
    recurring = "daily",
  })
  schedule.after(45, {
    action = "agent.disable",
    params = { id = "night-qa" },
  })

  local fetched = schedule.get(nightly.public_id)
  if fetched and fetched.enabled then
    local runs = schedule.runs(fetched.public_id, { limit = 5 })
    session.set("scheduler_recent_runs", runs.runs)
    schedule.update(fetched.public_id, {
      interval_seconds = 7200,
      overlap = "queue",
    })
    session.set("scheduler_job", fetched.public_id)
  end

  return ALLOW
end
```

Notes:

- `schedule.*` is backed by the daemon scheduler, not by ad hoc local state writes
- it only works in daemon-managed runtimes
- recurring jobs currently support overlap policies such as `skip`, `queue`, and `parallel`
- jobs may coordinate across related schedules with:
  - `work_key`
  - `max_concurrency`
- job persistence may be redirected with `state = ...`, `store = ...`, or `persistence = {...}`
- `schedule.after(...)`, `schedule.every(...)`, and `schedule.at(...)` accept either:
  - a bare prompt string
  - or a payload table such as `{ action = "agent.disable", params = { id = "night-qa" } }`
- `schedule.at(...)` accepts:
  - unix milliseconds
  - RFC3339 timestamps such as `"2030-01-02T03:04:05Z"`
  - local-time shorthand such as `"08:00"` or `"08:00:30"`
- `schedule.runs(public_id, opts?)` returns recent attempt history for one job
- `recurring = "daily"` or `recurring = "weekly"` may be combined with `schedule.at(...)`
- built-in scheduled daemon actions currently include:
  - `agent.enable`
  - `agent.disable`
  - `channel.enable`
  - `channel.disable`
  - `worklist.dispatch_next`
  - `worklist.release_stale`
- worklist scheduled actions use the same payload table shape:
  - `{ action = "worklist.dispatch_next", params = { name = "qa", where = { role = "browser" } } }`
  - `{ action = "worklist.release_stale", params = { name = "qa", stale_after_seconds = 600 } }`
- `worklist.dispatch_next`:
  - opens the named worklist in the job's targeted state/store context
  - claims the next eligible root item
  - dispatches prompt items as normal Turin tasks
  - dispatches action items through the same built-in or harness-defined action registry
- `worklist.release_stale`:
  - finds orphaned claimed root items in the named worklist
  - releases them back to pending state
- nested `worklist.*` action items are not recursively dispatched from `worklist.dispatch_next`
- scheduler job detail now also exposes:
  - `last_error_code`
  - `failure_count`
- missing harness-defined actions now surface as:
  - `last_error_code = "schedule_action_missing_handler"`
- custom scheduled action names may also be defined in the harness itself:

```lua
action.define("ops.enqueue_followup", function(this, params)
  runtime.schedule.create({
    prompt = params.prompt or "Follow up",
    after_seconds = params.after_seconds or 30,
  })
  return { status = "queued followup" }
end)
```

- declared action handlers receive a control-aware first parameter; Turin docs/examples conventionally name it `this`
- `this:pause({...})` is the lightweight way to checkpoint progress and continue later without introducing a separate workflow subsystem
- local custom composition points may be expressed with:
- `on("event.name", function(data, meta) ... end)`
- `emit("event.name", payload)`
- `action.run("name", payload)`
- `runtime.on("event.name", function(data, meta) ... end)`
- `runtime.emit("event.name", payload)`

```lua
action.define("bugs.create", function(this, params)
  return { id = "bug-" .. tostring(params.code) }
end)

on("qa.failed", function(failure)
  local created = action.run("bugs.create", failure)
  session.set("last_bug_id", created.id)
end)

function on_turn_prepare(turn)
  emit("qa.failed", { code = "checkout-smoke" })
  return ALLOW
end
```

- `on(...)` listeners are local, synchronous, and additive
- `emit(...)` returns the number of listeners invoked
- `on(...)` and `runtime.on(...)` support exact topics plus terminal wildcard
  topics such as `deploy.*`; `*` catches all topics
- system lifecycle hooks remain the explicit `on_*` functions; custom events are separate
- `runtime.on(...)` / `runtime.emit(...)` are the durable cross-agent variant:
  - handlers are declared at harness load time
  - declared topics are synced into a durable subscription index in daemon-owned `runtime.db`
  - emits create pending deliveries for subscribed agents
  - target agent peer runtimes wake and dispatch those handlers locally
  - cold agents are still routable because subscriber discovery does not depend on them already being loaded in memory
  - success deletes the delivery; failures remain visible for inspection/retry policy
  - durable coordination lives in daemon-owned `runtime.db`, not in harness `state`
  - `runtime.signals.subscribers(topic)` and `runtime.signals.list(opts?)` are the lightweight inspection helpers for that runtime coordination state

Example:

```lua
-- reviewer harness
runtime.on("code.ready", function(ready)
  worklist("reviews"):add({
    title = "Review " .. ready.branch,
    prompt = "Review " .. ready.branch
  })
end)

-- publisher harness
action.define("signals.publish", function(this, params)
  local delivered = runtime.emit("code.ready", {
    branch = params.branch
  })

  return this:complete({
    delivered = delivered
  })
end)
```

- when needed, `runtime.schedule.create(...)` / `update(...)` can also carry structured `content`, tool allowlists, and conflict policy just like `task.submit(...)`

### Durable worklists

For the design model behind worklists, see
`docs/concepts/worklists.md`.

```lua
function on_turn_prepare(turn)
  local sprint = worklist("sprint", {
    scope = "project:alpha",
  })

  local fix = sprint:add({
    title = "Fix login redirect",
    prompt = "Fix login redirect",
    content = {
      { type = "text", text = "Repro starts on /login" },
    },
    tools = {
      allow = { "shell_exec" },
    },
    conflict_policy = "detached",
  }, {
    metadata = { role = "dev" },
  })

  local qa = sprint:add({
    title = "Run checkout smoke test",
    action = "qa.run_smoke",
    params = { suite = "checkout" },
    priority = 10,
    metadata = { role = "qa" },
  })

  local claimed = sprint:next({
    where = { role = "qa" },
  })

  if claimed then
    claimed:done({ result = "queued" })
  end

  local progress = sprint:progress()
  session.set("sprint_progress", progress)
  return ALLOW
end
```

Notes:

- `worklist(name, opts?)` is DX sugar over `runtime.worklist.open(...)`
- worklist items may carry either:
  - prompt payloads
  - named action payloads
- prompt items may also carry structured `content`, `tools`, and `conflict_policy`
- item proxies expose both direct fields and a normalized `payload` view
- action item proxies expose the stored action name as `action_name`; `action` is the contextual action method
- worklists may be partitioned either by:
  - separate lists
  - or `where = {...}` filters on one shared list
- item proxies support nested work with the same helper surface via:
  - `item:add(...)`
  - `item:children()`
  - `item:next()`
- execution helpers are available on top of the shared prompt/action model:
  - `item:dispatch()`
  - `list:dispatch_next(opts?)`
- `item:dispatch()`:
  - enqueues prompt items onto the active session queue as normal Turin tasks
  - invokes action items through the declared action registry
- dispatch does not auto-complete the item; the harness still decides whether to call:
  - `item:done(...)`
  - `item:fail(...)`
- paused work can be inspected directly:
  - `list:paused()`
  - `list:paused({ due_only = true })`
  - `list:find({ where = { paused = true, pause_reason = "..." } })`
- `this:pause(...)` stores checkpoint metadata and moves the current work item into primary `paused` state; `item:requeue()` is the explicit path back to ordinary `pending`
- claim recovery helpers are available when an execution disappears without releasing work:
  - `list:orphaned({ stale_after_seconds = ... })`
  - `list:release_stale({ stale_after_seconds = ... })`
  - `item:heartbeat()`
- worklists are state-store backed, not daemon-scheduler backed:
  - `scope = ...`
  - `state = ...`
  - `store = ...`
  - `path = ...`
  all affect where durable worklist state lives

Reference-aware proxies can move through payload boundaries:

```lua
action.define_on("project", "review", function(this, project, params)
  local reviews = worklist("reviews", { scope = project })
  return reviews:add({
    title = "Review " .. tostring(params.branch),
    action = "review.branch",
    params = {
      project = ref(project),
      branch = params.branch,
    },
  })
end)

function on_turn_prepare(turn)
  local project = scope("project", "alpha")
  project:review({ branch = "feature-x" })
  return ALLOW
end
```

Passing a proxy directly sends `_ref` plus serializable public fields. `ref(proxy)`
sends only `_ref`, so receivers hydrate the current stored state.

### Harness UI surfaces

Harness UI intent lets a harness describe an app-like operator surface without
binding itself to `egui`, Ratatui, DOM, CSS, or any other renderer. The runtime
records semantic intent. Each client decides how to render, degrade, or ignore
that intent according to its capabilities.

Start with one app object and keep the first surface small:

```lua
local app = ui.app("Release Operator", {
  id = "release",
  about = "Coordinate release checks and approvals",
})

action.define("release.seed_demo_work", function(_ctx, params)
  return {
    status = "seeded",
    release = params.release,
    count = params.count,
  }
end)

app:home("Release Desk", function(screen)
  screen:worklist("Pending Approvals", {
    id = "pending-approvals",
    from = "release",
    where = { kind = "approval", status = "pending" },
    fields = { "title", "priority", "status" },
    intent = "approval",
    as = "table",
  })

  screen:form("Create Demo Work", {
    id = "seed-demo-form",
    action = "release.seed_demo_work",
    fields = {
      { name = "release", label = "Release", type = "text", default = "2026.06" },
      { name = "count", label = "Count", type = "integer", default = 3 },
    },
  })
end)

app:menu("Main", function(menu)
  menu:item("Release Desk", "home")
end)
```

Useful rules:

- `screen:list(...)` is the generic collection primitive; `screen:worklist(...)`
  is DX sugar for worklist-backed lists.
- `screen:worklist(...)` accepts either a bare worklist name such as
  `from = "release"` or an explicit semantic source such as
  `from = "worklists.release"`; both resolve to `worklists.release`.
- `intent` and `as` are advisory rendering hints. A desktop client may render
  cards or sheets, while the TUI may render the same data as compact tables.
- UI clients own local view state such as selected screen, selected row, form
  drafts, open panes, and filters. Persist durable workflow state through
  runtime primitives such as worklists, KV, memory, events, schedules, or
  actions.
- Dynamic methods such as `app:notice(...)`, `app:open(...)`, `app:show(...)`,
  `app:focus(...)`, `app:badge(...)`, and `app:refresh(...)` are client-facing
  suggestions, not runtime-owned screen state.
- Top-level dynamic helpers such as `ui.notice(...)` use the first declared
  `ui.app(...)` as the default app and fail clearly if no app has been declared.
  Prefer app-scoped calls when a harness declares multiple apps.
- Panes are useful for contextual surfaces that should not replace the active
  screen. Current clients render shown panes as local overlays. TUI panes
  support pane-local item, action, and form selection, but pane state remains
  terminal-local and separate from screen focus.
- Start with one screen, one useful list, and one action or form. Add reports,
  charts, activity, detail panes, and badges only when the workflow needs them.

The complete UI contract remains covered by an internal test fixture without
presenting that fixture as a ready-to-use workflow.

### Grant wrapper

```lua
function on_turn_prepare(turn)
  local result = runtime.governance.grant({
    ttl_ms = 5000,
    capabilities = {
      ["agent.submit"] = true,
      ["agent.await"] = true,
    }
  }, function()
    return runtime.agent("reviewer"):ask("Review this patch")
  end)

  session.set("grant_review", result)
  return ALLOW
end
```

### Time and JSON helpers

```lua
function on_turn_prepare(turn)
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

## Composition (`import`, `use`, `watch`)

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

### `use(name, opts?)`

Mount a behavior block during harness load.

```lua
use("blocks/harden_shell_execution")
use("blocks/track_user_manners", {
  when = function(hook, payload)
    return hook == "on_turn_prepare"
  end
})
```

Use this when the block’s job is to contribute hook behavior rather than return functions you call manually.

Supported block styles:

- script-style hooks:

```lua
function on_tool_call(call)
  ...
end
```

- returned-table hooks:

```lua
return {
  on_tool_call = function(call)
    ...
  end
}
```

### `use_scoped(name, opts?)`

Scoped behavior mount with the same governance root/delegation rules as `import_scoped(...)`.

```lua
use_scoped("plugins/harden_shell_execution", {
  root = "plugins_writable",
  capabilities = {
    ["runtime.db.query"] = true
  }
})
```

### `watch(path)`

Register extra harness-relative paths for hot reload.

```lua
watch("blocks")
watch("plugins")
```

Notes:

- `use(...)`, `use_scoped(...)`, and `watch(...)` are load-time only.
- Turin still watches the top-level harness directory by default.
- nested directories are only hot-reloaded if you explicitly watch them.

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
local function select_inference_context(turn)
  local prompt = string.lower(turn.prompt or "")

  if turn.token_limit > 0 and (turn.token_count / turn.token_limit) > 0.85 then
    return "fast"
  end

  if runtime.inference.available("reasoning")
    and (prompt:find("debug", 1, true) or prompt:find("trace", 1, true))
  then
    return "reasoning"
  end

  return "default"
end

function on_turn_prepare(turn)
  if turn.is_first_turn_in_task then
    turn.system_prompt = turn.system_prompt .. "\n\nAlways explain your plan before editing files."
  end

  local selected = select_inference_context(turn)
  if selected ~= "default" then
    turn.inference = selected
  end

  -- Structured preflight classification
  local route = turn:structured({
    prompt = turn.prompt or "",
    inference = "fast",
    name = "routing_decision",
    schema = {
      type = "object",
      properties = {
        inference = { type = "string", enum = { "default", "fast", "reasoning" } },
      },
      required = { "inference" },
      additionalProperties = false,
    },
  })

  if route.inference ~= "default" then
    turn.inference = route.inference
  end

  -- Request transport tuning for this turn
  turn.request_options = {
    headers = { ["x-run-purpose"] = "interactive" },
    request_timeout_seconds = 45,
    total_timeout_seconds = 90,
  }

  return ALLOW
end
```

Note: `turn.inference`, `turn.provider`, and other mutable fields are part of the `ContextWrapper` contract. `turn.model` is currently readable but not writable. `turn:structured(...)` is opt-in and does not change normal plain-text turn behavior unless you call it. See `docs/reference/hooks.md` for exact semantics.

If the latest user message includes attachments, inspect `turn.messages` directly. `turn.prompt` remains text-only and preserves image/file parts when you rewrite it.

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

Prefer the helper layer first in new harnesses, even though the canonical `runtime.*` substrate remains available.
When in doubt:

- use DX helpers for readability
- use `runtime.*` directly when you need exact primitive control or want the substrate shape explicitly

### Context selectors

```lua
local project = scope("agent", "coder", {
  namespace = "project_memory",
  visibility = "private",
})
```

Canonical equivalent:

```lua
local project = runtime.context("agent", "coder", {
  namespace = "project_memory",
  visibility = "private",
})
```

### Memory and KV

```lua
local project = scope("agent", "coder", {
  namespace = "project_memory",
  visibility = "private",
})

project:set("last_error", "E0425")

local hits = project:recall("compiler error parsing", { limit = 5 })
for _, row in ipairs(hits or {}) do
  log((row.score or 0) .. " " .. (row.content or ""))
end
```

Named stores and multi-source search:

```lua
local project = runtime.context("project", "rust")
local shared = runtime.context("global")

runtime.memory.store(
  "Borrow checker note from a shared KB",
  shared,
  { layer = "global" },
  { storage = "lexical_only", store = "rust_kb" }
)

local hits = runtime.memory.search("borrow checker", project, {
  include_metadata = true,
  sources = {
    { scope_kind = "project", scope_key = "rust" }, -- resolves through placements or state
    { store = "rust_kb", scope_kind = "global" },   -- explicit named store
  }
})
```

### Cache and code search

Build the index before querying it:

```bash
turin-map index
turin-map status
```

For a small local embedding model behind an OpenAI-compatible endpoint:

```toml
[providers.local_embeddings]
type = "openai"
base_url = "http://127.0.0.1:11434/v1"

[embeddings]
provider = "local_embeddings"
model = "your-small-embedding-model"
dimensions = 384
```

```bash
turin-map index
turin-map status
```

`turin-map` automatically reuses `./.turin/config.toml` when run from a Turin project root. Use `turin-map index --config path/to/.turin/config.toml` if the config lives elsewhere, and use explicit `--embedding-*` flags only when you want to override the configured profile for one run.

The quick success check is simple: `turin-map status` should report `Semantic: enabled (...)`. If it still says `disabled`, Turin will stay on lexical-only search until the local endpoint, model, and dimensions line up.

Then query it from the harness:

```lua
local spec = fs.read("SPEC.md")

local status = runtime.code.search.status(".")
if status.semantic and status.semantic.vector_format then
  log("semantic vectors: " .. status.semantic.vector_format)
end

local rows = runtime.code.search.hybrid(".", "capability decision", {
  languages = { "rust" },
  trace = true,
  strict = false,
})
if rows[1] and rows[1].trace then
  log("effective mode: " .. rows[1].trace.effective_mode)
end
```

Semantic/hybrid queries need both a semantic index and a configured embedding provider at query time.
The runtime embedding profile and index embedding profile must match on driver, base URL, model, and dimensions.
With `strict = false`, Turin falls back to lexical results when that path is unavailable.

### Multi-DB access

```lua
local handle = runtime.db.open({ path = "scratch/analysis.db" })

local changed = runtime.db.exec(
  "create table if not exists notes (id integer primary key, text text)",
  nil,
  { handle = handle.handle }
)

local rows = runtime.db.query(
  "select * from notes where id > :min_id",
  { min_id = 0 },
  { handle = handle.handle }
)
```

### Sparse graph selected paths

Use `graph.*` when a harness needs to record opt-in semantic relationships between graph nodes, branch heads, turns, or external references without dropping straight into the storage-shaped substrate. Ordinary sessions do not create graph rows by default.

```lua
local experiment = graph.new("experiment", "compare candidates")

local branch = agent.session.branch_create("candidate-a", {
  from_turn_index = 0,
})

experiment:add(graph.branch(branch), { role = "candidate" })

local target = experiment:newest("candidate")

agent.sidestep("Analyze this candidate path", {
  target = target,
})
```

The canonical substrate remains available when a harness wants exact control. `runtime.graph.path.select(...)` materializes graph edges that target `branch_head` or `turn` refs into an execution context target:

```lua
local group = runtime.graph.node.create({
  kind = "experiment",
  label = "compare candidates",
})

local branch = agent.session.branch_create("candidate-a", {
  from_turn_index = 0,
})

runtime.graph.edge.create({
  source = { kind = "graph_node", id = group.node_id },
  target = { kind = "branch_head", id = branch.branch_id },
  relation_kind = "contains",
  target_role = "candidate",
})

local target = runtime.graph.path.select({
  source = { kind = "graph_node", id = group.node_id },
  relation_kind = "contains",
  target_kind = "branch_head",
  target_role = "candidate",
  order = "newest_first",
  limit = 1,
})

agent.sidestep("Analyze this candidate path", {
  target = target,
})
```

If the harness already knows the exact sequence it wants, it can skip the graph-edge lookup and materialize an explicit ordered path directly:

```lua
local target = runtime.graph.path.select({
  refs = {
    { kind = "turn", id = "12" },
    { kind = "branch_head", id = branch.branch_id },
  },
})
```

### Peer-agent orchestration

```lua
local reviewer = runtime.agent("reviewer")

local task_id = reviewer:submit({
  prompt = "Review the proposed patch and list regressions",
  title = "regression review",
}, {
  capabilities = {
    ["db.query"] = true,
    ["db.exec"] = false,
    ["fs.read"] = true,
    ["fs.write"] = false,
  }
})

local result = reviewer:await(task_id, { timeout_ms = 30000 })
log(json.encode(result))
```

## Using Top-Level Aliases (Ergonomic)

Turin keeps ergonomic aliases for common workflows.

If you are starting from zero, use `turin init`, `turin quickstart`, `turin harness new`, and
`turin harness test` first, then come back here for the deeper patterns.
The fast bootstrap flow lives in `docs/getting-started/harness-cookbook.md`.

### Agent-scoped defaults

```lua
kv.set("task_state", "working")
local rows = memory.search("build failure")
local file = fs.read("README.md")
local hits = code.find("grant validation")
remember("Release notes should stay terse")
```

If you omit `store`, Turin resolves memory/KV placement through `[persistence.placements]`
before falling back to the primary `state` DB.

### Session/User scoped aliases

```lua
session.kv.set("step", "planning")
local profile = user.kv.get("profile")
```

These aliases rely on the active runtime identity; `user.*` requires `identity.user_id`.

## Governance-Aware Harness Design

Turin is flexibility-first, but you can design harnesses to cooperate with governance cleanly.

### 1. Prefer capability checks for feature toggles

```lua
local decision = access.check("db.exec")
if decision.allowed == false then
  return REJECT, "db writes not allowed in this profile"
end
```

### 2. Use temporary grants for explicit elevation

```lua
runtime.governance.grant({
  capabilities = { ["db.exec"] = true },
  ttl_ms = 15000,
  max_uses = 1,
  reason = "one-shot migration",
}, function()
  runtime.db.exec("delete from temp_rows where stale = 1")
end)
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

## Use `try(...)` when recovery is intentional

Public harness APIs raise on actual failure.
Use `try(...)` or `pcall(...)` only when the harness wants to recover explicitly.

`try(fn, ...)` returns `value, err`. It is useful when a missing file, unavailable
semantic index, denied optional capability, or transient database issue should not
abort the current turn.

```lua
local spec, err = try(fs.read, "SPEC.md")
if not spec then
  log("SPEC.md unavailable; continuing without it: " .. tostring(err))
end

local rows, search_err = try(runtime.code.search.hybrid, ".", "capability decision", {
  strict = false,
})
if not rows then
  log("code search unavailable: " .. tostring(search_err))
  rows = {}
end
```

## Prefer canonical APIs in new code

Aliases are great for ergonomics, but `runtime.*` is clearer and more future-proof for shared harness libraries.

## Debugging Harnesses

- Use `log("...")` for harness-level diagnostics
- Use `turin harness test` for mock-backed iteration and `turin daemon harness reload <id>` after editing a daemon-managed harness
- Use `on_kernel_event` for temporary deep observability
- Run `turin check` to validate config + harness syntax
- Use `scripts/live_minimax_smoke.sh` (manual/opt-in) for real endpoint testing

## Compatibility Note

Older Turin docs/examples used `turin.*` and `db.*` namespaces.
The canonical API is now `runtime.*` plus the aliases documented in `docs/reference/primitives.md`.
