# Harness Primitives / Standard Library

This document describes the current harness globals available in Turin’s Luau runtime.
This is the **canonical API surface reference** for harness authors (including both
the preferred `runtime.*` namespaces and the ergonomic top-level aliases).

## Result Convention (Important)

Harness-facing functions now use one public contract:

- success returns the useful value directly
- actual failure raises a Lua runtime error
- valid empty collections return empty collections
- valid missing singular values return `nil`
- use `try(fn, ...)` or `pcall(...)` when recovery is intentional

Examples:

```lua
local text = fs.read("README.md")

runtime.kv.set("foo", "bar", runtime.context("agent", "coder"))
```

Recovery example:

```lua
local rows, err = try(runtime.db.query, "select * from notes")
if not rows then
  log("query failed: " .. tostring(err))
end
```

Reach for `try(...)` when the harness can intentionally continue after a failed
call. For example, optional files, optional code indexes, or best-effort telemetry
should usually be wrapped; required state changes should usually use direct calls
so failures abort loudly.

## Verdict Constants (for hooks)

- `ALLOW`
- `REJECT`
- `ESCALATE`
- `MODIFY`

## DX Language Layer

Turin also ships a first-party DX layer in `src/harness/dx/`.

Important:

- `runtime.*` remains the canonical primitive layer
- DX helpers are ergonomic wrappers over those primitives
- DX helpers do not bypass governance/capability enforcement
- DX helpers follow the same public contract and raise on actual failure

Use the DX layer when you want cleaner harness code. Use `runtime.*` directly when you want the most explicit primitive behavior.

### `verdict`

- `verdict.allow() -> verdict_table`
- `verdict.reject(reason?) -> verdict_table`
- `verdict.escalate(reason?) -> verdict_table`
- `verdict.modify(value) -> verdict_table`
- `verdict.reject_if(condition, reason) -> verdict_table|nil`
- `verdict.escalate_if(condition, reason) -> verdict_table|nil`

Example:

```lua
return verdict.reject_if(call.name == "shell_exec", "shell disabled")
  or verdict.allow()
```

### Access helpers

- `allowed(capability, opts?) -> boolean`
- `needs(capability, opts?) -> true | error`
- `access.check(capability, opts?) -> decision_table`

`opts`:

```lua
{ agent_id = "reviewer" }
```

Notes:

- `allowed(...)` is the top-level boolean predicate
- `needs(...)` raises on denial; it does not return `(false, err)`
- `access.check(...)` returns the same decision shape as `runtime.governance.check(...)`
- helper-layer capability checks accept short forms such as:
  - `allowed("db.exec")`
  - `needs("agent.submit")`
  - `access.check("policy.set")`
- unqualified helper capability roots default to `runtime.`
- non-runtime capability roots such as `fs.read` and `fs.write` remain unchanged

### `session` / `user` DX helpers

These are layered on top of the existing scoped aliases.

- `session.remember(content, metadata?)`
- `session.recall(query, opts?)`
- `session.get(key)`
- `session.set(key, value)`
- `session.del(key)`
- `session.incr(key, by?) -> integer`

- `user.remember(content, metadata?)`
- `user.recall(query, opts?)`
- `user.get(key)`
- `user.set(key, value)`
- `user.del(key)`
- `user.incr(key, by?) -> integer`

`incr` behavior:

- missing key is treated as `0`
- stored integer strings are parsed
- return value is the new integer count
- malformed stored values raise
- overflow raises

### Top-level DX shortcuts

- `remember(content, metadata?, opts?)`
- `recall(query, opts?)`
- `scope(kind, key, opts?) -> scope_proxy`
- `graph.new(kind, label?, opts?) -> graph_node_proxy`
- `graph.node(id_or_row) -> graph_node_proxy`
- `graph.branch(id_or_row) -> graph_ref`
- `graph.turn(id_or_row) -> graph_ref`
- `graph.ref(kind, id_or_row) -> graph_ref`
- `action.define(name, fn) -> nil`
- `action.run(name, params?) -> value|nil`
- `on(name, fn) -> nil`
- `emit(name, payload?) -> integer`
- `runtime.on(name, fn) -> nil`
- `runtime.emit(name, payload?) -> integer`
- `runtime.signals.subscribers(topic) -> {agent_id...}`
- `runtime.signals.list(opts?) -> {signal_row...}`
- `schedule.after(seconds, payload, opts?) -> job`
- `schedule.every(seconds, payload, opts?) -> job`
- `schedule.at(timestamp, payload, opts?) -> job`
- `schedule.update(public_id, opts?) -> job`
- `schedule.get(public_id) -> job`
- `schedule.list(opts?) -> jobs`
- `schedule.runs(public_id, opts?) -> run_list`
- `schedule.enable(public_id) -> job`
- `schedule.disable(public_id) -> job`
- `schedule.delete(public_id) -> job`
- `worklist(name, opts?) -> list_proxy`
- `code.find(query, opts?)`
- `ui.app(title, opts?) -> app_proxy`
- `ui.notice(title, opts?)`
- `ui.open(target, opts?)`
- `ui.show(target, opts?)`
- `ui.badge(target, opts?)`
- `ui.focus(target)`
- `ui.refresh(binding)`

Notes:

- `remember(...)` / `recall(...)` use the same default agent-scoped memory as `memory.*`
- `code.find(...)` is a thin wrapper over `runtime.code.search.hybrid(...)`
- `code.find(...)` defaults to the Turin workspace root and accepts `opts.root` / `opts.index_path` overrides
- `ui.*` is the experimental UI Intent v0 surface; it records semantic UI intent for Turin clients rather than renderer-specific instructions
- `action.define(...)` and `on(...)` are load-time only registration helpers
- `action.run(...)` invokes a built-in or harness-defined action immediately in the current execution context
- `emit(...)` performs synchronous in-process listener dispatch and returns the number of listeners invoked
- custom events are local additive signals; they are distinct from system `on_*` hooks and from durable cross-agent signaling
- `on(...)` listeners receive domain data first and optional event metadata second
- `runtime.on(...)` is load-time only and registers cross-agent signal handlers for the current harness
- `runtime.on(...)` listeners receive domain data first and optional delivery metadata second
- `runtime.on(...)` also contributes the current harness topics to the durable subscription index in daemon-owned `runtime.db`
- `runtime.emit(...)` creates durable pending signal deliveries for subscribed agents, wakes their peer runtimes, and returns the number of target agents
- `runtime.emit(...)` requires daemon/runtime coordination to be available; it is not a purely local harness primitive
- `runtime.signals.subscribers(...)` lists the current durable subscribers for a topic from `runtime.db`
- `runtime.signals.list(...)` lists pending cross-agent signal rows from `runtime.db`
- cold agents remain routable because subscriber lookup comes from the durable topic index rather than only from currently loaded runtimes
- there is still no durable handler registry; once the target harness loads, its local `runtime.on(...)` handlers determine which functions run
- delivered cross-agent signals are deleted on success; failed deliveries remain visible with attempt metadata rather than becoming a replay log
- cross-agent signals are agent-scoped and asynchronous relative to the emitter; they are distinct from local `emit(...)`

Example:

```lua
remember("Compiler errors should stay concise")
local file = fs.read("SPEC.md")
local rows = code.find("capability decision")
local project = scope("project", "my-app", { namespace = "notes" })
```

### UI Intent v0

The `ui` namespace lets harnesses describe semantic UI intent for Turin clients.
Clients decide how to react based on their own capabilities. The API is
experimental and expected to evolve while the TUI and graphical app are built in
parallel.

Load-time example:

```lua
local app = ui.app("Release Operator", {
  id = "release",
  about = "Coordinate release checks",
})

app:home("Release Desk", function(screen)
  screen:list("Open Work", {
    id = "open-work",
    from = "worklists.release",
    where = { kind = "approval" },
    intent = "approval",
    as = "table",
  })

  screen:section("Controls", function(section)
    section:action("Run Smoke Tests", "qa.run_smoke", {
      id = "run-smoke",
      params = { suite = "smoke" },
    })
  end)

  screen:activity("Release Activity", {
    from = "signals.release",
  })
end)

app:menu("Main", function(menu)
  menu:item("Dashboard", "home")
end)
```

Dynamic intent example:

```lua
local app = ui.app("Release Operator", { id = "release" })

function on_turn_prepare(_ctx)
  app:notice("Release blocked", {
    body = "QA failed",
    level = "warning",
  })

  app:open("home")
  app:badge("home", { count = 1, level = "warning" })
  app:focus("open-work")
  app:refresh("worklists.release")

  return ALLOW
end
```

`app_proxy` structure methods:

- `app:home(title, fn, opts?)`
- `app:screen(id, title, fn, opts?)`
- `app:opens_with(screen_id)`
- `app:menu(title, fn)`
- `app:pane(id, title, fn, opts?)`

`app_proxy` dynamic intent methods:

- `app:notice(title, opts?)`
- `app:open(target, opts?)`
- `app:show(target, opts?)`
- `app:badge(target, opts?)`
- `app:focus(target)`
- `app:refresh(binding)`

`screen` / `section` / `pane` methods:

- `screen:list(title, opts?)`
- `screen:worklist(title, opts?)`
- `screen:section(title, fn)`
- `screen:text(text, opts?)`
- `screen:action(label, action, opts?)`
- `screen:activity(title, opts?)`
- `screen:detail(title, opts?)`
- `screen:form(title, opts)`
- `screen:report(title, opts?)`
- `screen:chart(title, opts?)`

Current v0 behavior:

- `ui.app(...)` returns an app object and records app intent
- `app:home(...)` is sugar for a `home` screen plus default screen intent
- `screen:list(...)` is the generic collection primitive
- `screen:worklist(...)` is DX sugar over `screen:list(...)` with `intent = "tasks"`
- `intent` and `as` fields are advisory signals; clients may degrade or ignore them
- load-time app, screen, pane, and menu intents are exposed on harness list/detail snapshots
- hook-time app-scoped methods such as `app:notice(...)` and `app:open(...)` emit ephemeral `ui.intent` session events
- dynamic UI intent events are client-facing signals; they are not persisted into session history
- actions reference named Turin actions; durable side effects should stay in actions/runtime primitives
- clients are responsible for rendering, degrading, or ignoring UI intent according to their capabilities

`scope_proxy` methods:

- `scope_proxy.remember(content, metadata?)`
- `scope_proxy.recall(query, opts?)`
- `scope_proxy.get(key)`
- `scope_proxy.set(key, value)`
- `scope_proxy.del(key)`
- `scope_proxy.incr(key, by?) -> integer`

Notes:

- `scope(...)` is DX sugar over:
  - `runtime.context(...)`
  - `memory.as(ctx)`
  - `kv.as(ctx)`
- it does not invent new scope semantics
- `scope("global")` is valid and targets the shared global selector
- non-global scopes require an explicit key

### DX `graph`

- `graph.new(kind, label?, opts?) -> graph_node_proxy`
- `graph.node(id_or_row) -> graph_node_proxy`
- `graph.branch(id_or_row) -> graph_ref`
- `graph.turn(id_or_row) -> graph_ref`
- `graph.ref(kind, id_or_row) -> graph_ref`

`graph_node_proxy` methods:

- `node:add(target, role_or_opts?) -> edge`
- `node:link(target, relation, opts?) -> edge`
- `node:find(role_or_opts?) -> target`
- `node:newest(role_or_opts?) -> target`
- `node:oldest(role_or_opts?) -> target`
- `node:all(role_or_opts?) -> target`

Notes:

- this is helper-layer sugar over:
  - `runtime.graph.node.create(...)`
  - `runtime.graph.edge.create(...)`
  - `runtime.graph.path.select(...)`
- `graph.node(...)` returns a richer node proxy, not just a bare ref table
- `graph.branch(...)` accepts either a `branch_id` string or a branch row with `branch.branch_id`
- `graph.turn(...)` accepts either a turn id or a table with `turn_id`
- `graph.ref(...)` is the generic escape hatch for less common reference kinds such as `external_path`
- node path helpers default to:
  - `relation_kind = "contains"`
  - `target_kind = "branch_head"`

Example:

```lua
local experiment = graph.new("experiment", "compare candidates")
local branch = agent.session.branch_create("candidate-a", { from_turn_index = 0 })

experiment:add(graph.branch(branch), { role = "candidate" })

local target = experiment:newest("candidate")
agent.sidestep("Analyze this candidate", {
  target = target,
})
```

### DX `runtime.db(...)`

`runtime.db` is callable in the DX layer:

- `runtime.db(selector) -> db_proxy`
- `runtime.db.with(selector, fn, opts?) -> <fn returns> | error`

`db_proxy` methods:

- `db:one(sql, params?, opts?) -> row|nil`
- `db:all(sql, params?, opts?) -> rows`
- `db:exec(sql, params?, opts?) -> changed_count`
- `db:close()`

Notes:

- `:one(...)` returns the first row or `nil`
- `runtime.db.with(...)` opens a handle, runs the callback, and closes the handle
- if the callback errors, the callback error wins over any close error

### DX `runtime.agent(...)`

`runtime.agent` is callable in the DX layer:

- `runtime.agent(agent_id) -> agent_proxy`

`agent_proxy` methods:

- `agent:submit(task, opts?) -> request_id`
- `agent:await(request_id, opts?) -> result`
- `agent:status() -> status`
- `agent:ask(prompt, opts?) -> output_string`

Notes:

- `:ask(...)` delegates to the canonical `runtime.agent.ask(...)` primitive
- peer-agent governance, child-agent allowlists, delegated capability ceilings, and active grant ceilings still apply

### DX `schedule`

- `schedule.after(seconds, payload, opts?) -> job`
- `schedule.every(seconds, payload, opts?) -> job`
- `schedule.at(timestamp, payload, opts?) -> job`
- `schedule.update(public_id, opts?) -> job`
- `schedule.get(public_id) -> job`
- `schedule.list(opts?) -> jobs`
- `schedule.enable(public_id) -> job`
- `schedule.disable(public_id) -> job`
- `schedule.delete(public_id) -> job`

`opts` for `after` / `every` / `at`:

```lua
{
  agent = "reviewer",
  overlap = "skip", -- or "queue" / "parallel"
  recurring = "daily", -- optional, mainly for schedule.at(...)
  work_key = "project:alpha:qa",
  max_concurrency = 1,
  enabled = true,
  state = "ops",
  store = { path = "./project.db" },
}
```

`payload` for `after` / `every` / `at` may be:

```lua
"Review the latest failed build"
```

or:

```lua
{
  action = "agent.disable",
  params = { id = "night-qa" },
}
```

`opts` for `update`:

```lua
{
  prompt = "Re-run QA sweep",
  after_seconds = 300,
  interval_seconds = 1200,
  overlap = "queue",
  work_key = "project:alpha:qa",
  max_concurrency = 2,
  enabled = true,
  state = "ops",
  store = { path = "./project.db" },
}
```

Notes:

- `schedule.*` is helper-layer sugar over the daemon-backed scheduler
- it requires a daemon-managed runtime and raises outside that environment
- `schedule.at(...)` accepts unix milliseconds, RFC3339 timestamp strings, and local-time shorthand such as `"08:00"`
- `recurring` currently supports:
  - `"daily"`
  - `"weekly"`
- `state = ...` / `store = ...` are shorthand for `persistence = { state = ..., store = ... }`
- `work_key` lets related jobs coordinate in the same concurrency lane
- `max_concurrency` bounds how many prompt jobs may run concurrently in that lane
- built-in action payloads currently support:
  - `agent.enable`
  - `agent.disable`
  - `channel.enable`
  - `channel.disable`
  - `worklist.dispatch_next`
  - `worklist.release_stale`
- worklist scheduled action params may include:
  - `name`
  - `scope`
  - `state`
  - `store`
  - `where`
  - `limit`
  - `stale_after_seconds`
- `worklist.dispatch_next` opens the named worklist, claims the next eligible root item, and dispatches it:
  - prompt items submit a normal Turin task
  - action items execute through the same built-in or harness-defined action registry
- `worklist.release_stale` releases orphaned claimed root items back to pending state
- nested `worklist.*` action items are rejected inside `worklist.dispatch_next`
- custom scheduled action names may be registered during harness load with:

```lua
action.define("qa.run_smoke", function(this, params)
  return { status = "queued " .. tostring(params.suite) }
end)
```

- declared action handlers receive `(control, params)`
- examples in Turin docs now conventionally name that first parameter `this`:
  - `action.define("name", function(this, params) ... end)`
- that action control object currently exposes:
  - `this.params`
  - `this.checkpoint`
  - `this.item`
  - `this:pause(opts)`
  - `this:pause_for(seconds, opts?)`
  - `this:complete(value?)`
  - `this:fail(value?)`
  - `this:cancel(value?)`
  - `this:is_cancelled()`
- `this.checkpoint` is also a helper table:
  - `this.checkpoint:get(key, default?)`
  - `this.checkpoint:all()`
- `this:pause(opts)` is the lightweight continuation helper:
  - persists checkpoint-style metadata on the current work item when available
  - transitions the current work item into primary `paused` state
  - optionally schedules a follow-up when `opts.resume_in_seconds` is provided
  - marks the item as paused so ordinary `worklist.next()` / `dispatch_next()` will skip it until the pause is due
- `this:pause_for(seconds, opts?)` is shorthand for `this:pause(...)` with `resume_in_seconds = seconds`
- `this:is_cancelled()` reflects the current cooperative session cancellation signal when the action is running inside a live task/session
- `opts` for `this:pause(...)` may include:
  - `because`
  - `note`
  - `checkpoint`
  - `resume_in_seconds`

### Reference-aware proxy payloads

Scope, worklist, and work item proxies can be passed through JSON payload
boundaries. The runtime encodes recognized proxies with a reserved `_ref` object
and decodes recognized references back into proxies.

This applies to declared action params/results, local events, durable runtime
signals, schedule create/update payloads, work item params/metadata, and
sidestep graph relation metadata.

```lua
local project = scope("project", "alpha")
local tickets = worklist("tickets", { scope = project })
local item = tickets:add({
  title = "Classify signup bug",
  action = "ticket.classify",
  params = { source = "smoke" },
})

item.priority = 50
action.run("ticket.enrich", { item = item })
runtime.emit("ticket.ready", { item = ref(item) })
```

Rules:

- `ref(proxy)` encodes only the reference identity and requires a reference-aware runtime proxy
- passing a proxy directly encodes `_ref` plus serializable public fields
- the receiver hydrates the reference first and then overlays the sent fields
- plain JSON without `_ref` remains plain JSON
- malformed or unknown `_ref` remains plain data
- recognized `_ref` whose backing object is missing raises an error
- overlay fields may not replace proxy methods
- runtime-internal proxy marker fields are not included in public payloads
- function, thread, userdata, vector, buffer, and error values are not JSON-serializable payload values

Reference-aware proxies also expose contextual action helpers:

```lua
action.define("project.archive", function(this, params)
  local project = params.subject
  project:set("status", "archived")
  return project
end)

local project = scope("project", "alpha")
project:action("archive")
```

`action.define_on(target, method, handler)` defines an action and attaches a
method to matching proxies:

```lua
action.define_on("project", "review", function(this, project, params)
  project:set("review_status", params.status or "queued")
  return project
end)

action.define_on(target.worklist("tickets"), "stats", function(this, list)
  return list:progress()
end)

action.define_on(target.workitem(), "requeue_with_note", function(this, item, params)
  item:update({ metadata = { note = params.note } })
  return item:requeue()
end)
```

Target forms:

- string shorthand: `"project"` means `target.scope("project")`
- `target.scope(name?)`
- `target.worklist(name?)`
- `target.workitem(name?)`

Specific targets override wildcard targets for the same method name. For
example, `target.workitem("tickets")` is more specific than `target.workitem()`.

- `schedule.delete(...)` rejects running jobs rather than orphaning active execution
- `schedule.update(...)` only changes the fields you provide; running attempts continue with the task that was already submitted

### DX `worklist`

For conceptual guidance and workflow examples, see
`docs/concepts/worklists.md`.

- `worklist(name, opts?) -> list_proxy`

`opts` for `worklist(...)`:

```lua
{
  scope = "project:alpha", -- or selector table
  state = "ops",
  store = { path = "./project.db" },
  metadata = { lane = "qa" },
}
```

`payload` for `list:add(...)` / `item:add(...)` may be:

```lua
"Fix login redirect"
```

or:

```lua
{
  title = "Fix login redirect",
  prompt = "Fix login redirect",
  content = {
    { type = "text", text = "Repro starts on /login" },
  },
  tools = {
    allow = { "shell_exec" },
  },
  conflict_policy = "detached",
}
```

or:

```lua
{
  title = "Run checkout smoke test",
  action = "qa.run_smoke",
  params = { suite = "checkout" },
}
```

Common `list_proxy` methods:

- `list:add(payload, opts?) -> item_proxy`
- `list:all(opts?) -> items`
- `list:pending(opts?) -> items`
- `list:paused(opts?) -> items`
- `list:active() -> item|nil`
- `list:next(opts?) -> item|nil`
- `list:current(opts?) -> item|nil`
- `list:find({ where = {...} }) -> item|nil`
- `list:progress() -> { done = n, total = n }`
- `list:empty() -> boolean`
- `list:orphaned(opts?) -> items`
- `list:release_stale(opts?) -> items`
- `list:dispatch_next(opts?) -> { item = item, result = result } | nil`

Common `item_proxy` methods:

- `item:add(payload, opts?) -> item_proxy`
- `item:children() -> items`
- `item:claim() -> item|nil`
- `item:heartbeat() -> item`
- `item:dispatch(opts?) -> result`
- `item:done(meta?) -> item`
- `item:fail(reason?) -> item`
- `item:requeue() -> item`
- `item:update(fields) -> item`

Notes:

- worklist items may be prompt-based or action-based
- prompt items may also carry `content`, `tools`, and `conflict_policy`
- item proxies expose both direct fields and a normalized `payload` table
- item proxies expose pause fields directly:
  - `paused`
  - `pause_reason`
  - `pause_until_unix_ms`
- `list:paused(opts?)` returns paused items; `opts.due_only = true` restricts that view to paused items whose resume window is already due
- `item:requeue()` clears pause markers and moves the item back to ordinary `pending`
- `opts.where` filters compare against built-in item fields, pause fields, and metadata keys
- stale-claim helpers use `opts.stale_after_seconds` and default to 300 seconds
- `item:dispatch(...)`:
  - enqueues prompt items onto the active session queue as normal Turin tasks
  - invokes action items through the declared action registry
- `list:dispatch_next(...)` claims the next eligible item and dispatches it immediately
- dispatch helpers do not auto-complete the item
- worklists are state-store backed, not daemon-scheduler backed

### DX `runtime.governance.grant(...)`

- `runtime.governance.grant(spec, fn) -> <fn returns> | error`

`spec` shape:

```lua
{
  capabilities = { ["db.query"] = true },
  ttl_ms = 10000,
  max_uses = 1,
  reason = "one-shot operation",
}
```

Behavior:

- issues a temporary grant
- runs `fn` under that active grant
- revokes the grant after the callback returns
- if the callback errors, the callback error wins over revoke errors
- helper-layer `capabilities` maps accept short runtime forms such as `db.query` or `agent.submit`
- file capabilities remain `fs.read` / `fs.write`

### DX `time`

- `time.since(ts) -> number`
- `time.after(ts, threshold) -> boolean`

Current semantics:

- `ts` may be a number or numeric string
- values are interpreted as Unix epoch seconds
- `time.since(...)` returns elapsed seconds as a Lua number
- `time.after(...)` compares elapsed seconds against the given threshold in seconds

### DX `fs`

- `fs.summary(path, opts?) -> string`
- `fs.read_json(path, opts?) -> value | error`
- `fs.write_json(path, value, opts?) -> true | error`

`opts` for `summary`:

```lua
{
  prompt = "Summarize only the constraints and obligations.", -- optional
  force = true,                                               -- optional
}
```

`opts` for `write_json`:

```lua
{ pretty = true }
```

Notes:

- `fs.summary(...)` is a high-level file-summary helper built on:
  - `fs.stat(...)`
  - `fs.read(...)`
  - session KV-backed invalidation
  - the active `turn:summarize(...)` capability during `on_turn_prepare(turn)`
- `fs.summary(...)` currently requires an active `on_turn_prepare(turn)` context and raises outside that hook.
- the cache key varies by file hash and summary prompt.
- these wrap `fs.read/write` and `json.decode/encode`
- existing `fs.read` / `fs.write` governance checks still apply

## System Globals

### `tool`

Harness-declared virtual tools.

- `tool.declare(name, spec) -> nil`
- `tool.call(name, args?, callback?) -> call_descriptor`
- `tool.sequence({ call_descriptor, ... }, callback?) -> sequence_descriptor`

`spec` shape:

```lua
{
  description = "Play an audio file with mpg123",
  params = {
    filename = { type = "string", required = true }
  },
  handler = function(args)
    return tool.call("shell_exec", {
      command = "mpg123 " .. shell.quote(args.filename)
    })
  end
}
```

Alternative full-schema form:

```lua
{
  description = "Lookup a structured value",
  input_schema = {
    type = "object",
    properties = {
      query = { type = "string" }
    },
    required = { "query" }
  },
  handler = function(args)
    return tool.call("read_file", { path = args.query })
  end
}
```

Notes:

- `tool.declare(...)` can only be called during harness load
- `params` is normalized into JSON Schema internally
- declared tools are exposed to the model in the normal provider tool list
- handlers execute in the harness VM and return nested tool-call descriptors for Turin to execute afterward
- handlers can use `runtime.*`, DX helpers, memory, KV, DB, and policy checks to decide what to dispatch
- `callback` is optional; for `tool.call(...)` it receives one result object, and for `tool.sequence(...)` it receives an array of result objects
- each result object includes `id`, `name`, `args`, `verdict`, `duration_ms`, `content`, and `is_error`
- callbacks may return:
  - a string
  - `{ content = "...", is_error = bool? }`
  - `tool.call(...)`
  - `tool.sequence(...)`
- handlers do not currently await nested tool results inline; callbacks run after Turin completes the nested native tool execution
- declaration order does not matter; virtual tool names are resolved after harness load completes
- virtual tools may call other virtual tools
- Turin rejects recursive virtual-tool chains and enforces a max virtual nesting depth of `8`

### `shell`

- `shell.quote(text) -> string`

Returns a POSIX-safe single-quoted shell fragment.

Example:

```lua
local cmd = "mpg123 " .. shell.quote(args.filename)
```

## `fs`

Filesystem helpers scoped to `harness.fs_root` (default: workspace root).

- `fs.read(path) -> string`
- `fs.write(path, content) -> bool`
- `fs.stat(path) -> stat_table` (raises on failure)
- `fs.summary(path, opts?) -> string` (raises on failure; only during `on_turn_prepare(turn)`)
- `fs.exists(path) -> bool`
- `fs.is_safe_path(path) -> bool`

Notes:

- `fs.read`/`fs.write` are governance-capability gated (`fs.read`, `fs.write`) when governance enforcement is enabled.
- `fs.stat(...)` currently uses the same `fs.read` capability gate because it reads file contents to compute a content hash.
- `fs.read`, `fs.write`, and `fs.stat` enforce the max harness file size before loading or writing file contents (kernel constant, default 10MB).
- Path traversal outside `harness.fs_root` is denied.

`fs.stat(path)` shape:

```lua
{
  path = "SPEC.md",
  bytes = 18234,
  hash = "...",
  previous_hash = nil or "...",
  seen_before = true or false,
  changed = true or false,
  modified_at = nil or 1714377000,
}
```

Notes:

- `hash` is a SHA-256 hash of the file's current bytes.
- `previous_hash`, `seen_before`, and `changed` are tracked per session.
- on first observation in a session, `seen_before = false` and `changed = true`.

Example:

```lua
local spec = fs.stat("SPEC.md")
if spec.changed then
  log("SPEC.md changed in this session")
end
```

`fs.summary(path, opts?)`:

```lua
local spec = fs.summary("SPEC.md")
local constraints = fs.summary("CONSTRAINTS.md", {
  prompt = "Summarize only the actionable constraints.",
})
```

Notes:

- `fs.summary(...)` caches summaries per session based on:
  - normalized file path
  - current file hash
  - summary prompt variant
- `opts.force = true` bypasses the cached summary and refreshes it.
- this helper currently depends on the active `on_turn_prepare(turn)` hook context because it reuses `turn:summarize(...)` underneath.

## `hash`

- `hash.sha256(text) -> string`

Returns the lowercase hexadecimal SHA-256 digest for the given string.

Example:

```lua
local digest = hash.sha256("hello")
```

## `json`

- `json.encode(value) -> string`
- `json.decode(string) -> value`

## `time`

- `time.epoch_seconds() -> integer`
- `time.now_utc() -> string` (Unix timestamp string, not ISO8601)

## `log`

- `log(message)`

Writes a harness-prefixed diagnostic line to stderr.

## `import(name)`, `import_scoped(name, opts?)`, `use(name, opts?)`, `use_scoped(name, opts?)`

Harness module import helpers.

- `import(name)` — unscoped module import (governance may restrict this)
- `import_scoped(name, opts)` — scoped module import with governance root assertion and optional delegated capability ceiling
- `use(name, opts?)` — mount a behavior block during harness load
- `use_scoped(name, opts?)` — mount a behavior block with governance root assertion and optional delegated capability ceiling

`opts` (for `import_scoped` / `use_scoped`, and optionally `use` where relevant):

```lua
{
  root = "core",            -- optional if governance.import.default_root is configured
  capabilities = {           -- optional delegated ceiling (downward-only)
    ["runtime.db.query"] = true,
    ["runtime.db.exec"] = false,
    ["runtime.db.*"] = true,
  },
  config = {                 -- `use(...)` only; exposed to the block as `block.config`
    strict = true
  },
  when = function(hook, payload) -- `use(...)` only; runtime gate, not dynamic registration
    return hook == "on_turn_prepare"
  end
}
```

Notes:

- Imported module functions are wrapped so governance checks run under the imported module/root subject context.
- Nested exported tables/functions are recursively wrapped.
- Nested imports cannot widen delegated capabilities beyond the importer’s delegation.
- `use(...)` and `use_scoped(...)` are load-time only; calling them from a hook is a runtime error.
- `use(...)` accepts either:
  - script-style hook blocks (`function on_turn_prepare(...) ... end`)
  - returned-table hook blocks (`return { on_turn_prepare = function(...) ... end }`)
- `use(...)` activates the block in the normal hook pipeline; `import(...)` stays side-effect free.

## `watch(path)`

Registers an extra harness-relative path for hot reload.

```lua
watch("blocks")
watch("plugins")
```

Notes:

- `watch(...)` is load-time only.
- Turin still watches the top-level harness directory by default.
- watched subpaths are explicit; nested trees are not watched unless you register them.

## Canonical Runtime API (`runtime.*`)

`runtime` is the preferred forward-looking namespace for harness code.

## `runtime.context`

### Callable selector builder

`runtime.context(...)` returns a normalized context selector table.

Supported signatures:

- `runtime.context({ tags = {...}, namespace = "...", visibility = "..." })`
- `runtime.context(scope, id?, opts?)`

Examples:

```lua
local ctx1 = runtime.context({
  tags = { "agent:coder", "tenant:acme" },
  namespace = "default",
  visibility = "private",
})

local ctx2 = runtime.context("agent", "coder")
local ctx3 = runtime.context("session", "sess_123", { namespace = "scratch" })
local ctx4 = runtime.context("global", nil, { visibility = "shared" })
```

Selector shape:

```lua
{
  tags = { "dimension:value", ... },
  namespace = "default",
  visibility = "private",
}
```

### `runtime.context.glob(pattern)`

Returns matching store aliases (wildcard `*` supported).

```lua
local aliases = runtime.context.glob("agent:*")
```

## `runtime.memory`

Canonical memory API with explicit selector.

- `runtime.memory.search(query, ctx, opts?)`
- `runtime.memory.store(content, ctx, metadata?, opts?)`
- `runtime.memory.feedback(memory_id, signal, ctx, opts?)`
- `runtime.memory.correct(memory_id, content, ctx, metadata?, opts?)`
- `runtime.memory.purge(ctx, opts?)`

`opts` for `search`:

- number (limit shorthand), or
- `{ limit = N, mode = "auto"|"lexical"|"semantic"|"hybrid", min_score = 0.0, include_metadata = false, include_superseded = false, strict = false, store = "alias"|selector_table, path = "relative/or/absolute.db", sources = { ... } }`

`opts` for `store` / `correct`:

- `{ storage = "auto"|"lexical_only"|"embedded", source_task = "...", tags = { ... }, store = "alias"|selector_table, path = "relative/or/absolute.db" }`

Notes:

- `storage="auto"` embeds when a provider is configured and otherwise stores lexical-only
- `storage="embedded"` requires an embedding provider
- `correct(...)` supersedes the referenced memory and stores a replacement row
- `purge(...)` defaults to `dry_run = true`
- use memory for searchable facts, notes, and learned context; see `docs/concepts/memory-vs-kv.md`
- when `store`/`path` is omitted, Turin resolves scoped state by:
  1. explicit per-call target
  2. matching `[persistence.placements]` rule
  3. primary `state` store
- `sources = { ... }` lets one search span multiple scopes and stores

Returns rows like:

```lua
{
  { content = "...", score = 0.93 },
  ...
}
```

## `runtime.kv`

Canonical KV API with explicit selector.

- `runtime.kv.get(key, ctx, opts?) -> string|nil`
- `runtime.kv.set(key, value, ctx, opts?) -> bool`
- `runtime.kv.delete(key, ctx, opts?) -> bool`

`opts` for KV:

- `{ store = "alias"|selector_table, path = "relative/or/absolute.db" }`

Notes:

- use KV for exact key-addressed state, flags, counters, mappings, and workflow state
- prefer key prefixes like `adrs/...` or `users/...` for KV-side grouping
- see `docs/concepts/memory-vs-kv.md` for the recommended convention

## `runtime.code`

Root-path-first code search API backed by `turin-map` indexes.

- `runtime.code.search.status(codebase, opts?) -> status`
- `runtime.code.search.lexical(codebase, query, opts?) -> rows`
- `runtime.code.search.semantic(codebase, query, opts?) -> rows`
- `runtime.code.search.hybrid(codebase, query, opts?) -> rows`

`codebase` can be:

- a string root like `"."` or `"repo"`
- `{ root = "...", index_path = "..." }`

Notes:

- build the index first with `turin-map index`
- from a Turin project root, `turin-map index` automatically reuses `./.turin/config.toml` and its `[embeddings]` / `[providers.*]` settings
- use `turin-map status` as the quick local check; successful semantic setup shows `Semantic: enabled (...)`
- use `turin-map index --config path/to/.turin/config.toml` when the config lives elsewhere
- use explicit `--embedding-*` flags only when you want to override the configured embedding profile for one run
- `runtime.code.search.status(...)` returns fact-level semantic metadata including `codebase_id`, `embedded_chunks`, `embedding_key`, `embedding_dimensions`, and `vector_format`
- semantic and hybrid queries require both semantic index capability and a query-time embedding provider
- semantic and hybrid queries also require the query-time embedding profile to match the index profile; `strict=false` falls back to lexical, `strict=true` errors
- `trace = true` adds per-row ranking metadata such as effective mode, fallback reason, candidate ranks, and RRF contributions
- when `strict=false`, missing semantic capability or missing embedding provider falls back to the best available lexical path
- when `strict=true`, those same cases return an error instead of falling back

## `runtime.db`

Dynamic database access API.

### Selectors

A DB target can be passed as:

- alias string (`"state"`, `"my_alias"`)
- path-like string (`"scratch/test.db"`, `"foo.db"`)
- table:
  - `{ handle = "..." }`
  - `{ path = "scratch/test.db" }`
  - `{ alias = "state" }`
  - `{ store = "state" }`
  - `{ selector = <context selector> }`
  - a raw context selector table (converted to alias)

### Functions

- `runtime.db.open(selector) -> handle_info`
- `runtime.db.close(handle_or_table) -> bool`
- `runtime.db.list() -> {handle_info...}`
- `runtime.db.query(sql, params?, opts?) -> rows`
- `runtime.db.exec(sql, params?, opts?) -> changed_count`

`handle_info` shape:

```lua
{
  handle = "db_...",
  path = "/abs/or/workspace/path.db",
  alias = "state" or nil,
  open_count = 1,
  idle_ms = 0,
}
```

`params` supports:

- positional array: `{ "alice", 42 }`
- named map: `{ name = "alice", count = 42 }` (names are normalized to `:name` if needed)

Rules:

- positional params must be dense 1-based arrays
- positional and named params cannot be mixed
- supported param value types: `nil`, boolean, number, string

`query` returns an array of JSON-like rows (Lua tables).
Blob values are encoded as:

```lua
{ __type = "blob", hex = "..." }
```

Runtime policy influences DB behavior (`db.allow_dynamic_open`, `db.path_scope`, `db.max_open_handles`, `db.idle_close_seconds`).

## `runtime.agent`

Peer-agent orchestration API.

- `runtime.agent.list() -> statuses`
- `runtime.agent.get_status(agent_id) -> status|nil`
- `runtime.agent.submit(agent_id, task, opts?) -> task_id`
- `runtime.agent.await(task_id, opts?) -> result`
- `runtime.agent.ask(agent_id, prompt, opts?) -> output`

`task` can be:

- string prompt
- `{ prompt = "...", title = "..." }`

`opts` for `submit`:

```lua
{
  capabilities = {
    ["runtime.db.query"] = true,
    ["runtime.db.exec"] = false,
  }
}
```

`opts` for `await`:

```lua
{ timeout_ms = 30000 }
```

`opts` for `ask`:

```lua
{
  title = "peer review",
  timeout_ms = 30000,
  capabilities = {
    ["db.query"] = true,
    ["db.exec"] = false,
  }
}
```

Governance integration:

- capability checks (`runtime.agent.submit`, `runtime.agent.await`, `runtime.agent.status`)
- `allowed_child_agents` allowlists
- delegated capability ceilings (downward-only)
- active grant ceilings may be inherited automatically
- DX `runtime.agent(...)` helpers accept short runtime capability names inside delegated `capabilities` maps

## `runtime.schedule`

Daemon-backed durable scheduler API.

- `runtime.schedule.create(opts) -> job`
- `runtime.schedule.update(opts) -> job`
- `runtime.schedule.get(public_id) -> job`
- `runtime.schedule.list(opts?) -> jobs`
- `runtime.schedule.runs(public_id, opts?) -> run_list`
- `runtime.schedule.enable(public_id) -> job`
- `runtime.schedule.disable(public_id) -> job`
- `runtime.schedule.delete(public_id) -> job`

`runtime.schedule.create(opts)`:

```lua
{
  prompt = "Follow up on the failed build", -- prompt jobs provide this
  action = "agent.disable",                 -- action jobs provide this instead
  params = { id = "night-qa" },
  content = {
    { type = "text", text = "Use the attached QA context" },
  },
  tools = {
    allow = { "shell_exec" },
  },
  conflict_policy = "detached",
  agent = "default",                        -- optional, defaults to current agent
  next_run_unix_ms = 1760000000000,        -- exact due time
  next_run = "2030-01-02T03:04:05Z",       -- RFC3339 or local-time shorthand
  after_seconds = 300,                     -- helper-style alternative
  interval_seconds = 3600,                 -- recurring interval
  recurring = "daily",                     -- anchored recurrence for next_run
  overlap_policy = "skip",                 -- default "skip"
  work_key = "project:alpha:heartbeat",
  max_concurrency = 1,
  enabled = true,
  persistence = {
    state = "ops",
    store = { path = "./project.db" },
  },
}
```

`runtime.schedule.update(opts)`:

```lua
{
  id = "0196f8fe-6e6a-7e1a-8da5-3f774f1a8d47", -- required
  prompt = "Follow up after deploy",
  action = "channel.disable",
  params = { id = "qa-telegram" },
  content = {
    { type = "text", text = "Preserve the prior test artifact context" },
  },
  tools = {
    allow = { "shell_exec" },
  },
  conflict_policy = "detached",
  next_run_unix_ms = 1760003600000,
  next_run = "08:00",
  after_seconds = 300,                         -- helper-style alternative
  interval_seconds = 1800,
  recurring = "weekly",
  overlap_policy = "queue",
  work_key = "project:alpha:qa",
  max_concurrency = 2,
  enabled = false,
  persistence = {
    state = "ops",
    store = { alias = "project_alpha" },
  },
}
```

Rules:

- specify one scheduling anchor:
  - `next_run`
  - `next_run_unix_ms`
  - `after_seconds`
  - or `interval_seconds` alone (first run scheduled relative to now)
- use at most one recurring mode:
  - `interval_seconds`
  - or `recurring`
- choose one job payload family:
  - prompt job:
    - `prompt`
    - optionally `content`, `tools`, `conflict_policy`
  - action job:
    - `action`
    - optional `params`
- `runtime.schedule.update(opts)` accepts the same scheduling anchor rules, but only for the fields you choose to replace
- `overlap_policy` currently supports:
  - `"skip"`
  - `"queue"`
  - `"parallel"`
- `recurring` currently supports:
  - `"daily"`
  - `"weekly"`
- `work_key` groups related prompt jobs into the same concurrency lane
- `max_concurrency` bounds concurrent prompt runs within that lane
- `agent` must be the current agent or another configured agent id
- `content`, `tools`, and `conflict_policy` follow the same shapes used by `task.submit(...)`
- built-in action names currently support:
  - `agent.enable`
  - `agent.disable`
  - `channel.enable`
  - `channel.disable`
  - `worklist.dispatch_next`
  - `worklist.release_stale`
- `worklist.dispatch_next` / `worklist.release_stale` accept:
  - `name` (required)
  - optional `scope`, `state`, `store`
  - optional `where`, `limit`
  - `stale_after_seconds` for stale-claim release
- `worklist.dispatch_next` only considers eligible root items and rejects nested `worklist.*` recursion
- other action names may be harness-defined, provided the target harness registered them with `action.define(...)`
- `runtime.schedule.list(...)` currently supports:
  - `{ agent = "reviewer" }`
- `runtime.schedule.runs(public_id, opts?)` currently supports:
  - `{ active_only = true }`
  - `{ limit = 10 }`
- job detail also includes:
  - `last_error_code`
  - `failure_count`

Notes:

- scheduler metadata and durable cross-agent signals live in the daemon-owned `runtime.db`
- local `on(...)` and durable `runtime.on(...)` support exact topics plus terminal wildcard topics such as `deploy.*`; `*` catches all topics
- scheduled execution still runs through Turin’s normal task path
- jobs may target arbitrary state/store contexts through `persistence`
- this namespace is unavailable outside daemon-managed runtimes
- updating a running job affects future scheduling metadata only; it does not rewrite the already-running task
- run history is per scheduled job and reflects actual attempts from `scheduled_job_runs`
- missing harness-defined actions surface as `last_error_code = "schedule_action_missing_handler"`

## `runtime.worklist`

State-store-backed durable work coordination API.

- `runtime.worklist.open(opts) -> list_proxy`

`runtime.worklist.open(opts)`:

```lua
{
  name = "sprint",                  -- required
  scope = "project:alpha",          -- optional string or selector table
  state = "ops",
  store = { path = "./project.db" },
  metadata = { lane = "qa" },
}
```

`list_proxy` methods:

- `list:add(payload, opts?) -> item_proxy`
- `list:all(opts?) -> items`
- `list:pending(opts?) -> items`
- `list:paused(opts?) -> items`
- `list:active() -> item|nil`
- `list:next(opts?) -> item|nil`
- `list:current(opts?) -> item|nil`
- `list:find({ where = {...} }) -> item|nil`
- `list:progress() -> { done = n, total = n }`
- `list:empty() -> boolean`
- `list:orphaned(opts?) -> items`
- `list:release_stale(opts?) -> items`
- `list:dispatch_next(opts?) -> { item = item, result = result } | nil`

`item_proxy` methods:

- `item:add(payload, opts?) -> item_proxy`
- `item:children() -> items`
- `item:claim() -> item|nil`
- `item:heartbeat() -> item`
- `item:dispatch(opts?) -> result`
- `item:done(meta?) -> item`
- `item:fail(reason?) -> item`
- `item:requeue() -> item`
- `item:update(fields) -> item`

Rules and notes:

- worklist items may be nested; `item:add(...)` writes child items under that item
- prompt items may be:
  - bare strings
  - or tables with `title`, `prompt`, optional `content`, `tools`, `conflict_policy`
- action items are tables with:
  - `action`
  - optional `params`
- both may also carry:
  - `priority`
  - `after`
  - `metadata`
- item proxies expose:
  - `paused`
  - `pause_reason`
  - `pause_until_unix_ms`
- action item proxies expose the stored action name as `action_name`; `action` is reserved for the contextual action method
- `list:paused({ due_only = true })` returns only paused items whose pause window has elapsed
- `item:requeue()` moves a paused item back to ordinary `pending`
- `opts.where` filters compare against:
  - `id`, `public_id`, `title`, `kind`, `status`, `priority`, `parent_id`
  - metadata keys
- `item:heartbeat()` refreshes the active claim heartbeat for the current execution
- `item:dispatch(...)`:
  - returns `{ dispatched = "task", task_id = ... }` for prompt items
  - returns `{ dispatched = "action", action = "...", result = ... }` for action items
- `list:dispatch_next(...)` claims the next eligible item and delegates to `item:dispatch(...)`
- dispatch helpers do not auto-complete the item
- `list:orphaned(...)` and `list:release_stale(...)` accept:
  - `stale_after_seconds`
  - optional `limit`
  - optional `where`
- item proxies expose a normalized `payload` table so prompt/action work mirrors scheduled job payloads

## `runtime.graph`

Sparse semantic graph overlay API.

- `runtime.graph.node.create(opts) -> node`
- `runtime.graph.node.list(opts?) -> nodes`
- `runtime.graph.edge.create(opts) -> edge`
- `runtime.graph.edge.list(opts?) -> edges`
- `runtime.graph.path.select(opts) -> context_target`

`runtime.graph.node.create` options:

```lua
{
  kind = "experiment",              -- required
  label = "compare candidates",
  origin_task_id = "task-123",
  origin_execution_id = "exec-456",
  metadata = { purpose = "speculation" },
  session_id = "session-ref",       -- optional; defaults to active session
}
```

`runtime.graph.edge.create` options:

```lua
{
  source = { kind = "graph_node", id = "..." }, -- required
  target = { kind = "branch_head", id = "..." }, -- required
  relation_kind = "contains",                    -- required
  source_role = "group",
  target_role = "candidate",
  origin_task_id = "task-123",
  origin_execution_id = "exec-456",
  metadata = { rank = 1 },
  session_id = "session-ref",
}
```

`runtime.graph.edge.list(opts?)` filters:

- `{ source = { kind = "...", id = "..." } }`
- `{ target = { kind = "...", id = "..." } }`
- `{ session_id = "session-ref" }`

`runtime.graph.path.select(opts)` options:

```lua
{
  source = { kind = "graph_node", id = "..." }, -- edge-query mode
  relation_kind = "contains",
  target_kind = "branch_head",
  target_role = "candidate",
  order = "oldest_first",                       -- or "newest_first"
  limit = 2,
  session_id = "session-ref",
}
```

Or:

```lua
{
  refs = {
    { kind = "turn", id = "12" },
    { kind = "branch_head", id = "..." },
  },                                            -- explicit-order mode
  session_id = "session-ref",
}
```

`runtime.graph.path.select(...)` returns an execution context target table:

```lua
{
  kind = "selected_path",
  turn_ids = { 12, 19, 27 }
}
```

Notes:

- graph rows are opt-in; ordinary serial sessions do not create them
- the sparse graph overlay does not replace turn ancestry or branch heads
- `runtime.graph.path.select(...)` accepts exactly one mode: `source` for graph-edge materialization or `refs` for explicit ordered refs
- source mode defaults to `order = "oldest_first"` and can optionally apply `limit`
- `runtime.graph.path.select(...)` currently materializes refs of kind `turn` and `branch_head`
- branch-head graph refs should use the same simple UUID value exposed as `branch.branch_id`
- validation for semantic meaning belongs in the harness/app layer; the runtime only validates what it must materialize safely

## `runtime.inference`

Inference-routing helper namespace.

- `runtime.inference.available(name) -> bool`

Checks whether a named inference context is configured for the current agent after applying any
agent-level inference overrides.

Example:

```lua
function on_turn_prepare(turn)
  if runtime.inference.available("reasoning") and (turn.prompt or ""):find("debug", 1, true) then
    turn.inference = "reasoning"
  end
  return ALLOW
end
```

## `runtime.policy`

Runtime policy storage API.

- `runtime.policy.get(key, scope?) -> json|nil`
- `runtime.policy.set(key, value, scope?) -> bool`

`scope` can be:

- `nil` (defaults to global)
- string (scope name)
- table:

```lua
{
  scope = "global",
  agent_id = "coder",
  session_id = "...",
  run_id = "...",
}
```

When identity fields are missing, Turin fills from the active runtime identity when available.

Common runtime policy keys (current):

- `spawn.enabled`
- `spawn.max_depth`
- `runtime.idle_timeout_seconds`
- `db.allow_dynamic_open`
- `db.path_scope`
- `db.max_open_handles`
- `db.idle_close_seconds`
- `queue.max_depth`
- `tool.exec_enabled`
- `hook.token_usage.reject_mode` (`informational` | `enforce_task` | `enforce_session`)

## `runtime.governance`

Governance observability and temporary grants.

### Observability

- `runtime.governance.profile() -> string` returns the configured governance profile label.
- `runtime.governance.snapshot(agent_id?) -> snapshot_json`
- `runtime.governance.agent(agent_id) -> snapshot_json`
- `runtime.governance.check(capability, agent_id?) -> decision_json`

### Temporary grants

- `runtime.governance.grant_issue(opts) -> grant`
- `runtime.governance.grant_get(grant_id) -> grant|nil`
- `runtime.governance.grant_revoke(grant_id) -> bool`
- `runtime.governance.with_grant(grant_id, fn) -> <fn returns>`

`grant_issue` options:

```lua
{
  capabilities = { ["runtime.db.exec"] = true }, -- required
  ttl_ms = 30000,
  max_uses = 1,
  reason = "approved migration",
}
```

Grant behavior:

- subject-scoped (agent/module/root context aware)
- TTL and max-use enforcement
- auditable (`governance_grant_issue/use/revoke`)
- active grant ceilings are applied to peer-agent delegation paths

## Ergonomic Aliases and Convenience APIs

## `memory` (default agent-scoped memory)

Uses a default selector derived from the active agent identity.
Requires an active session context.

- `memory.search(query, opts?)`
- `memory.store(content, metadata?, opts?)`
- `memory.feedback(memory_id, signal, opts?)`
- `memory.correct(memory_id, content, metadata?, opts?)`
- `memory.purge(opts?)`
- `memory.as(ctx) -> proxy`
  - proxy methods: `search`, `store`, `feedback`, `correct`, `purge`

Top-level shortcuts:

- `remember(content, metadata?, opts?)`
- `recall(query, opts?)`

All memory variants accept the same `store` / `path` search/store options as `runtime.memory.*`.

## `kv` (default agent-scoped KV)

Uses a default selector derived from the active agent identity.
Requires an active session context.

- `kv.get(key, opts?)`
- `kv.set(key, value, opts?)`
- `kv.delete(key, opts?)`
- `kv.as(ctx) -> proxy`
  - proxy methods: `get`, `set`, `delete`

All KV variants accept the same `store` / `path` options as `runtime.kv.*`.

## `code`

- `code.find(query, opts?)`

Thin wrapper over `runtime.code.search.hybrid(...)` with workspace-root defaulting.

## `session` and `user` aliases

Selector-derived scoped data aliases based on the active `RuntimeIdentity`.

- `session.memory.search/store`
- `session.kv.get/set/delete`
- `user.memory.search/store` (requires `identity.user_id`)
- `user.kv.get/set/delete` (requires `identity.user_id`)

## `agent` (top-level orchestration conveniences)

### Local queue/session helpers

- `agent.spawn(prompt, opts?) -> queue_token`
  - enqueues a local task in the current session queue
  - governed by `spawn.enabled`, `spawn.max_depth`, queue policy
- `agent.session.identity() -> identity_table`
- `agent.session.queue(prompt) -> bool`
- `agent.session.queue_next(prompt) -> bool`
- `agent.session.queue_all({prompts...}) -> bool`
- `agent.session.load(session_id) -> session_row|nil`
- `agent.session.list(limit?, offset?) -> rows`

### Peer-agent convenience

- `agent.submit(prompt, opts?) -> task_id`
  - submits to a peer agent and returns a request id for `runtime.agent.await(...)`
  - `opts.agent_id` (defaults to current configured agent id)
  - `opts.capabilities` (delegated ceiling)
  - `opts.execution`
- `agent.ask(prompt, opts?) -> output`
  - submits to a peer agent and awaits result in one call
  - `opts.agent_id` (defaults to current configured agent id)
  - `opts.timeout_ms`
  - `opts.capabilities` (delegated ceiling)
  - `opts.execution`

### Runtime retention

Use agent config or `runtime.policy.set("runtime.idle_timeout_seconds", value, scope?)` to control
when a runtime hibernates after a logical request completes.

- `0` => hibernate immediately after the request finishes
- positive integer => stay hot for that many idle seconds
- `nil` / `json.null` => keep the runtime hot indefinitely

## Notes on Old Namespaces

Turin’s canonical harness API is now `runtime.*` + top-level aliases described above.
Older documentation that references `turin.*` or `db.*` globals is outdated and should be treated as historical.
