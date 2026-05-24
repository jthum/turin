# DX Helpers

This page describes Turin's promoted helper-first authoring layer.

Use this page first when writing harnesses intended to read like prose.

Use [primitives.md](/home/jthum/Documents/Work/Code/turin/docs/reference/primitives.md) when you need exact substrate control or want the canonical `runtime.*` equivalent.

## Recommended Voice

Preferred style:

- direct helper calls
- callable proxies where available
- short runtime capability names in helper checks and delegated task/grant tables
- `target` in helper execution option tables

Examples:

```lua
function on_turn_prepare(turn)
  local reviewer = runtime.agent("reviewer")
  local spec = fs.summary("SPEC.md")

  if allowed("db.exec") then
    runtime.db.with("state", function(db)
      db:exec("insert into review_log(spec) values (?)", { spec })
    end)
  end

  local review = reviewer:ask("Review this spec:\n\n" .. spec)
  session.set("latest_review", review)
  return ALLOW
end
```

## Callback Naming Convention

Turin examples prefer naming callback parameters after what they are.

Recommended defaults:

- `on_turn_prepare(turn)`
- `on_tool_call(call)`
- `on_tool_result(result)`
- `on_session_start(session)`
- `on_task_start(task)`
- `on_token_usage(usage)`
- `action.define("name", function(this, params) ... end)`

For local events and cross-agent signals:

- callbacks receive domain data first and optional metadata second
- name the first argument after what it is when obvious:
  - `deploy`
  - `review`
  - `decision`
- when the payload is generic, `data` is the preferred fallback
- use `meta` for the second argument only when you actually need delivery/event details

Examples:

```lua
on("qa.failed", function(failure)
  log.info("QA failed: " .. tostring(failure.suite))
end)

runtime.on("code.ready", function(ready, meta)
  log.info("Ready branch: " .. tostring(ready.branch))
  log.info("From: " .. tostring(meta.source_agent_id))
end)
```

## Main Helpers

- `allowed(capability[, opts])`
- `needs(capability[, opts])`
- `scope(kind, key, opts?)`
- `ref(proxy)`
- `target.scope(name?)`
- `target.worklist(name?)`
- `target.workitem(name?)`
- `remember(...)`
- `recall(...)`
- `fs.summary(path, opts?)`
- `fs.stat(path)`
- `graph.new(...)`
- `graph.node(...)`
- `graph.branch(...)`
- `graph.turn(...)`
- `schedule.after(...)`
- `schedule.every(...)`
- `schedule.at(...)`
- `schedule.update(...)`
- `worklist(...)`
- `schedule.get(...)`
- `schedule.enable(...)`
- `schedule.disable(...)`
- `schedule.delete(...)`
- callable `runtime.db(...)`
- callable `runtime.agent(...)`
- `runtime.governance.grant(spec, fn)`
- `try(fn, ...)`

## Short Capability Forms

Short capability names elide `runtime.` only.

Examples:

- `allowed("db.exec")`
- `needs("agent.submit")`
- `access.check("policy.set")`

Supported helper-layer surfaces:

- `allowed(...)`
- `needs(...)`
- `access.check(...)`
- delegated capability tables in helper-style task/grant calls

Important boundary:

- `import_scoped(...)` and `use_scoped(...)` capability ceilings still use canonical strings such as `runtime.db.query`

## Callable Proxies

### Reference-aware object payloads

Scope, worklist, and work item proxies are reference-aware. They still cross action,
event, signal, and schedule boundaries as JSON, but the runtime includes a reserved
`_ref` identity marker and hydrates recognized references back into proxies on the
receiving side.

```lua
local project = scope("project", "alpha")
local sprint = worklist("sprint", { scope = project })
local task = sprint:add("Review checkout")

task.title = "Review checkout smoke flow"
action.run("review.enqueue", { task = task })

runtime.emit("review.ready", {
  project = ref(project),
  task = ref(task),
})
```

Notes:

- passing a proxy directly sends its public fields plus `_ref`; the receiver hydrates the proxy and overlays the sent fields
- `ref(proxy)` sends only `_ref`, so the receiver hydrates the current stored state
- plain JSON tables still behave as ordinary JSON
- malformed or unknown `_ref` objects stay plain data
- recognized `_ref` objects whose backing record no longer exists raise an error
- overlay fields may not replace proxy methods such as `action`, `done`, or `set`
- work item proxies expose the stored action name as `item.action_name`; `item.action` is the contextual action method

Contextual actions can be run without spelling the fully qualified action name:

```lua
action.define("project.elaborate", function(this, params)
  local project = params.subject
  project:set("summary", params.params.summary)
  return project
end)

local project = scope("project", "alpha")
project:action("elaborate", { summary = "Ready for review" })
```

Use `action.define_on(...)` when the action should attach as a method on matching
runtime proxies:

```lua
action.define_on("project", "elaborate", function(this, project, params)
  project:set("summary", params.summary)
  return project
end)

action.define_on(target.workitem("tickets"), "classify", function(this, item)
  item:update({ metadata = { class = "bug" } })
  return item
end)

action.define_on(target.workitem(), "escalate", function(this, item)
  return item:update({ priority = 100 })
end)
```

Targets:

- `"project"` is shorthand for `target.scope("project")`
- `target.scope()` matches all scope proxies
- `target.worklist("tickets")` matches the `tickets` worklist proxy
- `target.worklist()` matches all worklist proxies
- `target.workitem("tickets")` matches items in the `tickets` worklist
- `target.workitem()` matches all work item proxies
- when both generic and specific methods exist with the same name, the more specific target wins

### `runtime.agent(agent_id)`

Preferred methods:

- `agent:ask(prompt, opts?)`
- `agent:submit(task, opts?)`
- `agent:await(task_id, opts?)`
- `agent:status()`
- `agent:sidestep(prompt, "mode"|opts?)`

### `runtime.db(selector)`

Preferred methods:

- `db:one(sql, params?, opts?)`
- `db:all(sql, params?, opts?)`
- `db:exec(sql, params?, opts?)`
- `db:close()`

And:

- `runtime.db.with(selector, fn, opts?)`

### `schedule.*`

Preferred scheduler helpers:

- `schedule.after(seconds, payload, opts?)`
- `schedule.every(seconds, payload, opts?)`
- `schedule.at(timestamp, payload, opts?)`
- `schedule.update(public_id, opts?)`
- `schedule.get(public_id)`
- `schedule.list(opts?)`
- `schedule.runs(public_id, opts?)`
- `schedule.enable(public_id)`
- `schedule.disable(public_id)`
- `schedule.delete(public_id)`

Notes:

- this is a daemon-backed surface and requires a daemon-managed runtime
- scheduled jobs and durable cross-agent signals live in daemon-owned `runtime.db`
- jobs may still target different `state` / `store` persistence contexts
- `opts.work_key` lets related jobs share a concurrency lane
- `opts.max_concurrency` bounds how many prompt jobs in that lane may run at once
- `payload` may be either:
  - a bare prompt string
  - or a table such as `{ action = "agent.disable", params = { id = "night-qa" } }`
- `timestamp` may be:
  - unix milliseconds
  - an RFC3339 timestamp string
  - a local-time shorthand like `"08:00"` or `"08:00:30"`
- `opts.recurring` currently supports:
  - `"daily"`
  - `"weekly"`
- `opts.overlap` currently supports:
  - `"skip"`
  - `"queue"`
  - `"parallel"`
- `schedule.runs(public_id, opts?)` supports:
  - `{ active_only = true }`
  - `{ limit = 10 }`
- built-in action payloads currently support:
  - `agent.enable`
  - `agent.disable`
  - `channel.enable`
  - `channel.disable`
  - `worklist.dispatch_next`
  - `worklist.release_stale`
- worklist scheduled actions accept params such as:
  - `name`
  - `scope`
  - `state`
  - `store`
  - `where`
  - `limit`
  - `stale_after_seconds`
- `worklist.dispatch_next` opens the named worklist and dispatches the next eligible root item:
  - prompt items become normal Turin tasks
  - action items run through the same action registry as other scheduled actions
- `worklist.release_stale` releases orphaned claimed root items back to pending state
- nested `worklist.*` action items are rejected inside `worklist.dispatch_next`
- custom action names may be defined at harness load time with:
  - `action.define("qa.run_smoke", function(this, params) ... end)`
- declared actions may also be invoked directly with:
  - `action.run("qa.run_smoke", { suite = "checkout" })`
- local custom events use:
  - `on("qa.failed", function(data, meta) ... end)`
  - `emit("qa.failed", { suite = "checkout" })`
- cross-agent signals use:
  - `runtime.on("code.ready", function(data, meta) ... end)`
  - `runtime.emit("code.ready", { branch = "feature-x" })`
- declared action handlers receive a control-aware first parameter; Turin docs/examples conventionally name it `this`
  - `this.params`
  - `this.checkpoint`
  - `this.checkpoint:get(key, default?)`
  - `this.checkpoint:all()`
  - `this.item`
  - `this:pause(...)`
  - `this:pause_for(seconds, opts?)`
  - `this:complete(...)`
  - `this:fail(...)`
  - `this:cancel(...)`
  - `this:is_cancelled()`
- `this:pause(...)` is the preferred lightweight way to stop an action intentionally and continue later
- `this:pause(...)` now puts the current work item into primary `paused` state rather than treating it as ordinary `pending`
- `this:pause_for(seconds, opts?)` is the shorter form when the main intent is “pause and try again later”
- `this:is_cancelled()` lets long-running actions cooperate with session/task cancellation without inventing their own flag
- `on(...)` is load-time only and registers additive local listeners in registration order
- `emit(...)` dispatches those listeners synchronously in-process and returns the number of listeners invoked
- `on(...)` and `runtime.on(...)` support exact topics plus terminal wildcard topics such as `deploy.*`; `*` catches all topics
- local event listeners are intended to react by mutating state, calling `action.run(...)`, or scheduling/worklisting follow-up work
- `runtime.on(...)` is load-time only and declares that the current harness should receive durable cross-agent signals for that topic
- those declared topics are mirrored into a durable `runtime.db` subscription index on harness init/reload so cold agents remain discoverable
- `runtime.emit(...)` persists pending deliveries for subscribed agents, wakes their peer runtimes, and returns the number of target agents
- `runtime.signals.subscribers(topic)` is the cheap inspection helper for current durable topic subscribers
- `runtime.signals.list(opts?)` is the cheap inspection helper for pending cross-agent signal rows
- `runtime.emit(...)` requires daemon/runtime coordination; it is intentionally separate from purely local `emit(...)`
- cross-agent signal handlers still run locally inside the target harness after that harness boots, so they are best used to mutate state, schedule work, or call `action.run(...)`
- local `emit(...)` and cross-agent `runtime.emit(...)` are intentionally separate: one is synchronous in-process composition, the other is durable agent-to-agent signaling
- paused work items are skipped by ordinary `worklist.next()` / `dispatch_next()` until their pause window is due
- `list:paused({ due_only = true })` is the explicit inspection helper when you want paused items whose resume window has already elapsed
- `item:requeue()` is the explicit way to move paused work back to ordinary `pending`
- `opts.where` may also match pause fields:
  - `paused`
  - `pause_reason`
  - `pause_until_unix_ms`
- `runtime.schedule.create(...)` / `update(...)` may also carry structured `content`, `tools`, and `conflict_policy` fields when a scheduled prompt needs richer task input than a bare string
- `schedule.update(...)` changes only the fields you provide; it does not mutate the already-running attempt for a currently active job
- job detail surfaces scheduler health fields:
  - `last_error_code`
  - `failure_count`
- missing harness action handlers are reported as:
  - `last_error_code = "schedule_action_missing_handler"`

### `worklist(...)`

Preferred worklist helper:

- `worklist(name, opts?) -> list_proxy`

Common list methods:

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

Common item methods:

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

- worklist items may hold either:
  - prompt payloads
  - named action payloads
- prompt items may also carry structured:
  - `content`
  - `tools`
  - `conflict_policy`
- item proxies expose:
  - direct fields such as `prompt`, `action_name`, `params`, `content`, `tools`
  - and a normalized `payload` table
- `opts.where` filters against:
  - built-in fields like `title`, `kind`, `status`, `priority`
  - metadata keys stored on the item
- stale-claim helpers use `opts.stale_after_seconds` and default to 300 seconds
- `item:dispatch(...)` uses the shared durable payload model:
  - prompt items enqueue a normal local Turin task and return `{ dispatched = "task", task_id = ... }`
  - action items invoke the declared action handler and return `{ dispatched = "action", action = "...", result = ... }`
- `list:dispatch_next(...)` claims the next eligible item and immediately delegates to `item:dispatch(...)`
- dispatch helpers do not auto-complete the item
- worklists are backed by state stores, so `opts` may also carry:
  - `scope`
  - `state`
  - `store`
  - `path`

## Graph Helpers

Preferred graph authoring:

```lua
local experiment = graph.new("experiment", "compare candidates")
experiment:add(graph.branch(branch), { role = "candidate" })

local target = experiment:newest("candidate")
agent.sidestep("Analyze this candidate", {
  target = target,
})
```

Canonical substrate remains:

- `runtime.graph.node.*`
- `runtime.graph.edge.*`
- `runtime.graph.path.select(...)`

## Scope Helpers

Use built-in scopes when they fit:

- `session.*`
- `user.*`
- default `remember(...)` / `recall(...)`

Use `scope(...)` for custom domains:

```lua
local project = scope("project", "my-app")
project.remember("uses event sourcing")
project.set("version", "2.1")
```

## Error Handling

Promoted helper style should stay on the happy path.

Use:

- direct helper calls when failure should abort the turn
- `try(fn, ...)` or `pcall(...)` when recovery is intentional

Example:

```lua
local rows, err = try(runtime.db.query, "select * from notes")
if not rows then
  log("query failed: " .. tostring(err))
end
```

Use `try(...)` for optional inputs and best-effort work:

```lua
local spec, err = try(fs.read, "SPEC.md")
if not spec then
  log("optional spec missing: " .. tostring(err))
  spec = ""
end
```
