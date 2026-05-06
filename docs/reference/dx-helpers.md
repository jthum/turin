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
function on_turn_prepare(ctx)
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

## Main Helpers

- `allowed(capability[, opts])`
- `needs(capability[, opts])`
- `scope(kind, key, opts?)`
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
- scheduled jobs live in daemon-owned `jobs.db`
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
- custom action names may be defined at harness load time with:
  - `action.define("qa.run_smoke", function(params) ... end)`
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
  - direct fields such as `prompt`, `action`, `params`, `content`, `tools`
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
