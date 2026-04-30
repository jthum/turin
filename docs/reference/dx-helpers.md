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
  log.warn("query failed: " .. tostring(err))
end
```
