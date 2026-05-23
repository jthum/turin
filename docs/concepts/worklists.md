# Durable Worklists

Durable worklists are Turin's primitive for persistent, claimable work.

They are meant to make the simple case easy and the complex case possible. A
flat queue can be three lines of harness code. A project backlog with lanes,
dependencies, deterministic actions, human checkpoints, and stale-claim
recovery uses the same primitive. The extra behavior lives in item data, not in
a separate API for every workflow shape.

## Core Model

A worklist is a durable, ordered, claimable collection of work items.

Each item has:

- a title or prompt
- a status lifecycle
- optional prompt/action payload data
- optional metadata
- optional dependencies
- optional parent/child structure
- optional claim ownership and heartbeat state

Turin owns the durable mechanics:

- storage
- ordering
- dependency checks
- claim/release lifecycle
- stale-claim recovery
- prompt/action dispatch plumbing
- state/store/scope routing

Harness code owns the domain semantics:

- what the item means
- what "done" means
- how priorities are chosen
- which metadata fields matter
- when items are created
- whether hierarchy is useful
- how action and approval items are handled

The worklist does not infer completion. Completion is always explicit through
the harness, an action, or a user-approved workflow.

## Five Core Operations

Most worklist flows reduce to five operations:

```lua
list:add(payload, opts?) -- put something in
list:next(opts?)         -- claim the next eligible pending item
list:active()            -- inspect the current claimed item
item:done(meta?)         -- mark an item finished
item:fail(reason?)       -- mark an item failed
```

These operations are deliberately small:

- `add` creates durable work.
- `next` claims eligible work and records the claim.
- `active` lets recurring turns or scheduled heartbeats continue the same work.
- `done` and `fail` make completion explicit.

## Supporting Operations

The common helper surface adds inspection, recovery, and dispatch without
changing the core model:

```lua
list:current(opts?)       -- active item, or claim next
list:pending(opts?)       -- pending items
list:paused(opts?)        -- paused items
list:all(opts?)           -- all items regardless of status
list:find({ where = ... })-- find a specific item
list:progress()           -- { done = n, total = n }
list:empty()              -- no eligible pending items
list:orphaned(opts?)      -- claimed work with stale/missing execution
list:release_stale(opts?) -- release orphaned claims
list:dispatch_next(opts?) -- claim and dispatch one eligible item

item:add(payload, opts?)  -- add child work
item:children()           -- child items
item:claim()              -- claim this item directly
item:heartbeat()          -- refresh active claim heartbeat
item:dispatch(opts?)      -- dispatch prompt/action payload
item:requeue()            -- release claim / clear pause state
item:update(fields)       -- patch item fields
```

List and claim operations can filter by built-in fields, pause fields, and
metadata:

```lua
tasks:next({ where = { role = "qa" } })
tasks:pending({ where = { tag = "blocked" }, limit = 10 })
tasks:find({ where = { id = task_id } })
```

This is what lets different agents share one backlog without pretending that
all work belongs in one undifferentiated global queue.

## Payloads And Metadata

Worklist complexity is additive. If a field is unused, the worklist behaves like
a plain queue.

Common item fields:

| Field | Purpose | Simple case |
|---|---|---|
| `title` | Display label | Derived from prompt/action when omitted |
| `prompt` | Prompt work submitted as a Turin task | String payloads become prompt items |
| `content` | Structured multimodal prompt content | Omitted |
| `tools` | Per-task tool override | Omitted |
| `conflict_policy` | Runtime task conflict behavior | Default task policy |
| `action` | Named action to execute | Omitted |
| `params` | Action parameters | Omitted |
| `priority` | Higher values are claimed first | `0` |
| `after` | Dependency item ids | No dependencies |
| `metadata` | Harness-defined labels/state | Empty |

Metadata commonly carries domain-specific fields:

```lua
{
  role = "qa",
  needs = "browser",
  tag = "blocked",
  tier = 2,
  criteria = "Checkout succeeds in Firefox",
}
```

Turin stores metadata and can filter by it. The harness decides what it means.

## Flat Queue

The simplest worklist is a durable queue:

```lua
local patients = worklist("patients")

patients:add("Review John Doe", {
  metadata = { condition = "diabetes" },
})
patients:add("Review Jane Smith", {
  metadata = { condition = "hypertension" },
})

local patient = patients:next()
if patient then
  -- process the patient
  patient:done()
end
```

No hierarchy, no dependencies, no custom scheduler logic.

## Partitioning Work Across Agents

Different agents often need different views of the same project.

One option is separate worklists:

```lua
local dev = worklist("dev_tasks", { scope = "project:web-app" })
local qa = worklist("qa_tasks", { scope = "project:web-app" })
```

Another option is one shared backlog with filtered claiming:

```lua
local tasks = worklist("tasks", { scope = "project:web-app" })

tasks:add("Implement login form", {
  metadata = { role = "dev" },
})
tasks:add("Test login in browser", {
  metadata = { role = "qa", needs = "browser" },
})

local next_dev = tasks:next({ where = { role = "dev" } })
local next_qa = tasks:next({ where = { role = "qa" } })
```

Both approaches can live in one state DB or in different stores. The primitive
does not force that choice.

## Prompt And Action Items

Not every item requires inference. The same worklist can sequence prompts,
deterministic actions, and human checkpoints.

Prompt items submit normal Turin tasks:

```lua
tasks:add({
  title = "Review authentication module",
  prompt = "Review the authentication module for security issues.",
  tools = { allow = { "fs.read", "shell.exec" } },
  conflict_policy = "detached",
})
```

Action items invoke named harness actions:

```lua
tasks:add({
  title = "Run regression tests",
  action = "qa.run_tests",
  params = { suite = "checkout" },
  priority = 10,
})
```

Checkpoint items can be modeled as metadata and handled by the harness:

```lua
tasks:add("Approve sprint plan", {
  metadata = { kind = "approval" },
})
```

The worklist stores and claims these items. The harness decides whether to
dispatch, wait, ask the user, or mark them done.

## Dependencies

Dependencies are metadata-backed work ordering. `next` skips items whose
dependencies are not complete.

```lua
local compile = tasks:add("Compile source")
local test = tasks:add("Run tests", {
  after = { compile.id },
})
local deploy = tasks:add("Deploy", {
  after = { test.id },
})

local item = tasks:next() -- Compile source
item:done()

local next_item = tasks:next() -- Run tests
```

Fan-in works the same way:

```lua
local build_api = tasks:add("Build API")
local build_ui = tasks:add("Build UI")

tasks:add("Integration tests", {
  after = { build_api.id, build_ui.id },
})
```

The integration item becomes claimable only after both build items are done.

## Hierarchy

Items can have children. This gives nested projects, sprints, tasks, and
subtasks without a separate model.

```lua
local projects = worklist("projects")

local app = projects:add("My App", {
  metadata = { tier = 1 },
})
app:add("Fix login bug", {
  metadata = { criteria = "Login works" },
})
app:add("Add dark mode")

local project = projects:current()
local task = project:current()
if task then
  task:done()
end
```

Hierarchy is optional. A flat worklist and a nested worklist use the same helper
surface.

## Long-Running Work

Worklists pair naturally with scheduled jobs and recurring turns.

```lua
schedule.every(3600, "Review next patient", {
  overlap = "skip",
})

function on_turn_prepare(turn)
  local patients = worklist("patients")
  local patient = patients:active() or patients:next()

  if patient then
    turn.system_prompt = turn.system_prompt
      .. "\n\nCurrent patient work item: "
      .. patient.title
  else
    turn.system_prompt = turn.system_prompt
      .. "\n\nAll patients have been reviewed for this cycle."
  end

  return ALLOW
end
```

The scheduler decides when the harness wakes up. The worklist decides which
durable item is active. The harness decides what the active item means.

## Dispatch Pattern

`dispatch_next` is useful when the backlog itself contains prompt/action
payloads:

```lua
function on_turn_prepare(turn)
  local qa = worklist("qa", { scope = "project:web-app" })

  local dispatched = qa:dispatch_next({
    where = { role = "qa" },
  })

  if dispatched then
    -- Dispatch only starts the work. The harness still decides when to call
    -- done, fail, requeue, or pause.
    return ALLOW
  end

  turn.system_prompt = turn.system_prompt .. "\n\nNo QA work is pending."
  return ALLOW
end
```

Dispatch helpers do not auto-complete items. That keeps the completion decision
in the domain layer.

## Completion

Turin does not automatically decide that an item is complete.

Common completion patterns:

- An agent uses a tool/action that calls `item:done(...)`.
- A harness checks domain-specific criteria and calls `item:done(...)`.
- A user approves the result and the harness calls `item:done(...)`.
- A failed attempt calls `item:fail(reason)` or `item:requeue()`.
- A long-running item refreshes ownership with `item:heartbeat()`.

This explicitness is deliberate. It avoids hidden policy in the runtime and
keeps workflow judgment in harness code.

## Stale Claim Recovery

Claims are durable. If a runtime disappears while holding an item, the item can
be found and released later:

```lua
local stale = tasks:orphaned({ stale_after_seconds = 600 })
local released = tasks:release_stale({ stale_after_seconds = 600 })
```

This is useful for daemon-supervised agents, scheduled work, and channel-driven
sessions where a crash should not permanently trap work.

## Store And Scope

Worklists are state-store backed. They are not stored in one daemon-global queue
unless the harness chooses such a store.

```lua
local project_tasks = worklist("tasks", {
  scope = "project:alpha",
})

local ops_tasks = worklist("qa", {
  store = "ops",
})

local local_tasks = worklist("scratch", {
  store = { path = "./project.db" },
})
```

This lets harnesses keep:

- everything in one state DB
- project work in a project-local DB
- operational work in a dedicated store
- separate domains in separate stores

The worklist primitive should not force that choice.

## Relationship To Scheduler

The scheduler and worklists solve different problems:

- scheduler: when should something wake up or run?
- worklist: what durable work should be claimed next?

They complement each other:

- scheduled prompt jobs can enqueue recurring work
- scheduled action jobs can call `worklist.dispatch_next`
- scheduled maintenance can call `worklist.release_stale`
- overlap policy prevents repeated wakeups from piling onto the same lane
- worklist claiming preserves active work across heartbeats

## Current Implementation Notes

The durable schema lives in `src/persistence/schema.rs` and uses separate
`worklists` and `work_items` tables.

The current implementation includes:

- prompt and action item payloads
- hierarchy through parent item ids
- priority ordering
- dependency ids through `after`
- metadata filtering
- pause fields
- claim ownership and heartbeat fields
- state/store/scope routing
- daemon read APIs for inspection

For exact method signatures, see `docs/reference/primitives.md`. For ownership,
invariants, and focused tests, see `docs/architecture/maps/scheduler-worklists.md`.
