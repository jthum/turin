# API Surface

This page is the inventory for Turin's current public author-facing surface.

It is intentionally mechanical. The goal is to capture what exists today so naming, semantics, and DX can be reviewed from one place.

This page does not try to redesign the API.

## Surface Levels

Turin has three relevant layers:

1. Public authoring surface

- Lua hooks
- Lua globals
- canonical `runtime.*` namespaces
- ergonomic top-level aliases and DX helpers
- channel sidecar protocol

2. Canonical harness substrate

- the stable harness-facing primitives exposed through `runtime.*`
- this is the layer DX wrappers should compile down to

3. Rust implementation substrate

- persistence/state/kernel/runtime internals inside the Rust codebase
- important for implementation, but not the primary author-facing API

For DX review purposes, the main target is layer 1, while preserving the semantics of layer 2.

## Public Surfaces In Scope

### Harness Surface

This is the main author-facing Lua surface for harness writers.

- Hooks
- System globals
- Canonical `runtime.*`
- Top-level aliases
- DX convenience globals
- Declared virtual tools

Primary references:

- [hooks.md](/home/jthum/Documents/Work/Code/turin/docs/reference/hooks.md)
- [primitives.md](/home/jthum/Documents/Work/Code/turin/docs/reference/primitives.md)
- [common-shapes.md](/home/jthum/Documents/Work/Code/turin/docs/reference/common-shapes.md)

### Channel Sidecar Surface

This is a separate public protocol surface for channel adapter processes.

Primary reference:

- [channel-sidecars.md](/home/jthum/Documents/Work/Code/turin/docs/reference/channel-sidecars.md)

### State Schema Surface

This is not the main harness API, but it is part of Turin's durable substrate and should stay easy to inspect.

Primary reference:

- [schema.md](/home/jthum/Documents/Work/Code/turin/docs/reference/schema.md)

## Harness Inventory

### Hook Functions

Defined lifecycle hooks:

- `on_session_start(event)`
- `on_session_end(event)`
- `on_task_start(event)`
- `on_turn_start(event)`
- `on_turn_prepare(ctx)`
- `on_tool_call(call)`
- `on_tool_result(result)`
- `on_kernel_event(event)`
- `on_token_usage(event)`
- `on_plan_submit(event)`
- `on_task_complete(event)`
- `on_plan_complete(event)`
- `on_all_tasks_complete(event)`
- `on_inference_error(event)`

Reference:

- [hooks.md](/home/jthum/Documents/Work/Code/turin/docs/reference/hooks.md)

### Verdict Constants

Globals:

- `ALLOW`
- `REJECT`
- `ESCALATE`
- `MODIFY`

Reference:

- [primitives.md](/home/jthum/Documents/Work/Code/turin/docs/reference/primitives.md#verdict-constants-for-hooks)

### System Globals

Globals:

- `fs.*`
- `hash.*`
- `json.*`
- `time.*`
- `log.*`
- `try(fn, ...)`
- `import(name)`
- `import_scoped(name, opts?)`
- `use(name, opts?)`
- `use_scoped(name, opts?)`
- `watch(path)`

Reference:

- [primitives.md](/home/jthum/Documents/Work/Code/turin/docs/reference/primitives.md#system-globals)

### Canonical Runtime API

Namespaces under `runtime`:

- `runtime.context.*`
- `runtime.inference.*`
- `runtime.memory.*`
- `runtime.kv.*`
- `runtime.code.*`
- `runtime.db.*`
- `runtime.agent.*`
- `runtime.schedule.*`
- `runtime.worklist.*`
- `runtime.graph.*`
- `runtime.policy.*`
- `runtime.governance.*`
- `runtime.on(...)`
- `runtime.emit(...)`

Reference:

- [primitives.md](/home/jthum/Documents/Work/Code/turin/docs/reference/primitives.md#canonical-runtime-api-runtime)

### Daemon Control Plane

Current typed daemon/control-plane surfaces include:

- `daemon.ping`
- `daemon.status`
- `runtime.events.subscribe`
- `agent.*`
- `task.*`
- `schedule.create`
- `schedule.update`
- `schedule.get`
- `schedule.list`
- `schedule.runs`
- `schedule.enable`
- `schedule.disable`
- `schedule.delete`
- `worklist.list`
- `worklist.get`
- `worklist.items`
- `workitem.get`
- `session.*`
- `harness.*`
- `channel.*`

Notes:

- `worklist.*` / `workitem.get` control-plane queries are explicitly store-targeted because worklists live in arbitrary state/store backends, not in a daemon-owned global index like `jobs.db`
- current `worklist.*` daemon operations are read-only inspection APIs; mutation still lives in the harness/runtime worklist surface
- `worklist.items` currently supports:
  - `status`
  - `status = "paused"` for primary paused work items
  - `parent_id`
  - metadata-aware `where = { ... }`
  - `claimed_only`
  - `paused_only`
  - `due_only`
  - `limit`

### Top-Level Authoring Aliases

Author-facing aliases outside `runtime.*`:

- `memory.*`
- `kv.*`
- `code.*`
- `session.memory.*`
- `session.kv.*`
- `user.memory.*`
- `user.kv.*`
- `agent.*`

Reference:

- [primitives.md](/home/jthum/Documents/Work/Code/turin/docs/reference/primitives.md#ergonomic-aliases-and-convenience-apis)

### DX Convenience Globals

Current DX helper globals and helper-layer extensions:

- `verdict.*`
- `access.*`
- `remember(...)`
- `recall(...)`
- `scope(kind, key, opts?)`
- `graph.*`
- `schedule.*`
- `action.define(...)`
- `action.run(...)`
- `on(...)`
- `emit(...)`
- `session.remember(...)`
- `session.recall(...)`
- `session.get(...)`
- `session.set(...)`
- `session.del(...)`
- `session.incr(...)`
- `user.remember(...)`
- `user.recall(...)`
- `user.get(...)`
- `user.set(...)`
- `user.del(...)`
- `user.incr(...)`
- `code.find(...)`
- `fs.summary(...)`
- `fs.read_json(...)`
- `fs.write_json(...)`
- callable `runtime.agent("agent_id") -> proxy`
- `runtime.db.with(selector, fn, opts?)`
- `runtime.governance.grant(spec, fn)`
- `schedule.after(...)`
- `schedule.every(...)`
- `schedule.at(...)`
- `schedule.update(...)`
- `schedule.get(...)`
- `schedule.list(...)`
- `schedule.runs(...)`
- `schedule.enable(...)`
- `schedule.disable(...)`
- `schedule.delete(...)`
- `worklist(...)`

These are helper-layer conveniences over the canonical substrate and should be evaluated separately during DX review.

Reference:

- [harness-guide.md](/home/jthum/Documents/Work/Code/turin/docs/guides/harness-guide.md#writing-with-the-dx-layer)

### Virtual Tool Surface

Globals:

- `tool.define(name, spec)`
- `tool.call(name, args?)`
- `tool.sequence(calls, callback?)`
- `shell.quote(input)`

Reference:

- [harness-guide.md](/home/jthum/Documents/Work/Code/turin/docs/guides/harness-guide.md#declared-virtual-tools)
- [primitives.md](/home/jthum/Documents/Work/Code/turin/docs/reference/primitives.md#system-globals)

## Canonical Runtime Namespace Map

This is the current canonical map at the harness level.

### `runtime.context`

- `glob(pattern)`
- `build(...)`

### `runtime.memory`

- `search(query, opts?)`
- `store(content, metadata?, opts?)`
- `feedback(memory_id, signal, opts?)`
- `correct(memory_id, content, metadata?, opts?)`
- `purge(opts?)`

### `runtime.kv`

- `get(key, opts?)`
- `set(key, value, opts?)`
- `delete(key, opts?)`

### `runtime.code`

- `search.hybrid(codebase, opts?)`
- `search.lexical(codebase, opts?)`
- `search.semantic(codebase, opts?)`
- `status(codebase, opts?)`

### `runtime.db`

- `open(selector_or_opts)`
- `close(handle_or_opts)`
- `list()`
- `exec(sql, params?, opts?)`
- `query(sql, params?, opts?)`

### `runtime.agent`

- `identity()`
- `status(agent_id)`
- `submit(agent_id, task, opts?)`
- `ask(agent_id, task, opts?)`
- `await(task_id, opts?)`
- `status(task_id, opts?)`
- `cancel(task_id, opts?)`
- `sidestep(prompt, "mode"|opts?)`

### `runtime.graph`

- `node.create(opts)`
- `node.list(opts?)`
- `edge.create(opts)`
- `edge.list(opts?)`
- `path.select(opts)`

### `runtime.policy`

- `get(key, scope?)`
- `set(key, value, scope?)`

### `runtime.governance`

- `profile()`
- `snapshot(agent_id?)`
- `agent(agent_id)`
- `check(capability, agent_id?)`
- `grant_issue(opts)`
- `grant_get(grant_id)`
- `grant_revoke(grant_id)`
- `with_grant(grant_id, fn)`

Detailed signatures belong in:

- [primitives.md](/home/jthum/Documents/Work/Code/turin/docs/reference/primitives.md)

## Common Shapes That Need Separate Review

These shapes recur across the surface and should be reviewed as shared semantics, not one function at a time:

- public error/absence contract
- identity table
- session row
- branch row
- `context_target`
- graph ref
- graph node
- graph edge
- `selected_path`
- agent/task execution option tables
- memory/kv store selectors

Reference:

- [common-shapes.md](/home/jthum/Documents/Work/Code/turin/docs/reference/common-shapes.md)

## DX Review Implication

This inventory suggests the DX pass should review the harness surface in three buckets:

1. Canonical substrate naming and semantics
2. Repeated option/result shapes
3. Helper-layer readability and prose-like authoring quality

That lets Turin improve the authoring experience without destabilizing the underlying primitives.
