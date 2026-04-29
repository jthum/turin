# Common Shapes

This page captures the shared data shapes that recur across Turin's public harness surface.

It exists so the DX review can evaluate consistency at the shape level rather than only at the individual function level.

## Result Convention

Most Lua-facing APIs use:

- success: `value`
- failure: `nil, err`

Hooks are different because they return verdicts or modified payloads rather than tuple-style results.

Primary reference:

- [primitives.md](/home/jthum/Documents/Work/Code/turin/docs/reference/primitives.md#result-convention-important)

## Identity Table

Returned by surfaces such as `agent.session.identity()`.

Shape:

```lua
{
  session_id = "...",
  agent_id = "...",
  user_id = nil or "...",
  channel_id = nil or "...",
  tenant_id = nil or "...",
  run_id = nil or "...",
  extra = {
    -- string:string identity extensions
  },
}
```

This is the main author-facing identity shape for scoped helpers.

## Session Row

Returned by session-loading/listing helpers.

Shape:

```lua
{
  internal_id = 1,
  session_id = "...",
  agent_id = "default",
  metadata = nil or {...} or "...",
  created_at = "...",
}
```

## Branch Row

Returned by branch/session branch helpers.

Shape:

```lua
{
  branch_id = "...",
  name = "main",
  head_turn_index = 3,
  source_turn_id = nil or 12,
  origin_kind = "manual",
  origin_task_id = nil or "...",
  origin_execution_id = nil or "...",
  origin_metadata = nil or {...},
  active = true or false,
  deferred = true or false,
  created_at = "...",
}
```

Important naming distinction:

- `branch_head` is the structural primitive
- branch helper APIs currently return a row with `branch_id`
- DX review should preserve the underlying semantic distinction even if helper naming gets cleaner

## `context_target`

This is the umbrella execution-selection concept.

Current variants include:

- branch head
- turn id
- selected path
- external reference
- summary source

Example selected path shape:

```lua
{
  kind = "selected_path",
  turn_ids = { 12, 19, 27 },
}
```

Important distinction:

- `context_target` is the generic selector
- `selected_path` is one concrete materialized read-target shape

## Graph Ref

Used by `runtime.graph.*`.

Shape:

```lua
{
  kind = "graph_node" or "branch_head" or "turn" or "...",
  id = "...",
}
```

This is intentionally generic.

## Graph Node

Returned by `runtime.graph.node_create(...)` and `runtime.graph.nodes(...)`.

Shape:

```lua
{
  id = 1,
  node_id = "...",
  session_internal_id = 1,
  kind = "experiment",
  label = nil or "compare candidates",
  origin_task_id = nil or "...",
  origin_execution_id = nil or "...",
  metadata = nil or {...},
  created_at = "...",
}
```

## Graph Edge

Returned by `runtime.graph.edge_create(...)` and `runtime.graph.edges(...)`.

Shape:

```lua
{
  id = 1,
  edge_id = "...",
  session_internal_id = 1,
  source = { kind = "...", id = "..." },
  target = { kind = "...", id = "..." },
  relation_kind = "contains",
  source_role = nil or "group",
  target_role = nil or "candidate",
  origin_task_id = nil or "...",
  origin_execution_id = nil or "...",
  metadata = nil or {...},
  created_at = "...",
}
```

## `selected_path`

Returned by `runtime.graph.selected_path(...)`.

Shape:

```lua
{
  kind = "selected_path",
  turn_ids = { 12, 19, 27 },
}
```

Materialization modes currently include:

- source/filter mode
- explicit refs mode

This is a read-path shape, not a durable structural entity.

## Store / Selector Shapes

These recur across `runtime.memory.*`, `runtime.kv.*`, `runtime.db.*`, and related aliases.

Common ideas:

- state store alias
- explicit store path
- handle-based DB access
- selector-like context tables

These should likely be reviewed together during DX work rather than one namespace at a time.

## Why These Shapes Matter

The DX review should not only ask:

- "does this function name read well?"

It should also ask:

- "do these tables look consistent?"
- "are the important fields exposed at the right layer?"
- "does the same concept keep changing names?"

That is why these shapes have their own reference page.
