# Harness Primitives / Standard Library (v0.15.0)

This document describes the current harness globals available in Turin’s Luau runtime.

## Result Convention (Important)

Most stdlib functions follow a Lua tuple convention:

- success: `(value, nil)` or `(true, nil)`
- failure: `(nil, "error")` or `(false, "error")`

Examples:

```lua
local text, err = fs.read("README.md")
if not text then error(err) end

local ok, err = runtime.kv.set("foo", "bar", runtime.context("agent", "coder"))
if not ok then error(err) end
```

Some functions return plain values (e.g. `time.epoch_seconds()`) or raise Lua runtime errors for invalid argument shapes.

## Verdict Constants (for hooks)

- `ALLOW`
- `REJECT`
- `ESCALATE`
- `MODIFY`

## System Globals

## `fs`

Filesystem helpers scoped to `harness.fs_root` (default: workspace root).

- `fs.read(path) -> string|nil, err?`
- `fs.write(path, content) -> bool, err?`
- `fs.exists(path) -> bool`
- `fs.is_safe_path(path) -> bool`

Notes:

- `fs.read`/`fs.write` are governance-capability gated (`fs.read`, `fs.write`) when governance enforcement is enabled.
- `fs.write` enforces a max harness write size (kernel constant, default 10MB).
- Path traversal outside `harness.fs_root` is denied.

## `json`

- `json.encode(value) -> string|nil, err?`
- `json.decode(string) -> value|nil, err?`

## `time`

- `time.epoch_seconds() -> integer`
- `time.now_utc() -> string` (Unix timestamp string, not ISO8601)

## `log`

- `log(message)`

Writes a harness-prefixed diagnostic line to stderr.

## `import(name)` and `import_scoped(name, opts?)`

Harness module import helpers.

- `import(name)` — unscoped import (governance may restrict this)
- `import_scoped(name, opts)` — scoped import with governance root assertion and optional delegated capability ceiling

`opts` (for `import_scoped`):

```lua
{
  root = "core",            -- optional if governance.import.default_root is configured
  capabilities = {           -- optional delegated ceiling (downward-only)
    ["runtime.db.query"] = true,
    ["runtime.db.exec"] = false,
    ["runtime.db.*"] = true,
  }
}
```

Notes:

- Imported module functions are wrapped so governance checks run under the imported module/root subject context.
- Nested exported tables/functions are recursively wrapped.
- Nested imports cannot widen delegated capabilities beyond the importer’s delegation.

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
local aliases, err = runtime.context.glob("agent:*")
```

## `runtime.memory`

Canonical memory API with explicit selector.

- `runtime.memory.search(query, ctx, opts?)`
- `runtime.memory.store(content, ctx, metadata?, opts?)`

`opts` for `search`:

- number (limit shorthand), or
- `{ limit = N }`

Returns rows like:

```lua
{
  { content = "...", score = 0.93 },
  ...
}
```

## `runtime.kv`

Canonical KV API with explicit selector.

- `runtime.kv.get(key, ctx) -> string|nil, err?`
- `runtime.kv.set(key, value, ctx) -> bool, err?`
- `runtime.kv.delete(key, ctx) -> bool, err?`

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

- `runtime.db.open(selector) -> handle_info|nil, err?`
- `runtime.db.close(handle_or_table) -> bool, err?`
- `runtime.db.list() -> {handle_info...}|nil, err?`
- `runtime.db.query(sql, params?, opts?) -> rows|nil, err?`
- `runtime.db.exec(sql, params?, opts?) -> changed_count|nil, err?`

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

Runtime policy influences DB behavior (`db.allow_dynamic_open`, `db.path_scope`, `db.max_open_handles`, `db.idle_close_secs`).

## `runtime.agent`

Peer-agent orchestration API.

- `runtime.agent.list() -> statuses|nil, err?`
- `runtime.agent.get_status(agent_id) -> status|nil, err?`
- `runtime.agent.submit(agent_id, task, opts?) -> task_id|nil, err?`
- `runtime.agent.await(task_id, opts?) -> result|nil, err?`

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

Governance integration:

- capability checks (`runtime.agent.submit`, `runtime.agent.await`, `runtime.agent.status`)
- `allowed_child_agents` allowlists
- delegated capability ceilings (downward-only)
- active grant ceilings may be inherited automatically

## `runtime.policy`

Runtime policy storage API.

- `runtime.policy.get(key, scope?) -> json|nil, err?`
- `runtime.policy.set(key, value, scope?) -> bool, err?`

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

## `runtime.governance`

Governance observability and temporary grants.

### Observability

- `runtime.governance.profile() -> "open"|"balanced"|...`
- `runtime.governance.snapshot(agent_id?) -> snapshot_json`
- `runtime.governance.agent(agent_id) -> snapshot_json`
- `runtime.governance.check(capability, agent_id?) -> decision_json`

### Temporary grants

- `runtime.governance.grant_issue(opts) -> grant|nil, err?`
- `runtime.governance.grant_get(grant_id) -> grant|nil, err?`
- `runtime.governance.grant_revoke(grant_id) -> bool, err?`
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
- `memory.as(ctx) -> proxy`
  - proxy methods: `search`, `store`

## `kv` (default agent-scoped KV)

Uses a default selector derived from the active agent identity.
Requires an active session context.

- `kv.get(key)`
- `kv.set(key, value)`
- `kv.delete(key)`
- `kv.as(ctx) -> proxy`
  - proxy methods: `get`, `set`, `delete`

## `session` and `user` aliases

Selector-derived scoped data aliases based on the active `RuntimeIdentity`.

- `session.memory.search/store`
- `session.kv.get/set/delete`
- `user.memory.search/store` (requires `identity.user_id`)
- `user.kv.get/set/delete` (requires `identity.user_id`)

## `agent` (top-level orchestration conveniences)

### Local queue/session helpers

- `agent.spawn(prompt, opts?) -> queue_token|nil, err?`
  - enqueues a local task in the current session queue
  - governed by `spawn.enabled`, `spawn.max_depth`, queue policy
- `agent.session.identity() -> identity_table`
- `agent.session.queue(prompt) -> bool, err?`
- `agent.session.queue_next(prompt) -> bool, err?`
- `agent.session.queue_all({prompts...}) -> bool, err?`
- `agent.session.load(session_id) -> session_row|nil, err?`
- `agent.session.list(limit?, offset?) -> rows|nil, err?`

### Peer-agent convenience

- `agent.complete(prompt, opts?) -> output|nil, err?`
  - submits to a peer agent and awaits result in one call
  - `opts.agent_id` (defaults to current configured agent id)
  - `opts.timeout_ms`
  - `opts.capabilities` (delegated ceiling)
- `agent.send(agent_id, prompt) -> bool`
  - deprecated fire-and-forget convenience path

### Mode controls

- `agent.mode.get() -> "auto"|"stateful"|"stateless"`
- `agent.mode.set(mode) -> bool, err?`

## Notes on Old Namespaces

Turin’s canonical harness API is now `runtime.*` + top-level aliases described above.
Older documentation that references `turin.*` or `db.*` globals is outdated and should be treated as historical.
