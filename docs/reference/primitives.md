# Harness Primitives / Standard Library

This document describes the current harness globals available in Turin’s Luau runtime.
This is the **canonical API surface reference** for harness authors (including both
the preferred `runtime.*` namespaces and the ergonomic top-level aliases).

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

## DX Language Layer

Turin also ships a first-party DX layer in `src/harness/dx/`.

Important:

- `runtime.*` remains the canonical primitive layer
- DX helpers are ergonomic wrappers over those primitives
- DX helpers do not bypass governance/capability enforcement
- DX helpers often raise Lua runtime errors on denied/invalid operations instead of returning `(nil, err)`

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
- `cache.file(path, opts?)`
- `code.find(query, opts?)`

Notes:

- `remember(...)` / `recall(...)` use the same default agent-scoped memory as `memory.*`
- `cache.file(...)` is a thin wrapper over `runtime.cache.read(...)`
- `code.find(...)` is a thin wrapper over `runtime.code.search.hybrid(...)`
- `code.find(...)` defaults to the Turin workspace root and accepts `opts.root` / `opts.index_path` overrides

Example:

```lua
remember("Compiler errors should stay concise")
local file = cache.file("SPEC.md", { include_content = true })
local rows = code.find("capability decision")
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
- `agent:complete(prompt, opts?) -> output_string`

Notes:

- `:complete(...)` delegates to the canonical `runtime.agent.complete(...)` primitive
- peer-agent governance, child-agent allowlists, delegated capability ceilings, and active grant ceilings still apply

### DX `runtime.governance.grant(...)`

- `runtime.governance.grant(spec, fn) -> <fn returns> | error`

`spec` shape:

```lua
{
  capabilities = { ["runtime.db.query"] = true },
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

### DX `time`

- `time.since(ts) -> number`
- `time.after(ts, threshold) -> boolean`

Current semantics:

- `ts` may be a number or numeric string
- values are interpreted as Unix epoch seconds
- `time.since(...)` returns elapsed seconds as a Lua number
- `time.after(...)` compares elapsed seconds against the given threshold in seconds

### DX `fs`

- `fs.read_json(path, opts?) -> value | error`
- `fs.write_json(path, value, opts?) -> true | error`

`opts` for `write_json`:

```lua
{ pretty = true }
```

Notes:

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
local aliases, err = runtime.context.glob("agent:*")
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

- `runtime.kv.get(key, ctx, opts?) -> string|nil, err?`
- `runtime.kv.set(key, value, ctx, opts?) -> bool, err?`
- `runtime.kv.delete(key, ctx, opts?) -> bool, err?`

`opts` for KV:

- `{ store = "alias"|selector_table, path = "relative/or/absolute.db" }`

## `runtime.cache`

Session-aware content cache API.

- `runtime.cache.read(path, opts?) -> cache_row|nil, err?`
- `runtime.cache.invalidate(path, opts?) -> bool, err?`
- `runtime.cache.stats(opts?) -> stats|nil, err?`
- `runtime.cache.reset(opts?) -> report|nil, err?`

Notes:

- `runtime.cache.read(...)` inherits the same `fs.read` capability and safe-path rules as `fs.read(...)`
- `opts` for `read` include `session_id`, `include_content`, `include_previous`, `max_diff_lines`, and `token_estimate`
- `invalidate`, `stats`, and `reset` are governed separately under `runtime.cache.*`

## `runtime.code`

Root-path-first code search API backed by `turin-map` indexes.

- `runtime.code.search.status(codebase, opts?) -> status|nil, err?`
- `runtime.code.search.lexical(codebase, query, opts?) -> rows|nil, err?`
- `runtime.code.search.semantic(codebase, query, opts?) -> rows|nil, err?`
- `runtime.code.search.hybrid(codebase, query, opts?) -> rows|nil, err?`

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
- `runtime.agent.complete(agent_id, prompt, opts?) -> output|nil, err?`

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

`opts` for `complete`:

```lua
{
  title = "peer review",
  timeout_ms = 30000,
  capabilities = {
    ["runtime.db.query"] = true,
    ["runtime.db.exec"] = false,
  }
}
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

Common runtime policy keys (current):

- `spawn.enabled`
- `spawn.max_depth`
- `mode.default`
- `db.allow_dynamic_open`
- `db.path_scope`
- `db.max_open_handles`
- `db.idle_close_secs`
- `queue.max_depth`
- `tool.exec_enabled`
- `hook.token_usage.reject_mode` (`informational` | `enforce_task` | `enforce_session`)

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

## `cache`

- `cache.file(path, opts?)`

Thin wrapper over `runtime.cache.read(...)`.

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
