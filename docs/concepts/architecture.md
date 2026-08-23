# Turin Architecture

This document describes Turin’s current architecture after the canonical stdlib, multi-db/multi-agent runtime, and governance refactors.

For the non-obvious decisions behind the current shape, see `docs/adr/index.md`.

Current architectural shape:

- execution context is explicit rather than implicitly following one session-global head
- turns and branch heads remain the structural execution substrate
- semantic relationships that do not belong in the structural turn graph live in the sparse graph overlay

## Design Principles

1. **Kernel is execution physics, not policy.**
2. **Harness scripts define behavior.**
3. **Provider logic stays out of Turin core.**
4. **Capabilities and governance are opt-in overlays.**
5. **Internal architecture can refactor aggressively; public harness surfaces should evolve more deliberately.**

## High-Level Layers

### 1. Kernel (Rust)

The kernel owns:

- runtime composition and top-level control flow
- watcher ownership and harness reload orchestration
- peer-agent runtime orchestration
- runtime policy storage and governance evaluation

It does **not** embed workflow policy or prompt logic.

Execution-heavy behavior now lives primarily under `ExecutionHost`, which owns:

- session lifecycle
- task/plan loop orchestration
- turn execution and stream processing
- tool execution and auditing
- persistence and event durability
- harness hook invocation
- provider client initialization/use
- MCP runtime ownership for the current execution host

### 2. Harness Engine (Luau via `mlua`)

The harness engine:

- loads and composes Lua scripts/modules
- evaluates lifecycle hooks and returns verdicts
- exposes the stdlib/global runtime API
- hot-reloads harnesses atomically
- maintains execution context for governance subject attribution (module/root/import delegation/grants)

### 3. Provider SDK Layer (`inference-sdk-rust`)

Turin depends on normalized provider SDKs for:

- provider-specific request/response encoding
- streaming event normalization (`InferenceEvent`)
- compatibility quirks (Anthropic-compatible providers, etc.)

Turin maps normalized `InferenceEvent` values into `KernelEvent` stream events and remains provider-agnostic.

## Daemon Layer

Turin now also has a daemon/control layer built on top of the kernel.

The daemon owns:

- filesystem-backed registry scanning (`agents/`, `harnesses/`)
- live kernel replacement on daemon-level rescan
- Unix-socket NDJSON request/response handling
- event subscription streaming
- agent/harness/session/task control APIs

Key modules:

- `src/daemon/registry.rs`
  - scans filesystem-backed daemon state
  - synthesizes effective runtime config from bootstrap config plus daemon-managed agent/harness directories
- `src/daemon/state.rs`
  - owns live daemon state
  - mediates filesystem mutations, rescans, and runtime-facing control operations
- `src/daemon/server.rs`
  - Unix-socket server
  - NDJSON protocol dispatch
  - daemon watcher and event broadcast stream
- `src/daemon/protocol.rs`
  - request/response/event wire envelopes

Design rule:

- the filesystem is the persisted source of truth
- the daemon is the validated control surface over that filesystem state

Advanced users can still edit files directly. The daemon is expected to tolerate that and isolate bad agents/harnesses locally.

## Core Runtime Flows

### Session / Task / Turn Lifecycle

1. `Kernel::create_session()`
2. `Kernel::start_session()`
   - emits `session_start`
   - emits `governance_snapshot` audit event
   - calls `on_session_start`
3. `Kernel::run()`
   - dequeues tasks
   - emits `task_start`
   - calls `on_task_start`
   - runs task turn loop
4. Per turn:
   - emits `turn_start`
   - calls `on_turn_start`
   - builds mutable context and calls `on_turn_prepare`
   - streams provider events
   - executes tools if requested
   - emits `turn_end`
   - calls `on_turn_end`
5. Task terminal:
   - emits `task_complete`
   - calls `on_task_complete`
   - may emit `plan_complete`
   - may call `on_plan_complete`
6. Queue drained:
   - emits `all_tasks_complete`
   - calls `on_all_tasks_complete`
7. `Kernel::end_session()`
   - emits `session_end`
   - calls `on_session_end`
   - flushes durability lane

### Event Pipeline

Every significant action becomes a `KernelEvent`:

- `LifecycleEvent`
- `StreamEvent`
- `AuditEvent`

`persist_event()` performs:

1. optional protected audit pre-persist (immutable audit mode / `persist_before_hooks`)
2. `on_kernel_event` hook observation/interception
3. standard broadcast + durability persistence

### Tool Execution Pipeline

The tool execution path is split into dedicated `kernel::turn` submodules and includes:

- harness `on_tool_call` verdict evaluation
- optional user approval (`ESCALATE`)
- built-in tool execution
- audit events (`tool_exec_start`, `tool_exec_end`, `tool_result`)
- `on_tool_result` hook with `MODIFY` support before reinjection
- plan submission handling (`submit_plan`) with `on_plan_submit`
- governance enforcement fallback at kernel tool execution for high-risk built-in tools

## Module Layout (Current)

### Harness

- `crates/turin-harness-lua/src/harness/engine.rs`
  - script loading, module metadata, hook evaluation, hot reload
- `crates/turin-harness-lua/src/harness/globals.rs`
  - thin registration entrypoint + shared app state (`HarnessAppData`)
- `crates/turin-harness-lua/src/harness/context.rs`
  - `on_turn_prepare(turn)` userdata wrapper and request overrides
- `src/harness/stdlib/*`
  - canonical stdlib modules and shared binding helpers

#### `src/harness/stdlib/*` highlights

- `runtime_bindings.rs` — assembles `runtime.*`
- `runtime_context.rs` — `runtime.context` selector builder + glob
- `runtime_data.rs` — `runtime.memory`, `runtime.kv`
- `runtime_db.rs` — dynamic DB handles and SQL APIs
- `runtime_agent.rs` — peer-agent submit/await/status
- `runtime_policy.rs` — runtime policy read/write
- `runtime_governance.rs` — governance observability + grants
- `agent_bindings.rs` — top-level `agent.*` aliases (queue + peer convenience)
- `memory_kv_bindings.rs` — top-level `memory.*`, `kv.*`
- `session_user_aliases.rs` — scoped aliases for `session.*` / `user.*`
- `system_globals.rs` — `fs`, `json`, `time`, `log`, `import`, `import_scoped`, `use`, `use_scoped`, `watch`

### Kernel

- `src/kernel/mod.rs`
  - thin kernel shell + runtime composition/root wiring
- `src/kernel/execution_host.rs`
  - shared execution-owned state and agent/harness resolution helpers
- `src/kernel/init.rs`
  - execution-host initialization + kernel watcher startup
- `src/kernel/harness_manager.rs`
  - named harness registry, agent->harness binding, runtime lookup
- `src/kernel/harness_runtime.rs`
  - per-harness engine/app-data/watch-root lifecycle
- `src/kernel/session_lifecycle.rs`
  - execution-host-owned create/start/end session logic
- `src/kernel/event_persistence.rs`
  - execution-host-owned event broadcast + durability lane + `on_kernel_event`
- `src/kernel/harness_hooks.rs`
  - execution-host-owned `on_tool_call` / `on_token_usage` helpers
- `src/kernel/run_loop.rs`
  - execution-host-owned task queue orchestration and `run()` entrypoint
- `src/kernel/task_execution.rs`
  - execution-host-owned task execution orchestration
- `src/kernel/task_lifecycle.rs`
  - execution-host-owned task completion, inference error handling, plan completion callbacks
- `src/kernel/turn/*`
  - turn preflight, streaming, assistant finalization, tool execution pipeline
- `src/kernel/agent_manager/*`
  - peer runtime registry, peer task execution, result tracking
- `src/kernel/governance.rs`
  - profiles, capability evaluation, subjects, grants, snapshots
- `src/kernel/policy.rs`
  - runtime policy manager/storage
- `src/kernel/mcp_runtime.rs`
  - MCP runtime integration helpers

### Persistence

- `src/persistence/state.rs`
  - session/message/event/tool/KV/memory persistence
- `src/persistence/manager.rs`
  - public store manager surface
- `src/persistence/manager/path_support.rs`
  - selector/path resolution and path safety helpers
- `src/persistence/manager/cache_support.rs`
  - store handle cache/open/trim/eviction logic

### Code Search

Turin’s code search is intentionally split across two binaries/crates:

- `turin`
  - owns the runtime read/query path
  - exposes `runtime.code.search.*` and DX wrappers like `code.find(...)`
  - validates `index_meta` plus stable read views before querying
  - negotiates lexical/semantic/hybrid behavior at query time
- `turin-map`
  - owns the index write/update/remove/rebuild lifecycle
  - walks the filesystem, applies ignore rules, chunks code, and enriches embeddings
  - writes a root-path-first code index DB that Turin can read directly
- `turin-code-index`
  - shared contract/read/write crate used by both binaries
  - keeps the runtime/query contract and write-path schema aligned

Key design rules:

- codebase discovery is root-path-first, not `codebase_id`-first
- `codebase_id` is optional metadata only
- Turin never shells out to `turin-map` per query
- contract validation failures are hard errors, not silent fallback
- semantic storage format is observable fact (`vector_format`) rather than hidden implementation detail

Current code-search quality behavior:

- lexical search uses a mixed strategy
  - deterministic definition-oriented ranking for symbol lookups
  - Tantivy/Turso FTS for multi-token natural-language and path-like queries
- semantic search uses shared embedding-provider infrastructure from `inference-sdk-rust`
- hybrid search uses explicit RRF fusion
- `trace = true` exposes effective mode, fallback reason, candidate ranks, and fusion metadata for tuning/debugging

## Data Model and Identity

### Runtime Identity

`RuntimeIdentity` carries subject identity across hooks/events/runtime APIs:

- `session_id` (always)
- `agent_id` (always for Turin-managed sessions)
- optional: `user_id`, `channel_id`, `tenant_id`, `run_id`
- `extra` map for future/extended identity dimensions

Identity is included in lifecycle hook payloads and durable lifecycle events.

### Context Selectors (Memory/KV/Data)

Canonical scoped data APIs use a selector shape:

```lua
{
  tags = { "agent:coder", "tenant:acme" },
  namespace = "default",
  visibility = "private"
}
```

Selectors are normalized and converted to store aliases for persistence lookup.

## Multi-DB Architecture

Turin now supports multiple logical state stores via `StoreManager`.

### Store Selection

A store can be selected by:

- alias (`"state"`, custom alias names)
- path (`"scratch/test.db"`)
- handle (`{ handle = "..." }`)
- selector-derived alias (`{ selector = {...} }`)

### Store Handle Model

`runtime.db.open(...)` returns a handle record including:

- `handle`
- `path`
- `alias` (if any)
- `open_count`
- `idle_ms`

The store manager tracks handles and trims idle/open caches according to runtime policy.

## Multi-Agent Architecture

Peer agents are managed by `AgentManager`.

### Runtime Registry

Each peer agent runtime is started lazily and tracked in a runtime registry.

Features:

- async task submission (`submit`)
- optional fire-and-forget send (`send`)
- result awaiting (`await_result`)
- status listing/inspection
- idle shutdown and restart

Peer runtime execution is now centered in `src/kernel/agent_manager/peer_runtime.rs`, where a peer runtime owns an `ExecutionHost` rather than a full `Kernel`. That keeps peer sessions on the same execution model as direct sessions while still letting the top-level kernel own watcher/control concerns.

## Multi-Harness Reload Model

- Each configured harness is represented by its own `HarnessRuntime`.
- File watching is configured from:
  - the harness directory itself
  - any explicit `watch(...)` roots declared during harness load
- A file change now reloads only the owning harness runtime(s), not every runtime in the process.
- After reload, watcher roots are rebuilt from the reloaded harness graph so changes to `watch(...)` declarations take effect without restart.

This keeps reload semantics simple:

- reload scope is per harness runtime
- reload operation is still full-runtime and atomic
- no partial file-level hot patching is attempted

### Delegation and Governance Integration

Peer dispatch can carry delegated capability ceilings. Effective authority is constrained by:

- governance profile / enforcement state
- caller subject capabilities
- caller grants (if active)
- agent max capabilities / profiles
- child-agent allowlists

## Governance Architecture (Opt-In)

Governance is implemented as an overlay, not a rewrite of kernel behavior.

### Key Concepts

- **Profile** (`open`, `balanced`, `governed`, `custom`)
- **Capability checks** (`runtime.db.exec`, `shell.exec`, etc.)
- **Subject context** (agent/module/root/import delegation/grant)
- **Import policy** (`legacy`, `mixed`, `scoped`)
- **Agent ceilings** (`capability_profile`, `max_capabilities`, `allowed_child_agents`)
- **Temporary grants** (TTL/max uses, auditable)

### Enforcement Layers

1. **Stdlib binding layer** (primary, ergonomic errors)
2. **Kernel tool execution fallback** (defense-in-depth)
3. **Import proxy scoping** (module/root/delegation attribution)

### Audit Modes

- `off`
- `observational`
- `immutable`

In immutable mode (or with `persist_before_hooks=true`), audit events persist before `on_kernel_event` can reject them.

## Provider-Agnostic Compatibility Strategy

Turin must remain provider-agnostic.

- Turin consumes normalized SDK events (`InferenceEvent`) and emits `KernelEvent` stream events.
- Provider-specific quirks (Anthropic-compatible variations, request normalization issues, etc.) are fixed in `inference-sdk-rust`.
- Turin preserves richer normalized data (thinking blocks and signatures) so the SDK can roundtrip provider-specific requirements without Turin branching by provider.

## Testing and Validation Layers

Turin uses multiple validation layers:

- unit tests and integration tests (`cargo test`)
- static linting (`cargo clippy --all-targets -- -D warnings`)
- release builds (`cargo build --release`)
- manual live provider smoke tests (`scripts/live_minimax_smoke.sh`)

See `docs/operations/testing.md` and `docs/operations/live-provider-testing.md`.
