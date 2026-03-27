# Governance and Capabilities

Turin’s governance system is **opt-in** and **flexibility-first**.

It is designed to let you run:

- fully open autonomous agents in isolated environments
- tightly governed agents with explicit capability ceilings and audit guarantees
- mixed systems where trusted and self-evolving harness modules coexist

## Guiding Principles

1. **Security is opt-in, not mandatory.**
2. **Dangerous behavior must be possible when intentionally configured.**
3. **Restrictions should be explicit and observable.**
4. **Delegation is downward-only (no privilege amplification).**
5. **Provider-specific quirks do not belong in governance logic.**

## Governance Profiles

Configured in `[governance]`:

```toml
[governance]
profile = "open"         # open | balanced | governed | custom
enforcement_enabled = false
```

### `open`

- Maximum flexibility
- Governance checks may still be observable, but enforcement is commonly disabled
- Good for experimentation and isolated environments

### `balanced`

- Safer default capability posture
- Designed for day-to-day use where some restrictions are helpful
- Still overrideable with explicit config/runtime policy choices

### `governed`

- Strict capability enforcement posture
- Intended for controlled environments and stronger audit expectations
- Commonly paired with scoped imports and immutable audit mode

### `custom`

- Use your own capability profile tables and governance config
- Turin still provides the enforcement engine; you define the shape

## Enforcement Toggle

Governance capability checks are only enforced when enabled:

```toml
[governance]
enforcement_enabled = true
```

When `false`, Turin remains open/flexible while still exposing governance observability APIs.

## Capability Model

Capabilities are strings such as:

- `runtime.db.query`
- `runtime.db.exec`
- `runtime.agent.submit`
- `runtime.policy.set`
- `runtime.governance.grant.issue`
- `fs.read`
- `fs.write`
- `shell.exec`
- `harness.import.scoped`
- `harness.import.unscoped`

Wildcard rules are supported in many places using `prefix.*`, e.g.:

- `runtime.db.*`
- `runtime.agent.*`

## Current Capability Registry (Comprehensive)

This section lists the **currently enforced/checked capability names** in the current Turin baseline.
If a capability is not listed here, Turin may still expose related functionality, but it is not
currently checked by the governance enforcement path as a named capability.

### Runtime DB

- `runtime.db.open`
- `runtime.db.close`
- `runtime.db.list_handles`
- `runtime.db.query`
- `runtime.db.exec`

### Runtime Agent

- `runtime.agent.status`
  - used by `runtime.agent.list(...)`
  - used by `runtime.agent.get_status(...)`
- `runtime.agent.submit`
- `runtime.agent.await`
- `runtime.agent.spawn`
  - used by top-level alias `agent.spawn(...)`

### Runtime Policy

- `runtime.policy.set`

Notes:
- `runtime.policy.get` is intentionally readable without a governance capability gate in the current design.

### Runtime Governance (Temporary Grants)

- `runtime.governance.grant.issue`
- `runtime.governance.grant.get`
- `runtime.governance.grant.revoke`
- `runtime.governance.grant.use`

Notes:
- `runtime.governance.profile/snapshot/check/agent` are observability APIs and are not capability-gated in the current design.

### Harness Import

- `harness.import.unscoped`
- `harness.import.scoped`

These apply at the import / behavior-mount boundary (`import(...)`, `import_scoped(...)`, `use(...)`, `use_scoped(...)`) when governance enforcement is enabled.

### Filesystem and Built-in Tools

- `fs.read`
- `fs.write`
- `shell.exec`

Usage:
- `fs.read` and `fs.write` gate top-level harness `fs.*` calls.
- Kernel built-in tool fallback maps:
  - `read_file` -> `fs.read`
  - `write_file` / `edit_file` / `apply_patch` -> `fs.write`
  - `shell_exec` -> `shell.exec`

### Useful Wildcard Prefixes

Wildcard rules are supported for many capability tables and profiles:

- `runtime.db.*`
- `runtime.agent.*`
- `runtime.governance.grant.*`
- `harness.import.*`

Wildcard behavior follows longest-prefix matching (`prefix.*`) and composes with profile/root/agent/import/grant ceilings.

## Effective Capability Evaluation (Conceptual)

Turin evaluates capabilities against a **subject** and multiple ceilings.
A practical mental model is:

`effective = profile ∩ agent_ceiling ∩ root_ceiling ∩ import_delegation ∩ active_grant ∩ per-call delegation`

Not every term is always present. The important rule is:

- **authority only narrows as you go down the stack**

## Governance Subject Context

Capability checks may be attributed to more than just an agent.
Turin tracks context such as:

- agent id
- active harness module name
- active harness root name (for scoped imports)
- active import delegated capabilities
- active temporary grant id

This matters for import scoping, grants, and auditability.

## Import And Block Governance (`import` / `use`)

Configured in `[governance.import]`:

```toml
[governance.import]
mode = "scoped"                # legacy | mixed | scoped
default_root = "core"          # optional
allow_unscoped_in_open = true   # open profile escape hatch
```

### Modes

#### `legacy`

- `import(...)` allowed
- `use(...)` allowed
- `import_scoped(...)` disabled
- `use_scoped(...)` disabled
- Best for older or simple harnesses

#### `mixed`

- both `import(...)` / `import_scoped(...)` and `use(...)` / `use_scoped(...)` allowed
- good migration path

#### `scoped`

- unscoped `import(...)` / `use(...)` can be disabled
- `import_scoped(...)` / `use_scoped(...)` require `opts.root` or `default_root`
- best for compartmentalized harness systems

Relevant capability names:

- `harness.import.unscoped`
- `harness.import.scoped`
- `harness.use.unscoped`
- `harness.use.scoped`

### Governance Roots

Define roots in `[governance.roots]`:

```toml
[governance.roots.core]
path = ".turin/harnesses/core"
writable_hint = false
default_profile = "balanced"

[governance.roots.plugins]
path = ".turin/harnesses/plugins"
writable_hint = true
max_capabilities = { "runtime.db.query" = true, "runtime.db.exec" = false, "fs.write" = true }
```

Notes:

- `path` is used for root attribution (longest matching configured path wins)
- `writable_hint` is metadata/intent (useful for documentation and policy design)
- `max_capabilities` can ceiling imported modules from that root

### Import-Scoped Delegation

You can constrain imported modules further at the import call site:

```lua
local plugin = import_scoped("plugins/indexer", {
  root = "plugins",
  capabilities = {
    ["runtime.db.query"] = true,
    ["runtime.db.exec"] = false,
    ["fs.read"] = true,
    ["fs.write"] = true,
  }
})
```

Turin enforces:

- delegated values must be booleans
- nested imports cannot widen beyond importer delegation
- wildcard rules (`prefix.*`) are supported
- imported calls run under imported module/root subject attribution

## Agent Governance

Configured in `[governance.agents]`.
These rules apply to peer-agent dispatch and runtime actions attributed to those agents.

```toml
[governance.agents.reviewer]
capability_profile = "reviewer_read_only"
allowed_child_agents = []
max_capabilities = { "runtime.db.exec" = false }

[governance.agents.coder]
capability_profile = "coder"
allowed_child_agents = ["reviewer"]
```

### Capability profiles

Define named capability profiles in `[governance.capability_profiles]`:

```toml
[governance.capability_profiles.reviewer_read_only]
"runtime.db.query" = true
"runtime.db.exec" = false
"fs.read" = true
"fs.write" = false
"runtime.policy.set" = false

[governance.capability_profiles.coder]
"runtime.db.*" = true
"fs.read" = true
"fs.write" = true
```

### `allowed_child_agents`

If non-empty, it acts as an allowlist for peer-agent dispatch from that agent.
Turin enforces this on:

- `runtime.agent.submit(...)`
- `agent.complete(...)`
- `agent.send(...)`

## Temporary Grants

Temporary grants are optional, kernel-managed capability ceilings that can be activated for a short window.

Enable them in config:

```toml
[governance.grants]
enabled = true
max_ttl_ms = 60000
require_audit_reason = true
```

Runtime APIs (capability-gated):

- `runtime.governance.grant_issue(opts)`
- `runtime.governance.grant_get(grant_id)`
- `runtime.governance.grant_revoke(grant_id)`
- `runtime.governance.with_grant(grant_id, fn)`
- `runtime.governance.grant(spec, fn)` (DX wrapper over issue/use/revoke)

Example:

```lua
local grant, err = runtime.governance.grant_issue({
  capabilities = { ["runtime.db.exec"] = true },
  ttl_ms = 10000,
  max_uses = 1,
  reason = "one-shot cleanup",
})

if grant then
  runtime.governance.with_grant(grant.grant_id, function()
    local changed, derr = runtime.db.exec("delete from temp where stale = 1")
    if not changed then error(derr) end
  end)
end
```

Grant properties:

- subject-scoped
- TTL enforced
- `max_uses` enforced
- auditable issue/use/revoke events
- active grant ceilings propagate to peer-agent delegation paths (downward-only)

### DX governance helpers

The DX layer adds a few helpers that are governance-relevant but do not change enforcement semantics:

- `allowed(capability[, opts]) -> boolean`
- `needs(capability[, opts]) -> true | error`
- `access.check(capability[, opts]) -> decision_table`
- `runtime.governance.grant(spec, fn)`

Example:

```lua
if not allowed("runtime.db.exec") then
  return verdict.reject("db exec denied")
end

local output = runtime.governance.grant({
  ttl_ms = 5000,
  capabilities = {
    ["runtime.agent.submit"] = true,
    ["runtime.agent.await"] = true,
  },
}, function()
  return runtime.agent("reviewer"):complete("Review this patch")
end)
```

Important semantics:

- `needs(...)` raises a Lua runtime error on denial
- `runtime.governance.grant(...)` is only convenience syntax; capability enforcement remains unchanged
- `runtime.governance.grant(...)` prioritizes callback errors over revoke errors

## Audit Modes

Configured in `[governance.audit]`:

```toml
[governance.audit]
mode = "immutable"                 # off | observational | immutable
include_capability_context = true
persist_before_hooks = true         # optional override (defaults based on mode)
```

### `off`

- no special audit behavior beyond normal event persistence

### `observational`

- governance snapshot/capability context can be surfaced for observability
- `on_kernel_event` still behaves normally

### `immutable`

- audit events are persisted before `on_kernel_event`
- `REJECT` from `on_kernel_event` cannot suppress already-protected audit persistence
- gives stronger trust guarantees without removing harness flexibility elsewhere

## What Governance Currently Enforces

When `enforcement_enabled = true`, Turin applies capability checks in multiple layers.

### Stdlib layer (primary)

Examples:

- `runtime.db.*`
- `runtime.agent.*`
- `runtime.policy.set`
- `runtime.governance.grant_*`
- `fs.read`, `fs.write`
- top-level `agent.*` peer dispatch paths

### Kernel tool fallback (defense in depth)

High-risk built-in tools are also checked at execution time, including:

- `read_file` -> `fs.read`
- `write_file` / `edit_file` / `apply_patch` -> `fs.write`
- `shell_exec` -> `shell.exec`

This prevents bypass if a model emits direct tool calls that do not pass through the stdlib.

## Runtime Policy Knobs (Selected, Current)

Runtime policy is intentionally flexible and can remain harness-mutable when you choose to allow
`runtime.policy.set`. Relevant current knobs include:

- `spawn.enabled`
- `spawn.max_depth`
- `mode.default`
- `db.allow_dynamic_open`
- `db.path_scope`
- `db.max_open_handles`
- `db.idle_close_secs`
- `queue.max_depth`
- `tool.exec_enabled`
- `hook.token_usage.reject_mode`
  - `informational` (default)
  - `enforce_task`
  - `enforce_session`

This hook token-usage mode is a good example of Turin’s philosophy:
- default remains flexible/informational
- stricter enforcement is opt-in via policy

## Runtime Observability APIs

Use these from harness code to inspect effective governance state:

```lua
local profile = runtime.governance.profile()
local snapshot = runtime.governance.snapshot()
local agent_view = runtime.governance.agent("reviewer")
local decision = runtime.governance.check("runtime.db.exec")
```

These are useful for adaptive behavior (e.g., degrade gracefully when a capability is unavailable).

## Recommended Patterns

### 1. Open by default, tighten by deployment

- local/dev sandbox: `profile = "open"`, `enforcement_enabled = false`
- shared/dev infra: `balanced`
- production/controlled workflows: `governed` + scoped imports + immutable audit

### 2. Split harness roots by trust level

- `core` (read-only, user-controlled)
- `plugins` (writable, self-evolving)
- use `import_scoped` to enforce root and delegation ceilings

### 3. Use per-agent capability profiles

Separate agents by role instead of relying only on prompt discipline:

- `planner` (read-only, no shell)
- `coder` (fs write + tooling)
- `reviewer` (read-only)
- `deployer` (temporary grants only)

### 4. Prefer temporary grants over permanent broad allowances

Use grants for short-lived elevated actions rather than widening persistent capabilities.

## Configuration Example (Governed Setup)

```toml
[governance]
profile = "governed"
enforcement_enabled = true

[governance.audit]
mode = "immutable"
include_capability_context = true

[governance.import]
mode = "scoped"
default_root = "core"

[governance.roots.core]
path = ".turin/harnesses/core"
writable_hint = false

[governance.roots.plugins]
path = ".turin/harnesses/plugins"
writable_hint = true
max_capabilities = { "runtime.db.query" = true, "runtime.db.exec" = false, "fs.write" = true }

[governance.capability_profiles.reviewer]
"runtime.db.query" = true
"runtime.db.exec" = false
"fs.read" = true
"fs.write" = false
"shell.exec" = false

[governance.agents.reviewer]
capability_profile = "reviewer"
allowed_child_agents = []

[governance.grants]
enabled = true
max_ttl_ms = 60000
require_audit_reason = true
```

## Limitations and Notes

- Governance is intentionally opt-in; if enforcement is off, capability checks are observational only.
- Runtime policy (`runtime.policy.set`) remains powerful by design, but can be gated by capability checks and profile/agent/root ceilings.
- Some runtime behavior is still controlled by policy keys and config rather than a fully centralized knob registry; Turin is moving toward increasingly explicit knobs over time.
