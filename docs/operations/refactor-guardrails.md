# Refactor Guardrails

This document defines the safety net for Turin's runtime/subsystem refactor.

The goal is not to freeze implementation details. Internal APIs, crate boundaries,
and config shapes can change while Turin has no active users. The goal is to make
sure the refactor preserves capabilities, security posture, persistence behavior,
and operational behavior.

## Operating Rule

Every refactor phase should have:

- a scoped capability inventory
- targeted characterization tests before broad rewrites
- focused integration tests for the changed boundary
- a perf/footprint baseline when the change touches runtime state, channels, or persistence
- one checkpoint commit after tests pass

Avoid large, mixed commits that combine behavior changes, file moves, and cleanup.

## Baseline Before Refactor

Run this once before the first major phase:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace --lib --tests
```

If disk space is tight, prefer targeted tests during iteration and keep `CARGO_TARGET_DIR=target`
for auxiliary tools so builds share the root cache.

Capture at least these perf baselines:

```bash
CARGO_TARGET_DIR=target cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  channel-scale --sessions 1 --messages-per-session 1000 --checkpoints 10,100,200,1000

CARGO_TARGET_DIR=target cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  channel-scale --sessions 2 --messages-per-session 1000 --checkpoints 10,100,200,1000 \
  --message-bytes 512 --response-bytes 2048

CARGO_TARGET_DIR=target cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  hot-history --turns 100 --payload-bytes 1048576 --sample-every 10
```

Interpret perf reports as trend baselines. RSS/PSS numbers vary by machine and build
profile; compare runs on the same machine with the same build profile.

## Capability Inventory

Before a subsystem is peeled out, list the public behavior it must preserve.

Minimum inventory:

- supported config shape
- durable state written
- daemon protocol requests/responses
- harness APIs registered
- tools exposed
- events emitted
- security or governance checks enforced
- recovery/restart behavior
- failure behavior for invalid input

This inventory should be captured in tests before moving code.

## Phase Gates

### 1. Channel Runtime

Scope:

- shared runner behavior
- channel access state
- channel session binding
- progress/streaming behavior
- daemon sidecar supervision
- platform adapter boundaries

Required tests:

- mock channel runner round trip
- same-conversation serialization
- different-conversation parallelism
- access allow/deny/pairing behavior
- channel-owned state path behavior
- progress updates before final response
- invalid channel config isolation
- daemon runtime status and restart/error reporting

Perf baselines:

- `channel-scale` with 1, 2, and 5 logical sessions
- tiny metadata baseline with `PONG`
- realistic text baseline with configured `--message-bytes` and `--response-bytes`

### 2. Bounded Hot History

Scope:

- in-memory session history
- persisted transcript source of truth
- request context reconstruction
- old tool-result compaction
- idle runtime release

Required tests:

- persisted session can resume after hot window eviction
- tool-call adjacency is preserved across window boundaries
- branch checkout can rematerialize older turns
- large tool results do not remain hot beyond configured policy
- debug/high-memory profile can keep larger history
- context compaction remains deterministic enough for existing behavior

Perf baselines:

- `hot-history` with large file/tool payloads
- repeated channel messages with realistic response size
- idle-after-work snapshot once idle release exists

### 3. Scheduler

Scope:

- recurrence calculation
- overlap policy
- dispatch into runtime tasks
- durable job/run records
- harness namespace registration

Required tests:

- recurrence never moves backwards
- overlap policies preserve current semantics
- scheduled tasks persist and resume across daemon restart
- disabled/invalid jobs fail isolated
- harness schedule APIs remain available

### 4. Worklists

Scope:

- worklist schema
- claims/heartbeats/stale recovery
- dependencies and hierarchy
- prompt/action payload dispatch
- harness namespace registration

Required tests:

- claim lifecycle
- stale claim recovery
- dependency blocking/unblocking
- prompt/action dispatch behavior
- daemon/task integration
- invalid worklist state recovery

### 5. Memory

Scope:

- memory record storage
- lexical/semantic/hybrid recall
- feedback/correct/purge lifecycle
- `remember`/`recall` tools
- harness memory namespace

Required tests:

- exact KV behavior remains exact
- memory search behavior remains scoped
- purge/correct lifecycle is durable
- embedding-disabled lexical fallback still works
- governance and tool policy still gate memory writes

### 6. MCP / External Tools

Scope:

- MCP client/process lifecycle
- tool exposure
- governance/tool policy
- shutdown and cleanup

Required tests:

- MCP remains opt-in
- denied MCP tools are not exposed
- subprocess cleanup on drop/shutdown
- invalid MCP config fails isolated
- audit events include external tool activity

## Security Regression Set

These tests are high priority and should not be weakened during refactor:

- path traversal fails for filesystem tools
- symlink escape behavior remains safe
- denied tools cannot execute
- child/peer capabilities cannot exceed parent ceilings
- imports cannot bypass governance ceilings
- unauthorized channel users are denied or queued for pairing
- oversized/invalid inbound channel payloads fail explicitly
- remote bridge requires explicit secure configuration
- high-risk tool execution remains auditable

## Test Placement Rules

Use these placement rules while splitting god files:

- private pure helper tests can stay in-module
- public behavior tests should move to `tests/` or crate integration tests
- fixture-heavy channel tests should live outside production adapter files
- reusable setup belongs beside the owning integration suite or in crate-local `test_support`
- capability characterization tests should be named for behavior, not implementation

## Checkpoint Commit Pattern

For each phase:

1. add or strengthen tests around current behavior
2. commit: `Add <subsystem> characterization tests`
3. move/split code without changing behavior
4. run targeted tests
5. commit: `Extract <subsystem> boundary`
6. simplify/DRY after the move
7. run targeted tests plus relevant perf baseline
8. commit: `Simplify <subsystem> internals`

If a behavior intentionally changes, update the capability charter, docs, and tests in
the same checkpoint.
