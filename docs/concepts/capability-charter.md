# Capability Charter

Turin is a programmable runtime for AI agents. Its central promise is that model output does not directly become system action. Inference proposes, the harness decides, and the kernel enforces.

This charter defines the capabilities Turin intends to preserve and the boundaries that should remain explicit as the codebase evolves.

## Core Promise

Turin should provide:

- durable agent sessions and turns
- explicit tool execution
- programmable harness policy
- typed events and audit records
- local-first persistence
- configurable provider routing
- inspectable daemon operation
- optional channel sidecars
- capability-aware governance

The runtime may change internal structure aggressively, but these capabilities should not disappear accidentally.

## Capability Surfaces

### Inference

The inference layer proposes text, structured output, and tool calls.

Rules:

- provider-specific quirks belong in provider SDKs or provider adapters
- Turin should consume normalized inference events
- provider selection should be configuration-driven
- context-window behavior should be explicit and observable

### Harness

The harness is the programmable policy and workflow layer.

It owns:

- lifecycle hooks
- tool-call verdicts
- context preparation
- workflow orchestration
- memory and state policy
- governance-aware imports

Rules:

- harness scripts should run inside a sandboxed Luau VM
- harness APIs should be explicit and documented
- imports and delegated capabilities should not bypass governance ceilings
- high-risk behavior should be auditable

### Kernel

The kernel is execution physics.

It owns:

- session lifecycle
- task and turn execution
- tool execution pipeline
- event emission
- persistence coordination
- enforcement fallback for high-risk operations

Rules:

- workflow policy should not be hard-coded into the kernel
- kernel events should be durable and inspectable
- the kernel should remain small enough to reason about

### Tools

Tools are explicit external-action capabilities.

Current built-in tool risk classes:

- read: `read_file`, `recall`
- write: `write_file`, `edit_file`, `apply_patch`, `remember`, `submit_plan`
- process: `shell_exec`
- network: `web_fetch`, `web_search`
- integration: `bridge_mcp`

Rules:

- default tool exposure must be documented and tested
- child scopes must not expand beyond parent grants
- process and integration tools should be treated as high-risk
- tool execution should be observable through events and persistence

Current code default: `apply_patch` and `bridge_mcp` are opt-in, while `shell_exec` is default-exposed. If this changes, docs and characterization tests should change together.

### Persistence

Persistence is not just storage; it is the durable execution record.

It owns:

- sessions
- turns
- messages
- tool executions
- events
- branches
- KV/state
- memory records
- subsystem-specific durable state

Rules:

- persisted state should be typed where practical
- schema changes should be intentional
- store placement should be explicit
- audit-relevant events should survive process restarts

### Daemon

The daemon is the long-running control plane.

It owns:

- filesystem-backed runtime registry
- live runtime state
- task/session/channel control
- local IPC protocol
- event subscriptions
- sidecar supervision

Rules:

- filesystem state remains the authoritative dynamic registry
- invalid agents/harnesses/channels should be isolated, not globally fatal
- protocol surfaces should be typed and versioned
- remote access must be explicitly secured

### Channels

Channel sidecars adapt external messaging systems into Turin tasks and outbound messages.

Rules:

- channel credentials should stay in environment/configured secret references
- inbound user/channel admission should be explicit
- channel settings should validate before activation
- channel sidecars should be restartable and observable
- shared channel behavior should live in shared channel runtime primitives

### Governance

Governance makes capability decisions explicit.

It owns:

- profiles
- capability ceilings
- agent/root/module subjects
- imports and delegated capabilities
- temporary grants
- audit snapshots

Rules:

- capability escalation must be impossible by default
- grants must not exceed issuer ceilings
- governance failures should be visible
- observability should not depend on prompt text

## What Turin Should Refuse

Turin should refuse:

- silent capability escalation
- hidden shell/process execution
- unaudited high-risk tool execution
- remote control without authentication
- ambiguous dynamic config behavior
- provider-specific logic leaking into harness policy
- sidecar behavior that bypasses daemon validation

## Refactor Guardrail

Breaking APIs, config shapes, or crate boundaries can be acceptable while Turin has no active users. Losing capabilities accidentally is not.

Before major refactors, maintain a capability inventory covering:

- tool exposure
- harness APIs
- daemon protocol
- scheduler behavior
- worklist behavior
- memory behavior
- code search behavior
- governance behavior
- channel behavior
- persistence behavior
- security constraints

This inventory should be backed by characterization tests.

