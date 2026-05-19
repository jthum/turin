# Project Quality Bar

Turin should feel like a deliberate systems project, not just a working pile of generated code. This document defines the quality bar contributors and agents should preserve.

## The Standard

A senior engineer should be able to open any module and understand its role in under two minutes.

That does not mean every implementation is simple. It means each module has a clear purpose, clear boundaries, and clear invariants.

## Design Values

### Legibility

Code should communicate intent through names, types, and module layout.

Prefer:

- `RuntimeSnapshot`
- `WorkspacePathPolicy`
- `ChannelAdmissionPolicy`
- `ToolRiskClass`
- `HarnessExtension`

over scattered booleans and helper functions that hide the domain concept.

### Cohesion

Each module should have one obvious reason to change.

Avoid files that mix:

- config parsing
- API clients
- rendering
- media handling
- runtime state
- tests
- protocol construction

### Typed Invariants

Use Rust types to make invalid states harder to represent.

Good candidates:

- non-empty IDs
- validated paths
- parsed session scopes
- tool risk classes
- capability names/rules
- typed channel settings
- runtime snapshots

### Explicit Risk

High-risk behavior should be visible in names, config, docs, and tests.

High-risk surfaces include:

- shell/process execution
- MCP bridging
- filesystem writes
- remote control
- channel ingress
- unrestricted web access

### Small Public APIs

Crates and subsystems should expose narrow, intentional APIs. Implementation details should remain private.

### Tests As Contracts

Tests should describe behavior and invariants, not just implementation details.

High-value test types:

- capability characterization tests
- conformance tests
- security/negative tests
- property tests for invariants
- integration tests for runtime flows

## Agent Maintenance Guidance

Turin will continue to be maintained by coding agents. Good structure helps agents as much as humans.

Agents perform better when:

- files are small enough to fit in context
- ownership is obvious
- repeated patterns are centralized
- tests are named by behavior
- APIs are typed and narrow
- module names point to domain concepts

Agents perform worse when:

- files are huge
- repeated logic appears in multiple places
- helpers are vague
- validation is scattered
- tests are mixed into unrelated production code
- there are multiple ways to do the same thing

Hand-crafted code is therefore not only a human aesthetic goal. It is a maintenance strategy for agent-assisted development.

## Rules Of Thumb

- No new god files.
- New capabilities need characterization tests.
- New high-risk tools need risk classification.
- Docs must match default behavior.
- New channel behavior should use shared channel primitives where possible.
- Config parsing should produce typed settings early.
- Prefer behavior-preserving refactors before feature expansion in crowded modules.
- Keep comments focused on boundaries, invariants, and trust assumptions.

## What Good Looks Like

Good Turin code should feel:

- deliberate
- navigable
- cohesive
- precise
- hard to misuse
- easy to test
- easy to review
- friendly to future agents

The goal is not fewer lines at any cost. The goal is a codebase where fewer lines are needed because the right abstractions carry the intent.
