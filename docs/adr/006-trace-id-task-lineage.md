# ADR 006: Preserve Explicit Trace IDs Across Task Lineage

- Status: Accepted
- Date: 2026-03-03

## Context

As Turin added:

- multi-agent orchestration
- daemon task inspection
- cancellation controls
- future channels/control-plane requirements

it became important to correlate:

- parent and child tasks
- peer-agent hops
- persisted lifecycle events
- runtime logs

Task IDs alone were not enough because child/peer tasks legitimately create new task identities.

## Decision

Add explicit `trace_id` lineage to queued tasks and propagate it through:

- task snapshots
- peer task results
- lifecycle events
- hook payloads
- runtime logs

Child/peer tasks inherit the parent trace unless they intentionally start a new lineage.

## Consequences

Positive:

- logs and persisted events can be correlated across peer-agent boundaries
- daemon task inspection has real execution lineage
- channels/control-plane work now has a clean correlation substrate

Negative:

- more fields threaded through runtime types
- task expansion paths must be explicit about trace inheritance

## Rejected alternatives

- rely only on task IDs
  - rejected because lineage breaks at peer/task fan-out boundaries
- add trace only to logs
  - rejected because it would not help daemon APIs, events, or persisted inspection
