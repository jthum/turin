# ADR 005: Support Both Cooperative Cancel and Forceful Kill

- Status: Accepted
- Date: 2026-03-03

## Context

Daemon mode needs real operator stop controls.

Practical user need:

- stop bad or misunderstood inference quickly
- avoid wasting tokens or continuing incorrect tool actions

A single stop mechanism was not enough:

- purely cooperative cancel may still wait at provider/tool boundaries
- purely forceful kill is too destructive for normal use

## Decision

Expose two distinct controls:

- `session.cancel`
  - cooperative
  - cancels queued work immediately
  - moves running work into a `cancelling` state and completes it as `cancelled` at real execution boundaries
- `session.kill`
  - forceful
  - aborts the active runtime immediately
  - marks work as `killed`
  - recreates the runtime/session on demand later

`task.cancel` is truthful as well:

- queued tasks cancel immediately
- running tasks transition through the cooperative cancellation path

## Consequences

Positive:

- operator stop controls are honest
- safe and forceful stop semantics are explicit
- CLI/control-plane can expose real “Stop” behavior

Negative:

- more lifecycle states to reason about
- provider/tool boundaries still determine how fast cooperative cancel lands

## Rejected alternatives

- fake cancel that only flips metadata while execution continues
  - rejected as misleading
- forceful kill only
  - rejected because it is too destructive as the default operator stop path
