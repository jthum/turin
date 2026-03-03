# ADR 002: Filesystem-Authoritative Daemon State

- Status: Accepted
- Date: 2026-03-03

## Context

Daemon mode needed dynamic agent and harness management with persistence.

There were three obvious options:

1. keep all live daemon state in `turin.toml`
2. introduce a hidden daemon-managed registry/state file
3. make the filesystem layout itself the persisted state

Turin also needed to preserve:

- direct file editing by advanced users
- fault isolation for bad agents/harnesses
- local-first, inspectable behavior

## Decision

Use the filesystem layout itself as the persisted daemon state.

The current model is:

- `turin.toml`
  - bootstrap/global config only
- `agents/<id>/agent.toml`
  - daemon-managed agent config
- `agents/<id>/harness/`
  - local harness for that agent
- `harnesses/<id>/`
  - optional shared harness programs

The daemon is the validated control surface over that filesystem state, but direct file edits remain allowed.

## Consequences

Positive:

- one source of truth
- no hidden registry drift
- easy inspection, Git use, and manual editing
- clear fault isolation: one bad agent/harness does not poison the whole runtime

Negative:

- daemon code must tolerate partial or invalid filesystem edits
- runtime mutation requires careful atomic file writes

## Rejected alternatives

- `turin.toml` as the mutable live registry
  - rejected because it turns bootstrap config into a runtime database in disguise
- hidden managed state file alongside the filesystem layout
  - rejected because it duplicates truth and creates precedence/merge problems
