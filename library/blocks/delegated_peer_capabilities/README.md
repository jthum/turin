# Delegated Peer Capabilities

This example shows a main harness delegating a narrow capability set to a reviewer peer agent through `runtime.agent(...):ask(...)`.

Use this pattern when:

- a peer agent should be able to inspect data without inheriting the caller's full runtime surface
- you want explicit capability shaping at the call site instead of a broader grant wrapper
- the peer should be able to read/query but not mutate runtime state

## Files

- `harness/main.lua` — main agent harness
- `agents/reviewer/main.lua` — reviewer harness

## Copy Into a Project

1. Copy `harness/main.lua` into your main harness directory.
2. Copy `agents/reviewer/main.lua` into a reviewer harness directory.
3. Configure a `reviewer` agent in `turin.toml` pointing at that reviewer harness directory.
4. If you use governance enforcement, allow the parent agent to call `reviewer`.

## What It Does

- calls `runtime.agent("reviewer"):ask(...)`
- delegates only `db.query`
- verifies inside the reviewer harness that:
  - `db.query` is allowed
  - `db.exec` is denied by the delegated ceiling
- writes `.turin/runtime/delegated-review.txt` with the reviewer output
- writes `.turin/runtime/delegated-review-input.txt` with the delegated prompt
