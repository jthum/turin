# Example Library

Turin ships a copyable example library under `examples/`.

These examples are meant to be:

- real enough to start from
- small enough to understand quickly
- validated by tests so they do not silently drift

## Where To Look

- `examples/README.md`
- `examples/harnesses/openclaw_style_workspace/`
- `examples/harnesses/governed_peer_review/`
- `examples/harnesses/durable_journal/`

## Example Pack Layout

Most examples follow this structure:

- `workspace/` — files that belong at the workspace root
- `harness/` — files for the main harness directory
- `agents/<agent_id>/` — peer-agent harness directories
- `README.md` — copy instructions and intent

This keeps examples practical instead of burying them in synthetic test-only layouts.

## How Examples Are Tested

The example packs are exercised by:

```bash
cargo test --test example_harness_examples
```

That suite copies the example files into a temporary workspace, runs Turin with a mock provider setup, and checks for the expected behavior or artifacts.

## Current Examples

## OpenClaw-Style Workspace

`examples/harnesses/openclaw_style_workspace/`

Pattern:

- repository-level markdown contracts (`SOUL.md`, `AGENTS.md`)
- harness injects them into the effective prompt
- runtime writes a prompt-contract snapshot for debugging

This is the closest current Turin pattern to an OpenClaw-style markdown-driven runtime.

## Governed Peer Review

`examples/harnesses/governed_peer_review/`

Pattern:

- main agent delegates to a reviewer peer agent
- delegation runs under a temporary grant
- the reviewer output is folded back into the main agent's prompt shaping

## Durable Journal

`examples/harnesses/durable_journal/`

Pattern:

- store operational notes durably in the runtime DB
- keep a memory trace and a durable SQL record
- emit a simple runtime artifact for inspection
