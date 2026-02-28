# OpenClaw-Style Workspace

This example shows the simplest OpenClaw-style Turin harness:

- behavior is shaped by checked-in markdown files
- the harness reads `SOUL.md` and `AGENTS.md`
- the assembled contract is injected into the system prompt every turn

This is useful when you want the runtime to stay generic while the repository-level markdown files define personality, workflow, and delegation rules.

## Files

- `workspace/SOUL.md`
- `workspace/AGENTS.md`
- `harness/main.lua`

## Copy Into a Project

1. Copy `workspace/SOUL.md` and `workspace/AGENTS.md` to the workspace root.
2. Copy `harness/main.lua` into your harness directory, for example `.turin/harnesses/main.lua`.

## What It Does

- fails early if either markdown contract is missing
- appends both files to the effective system prompt
- writes `.turin/runtime/openclaw-contract.md` for inspection/debugging
- writes `.turin/runtime/openclaw-last-prompt.txt` with the latest user prompt

That runtime snapshot is intentional. It makes prompt-shaping observable and debuggable.
