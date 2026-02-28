# Spec Writer

This block is a focused specification harness.

It loads a checked-in spec-writing contract, persists each request, and shapes the active agent toward turning rough ideas into concrete implementation specs.

## Files

- `workspace/IDEA.md`
- `workspace/ACCEPTANCE.md`
- `workspace/CONTEXT.md`
- `harness/main.lua`

## What It Does

- loads spec-writing instructions from checked-in files
- writes runtime artifacts under `.turin/runtime/spec-writer/`
- records requests in `spec_writer_runs`
- shapes the active model toward concrete, executable specs
