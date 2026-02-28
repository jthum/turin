# Example Library

This directory contains copyable Turin example packs.

These examples are not throwaway snippets. They are designed to be:

- readable
- directly copyable into a project
- continuously validated by tests

## Layout Convention

Each example folder may contain:

- `README.md` — what the example is for and how to copy it
- `workspace/` — files that live at the workspace root
- `harness/` — files that go into the main harness directory
- `agents/<agent_id>/` — files that go into peer-agent harness directories

## Validation

Examples are exercised by:

```bash
cargo test --test example_harness_examples
```

That keeps the library honest: if an example stops working, CI/local validation catches it.

## Current Examples

- `examples/harnesses/openclaw_style_workspace/`
- `examples/harnesses/governed_peer_review/`
- `examples/harnesses/durable_journal/`
