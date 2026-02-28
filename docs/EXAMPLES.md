# Examples

`examples/` is no longer the home of Turin's serious ready-to-use harnesses.

Those now live in the Harness Library under `library/`:

- `library/blocks/`
- `library/workflows/`

See:

- `examples/README.md`
- `library/README.md`
- `docs/HARNESS_LIBRARY.md`

## What Belongs In `examples/`

Use `examples/` for:

- small instructional snippets
- narrowly scoped docs companions
- tiny demonstrations of a specific primitive or pattern
- one-off material that would be too small or too incomplete for the Harness Library

Use `library/` for:

- production-grade baselines
- reusable harness blocks
- full workflow harnesses
- entries that should remain copyable, practical, and continuously validated

## Validation

The Harness Library is exercised by:

```bash
cargo test --test example_harness_examples
```

That suite copies library entries into temporary workspaces, runs Turin with a mock provider setup, and checks for the expected behavior or produced artifacts.
