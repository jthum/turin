# Performance Guards

This workspace uses release-mode performance tests as CI guardrails:

1. Fixed budget caps.
2. Historical baseline regression checks.

## Why this exists

1. Catch accidental performance regressions in stream/event hot paths.
2. Keep guardrails lightweight with no new runtime dependencies.
3. Preserve flexibility while keeping behavior and cost bounded.

## Guarded paths

Current performance checks live in:

- `core/tests/perf_budget.rs`
- baseline data: `core/perf_baseline.json`

They cover:

1. Event-order validation throughput (`validate_event_sequence`).
2. Text stream assembly throughput (`InferenceResult::from_stream`).
3. Tool delta assembly and JSON parse throughput (`InferenceResult::from_stream` with large tool args).

## CI integration

The CI workflow runs:

```bash
cargo test -p inference-sdk-core --release --test perf_budget -- --ignored
```

The tests are ignored by default for local `cargo test` runs and executed in the dedicated performance gate.

## Updating Baselines and Budgets

If a legitimate refactor changes performance characteristics:

1. Measure before and after with release builds.
2. Update `core/perf_baseline.json` only with evidence.
3. Update hard caps in `core/tests/perf_budget.rs` only when justified.
4. Document the reason in the PR description.
