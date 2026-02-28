# Testing Turin

This document covers Turin’s testing strategy and the commands used to validate changes locally.

## Test Layers

Turin validation is layered on purpose.

### 1. Static + local deterministic checks (default)

- `cargo check`
- `cargo test`
- `cargo clippy --all-targets -- -D warnings`
- `cargo build --release`

These should be your default validation loop.

### 2. Live endpoint validation (manual / opt-in)

- `scripts/live_minimax_smoke.sh`
- project-specific real-harness runs using your provider credentials

Live tests are never run automatically by Turin’s standard cargo commands.

## Core Local Validation Commands

## Fast iteration

```bash
cargo check
cargo test
```

## Strict linting

```bash
cargo clippy --all-targets -- -D warnings
```

## Release build (size/perf sanity)

```bash
cargo build --release
stat -c '%s' target/release/turin
file target/release/turin
```

## Running Turin Manually

## One-shot run

```bash
target/release/turin run --config turin.toml --prompt "Summarize this repository"
```

Options worth using while debugging:

- `--log-level debug`
- `--json` (NDJSON event stream to stdout)
- `--model ...`
- `--provider ...`

## REPL mode

```bash
target/release/turin repl --config turin.toml
```

Useful REPL slash commands include `/reload` (reload harness scripts).

## Config and Harness Validation

## `turin check`

Static validation of config + harness scripts:

```bash
target/release/turin check --config turin.toml
```

Use this before live runs when editing harness scripts heavily.

## Hook and Harness Regression Testing

Turin’s test suite includes dedicated coverage for:

- hook lifecycle behavior
- harness verdict composition
- canonical stdlib APIs (`runtime.*`)
- governance profiles/capabilities/import scoping
- temporary grants and immutable audit semantics
- peer-agent orchestration
- path traversal/security checks
- copyable example packs under `examples/`

Most behavior should be validated there before spending provider quota.

## Live Provider Validation (Manual / Opt-In)

## MiniMax live suites (Anthropic-compatible or OpenAI-compatible)

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite smoke
```

OpenAI-compatible mode:

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --api-format openai --suite smoke
```

Broader core coverage (recommended before releases or public trials):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite core
```

Soak coverage (repeats the core case set; default `--repeat 3`):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite soak
```

OpenAI-compatible core/soak:

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --api-format openai --suite core
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --api-format openai --suite soak
```

Custom case selection:

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env \
  --cases basic,tool_read,tool_error,tool_write_read,peer_complete_caps
```

Request debug dumps (SDK layer):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --debug-requests
```

`--debug-requests` is primarily useful in Anthropic-compatible mode.

Machine-readable summary report (good for sharing failures):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite core --report-json live-report.json
```

Repeat a suite/custom case set:

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite core --repeat 2
```

See `docs/LIVE_PROVIDER_TESTING.md` for environment setup and troubleshooting.

## Suggested Validation Workflow for Major Changes

### Runtime / kernel / stdlib changes

1. `cargo test`
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo build --release`
4. optional live `smoke` suite

### Governance changes

1. unit tests (`kernel::governance`)
2. harness integration tests (`tests/harness_tests.rs`)
3. `cargo clippy --all-targets -- -D warnings`
4. optional governed-mode live `smoke`/`core` suite (project-specific harness/config)

### Provider compatibility debugging

1. reproduce in `inference-sdk-rust` first (faster loop)
2. patch and validate SDK tests
3. rerun Turin smoke script

This keeps provider-specific logic out of Turin and speeds debugging.

## Common Warnings and How to Interpret Them

### `FTS5 extension not available. Hybrid search will be degraded.`

- Meaning: FTS5 search acceleration is unavailable in the current SQLite/libSQL build
- Impact: memory search degrades gracefully
- Relevance to provider testing: none (not a provider/network issue)

### `Event broadcast skipped — no active receivers` (debug)

- Meaning: no in-memory subscribers were attached to the broadcast channel
- Impact: no problem for normal one-shot runs (durability lane still handles persistence if enabled)

## Keeping Tests Fast While Iterating

- Prefer focused test files while developing:
  - `cargo test --test harness_tests`
  - `cargo test --test agent_loop_tests`
  - `cargo test --test dx_harness_examples`
  - `cargo test --test example_harness_examples`
- Run full `cargo test` before committing
- Reserve live tests for behavior that depends on real provider responses

## Release Validation Checklist (Recommended)

Before cutting a release tag:

1. `cargo test`
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo build --release`
4. record binary size
5. run opt-in live suites against at least one real provider/proxy
6. record provider/model + cases passed in release notes or docs
7. verify docs/changelog/version consistency
