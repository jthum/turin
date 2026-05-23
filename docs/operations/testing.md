# Testing Turin

This document covers Turin’s testing strategy and the commands used to validate changes locally.

For major runtime/subsystem refactors, use `docs/operations/refactor-guardrails.md`
alongside this page. That page defines the capability inventory, characterization
tests, and perf baselines expected before each refactor phase.

## Testing Philosophy

Turin's tests are meant to protect capabilities, security boundaries, and
authoring contracts while still allowing aggressive internal refactors. Test
LOC is not part of the shipped-runtime LOC budget, so prefer meaningful coverage
over minimal coverage.

Use tests to answer four questions:

- does the public capability still work after internal code moves?
- does a risky input fail safely and explicitly?
- does a durable protocol/storage/harness contract keep its shape?
- does a performance-sensitive change preserve behavior before it changes
  footprint or latency?

Do not add tests only to raise counts. Add tests when they lock down behavior a
future maintainer or agent could plausibly break.

## Test Categories

Use the smallest layer that proves the intended behavior:

- Unit tests: pure helpers, parsers, policy matching, recurrence math, path
  normalization, result shaping.
- Integration tests: harness + kernel + persistence, daemon state, channel
  runner, scheduler/worklist dispatch, session lifecycle.
- Characterization/conformance tests: stable subsystem contracts such as daemon
  protocol shapes, harness namespace behavior, channel sidecar handshake, tool
  policy inheritance, and security defaults.
- Golden/fixture tests: user-visible rendering, generated config, examples, and
  manifest output where stable formatting matters.
- Property tests: invariants over broad input spaces, especially path safety,
  capability ceilings, branch graph paths, schedule recurrence, worklist state
  transitions, and tool-policy subset rules.
- Security negative tests: traversal, denied capabilities/tools, malicious
  imports, unauthorized channel users, invalid config, oversized inputs, and
  remote exposure checks.
- Smoke/live tests: real providers and live channels. These are manual or
  explicitly opt-in, never part of ordinary `cargo test`.

Embedded tests are fine for private helpers. Move scenario-heavy, fixture-heavy,
or public-contract tests into integration tests or crate-level test modules so
production files remain readable.

## Test Layers

Turin validation is layered on purpose.

### 1. Static + local deterministic checks (default)

- `cargo check`
- `cargo test`
- `cargo clippy --all-targets -- -D warnings`
- `cargo build --release`

These should be your default validation loop.

Prefer `cargo check` plus targeted tests during normal iteration.
Reserve `cargo build --release` for checkpointing binary size or release-quality sanity.

### 1.5. CI-sensitive stress loop (manual / targeted)

Some integration regressions are timing-sensitive and show up on slower or more
contended runners before they show up locally. Use the local stress runner when
touching branching persistence, daemon restart behavior, or daemon-owned
Telegram channel flows, or peer-agent example persistence:

```bash
scripts/ci_stress.sh --repeat 30
```

Target only the known-sensitive cases if you want a faster pass:

```bash
scripts/ci_stress.sh --tests daemon-restart,telegram-roundtrip
```

```bash
scripts/ci_stress.sh --tests governed-peer-review,delegated-peer-review
```

Supported stress targets:

- `daemon-restart`
- `telegram-roundtrip`
- `telegram-streaming`
- `governed-peer-review`
- `delegated-peer-review`

The script runs each selected test in a loop and preserves the failing
iteration's log on first error.

### 2. Live endpoint validation (manual / opt-in)

- `scripts/live_minimax_smoke.sh`
- `scripts/live_discord_channel_smoke.sh`
- `scripts/live_telegram_channel_smoke.sh`
- project-specific real-harness runs using your provider credentials

Live tests are never run automatically by Turin’s standard cargo commands.

### 3. Performance and footprint baselines (manual / local)

- `tools/perf-suite` hot-history scenarios
- `tools/perf-suite` fake-channel and channel-scale scenarios
- release binary size snapshots from `target/release/turin`

These are not pass/fail tests yet. Treat them as baseline reports to compare
before and after runtime refactors, especially bounded hot history, daemon
session management, and channel-runner changes.

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

## Performance and Footprint Baselines

The repo-local perf suite uses mocked inference so it can stress Turin without
spending provider tokens.

Use the shared target directory to avoid creating a second large
`tools/perf-suite/target` tree:

```bash
CARGO_TARGET_DIR=target cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  channel-scale \
  --sessions 2 \
  --messages-per-session 1000 \
  --checkpoints 10,100,200,1000 \
  --message-bytes 512 \
  --response-bytes 1024
```

For mostly metadata overhead, deliberately shrink payloads:

```bash
CARGO_TARGET_DIR=target cargo run --manifest-path tools/perf-suite/Cargo.toml -- \
  channel-scale \
  --sessions 1 \
  --messages-per-session 2000 \
  --message-bytes 16 \
  --response-bytes 4
```

Reports are written to `.workspace/perf-reports/`. Capture one report before a
runtime refactor and another after the change, then compare RSS/PSS, DB main
file, WAL, SHM, total state bytes, and elapsed time at the same checkpoints.

## Running Turin Manually

## One-shot run

```bash
target/release/turin run --config .turin/config.toml --prompt "Summarize this repository"
```

Options worth using while debugging:

- `--log-level debug`
- `--json` (NDJSON event stream to stdout)
- `--model ...`
- `--provider ...`

## REPL mode

```bash
target/release/turin repl --config .turin/config.toml
```

Useful REPL slash commands include `/reload` (reload harness scripts).

## Config and Harness Validation

## `turin check`

Static validation of config + harness scripts:

```bash
target/release/turin check --config .turin/config.toml
```

Use this before live runs when editing harness scripts heavily.

## Hook and Harness Regression Testing

Turin’s test suite includes dedicated coverage for:

- hook lifecycle behavior
- harness verdict composition
- canonical stdlib APIs (`runtime.*`)
- code-search primitives (`runtime.code.search.*`)
- governance profiles/capabilities/import scoping
- temporary grants and immutable audit semantics
- peer-agent orchestration
- path traversal/security checks
- DX wrappers such as `remember` and `code.find`
- harness library entries under `library/`

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
  --cases basic,tool_read,tool_error,tool_write_read,peer_ask_caps
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

See `docs/operations/live-provider-testing.md` for environment setup and troubleshooting.

## Discord Channel Live Validation (Manual / Opt-In)

When validating daemon-owned channel runtimes against real Discord:

```bash
scripts/live_discord_channel_smoke.sh \
  --channel-id "$DISCORD_CHANNEL_ID" \
  --token-env-name DISCORD_BOT_TOKEN \
  --transport gateway
```

This script provisions a temporary workspace, starts the daemon, creates a
`kind=discord` channel, and verifies that runtime reaches `running`.

## Telegram Channel Live Validation (Manual / Opt-In)

When validating daemon-owned channel runtimes against real Telegram:

```bash
scripts/live_telegram_channel_smoke.sh \
  --chat-id "$TELEGRAM_CHAT_ID" \
  --token-env-name TELEGRAM_BOT_TOKEN
```

This script provisions a temporary workspace, starts the daemon, creates a
`kind=telegram` channel, and verifies that runtime reaches `running`.

Notes:

- Requires a real Telegram bot token in the specified env var.
- Requires a real numeric Telegram chat id.
- If the bot still has an active webhook configured, long polling will fail until the webhook is removed.

## Suggested Validation Workflow for Major Changes

Default local gate for parity with CI:

```bash
scripts/prepush_ci.sh
```

This runs the shared repo gate:

- `cargo fmt --all --check`
- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets -- --include-ignored`

For full release-build parity, use:

```bash
scripts/prepush_ci.sh ci
```

### Runtime / kernel / stdlib changes

1. `scripts/prepush_ci.sh`
2. `cargo build --release`
4. optional live `smoke` suite

### Governance changes

1. unit tests (`kernel::governance`)
2. harness integration tests (`tests/harness_tests.rs`)
3. `scripts/prepush_ci.sh clippy`
4. optional governed-mode live `smoke`/`core` suite (project-specific harness/config)

### Provider compatibility debugging

1. reproduce in `inference-sdk-rust` first (faster loop)
2. patch and validate SDK tests
3. rerun Turin smoke script

This keeps provider-specific logic out of Turin and speeds debugging.

## Common Warnings and How to Interpret Them

### `Event broadcast skipped — no active receivers` (debug)

- Meaning: no in-memory subscribers were attached to the broadcast channel
- Impact: no problem for normal one-shot runs (durability lane still handles persistence if enabled)

## Keeping Tests Fast While Iterating

- Prefer focused test files while developing:
  - `cargo test --test harness_tests`
  - `cargo test --test agent_loop_tests`
  - `cargo test --test dx_harness_examples`
  - `cargo test --test example_harness_examples`
- Focused commands for recent code-search and DX work:
  - `cargo test -p turin-code-index -- --nocapture`
  - `cargo test -p turin-code-index real_repo_smoke -- --ignored --nocapture`
  - `cargo test -p turin-map -- --nocapture`
  - `cargo test --test project_cli_integration_tests -- --nocapture`
  - `cargo test test_runtime_code_search_ --test harness_tests -- --nocapture`
  - `cargo test test_dx_fixture_code_cache_shortcuts --test dx_harness_examples -- --nocapture`
  - `cargo test test_dx_fixture_code_search_fallback --test dx_harness_examples -- --nocapture`
  - `cargo test test_dx_fixture_workspace_review_assistant --test dx_harness_examples -- --nocapture`
  - `cargo test test_lexical_only_hybrid_fallback_prefers_best_text_match --lib`
  - `cargo test --test daemon_integration_tests -- --nocapture`
  - `cargo test --test daemon_cli_integration_tests -- --nocapture`
  - `cargo test daemon_managed_subscription_reconnects_after_restart --test daemon_integration_tests -- --nocapture`

For a manual local-embeddings smoke check:
  - start your local OpenAI-compatible embeddings endpoint
  - `turin-map index`
  - `turin-map status`
  - confirm the status output says `Semantic: enabled (...)`

## Phase 4 Closeout Commands

When validating code-search integration changes specifically:

```bash
cargo test -p turin-code-index -- --nocapture
cargo test -p turin-code-index real_repo_smoke -- --ignored --nocapture
cargo test test_runtime_code_search_ --test harness_tests -- --nocapture
```

What these cover:

- direct index contract validation and fallback behavior
- lexical/semantic/hybrid retrieval quality on a real repo slice
- runtime-facing `runtime.code.search.*` behavior, including fallback and trace metadata

If you are tuning ranking behavior, use `trace = true` in harness/runtime calls so you can inspect:

- requested vs effective mode
- fallback reason
- lexical/semantic candidate ranks
- RRF contributions for hybrid results

- Run full `cargo test` before committing
- Reserve live tests for behavior that depends on real provider responses

## Phase 6 / 7 CLI Smoke

When validating the current onboarding and harness-authoring UX specifically:

```bash
cargo test --test project_cli_integration_tests -- --nocapture
```

What this covers:

- `turin init` scaffolding and `.gitignore` update
- `turin quickstart` bootstrapping and first mock-backed run
- `turin harness new`
- `turin harness test`

## Local SDK Development Without Breaking CI

Keep the checked-in `.cargo/config.toml` CI-safe. Put machine-specific path patches in `~/.cargo/config.toml` instead:

```toml
[patch."https://github.com/jthum/inference-sdk-rust"]
inference-sdk-core = { path = "/home/you/src/inference-sdk-rust/core" }
anthropic-sdk = { path = "/home/you/src/inference-sdk-rust/anthropic" }
openai-sdk = { path = "/home/you/src/inference-sdk-rust/openai" }
inference-sdk-registry = { path = "/home/you/src/inference-sdk-rust/registry" }

[patch."https://github.com/jthum/mcp-sdk-rust"]
mcp-sdk = { path = "/home/you/src/mcp-sdk-rust" }
```

Turin’s `Cargo.toml` should stay pinned to GitHub commits or tags for reproducible CI and release builds. Cargo keeps git dependencies in a local cache, so normal local builds do not refetch them every time.

## Release Validation Checklist (Recommended)

Before cutting a release tag:

1. `scripts/prepush_ci.sh`
2. `cargo build --release`
4. record binary size
5. run opt-in live suites against at least one real provider/proxy
6. record provider/model + cases passed in release notes or docs
7. verify docs/changelog/version consistency
