# Live Provider Testing (Manual / Opt-In)

Turin does not run live endpoint tests during `cargo test` or `cargo build`.

This document covers how to validate Turin against real providers manually, including MiniMax via either its Anthropic-compatible or OpenAI-compatible endpoints.

## Why Live Tests Are Separate

Live tests are valuable, but they are not deterministic:

- network failures
- rate limits
- provider-side behavior changes
- account quotas
- model drift

Turin therefore treats live testing as an **opt-in validation layer** on top of the normal unit/integration test suite.

## Manual Live Suites (MiniMax)

Turin includes a manual live suite script:

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite smoke
```

Anthropic-compatible (default wire format) and OpenAI-compatible modes are both supported:

```bash
# Anthropic-compatible (default)
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite smoke

# OpenAI-compatible
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --api-format openai --suite smoke
```

It is **not** run automatically by:

- `cargo build`
- `cargo test`
- `cargo clippy`

### Suite presets

- `smoke` (default) — fast confidence checks (cheap, suitable before/after most changes)
- `core` — broader end-to-end validation across governance/multi-db/multi-agent/grants/audit
- `all` — currently the same as `core` (reserved for future expansion, soak/provider matrix additions)

Run a suite:

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite core
```

Write a machine-readable summary (useful when sharing results back for debugging):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite core --report-json live-report.json
```

Or print the JSON summary to stdout at the end:

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite core --report-json -
```

### Supported cases

- `basic` — exact `PONG` response smoke test
- `tool_read` — `read_file` tool roundtrip
- `tool_error` — failing tool call + recovery path
- `tool_write_read` — multi-tool (`write_file` + `read_file`) roundtrip
- `governed_denial` — harness-driven governance denial sentinel + successful inference turn
- `peer_agent` — harness-driven peer-agent `agent.complete(...)` roundtrip + successful main-agent turn
- `queue_steer` — harness-driven queue steering via `on_all_tasks_complete` follow-up prompt injection
- `runtime_db` — harness-driven `runtime.db.open/list/exec/query/close` + sqlite verification
- `grant_flow` — temporary grant issue/use/revoke + durable audit event verification
- `token_reject_task` — live `on_token_usage` `REJECT` with `hook.token_usage.reject_mode=enforce_task`
- `immutable_audit` — immutable audit persists rejected `governance_snapshot` audit event
- `peer_grant` — temporary grant ceiling propagation into peer-agent submit/await path

Run specific cases:

```bash
scripts/live_minimax_smoke.sh \
  --env-file ~/Documents/minimax.env \
  --cases basic,tool_read,tool_error,tool_write_read,governed_denial,peer_agent,queue_steer
```

### What `core` covers (confidence-building set)

The `core` suite is designed to validate Turin’s real value surface against a live endpoint:

- live inference + streaming
- tool roundtrips (`read_file`, `write_file`)
- tool failure and recovery
- governance enforcement denials
- peer-agent orchestration
- queue steering / follow-up task injection
- runtime DB API (`runtime.db.*`)
- temporary grants + grant audit events
- token-usage hook enforcement mode (`enforce_task`)
- immutable audit persistence semantics
- grant ceiling propagation to peer agents

## Required Environment Variables

Anthropic-compatible mode (default) expects:

- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_MODEL`

Recommended setup:

```bash
# ~/Documents/minimax.env
export ANTHROPIC_API_KEY='...'
export ANTHROPIC_BASE_URL='https://api.minimax.io/anthropic'
export ANTHROPIC_MODEL='MiniMax-M2.5'
```

The script normalizes the base URL to include `/v1` if missing.
For MiniMax, Turin’s Anthropic provider path handling expects the effective base URL to end in `/v1`.

OpenAI-compatible mode (`--api-format openai`) expects:

- `OPENAI_BASE_URL` (required)
- `OPENAI_MODEL` (optional; falls back to `ANTHROPIC_MODEL`)
- `OPENAI_API_KEY` (optional; falls back to `ANTHROPIC_API_KEY`)

Example:

```bash
# ~/Documents/minimax.env (can coexist with Anthropic-compatible vars)
export OPENAI_BASE_URL='https://api.minimax.io/v1'
# Optional if reusing the same key/model:
# export OPENAI_API_KEY='...'
# export OPENAI_MODEL='MiniMax-M2.5'
```

## Anthropic-Compatible Base URL Notes

For Turin’s Anthropic provider path semantics:

- Turin appends `/messages` to the configured Anthropic base URL
- therefore the configured base URL should be the prefix **before** `/messages`

Examples:

- Anthropic official: `https://api.anthropic.com/v1`
- MiniMax Anthropic-compatible: `https://api.minimax.io/anthropic/v1`

## OpenAI-Compatible Base URL Notes

For MiniMax’s OpenAI-compatible endpoint, configure:

- `https://api.minimax.io/v1`

## Debugging Request/Response Compatibility

If you need to debug Anthropic-compatible wire format issues (tool roundtrips, thinking blocks, etc.):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --debug-requests
```

This enables `ANTHROPIC_SDK_DEBUG_REQUESTS=1` for the normalized SDK request dumps.
(`--debug-requests` is most useful in Anthropic-compatible mode.)

## Validation Strategy (Recommended)

Run live checks in layers:

1. **SDK-level repro/example** (faster iteration)
   - use `inference-sdk-rust` examples for provider wire-format debugging
2. **Turin `smoke` suite** (fast end-to-end kernel + harness + tools)
3. **Turin `core` suite** (broader end-to-end feature validation)
4. **Project-specific harness validation** (your real harness stack and governance profile)

This keeps provider compatibility debugging out of Turin core while still validating Turin end-to-end.

## Recommended Confidence Bar Before Asking Others to Try Turin

At minimum, publish a provider/model + scenario statement based on real runs.

Recommended baseline:

1. `cargo test`
2. `cargo clippy --all-targets -- -D warnings`
3. `cargo build --release`
4. `scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite smoke`
5. `scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --suite core`

Then document:

- provider + model used (for example, MiniMax M2.5 Anthropic-compatible or OpenAI-compatible)
- exact base URL format (including `/v1` when required)
- which live cases passed
- known caveats / experimental surfaces

## Latest Known-Good Live Validation Baselines (MiniMax)

### Anthropic-Compatible Endpoint

Recorded on: `2026-02-25T07:58:06Z`  
Turin commit: `146fe3e`  
Provider/model: `MiniMax-M2.5` (Anthropic-compatible)  
Base URL: `https://api.minimax.io/anthropic/v1`

Suite:

- `core`

Result:

- `12 passed, 0 failed`

Passed cases:

- `basic`
- `tool_read`
- `tool_error`
- `tool_write_read`
- `governed_denial`
- `peer_agent`
- `queue_steer`
- `runtime_db`
- `grant_flow`
- `token_reject_task`
- `immutable_audit`
- `peer_grant`

This baseline demonstrates end-to-end live validation across Turin’s core runtime value surface:

- live inference + streaming
- tools (success + failure + multi-tool)
- governance enforcement
- peer-agent orchestration
- queue steering
- dynamic multi-db runtime APIs
- temporary grants + grant audit events
- token-usage enforcement modes
- immutable audit persistence
- grant-ceiling propagation to peers

### OpenAI-Compatible Endpoint

Recorded on: `2026-02-25T09:41:12Z`  
Turin commit: `f0bfa29`  
Provider/model: `MiniMax-M2.5` (OpenAI-compatible)  
Base URL: `https://api.minimax.io/v1`

Suite:

- `core`

Result:

- `12 passed, 0 failed`

Passed cases:

- `basic`
- `tool_read`
- `tool_error`
- `tool_write_read`
- `governed_denial`
- `peer_agent`
- `queue_steer`
- `runtime_db`
- `grant_flow`
- `token_reject_task`
- `immutable_audit`
- `peer_grant`

This baseline demonstrates the same end-to-end Turin core-runtime surface against MiniMax’s OpenAI-compatible wire protocol, including:

- inference + streaming
- tool roundtrips (success + error + multi-tool)
- governance enforcement
- peer-agent orchestration and peer grant propagation
- queue steering
- runtime DB APIs
- temporary grants and immutable audit behavior

Note: MiniMax OpenAI-compatible tool-call support required normalized SDK OpenAI stream handling fixes for provider-specific final usage chunk shape (`usage` + non-empty `choices`) and explicit `tool_choice: "auto"` when tools are present.

## Troubleshooting

### `404 Not Found` on Anthropic-compatible endpoint

Likely base URL path mismatch.
Ensure the effective base URL includes `/v1`.

### `tool call and result not match` (provider error)

This usually indicates a provider compatibility issue in the SDK wire format for tool-result turns.
Debug at the SDK layer (`inference-sdk-rust`) and rerun Turin smoke once patched.

### `dns error` / `Temporary failure in name resolution`

This is an environment/network resolution issue, not a Turin/provider-wire-format issue.

The live suite script retries `turin run` automatically for transient transport failures, but persistent DNS failures will still fail the case after retries.

### Benign warning: `FTS5 extension not available`

This warning affects hybrid search quality/fallback behavior, not provider compatibility.
It does not block live inference/tool roundtrips.

## Security Notes

- Do not paste API keys into chat or commit them to the repo.
- Prefer env files outside the repo (`chmod 600`).
- Rotate keys if they were accidentally echoed in terminal logs.
