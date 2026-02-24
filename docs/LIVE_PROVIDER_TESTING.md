# Live Provider Testing (Manual / Opt-In)

Turin does not run live endpoint tests during `cargo test` or `cargo build`.

This document covers how to validate Turin against real providers manually, including Anthropic-compatible proxies such as MiniMax.

## Why Live Tests Are Separate

Live tests are valuable, but they are not deterministic:

- network failures
- rate limits
- provider-side behavior changes
- account quotas
- model drift

Turin therefore treats live testing as an **opt-in validation layer** on top of the normal unit/integration test suite.

## Manual Smoke Script (MiniMax / Anthropic-Compatible)

Turin includes a manual smoke script:

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env
```

It is **not** run automatically by:

- `cargo build`
- `cargo test`
- `cargo clippy`

### Supported cases

- `basic` — exact `PONG` response smoke test
- `tool_read` — `read_file` tool roundtrip
- `tool_error` — failing tool call + recovery path
- `tool_write_read` — multi-tool (`write_file` + `read_file`) roundtrip

Run specific cases:

```bash
scripts/live_minimax_smoke.sh \
  --env-file ~/Documents/minimax.env \
  --cases basic,tool_read,tool_error,tool_write_read
```

## Required Environment Variables

The script expects:

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

## Anthropic-Compatible Base URL Notes

For Turin’s Anthropic provider path semantics:

- Turin appends `/messages` to the configured Anthropic base URL
- therefore the configured base URL should be the prefix **before** `/messages`

Examples:

- Anthropic official: `https://api.anthropic.com/v1`
- MiniMax Anthropic-compatible: `https://api.minimax.io/anthropic/v1`

## Debugging Request/Response Compatibility

If you need to debug Anthropic-compatible wire format issues (tool roundtrips, thinking blocks, etc.):

```bash
scripts/live_minimax_smoke.sh --env-file ~/Documents/minimax.env --debug-requests
```

This enables `ANTHROPIC_SDK_DEBUG_REQUESTS=1` for the normalized SDK request dumps.

## Validation Strategy (Recommended)

Run live checks in layers:

1. **SDK-level repro/example** (faster iteration)
   - use `inference-sdk-rust` examples for provider wire-format debugging
2. **Turin smoke script** (end-to-end kernel + harness + tools)
3. **Project-specific harness validation** (your real harness stack and governance profile)

This keeps provider compatibility debugging out of Turin core while still validating Turin end-to-end.

## Troubleshooting

### `404 Not Found` on Anthropic-compatible endpoint

Likely base URL path mismatch.
Ensure the effective base URL includes `/v1`.

### `tool call and result not match` (provider error)

This usually indicates a provider compatibility issue in the SDK wire format for tool-result turns.
Debug at the SDK layer (`inference-sdk-rust`) and rerun Turin smoke once patched.

### Benign warning: `FTS5 extension not available`

This warning affects hybrid search quality/fallback behavior, not provider compatibility.
It does not block live inference/tool roundtrips.

## Security Notes

- Do not paste API keys into chat or commit them to the repo.
- Prefer env files outside the repo (`chmod 600`).
- Rotate keys if they were accidentally echoed in terminal logs.
