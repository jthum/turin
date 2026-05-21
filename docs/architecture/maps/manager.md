# Manager Map

## Purpose

`turin-manager` is the operator setup and troubleshooting CLI. It owns first-run config generation, local environment file updates, generic channel setup flows driven by channel manifests, channel inventory/status views, and doctor checks.

Keep this crate as an operator-facing orchestration layer. It should not duplicate daemon runtime behavior, channel adapter internals, or control-client transport details.

## Files

- `crates/turin-manager/src/main.rs`
  - Clap command parsing and thin dispatch into setup flows.
- `crates/turin-manager/src/setup.rs`
  - Public setup facade and command arg structs used by `main.rs`.
- `crates/turin-manager/src/setup/init.rs`
  - `turin-manager init`: default provider/model prompts, generated Turin config, starter harness, optional `.env` update.
- `crates/turin-manager/src/setup/doctor.rs`
  - `turin-manager doctor`: config presence, configured channel discovery, sidecar/secret checks, daemon reachability, channel runtime health.
- `crates/turin-manager/src/setup/channels.rs`
  - `turin-manager channels list/configure/status`: sidecar discovery, manifest-driven setup prompts, auth-flow polling, channel config rendering, runtime status table.
- `crates/turin-manager/src/files.rs`
  - Config-path/layout resolution, channel config discovery/rendering, planned writes, TOML/env merge helpers, and redacted diffs.
- `crates/turin-manager/src/runner.rs`
  - Re-export boundary for `turin-channel-host` sidecar discovery, manifest, validation, and auth-flow helpers.

## Data Flow

Init:

1. Prompt for provider/model/system prompt and optional API key.
2. Generate `turin.toml` and starter harness content.
3. Optionally merge API key into `.env` next to the config.
4. Present planned diffs, then write after confirmation.

Channel configure:

1. Parse the requested channel kind and inspect the sidecar manifest.
2. Prompt for channel id and agent id.
3. Walk manifest secrets, config fields, visibility rules, validation checks, and auth flows.
4. Validate final settings through the sidecar.
5. Render the channel config and optional `.env` updates, then write after confirmation.

Doctor/status:

1. Load configured channel files from the resolved channels directory.
2. Check sidecar availability and required secrets.
3. Connect through `turin-control-client` for daemon status when available.
4. Print operator-facing diagnostics without mutating runtime state.

## Invariants

- User-facing command names, prompts, and generated config shape are part of manager DX; change them deliberately.
- Channel setup must remain manifest-driven so new sidecars can be configured without hard-coding adapter-specific prompts in manager.
- Secret values may be written to disk only through the planned-write flow and redacted display contents.
- Runtime status must come through `turin-control-client`; manager should not speak daemon protocol directly.
- Sidecar discovery, settings validation, and auth-flow IPC should use `turin-channel-host` helpers via `runner.rs`.
- `files.rs` owns channel config rendering and `.env` merge mechanics; setup modules should not hand-edit TOML/env strings.

## Common Changes

Add a manager command:

1. Add the Clap command in `main.rs`.
2. Add a command arg struct and run function in the relevant setup module, or a new setup child module if the command has its own lifecycle.
3. Keep `setup.rs` as a facade, not a new mixed implementation file.
4. Add focused unit coverage when the command has parseable/renderable behavior.

Change channel setup behavior:

1. Update `setup/channels.rs`.
2. Keep adapter-specific behavior in manifests/sidecars unless the behavior is genuinely generic.
3. Run manager tests and channel-host checks when sidecar discovery or auth-flow behavior changes.

Change generated config or file rendering:

1. Update `setup/init.rs` for init prompts/defaults.
2. Update `files.rs` for TOML/env rendering.
3. Add tests for generated snippets or merge behavior.

## Tests

Focused checks:

```sh
cargo test -p turin-manager
cargo check -p turin-manager
```

Related checks when touching sidecar discovery or control-client behavior:

```sh
cargo test -p turin-channel-host
cargo test -p turin-control-client
```

Basic checks:

```sh
cargo fmt --all -- --check
git diff --check
```
