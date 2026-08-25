# CLI Dispatch Map

## Purpose

The CLI dispatch layer maps parsed `clap` commands to command implementations. It should stay thin: parse-time shape belongs in `src/cli.rs`, runtime behavior belongs in `src/commands/*`, and dispatch only translates command variants into those calls.

This subsystem should preserve three guarantees:

- CLI argument semantics are defined by `clap` structs/enums, not by ad hoc post-parse checks in dispatch when `clap` can express them
- dispatch should not duplicate daemon/runtime business logic
- daemon subcommands should route through the same command implementation functions used by tests and wrapper-facing paths

## Files

- `crates/turin-cli/Cargo.toml`
  - Owns the `turin` and `turin-remote` executable targets. Lua is an explicit product
    dependency alongside the engine-neutral core.
- `crates/turin-cli/src/composition.rs`
  - Product composition root that injects `turin-harness-lua` into direct kernel
    construction and daemon startup.
- `crates/turin-cli/src/main.rs`
  - Executable entry point and CLI module assembly.
- `src/cli.rs`
  - Top-level `Cli`, root commands, harness commands, and root command argument groups.
- `src/cli/daemon.rs`
  - Daemon command shape, daemon subcommands, and shared daemon argument groups.
- `src/dispatch.rs`
  - Top-level command routing: run/repl/script/init/quickstart/check/doctor/harness/daemon.
- `src/dispatch/daemon.rs`
  - Daemon command routing for control, agents, tasks, harnesses, and sessions.
- `src/commands/*`
  - Actual command behavior, IO, daemon client calls, rendering, and runtime interactions.
- `src/commands/check.rs`
  - Aggregated project validation and local readiness diagnostics for `turin check`
    and `turin doctor`.

## Data Flow

1. `crates/turin-cli/src/main.rs` parses `Cli` with `clap`.
2. `dispatch::run` routes root commands.
3. `dispatch/daemon.rs` routes daemon subcommands and converts CLI-only convenience values into daemon command payloads.
4. `src/commands/*` performs the operation and rendering.

## Invariants

- Keep root dispatch and daemon dispatch separate.
- Do not put daemon protocol or state logic in dispatch.
- CLI-only JSON payload construction should be small and visible near the subcommand it supports.
- Kernel and daemon construction must go through `composition.rs`; command modules
  must not select or instantiate a scripting engine themselves.
- `turin check` treats invalid config and harness runtimes as blocking failures,
  but missing optional credentials and directories as explicit warnings.
- `turin doctor` extends the same report with a non-mutating daemon probe. An
  offline daemon is a warning because direct CLI execution remains supported;
  incompatible or otherwise invalid daemon responses are failures.
- JSON diagnostics must retain the same severity and exit behavior as human output.
- `daemon task submit` selects exactly one target: `--agent <id>` for a new
  session or `--session-id <id>` for an existing live session. Keep this
  relationship in `clap` rather than deferred runtime validation.
- `daemon stop` is a bounded lifecycle operation: after accepting the stop
  request, it waits for the configured endpoint to become unreachable before
  reporting success.

## Common Changes

Add a root command:

1. Add the shape in `src/cli.rs`.
2. Add top-level routing in `src/dispatch.rs`.
3. Put behavior in `src/commands/*`.

Add a daemon subcommand:

1. Add the shape in `src/cli/daemon.rs`.
2. Add routing in `src/dispatch/daemon.rs`.
3. Prefer an existing `commands::daemon::*` helper or add one there.

## Tests

Focused tests:

```sh
cargo test -p turin-cli --bin turin parse_reference_diagnostic_commands
cargo test -p turin-cli --bin turin parse_daemon_task_and_bounded_stop_commands
cargo test -p turin-cli --bin turin commands::check::tests
cargo test -p turin-cli --test daemon_cli_integration_tests -- --test-threads=1
```

Basic checks:

```sh
cargo check -p turin-cli --all-targets
cargo fmt --all -- --check
git diff --check
```
