# CLI Dispatch Map

## Purpose

The CLI dispatch layer maps parsed `clap` commands to command implementations. It should stay thin: parse-time shape belongs in `src/cli.rs`, runtime behavior belongs in `src/commands/*`, and dispatch only translates command variants into those calls.

This subsystem should preserve three guarantees:

- CLI argument semantics are defined by `clap` structs/enums, not by ad hoc post-parse checks in dispatch when `clap` can express them
- dispatch should not duplicate daemon/runtime business logic
- daemon subcommands should route through the same command implementation functions used by tests and wrapper-facing paths

## Files

- `src/cli.rs`
  - Top-level `Cli`, root commands, harness commands, and root command argument groups.
- `src/cli/daemon.rs`
  - Daemon command shape, daemon subcommands, and shared daemon argument groups.
- `src/dispatch.rs`
  - Top-level command routing: run/repl/script/init/quickstart/check/harness/daemon.
- `src/dispatch/daemon.rs`
  - Daemon command routing for control, agents, tasks, harnesses, channels, sessions, and CLI channel-setting parsing.
- `src/commands/*`
  - Actual command behavior, IO, daemon client calls, rendering, and runtime interactions.

## Data Flow

1. `main.rs` parses `Cli` with `clap`.
2. `dispatch::run` routes root commands.
3. `dispatch/daemon.rs` routes daemon subcommands and converts CLI-only convenience values into daemon command payloads.
4. `src/commands/*` performs the operation and rendering.

## Invariants

- Keep root dispatch and daemon dispatch separate.
- Do not put daemon protocol or state logic in dispatch.
- CLI-only JSON payload construction should be small and visible near the subcommand it supports.
- Channel `--setting KEY=VALUE` parsing accepts JSON values when possible and strings otherwise.

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
cargo test -p turin --bin turin parse_cli_settings
```

Basic checks:

```sh
cargo check -p turin
cargo fmt --all -- --check
git diff --check
```
