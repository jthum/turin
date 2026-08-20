# Daemon Registry Map

## Purpose

The daemon registry loads filesystem-backed agents and shared harnesses into
the effective runtime configuration. It owns config file IO, directory
scanning, isolated issue reporting, effective config assembly, and status
snapshot shaping.

External clients and channels are not registry entries. Their
configuration, credentials, health, and process lifecycle stay outside Turin
daemon.

## Files

- `src/daemon/registry.rs`
  - Facade and local re-exports for registry types and operations.
- `src/daemon/registry/types.rs`
  - Agent/harness snapshot rows, discovered entries, and load results.
- `src/daemon/registry/files.rs`
  - Agent TOML read/write helpers and atomic replacement.
- `src/daemon/registry/scan.rs`
  - Agent and harness scanning plus isolated validation issues.
- `src/daemon/registry/effective.rs`
  - Bootstrap plus registry-entry merge into effective `TurinConfig`.
- `src/daemon/registry/snapshot.rs`
  - Conversion into daemon-facing status rows.
- `src/daemon/state/registry_ops.rs`
  - Agent and shared-harness mutations followed by rescan.

## Data Flow

1. Resolve agent and shared-harness directories from Turin config.
2. Scan and validate shared harnesses independently.
3. Scan agents, resolving local or shared harness bindings.
4. Sort entries and issues for deterministic status.
5. Merge enabled entries into a cloned bootstrap config and validate it.

## Invariants

- One broken agent or harness surfaces as an issue without preventing unrelated
  valid entries from loading.
- `default` remains reserved for the bootstrap agent.
- An agent cannot declare both a shared harness and local `harness/` directory.
- Agent writes use temp-file plus atomic rename.
- Snapshot shaping performs no additional scanning or validation.
- Channel paths do not trigger daemon rescans or registry issue events.

## Tests

```sh
cargo test -p turin --lib daemon::registry::tests
cargo test -p turin --lib daemon::state::tests
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```
