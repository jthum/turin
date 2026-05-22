# Daemon Registry Map

## Purpose

The daemon registry loads filesystem-backed agents, shared harnesses, and channels into the effective daemon runtime configuration. It owns config file IO, directory scanning, isolated issue reporting, effective config assembly, and status snapshot shaping.

This subsystem should preserve three guarantees:

- one broken agent, harness, or channel should surface as a registry issue without preventing unrelated valid entries from loading
- filesystem-backed agent/channel config writes should be atomic replacements
- the effective runtime config should be derived from the bootstrap config plus enabled registry entries, not from ad hoc call-site merging

## Files

- `src/daemon/registry.rs`
  - Facade and public re-exports for registry types and operations.
- `src/daemon/registry/types.rs`
  - Registry snapshot rows, discovered filesystem entries, load result, and agent/channel file config shapes.
- `src/daemon/registry/files.rs`
  - TOML config read/write helpers for agent and channel directories.
- `src/daemon/registry/scan.rs`
  - Directory scanning, harness validation, agent/channel discovery, and channel persistence/inference validation.
- `src/daemon/registry/effective.rs`
  - Bootstrap plus registry-entry merge into the effective `TurinConfig`.
- `src/daemon/registry/snapshot.rs`
  - Conversion from internal `RegistryLoad` into daemon-facing status snapshot rows.
- `src/daemon/state/registry_ops.rs`
  - Mutating daemon operations that call registry file helpers, rescan, and return daemon details.

## Data Flow

Registry load:

1. Resolve agents, shared harnesses, and channels directories from the bootstrap config.
2. Scan shared harnesses first and validate each harness in isolation.
3. Scan agent directories, resolving either local harnesses or shared harness references.
4. Scan channel directories after agents, so channel `agent_id` references can be validated.
5. Sort loaded entries and issues for stable status output.

Effective config:

1. Clone the bootstrap config.
2. Clear generated `agents` and `harnesses`.
3. Insert shared harness definitions.
4. Insert enabled agent configs and their local harness definitions.
5. Validate the resulting config before daemon runtime use.

Registry mutation:

1. `registry_ops.rs` reads or writes an agent/channel `config.toml`.
2. Writes go through a temp file and atomic rename.
3. The daemon rescans the registry.
4. Callers receive refreshed detail or issue output from the loaded registry state.

## Invariants

- `default` is reserved for the bootstrap/root agent and is not a filesystem-backed agent id.
- An agent cannot declare both a shared harness and a local `harness/` directory.
- A channel must reference either the bootstrap agent or a discovered filesystem-backed agent.
- Channel persistence selectors should be validated through the bootstrap persistence config helpers.
- Channel inference overrides should validate both shallow shape and merged effective provider/model completeness.
- Snapshot shaping should not perform additional scanning or validation.

## Common Changes

Change registry file shape:

1. Update `types.rs`.
2. Update `files.rs` only if IO semantics change.
3. Update `registry_ops.rs` if daemon mutation inputs need mapping changes.
4. Run registry and daemon-state registry tests.

Change scan behavior:

1. Update `scan.rs`.
2. Keep broken-entry isolation behavior.
3. Add a focused test in `src/daemon/tests/registry.rs`.

Change daemon-facing status shape:

1. Update `types.rs` and `snapshot.rs`.
2. Check daemon server/client/control-client consumers.

## Tests

Focused tests:

```sh
cargo test -p turin --lib daemon::registry::tests
cargo test -p turin --lib daemon::state::tests::channel_create_disable_update_and_delete_are_filesystem_backed
cargo test -p turin --lib daemon::state::tests::agent_can_bind_shared_harness_and_switch_back_to_local
cargo test -p turin --lib daemon::state::tests::shared_harness_create_and_delete_are_filesystem_backed
```

Basic checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```
