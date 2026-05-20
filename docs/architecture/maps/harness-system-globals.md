# Harness System Globals Map

## Purpose

`system_globals.rs` registers the Lua globals that every harness sees: imports, load-time `use`, file access, hashes, JSON, time, logging, and `try`.

This file is security-sensitive because it defines how harness code loads other harness modules and touches the filesystem. Keep policy checks explicit and shared rather than copied per operation.

## Files

- `src/harness/stdlib/system_globals.rs`
  - Lua globals: `import`, `import_scoped`, `use`, `use_scoped`, `watch`, `try`, `fs`, `hash`, `json`, `time`, and `log`.
  - Import/use policy enforcement and scoped capability delegation.
  - File path safety and `fs.stat` session hash tracking.
- `src/harness/engine.rs`
  - Module loading, loaded-module registry, load phase, and watch roots.
- `src/harness/stdlib/governance_support.rs`
  - Active subject and capability enforcement.
- `src/tools.rs`
  - Safe path resolution used by filesystem globals.

## Data Flow

Imports:

1. `import*` resolves or loads a module under the harness directory.
2. The import policy checks governance mode, scoped/unscoped capability, requested root, and root attribution.
3. Exported functions are wrapped so active module/root/capability context is restored around calls.

Use blocks:

1. `use*` is allowed only during harness load.
2. The module is loaded as an active block with optional config and `when` predicate.
3. Scoped capability delegation is capped by the active importer.

Filesystem globals:

1. `fs.read`, `fs.write`, and `fs.stat` require filesystem capabilities.
2. Paths must resolve safely under the harness filesystem root.
3. `fs.stat` hashes content and stores previous hashes in session-scoped KV when a session is active.

## Invariants

- `watch`, `use`, and `use_scoped` are load-time only.
- Scoped import/use policy should stay in one shared path.
- Unscoped import/use behavior must respect `governance.import.mode` and the Open-profile override.
- `import_scoped` and `use_scoped` must reject requested roots that cannot be attributed to the module.
- Delegated import capabilities cannot widen the parent delegation.
- Imported function wrappers must restore previous module/root/capability context after the call.
- Filesystem APIs must not bypass safe path resolution.

## Tests

Focused tests:

```sh
cargo test -p turin --test harness_tests test_import_scoped_tracks_imported_module_subject_and_root
cargo test -p turin --test harness_tests test_governed_scoped_import_mode_blocks_unscoped_import
cargo test -p turin --test harness_tests test_governed_scoped_import_mode_blocks_unscoped_use
cargo test -p turin --test harness_tests test_use_scoped_root_mismatch_fails_harness_init
cargo test -p turin --test harness_tests test_import_scoped_capability_delegation_is_downward_only
cargo test -p turin --test harness_tests test_use_scoped_capability_delegation_is_downward_only
cargo test -p turin --test harness_tests test_nested_import_cannot_widen_import_delegation
```

Basic checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current pass keeps the globals in one file but removes duplicate import/use governance logic. `enforce_module_policy` owns the shared decision tree, while `enforce_import_policy` and `enforce_use_policy` keep call sites readable.
