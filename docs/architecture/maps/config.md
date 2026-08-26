# Config Map

## Purpose

`kernel::config` owns TOML parsing, defaults, validation, path normalization, and the public config schema used by the runtime.

Keep top-level `config.rs` as the entry point for `TurinConfig`, agent/provider/daemon/harness schema, loading, and cross-cutting path normalization. Move domain-specific config semantics into child modules when they have their own validation, resolution, or merge rules.

## Files

- `src/kernel/config.rs`
  - Top-level `TurinConfig`, core runtime/agent/provider/harness/daemon/remote schema, config loading, layout/path normalization, and inference route entry points.
- `src/kernel/config/defaults.rs`
  - Default values used by serde and manual defaults.
- `src/kernel/config/environment.rs`
  - Immutable values captured from the configured workspace env file without mutating process state.
- `src/kernel/config/inference.rs`
  - Inference contexts, overrides, route resolution, hot-history, and compaction config.
- `src/kernel/config/layout.rs`
  - Layout config and resolved runtime layout.
- `src/kernel/config/persistence.rs`
  - Raw persistence config, store target aliases/paths, scope placements, and resolved persistence paths.
- `src/kernel/config/validation.rs`
  - Cross-field validation for config values.
- `src/kernel/config/tests.rs`
  - Config parsing, validation, and resolution tests.

## Invariants

- Public config types should continue to be re-exported from `crate::kernel::config` unless there is a deliberate API break.
- Defaults used in serde attributes must remain available to the modules that reference them.
- Validation stays centralized in `validation.rs` when it checks cross-field behavior.
- Persistence target resolution should stay in `config/persistence.rs`; call sites should not duplicate alias/path selection rules.
- `from_file` captures the adjacent env file without mutating process state, then normalizes
  runtime paths before validation. Process environment values take precedence when credentials
  are resolved.
- Captured environment is loaded configuration state owned by `TurinConfig`; `LayoutConfig`
  declares only the location of the env file and must not retain loaded values.
- `from_str` parses and validates without filesystem path normalization.
- Plain daemon filesystem paths share one normalization helper; the daemon endpoint stays separate because local IPC endpoint resolution has different semantics.
- `runtime.linked_runtime_lanes` is a positive startup-stable global default; an
  agent's positive `linked_runtime_lanes` overrides it for that profile.
- External client and channel configuration is not part of
  `TurinConfig`. Turin config owns runtime behavior, not client credentials,
  bindings, access policy, or process lifecycle.
- Governance policy is explicit runtime configuration. Turin core does not
  interpret named presets; `turin-manager` templates expand into concrete
  enforcement, audit, import, and capability fields.

## Common Changes

Change persistence config:

1. Update `src/kernel/config/persistence.rs`.
2. Update validation if the new field has cross-field constraints.
3. Add parsing/resolution coverage in `src/kernel/config/tests.rs`.

Change inference routing config:

1. Update `src/kernel/config/inference.rs`.
2. Preserve fallback and cycle-detection tests.
3. Run config and inference-route focused tests.

Change path normalization:

1. Update `src/kernel/config.rs` and/or `layout.rs`.
2. Add `from_file` or resolved-layout coverage, not only `from_str` parsing tests.
3. Check daemon path defaults and persistence path resolution together.

## Tests

Focused tests:

```sh
cargo test -p turin config --lib
```

Basic checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current pass moved persistence schema and selector resolution out of the top-level config file into `config/persistence.rs`, next to resolved persistence path logic. This keeps the public API stable while making persistence config ownership explicit.
