# Code Search Map

## Purpose

Code search lets harness code query a prebuilt Turin code index through `runtime.code.search`.

This subsystem should preserve four guarantees:

- runtime search never builds or mutates the index
- codebase roots and index paths are validated by the shared reader
- semantic and hybrid search only use embeddings when the runtime provider matches the index profile
- non-strict semantic/hybrid calls fall back predictably instead of failing when embeddings are unavailable

## Files

- `crates/turin-harness-lua/src/harness/stdlib/runtime_code.rs`
  - Lua-facing `runtime.code.search` namespace, governance capability checks, option parsing, embedding-provider negotiation, runtime fallback trace annotation.
- `crates/turin-code-index/src/code_index_reader.rs`
  - Shared read API for status, lexical search, semantic search, hybrid search, tracing, and request structs.
- `crates/turin-code-index/src/code_index_reader/query.rs`
  - SQL construction, search-mode negotiation against index capabilities, lexical scoring, semantic vector query, and hybrid fusion.
- `crates/turin-code-index/src/code_index_reader/resolve.rs`
  - Codebase selector resolution, index path validation, schema contract checks, and index metadata loading.
- `crates/turin-code-index-writer`
  - Index writer/chunker/store implementation.
- `crates/turin-map`
  - CLI/binary that builds or refreshes code indexes.
- `tests/harness_tests.rs`
  - Runtime Lua integration coverage for code search.
- `tests/dx_harness_examples.rs`
  - DX fixture coverage for code-search fallback behavior.

## Data Flow

Status:

1. Lua calls `runtime.code.search.status(codebase, opts)`.
2. Runtime checks `runtime.code.search.status` capability.
3. `runtime_code.rs` parses the codebase selector.
4. `turin-code-index` resolves the root/index path, validates schema metadata, and returns index status.

Search:

1. Lua calls `runtime.code.search.lexical`, `.semantic`, or `.hybrid`.
2. Runtime checks the matching governance capability.
3. Lua options become a `CodeSearchRequest`.
4. Runtime resolves query embedding policy:
   - lexical never embeds
   - semantic/hybrid require a runtime embedding provider only when strict fallback is disabled
   - provider key and dimensions must match the index metadata
   - non-strict mismatch or missing provider falls back to lexical
5. `turin-code-index` validates the index and negotiates against index capabilities.
6. The reader executes lexical, semantic, or hybrid search.
7. Runtime adds requested/effective mode and runtime fallback reason to trace rows when `trace=true`.

Indexing:

1. `turin-map` or writer code builds the DB.
2. Runtime search opens the existing DB read-side only.
3. The runtime must not silently create or refresh indexes during harness execution.

## Invariants

- `runtime_code.rs` owns Lua API shape and governance checks; `turin-code-index` owns index validation and search behavior.
- Root/index path policy should stay in `crates/turin-code-index/src/code_index_reader/resolve.rs`.
- Runtime embedding-profile mismatch should fail only when `strict=true`.
- Runtime fallback reasons are runtime-local:
  - `missing_embedding_provider`
  - `embedding_profile_mismatch`
- Index capability fallback is reader-local and should remain distinguishable from runtime fallback.
- Trace output should preserve both requested and effective modes.
- Runtime code search should stay read-only; index writing belongs to `turin-map` / `turin-code-index-writer`.

## Common Changes

Change Lua API shape:

1. Update `crates/turin-harness-lua/src/harness/stdlib/runtime_code.rs`.
2. Preserve capability names unless the governance model is being intentionally changed.
3. Run the runtime code search harness tests.

Change fallback or embedding-profile behavior:

1. Update `resolve_query_embedding` in `crates/turin-harness-lua/src/harness/stdlib/runtime_code.rs`.
2. Add/adjust harness tests for strict and non-strict behavior.
3. Run:

```sh
cargo test -p turin --test harness_tests test_runtime_code_search_falls_back_without_embedding_provider
cargo test -p turin --test harness_tests test_runtime_code_search_api_round_trip
```

Change lexical/semantic/hybrid ranking:

1. Update `crates/turin-code-index/src/code_index_reader/query.rs`.
2. Add reader-level tests before changing runtime harness code.
3. Run:

```sh
cargo test -p turin-code-index
```

Change index schema or metadata:

1. Update writer and reader together.
2. Keep schema revision and metadata validation explicit.
3. Run:

```sh
cargo test -p turin-code-index
cargo test -p turin-code-index-writer
```

## Tests

Runtime harness tests:

```sh
cargo test -p turin --test harness_tests test_runtime_code_search_api_round_trip
cargo test -p turin --test harness_tests test_runtime_code_search_falls_back_without_embedding_provider
```

Reader/writer tests:

```sh
cargo test -p turin-code-index
cargo test -p turin-code-index-writer
```

Basic compile/format checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current pass kept code search in the root runtime crate, but made the root-facing code thinner:

- search-mode registration is now shared for lexical, semantic, and hybrid
- runtime embedding/provider negotiation is isolated in `resolve_query_embedding`
- profile mismatch detection and trace annotation are named helpers

This is a useful intermediate shape before any future `turin-code-search` crate. A separate crate would need a clean extension boundary for harness namespace registration and access to the runtime embedding provider abstraction. That boundary is not worth forcing until the harness extension model is more explicit.
