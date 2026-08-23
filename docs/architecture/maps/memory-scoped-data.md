# Memory And Scoped Data Map

## Purpose

The memory and scoped-data layer gives harness Lua code, native tools, and runtime namespaces one durable way to store scoped KV values and long-lived memories.

This subsystem should preserve four guarantees:

- memory and KV operations resolve to the correct scope and backing store
- embedding-backed memory is optional unless the caller explicitly requires it
- public Lua APIs follow the shared direct-value / raise-on-failure contract
- native tools and harness globals share backend policy instead of drifting

## Files

- `src/harness/stdlib/scoped_data_backend/mod.rs`
  - Shared request structs, scope encoding, selector validation, store opening, and memory metadata augmentation.
- `src/harness/stdlib/scoped_data_backend/memory.rs`
  - Canonical memory store/search/feedback/correct/purge behavior.
- `src/harness/stdlib/scoped_data_backend/kv.rs`
  - Canonical scoped KV get/set/delete behavior.
- `src/harness/stdlib/binding_common.rs`
  - Lua option parsing, search-source parsing, store selector resolution, and Lua row conversion.
- `src/harness/stdlib/memory_kv_bindings/mod.rs`
  - Shared Lua bridge/result adapters for memory and KV backends.
- `src/harness/stdlib/memory_kv_bindings/memory.rs`
  - Global `memory.*` and `memory.as(ctx).*` APIs.
- `src/harness/stdlib/memory_kv_bindings/kv.rs`
  - Global `kv.*` and `kv.as(ctx).*` APIs.
- `src/harness/stdlib/runtime_data.rs`
  - Canonical `runtime.memory.*` and `runtime.kv.*` APIs that require explicit context selectors.
- `src/harness/stdlib/session_user_aliases.rs`
  - Convenience `session.memory`, `session.kv`, `user.memory`, and `user.kv` aliases.
- `src/tools/builtins/memory_tools.rs`
  - Native `remember` and `recall` tools backed by the same memory backend.
- `src/persistence/search.rs`
  - Memory search plus bounded, read-only memory inspection used by operator clients.
- `src/daemon/state/memories.rs`
  - Maps inspection rows into the typed daemon `memory.list` response.

## Data Flow

Memory store/correct:

1. Lua or native tool options become a `MemoryStoreRequest`.
2. The caller's `ContextSelector` is normalized and resolved to a `scope_kind` plus encoded `scope_key`.
3. Store selector placement is resolved before opening a `StateStore`.
4. `scoped_data_backend::memory` resolves storage policy:
   - `auto` embeds only when an embedding provider exists
   - `lexical_only` stores without vectors
   - `embedded` requires an embedding provider
5. Metadata is augmented with Turin-owned fields when `source_task` or tags are supplied.
6. The persistence store inserts the row or supersedes the original row during correction.

Memory search:

1. Lua or native tool options become a `MemorySearchRequest`.
2. Empty queries return an empty result set.
3. Search mode resolves once:
   - `auto` becomes hybrid with embeddings and lexical without embeddings
   - `semantic` and `hybrid` fall back to lexical unless `strict=true`
4. Explicit multi-source searches use each source's selector; otherwise the caller scope is the only source.
5. Per-source results are merged, sorted by score and creation time, then truncated to the requested limit.

Lua bridge:

1. Namespace registration code owns the public API shape and argument order.
2. Shared bridge helpers in `memory_kv_bindings/mod.rs` own backend invocation and public result shaping.
3. Backend errors become Lua runtime errors at the public API boundary; authors can recover with `try(fn, ...)`.
4. Successful rows are converted by `binding_common`.

## Invariants

- `selector_scope_ref` is the scope encoding source of truth.
- Non-private selector visibility is rejected unless a future policy explicitly enables it.
- Search `strict=true` must fail for semantic/hybrid modes when embeddings are unavailable.
- Search `strict=false` must preserve lexical fallback when embeddings are unavailable.
- `embedded` storage must fail without an embedding provider for both store and correct.
- Store placement must be resolved before opening the state store.
- Multi-source memory search should not apply the caller's store selector after sources are resolved.
- Lua-facing memory/KV APIs should share bridge helpers; do not copy backend invocation blocks into every namespace.
- Native `remember` and `recall` should continue to call the scoped-data backend directly, not reimplement persistence semantics.
- Operator inspection must not update retrieval count or last-retrieved timestamps and must never expose embedding blobs.
- Applying memory feedback must update the ranking weight and append its audit event in one
  transaction. Concurrent deltas accumulate from the persisted weight rather than overwriting it.
- Correcting memory must insert the replacement and supersede the original in one transaction.
  Competing corrections may commit at most one replacement.

## Common Changes

Change memory search semantics:

1. Update `src/harness/stdlib/scoped_data_backend/memory.rs`.
2. Add focused backend tests for fallback, strict behavior, and multi-source ordering.
3. Run `cargo test -p turin scoped_data_backend::tests::memory --lib`.

Change Lua memory/KV API shape:

1. Update the relevant registration file:
   - global APIs in `memory_kv_bindings/memory.rs` or `memory_kv_bindings/kv.rs`
   - explicit runtime APIs in `runtime_data.rs`
   - session/user aliases in `session_user_aliases.rs`
2. Keep shared backend bridge/result behavior in `memory_kv_bindings/mod.rs`.
3. Run the harness tests listed below.

Change store placement behavior:

1. Update selector/store resolution in `binding_common.rs` or `scoped_data_backend/mod.rs`.
2. Preserve explicit store, placement-routed store, and multi-source tests.
3. Run the runtime memory harness tests listed below.

Change native memory tools:

1. Update `src/tools/builtins/memory_tools.rs`.
2. Keep parsing/tool output separate from backend persistence behavior.
3. Run `cargo test -p turin memory_tools --lib`.

Change operator memory inspection:

1. Keep the persistence query bounded and observational in `src/persistence/search.rs`.
2. Update the typed daemon projection in `src/daemon/state/memories.rs`.
3. Run `cargo test -p turin memory_inspection_is_bounded_filtered_and_does_not_record_retrieval`.

## Tests

Focused backend tests:

```sh
cargo test -p turin scoped_data_backend::tests::memory --lib
cargo test -p turin concurrent_memory_ --lib
cargo test -p turin memory_feedback_rolls_back_weight_when_audit_insert_fails --lib
cargo test -p turin memory_correction_rolls_back_replacement_when_supersession_fails --lib
```

Harness integration tests:

```sh
cargo test -p turin --test harness_tests test_stdlib_context_api_kv_memory_and_tier2
cargo test -p turin --test harness_tests test_runtime_memory
```

Basic compile/format checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current pass made two deliberate changes:

- `scoped_data_backend::memory` now centralizes embedding policy, search-mode fallback, public memory ID parsing, and feedback delta calculation.
- Lua memory/KV namespaces now share bridge/result helpers instead of duplicating async backend invocation blocks across `runtime.memory`, global `memory`, `session.memory`, `user.memory`, and KV aliases.

This reduced source duplication while keeping public APIs and behavior intact. The remaining larger cleanup opportunity is API registration ergonomics: the namespace registration code still repeats similar option-resolution shapes, but extracting that further should only happen if it keeps argument order and session-specific checks obvious.
