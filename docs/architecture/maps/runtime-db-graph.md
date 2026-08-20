# Runtime DB And Graph Map

## Purpose

The runtime DB and graph namespaces expose controlled persistence access to harness code:

- `runtime.db.*` gives Lua explicit SQL access to configured Turin state stores.
- `runtime.graph.*` records and queries sparse relationships between sessions, turns, branches, and harness-defined graph refs.

These namespaces are powerful. They should preserve three guarantees:

- every public operation is guarded by a governance capability
- dynamic DB path access respects runtime DB policy
- graph selected paths only materialize valid turns for the resolved session

## Files

- `src/harness/stdlib/runtime_db.rs`
  - Lua-facing `runtime.db` API, DB target resolution, open handle listing/closing, SQL query/exec dispatch, and SQL row conversion.
- `src/harness/stdlib/runtime_graph.rs`
  - Lua-facing `runtime.graph` API, graph node/edge conversion, graph ref parsing, selected-path materialization, and graph session resolution.
- `src/harness/stdlib/db_support.rs`
  - Shared DB selector parsing, SQL param parsing, store path policy helpers, and SQL value conversion.
- `src/harness/dx/db.rs`
  - DX helpers layered over `runtime.db`.
- `src/harness/dx/graph.rs`
  - DX helpers layered over `runtime.graph`.
- `src/persistence/state/graph.rs`
  - Graph persistence operations.
- `src/persistence/state.rs`
  - State store connection access used by runtime DB.

## Data Flow

Runtime DB:

1. Lua calls `runtime.db.open`, `close`, `list`, `query`, or `exec`.
2. The namespace checks the matching governance capability.
3. DB selectors are parsed by `db_support`.
4. Runtime policy resolves path scope, cache trimming, and dynamic-open denial.
5. `query` and `exec` open a scoped state store connection and dispatch SQL with parsed params.
6. `query` converts rows to JSON-shaped Lua tables; `exec` returns changed row count.

Runtime graph:

1. Lua calls `runtime.graph.node.*`, `edge.*`, or `path.select`.
2. The namespace checks read or write governance capability.
3. The active or requested session reference resolves to a state store and internal session id.
4. Node/edge operations create or list graph rows scoped to that session.
5. `path.select` materializes graph refs or source edges into turn ids.

## Invariants

- `runtime.db.query` and `runtime.db.exec` must share DB target/policy resolution.
- Dynamic path opens must be denied when `db.allow_dynamic_open=false`.
- SQL param parsing should stay in `db_support.rs`; runtime DB should not parse params ad hoc.
- `runtime.graph.write` guards node/edge creation.
- `runtime.graph.query` guards list/path operations.
- Graph refs that materialize to turns must belong to the resolved session.
- Selected paths must reject duplicate materialized turns.
- Runtime graph is sparse relationship metadata; durable transcript/session state remains owned by persistence/session modules.
- Every state-store connection enables foreign-key enforcement. Runtime DB callers may issue
  advanced SQL, but writes that violate declared state relationships fail at the database boundary.
- Core schema, FTS schema, and schema-version recording initialize in one transaction. A failed
  bootstrap must leave the database retryable rather than stranded as an unversioned partial schema.
- Persisted values represented as unsigned domain counters, indexes, dimensions, or durations are
  range-checked while mapping rows. Negative values fail as typed persistence-integrity errors and
  must never wrap into large unsigned values.
- Persisted turn depths, ancestry links, and branch-head targets are validated as they are
  materialized. Missing or cross-session graph records return a typed persistence-integrity error;
  Turin does not scan the complete database or attempt automatic repair.
- Parent/child session ownership is structural persistence data, not a semantic graph
  edge or JSON metadata. It uses normalized session columns and dedicated indexes so
  peer-thread lookup does not scan or parse metadata.
- Opaque client-origin provenance is likewise a normalized nullable session field
  with a partial index. It organizes root-session discovery and is not a semantic
  graph edge, authenticated identity, or authority grant.
- The operator-facing Session Graph visualizes the durable turn tree and branch
  heads. It is not a renderer for `runtime.graph.*`; a future semantic overlay
  should remain visually and contractually distinct.

## Common Changes

Change SQL selector or param behavior:

1. Prefer updating `src/harness/stdlib/db_support.rs`.
2. Keep `runtime_db.rs` focused on capability checks and dispatch.
3. Run runtime DB harness tests.

Change runtime DB query/exec behavior:

1. Update `src/harness/stdlib/runtime_db.rs`.
2. Keep query and exec using the shared target/open path.
3. Run:

```sh
cargo test -p turin --test harness_tests test_runtime_db_api_and_context_glob
cargo test -p turin --lib runtime_db
```

Change graph selected-path behavior:

1. Update selected-path helpers in `src/harness/stdlib/runtime_graph.rs`.
2. Preserve duplicate-turn rejection and session ownership checks.
3. Run:

```sh
cargo test -p turin runtime_graph --lib
cargo test -p turin --test harness_tests test_runtime_graph
```

## Tests

Focused tests:

```sh
cargo test -p turin --test harness_tests test_runtime_db_api_and_context_glob
cargo test -p turin --lib runtime_db
cargo test -p turin runtime_graph --lib
cargo test -p turin --test harness_tests test_runtime_graph
```

Basic compile/format checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current pass kept DB and graph in the root harness stdlib but reduced repetition:

- `runtime_db.rs` now shares DB target resolution and SQL query/exec connection handling.
- `runtime_graph.rs` now shares optional-field and metadata conversion helpers for node/edge Lua rows.

This was intentionally a small lean pass. A larger extraction should wait until the harness extension model is clearer, because these namespaces are tightly coupled to governance, runtime policy, active session identity, and state-store access.
