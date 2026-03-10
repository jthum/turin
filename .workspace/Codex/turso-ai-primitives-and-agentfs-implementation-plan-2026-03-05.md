# Turin Implementation Plan: Turso-Native AI Primitives + Optional AgentFS

Date: 2026-03-05
Status: Proposed (ready for implementation after doc review)
Owner: Codex

## 1) Strategic Goal

Make Turin meaningfully Turso-native and AI-first by shipping:
1. Turso 0.5 native search foundation (breaking reset).
2. Memory v2 primitives with explicit lifecycle and compact public IDs.
3. Content cache primitives that save tokens without creating a second filesystem policy path.
4. Code search primitives in Turin plus an external `turin-map` indexing pipeline.
5. Optional AgentFS filesystem backend for true copy-on-write isolation.
6. First-party delightful DX wrappers so simple harnesses stay short and readable.

## 2) Hard Constraints (Locked)

1. Breaking changes are allowed and preferred now.
2. No backward-compatibility work for pre-v2 memory/cache/code-search behavior.
3. No migration support for old DB schema or old public contracts.
4. Old DBs may be deleted and recreated.
5. No compatibility shims, legacy aliases, or dual-path behavior for old semantics.
6. Public contract authority lives in the contract spec; this plan does not duplicate result shapes.
7. Kernel behavior must stay explicit and overrideable; hidden policy is not acceptable.

## 3) Design Principles

1. Simple should be easy; difficult should be possible.
2. Canonical APIs come first (`runtime.*`), but delightful DX is a release criterion, not an afterthought.
3. Sane defaults should keep harness scripts short, readable, and close to natural language.
4. Advanced behavior should be opt-in and explicit.
5. Prefer policy-by-harness over hard-coded kernel policy.
6. Prefer SQL-backed deterministic state over ad-hoc in-memory behavior.
7. Store public IDs compactly: use UUIDv7-backed 16-byte storage where practical and format strings only at API boundaries.

## 4) Locked Architecture Decisions

### 4.1 Memory and IDs

1. Memory v2 is a breaking public contract; `runtime.memory.store` returns a memory record, not `bool`.
2. Memory public IDs are opaque UUIDv7 strings in compact/simple form with no type prefix.
3. Internal numeric row IDs remain internal-only and are never part of the public contract.
4. `runtime.memory.store` uses explicit storage modes: `"auto"`, `"lexical_only"`, `"embedded"`.
5. `storage="auto"` embeds when a provider is available, otherwise stores lexical-only.
6. `storage="embedded"` errors when embeddings are unavailable.
7. `runtime.memory.correct` always supersedes; it never deletes. Physical deletion belongs only to purge.

### 4.2 Content Cache

1. `runtime.cache.read` is a read-through wrapper over `fs.read` semantics:
   - same `harness.fs_root` and safe-path rules
   - same `fs.read` governance capability enforcement
   - no second read capability name
2. Cache session identity uses Turin session public IDs in their existing compact UUID form.
3. Cache result shapes are defined only in the contract spec.

### 4.3 Code Search

1. `turin-map` is a separate binary (same repo, separate crate/bin) and owns write/index lifecycle.
2. Turin runtime owns the read/search path for `runtime.code.search.*` by reading index DBs directly.
3. Codebase discovery is root-path based; callers pass either a root path string or `{ root = ..., index_path? = ... }`.
4. Index DB location defaults to `<codebase_root>/.turin/codebase.db` (gitignored).
5. Turin core must stay lean: no tree-sitter or grammar dependencies in the runtime binary.
6. `turin-map` may include broad language support.
7. Coupling is contract-based and minimal: root-path-first `index_meta`, stable read views, and feature negotiation.
8. `index_meta.codebase_id` is optional metadata only; it is not required for discovery or validation.
9. No runtime subprocess fallback when direct contract validation fails; fail loudly and clearly.

### 4.4 External Inspiration (Locked Direction, Not Dependency Lock-In)

We should borrow ideas aggressively from Turso's code-search article and Codemogger where they improve Turin, without adopting either schema or tool as Turin's contract authority.

1. Keep Turin's public contract root-path-first with `index_meta` and stable read views.
2. Do not couple Turin runtime behavior to Codemogger's schema, CLI, SDK, or release cadence.
3. Use Codemogger and the Turso article as implementation inspiration and quality benchmarks, not as a dependency plan.
4. Prefer AST/symbol-aware chunking in `turin-map` for supported languages rather than permanent reliance on naive fixed-line chunking.
5. Respect `.gitignore` and skip obvious build/vendor directories by default, while keeping override hooks available.
6. Prefer incremental indexing by file hash and targeted stale cleanup over full rebuilds on every update.
7. Prefer weighted lexical search across `name`, `signature`, and related definition fields so definition lookup is better than grep-style substring matching.
8. Prefer RRF-style hybrid fusion over ad-hoc raw score mixing when semantic and lexical search are both available.
9. When semantic indexing lands, prefer compact vector storage by default (for example `vector8(384)` where quality is acceptable) and make richer storage opt-in.
10. Keep `turin-map` implementation choices subordinate to Turin's DX goals: the common path should still be "index root, search root" with minimal ceremony.

## 5) Workstreams and Phases

## Phase 0 — Turso 0.5 Foundation (Breaking Reset)

### Scope

1. Upgrade the `turso` crate to 0.5.
2. Replace legacy FTS bootstrap logic with the strict native path.
3. Remove the fallback LIKE search branch and the related docs/tests.
4. Align schema/init logic with the breaking reset stance.
5. Ensure memory can operate cleanly in lexical-only mode when embeddings are unavailable.

### Deliverables

1. New schema/init path with no old-version migration code.
2. Deterministic lexical/vector/hybrid search path.
3. Updated docs stating DB reset and public-contract breakage clearly.

### Acceptance Criteria

1. `cargo test`, `cargo clippy`, and `cargo build --release` pass.
2. No runtime warning path about missing FTS5 in standard Turso setup.
3. Memory store/search work with embeddings disabled via lexical-only behavior.
4. All legacy fallback branches are removed.

---

## Phase 1 — Memory v2 Core Contract

### Scope

1. Introduce the breaking Memory v2 public contract from the spec.
2. Add compact public memory IDs (UUIDv7, no prefix) while keeping numeric row IDs internal.
3. Add ranking and retrieval fields:
   - `weight`
   - `last_retrieved_at`
   - `retrieval_count`
4. Add explicit storage modes:
   - `"auto"`
   - `"lexical_only"`
   - `"embedded"`
5. Update search ranking to combine lexical/semantic contribution with weight and recency signals.

### Acceptance Criteria

1. `runtime.memory.store` returns a record, not `bool`.
2. `storage="auto"` works without embeddings and stores lexical-only.
3. `runtime.memory.search` supports `auto`, `lexical`, `semantic`, and `hybrid` with strict vs graceful fallback behavior per spec.
4. Public ID, timestamp, and ranking behavior have deterministic tests.
5. Old memory contract behavior is removed rather than preserved.

---

## Phase 2 — Content Cache Primitive

### Scope

1. Add `runtime.cache.read`, `runtime.cache.invalidate`, `runtime.cache.stats`, and `runtime.cache.reset`.
2. Implement session-aware read tracking keyed by Turin session public ID.
3. Reuse existing `fs.read` path resolution, root enforcement, and governance checks.
4. Provide compact unchanged responses, forced-content behavior, and unified diff handling.

### Acceptance Criteria

1. Re-reading an unchanged file omits full content by default and increments savings counters.
2. `include_content=true` forces full content even for unchanged reads.
3. Session isolation is deterministic and tested.
4. `runtime.cache.read` cannot bypass `fs.read` path or governance restrictions.
5. Cache stats/reset behavior is deterministic and tested.

---

## Phase 3 — Memory v2 Lifecycle APIs

### Scope

1. Add `runtime.memory.feedback`, `runtime.memory.correct`, and `runtime.memory.purge`.
2. Add correction chain and supersession tables as needed.
3. Add retrieval attribution support as needed for ranking/lifecycle.
4. Keep purge as the only physical deletion path.

### Acceptance Criteria

1. Numeric and symbolic feedback signals work deterministically.
2. `runtime.memory.correct` always creates a replacement and marks the prior record superseded.
3. `runtime.memory.search` hides superseded memories by default.
4. `runtime.memory.purge` defaults to `dry_run=true`.
5. Feedback/correct/purge integration tests pass.

---

## Phase 3.5 — Embedding Substrate Unification

### Scope

1. Move Turin runtime and `turin-map` embedding generation onto a single provider-agnostic embedding abstraction.
2. Stop hard-coding OpenAI-vs-noop embedding behavior in Turin-specific layers.
3. Support OpenAI-compatible embedding endpoints with configurable `base_url`, so local providers can work without a separate Turin-only adapter path.
4. Make embedding dimensions explicit and configurable rather than assuming `1536` everywhere.
5. Store embedding metadata needed for safe mixed-mode operation:
   - provider/config key
   - embedding dimensions
   - enough metadata to detect incompatible query/index or query/memory combinations
6. Ensure no-embedding operation remains first-class:
   - no config => lexical-only behavior
   - configured embeddings => best available semantic/hybrid behavior

### Acceptance Criteria

1. Turin runtime and `turin-map` share one embedding provider abstraction instead of parallel custom implementations.
2. OpenAI-compatible local embedding servers can be configured without Turin-specific code changes.
3. Memory and code-search storage/search paths can operate with dimensions other than `1536`.
4. Missing embedding configuration degrades gracefully to lexical-only behavior where the contract says it should.
5. Docs include a local embedding quickstart path.

---

## Phase 4 — Code Search Integration (`turin-map` write path, Turin read path)

### Scope

1. Add external `turin-map` binary for indexing lifecycle:
   - index/update
   - remove
   - rebuild
   - embedding enrichment where configured
   - incremental reindex based on file hashes rather than unconditional full re-scan/re-embed
2. Turin runtime implements:
   - `runtime.code.search.lexical(codebase, query, opts?)`
   - `runtime.code.search.semantic(codebase, query, opts?)`
   - `runtime.code.search.hybrid(codebase, query, opts?)`
   - `runtime.code.search.status(codebase, opts?)`
3. Turin resolves codebases from explicit root-based selectors.
4. Turin validates `index_meta` and stable read views, then feature-negotiates.
5. Status diagnostics return facts, not policy:
   - `updated_at`
   - `index_age_seconds`
   - `capabilities`
   - `index_path`
6. `turin-map` indexing quality work:
   - respect `.gitignore`
   - skip obvious generated/build/vendor directories by default
   - evolve from fallback chunking toward symbol-aware chunk extraction for supported languages
   - split oversized symbols/chunks instead of storing very large monoliths
7. `turin-map` lexical quality work:
   - weight `name` and `signature` above generic snippet/path text
   - bias results toward definitions rather than arbitrary mentions
8. `turin-map` semantic/hybrid quality work:
   - use the shared embedding substrate from Phase 3.5
   - prefer compact quantized vector storage by default where quality remains acceptable
   - use RRF-style fusion when both lexical and semantic candidate sets are available
9. Codemogger-style behavior may be used as a reference benchmark for retrieval quality and indexing ergonomics, but Turin keeps its own contract and database shape.

### Acceptance Criteria

1. Turin binary size impact from code indexing deps is minimal (no parser grammar deps in Turin runtime).
2. Turin searches discovered index DBs directly without spawning subprocesses per query.
3. Strict vs graceful fallback behavior is clear and tested.
4. `turin-map` can build and refresh index DBs for supported languages.
5. Contract validation failures return explicit errors with remediation guidance.
6. Incremental indexing skips unchanged files deterministically based on persisted file identity/hash state.
7. Lexical search returns definition-oriented results for identifier lookups instead of grep-like noise.
8. Hybrid ranking behavior is explicit, tested, and not based on opaque raw score mixing.
9. Real-repo smoke tests exist for at least one medium codebase, and the results are reviewed against a Codemogger/Turso-inspired quality bar even if Turin does not match them fully in the first release.

---

## Phase 5 — AgentFS Design Spike (Non-Blocking)

### Scope

1. Define the `FsBackend` abstraction boundary in design/docs.
2. Identify exactly which existing file operations must route via the backend trait.
3. Validate AgentFS feasibility risks without blocking phases 0-4.

### Acceptance Criteria

1. Architecture/design doc is complete enough to start implementation later.
2. No mandatory runtime changes that block phases 0-4.

---

## 6) Cross-Cutting Tasks

1. Governance integration
   - Add capability gating for:
     - `runtime.memory.search`
     - `runtime.memory.store`
     - `runtime.memory.feedback`
     - `runtime.memory.correct`
     - `runtime.memory.purge`
     - `runtime.cache.invalidate`
     - `runtime.cache.stats`
     - `runtime.cache.reset`
     - `runtime.code.search.lexical`
     - `runtime.code.search.semantic`
     - `runtime.code.search.hybrid`
     - `runtime.code.search.status`
   - `runtime.cache.read` inherits `fs.read` capability and root enforcement instead of introducing a duplicate read gate.

2. DX layer integration
   - Canonical APIs remain explicit and complete.
   - Thin DX wrappers ship with the release; they are not a later nice-to-have.
   - Wrapper naming should read cleanly in harness scripts and stay overrideable.
   - Initial DX targets:
     - `memory.search(query, opts?)`
     - `memory.store(content, metadata?, opts?)`
     - `remember(content, metadata?, opts?)`
     - `recall(query, opts?)`
     - `cache.file(path, opts?)`
     - `code.find(query, opts?)` with workspace-root default and explicit override when needed
   - Wrappers must be straight delegates with no hidden side effects and no governance bypass.

3. Docs
   - Update `docs/PRIMITIVES.md`, `docs/HARNESS_GUIDE.md`, `docs/ARCHITECTURE.md`, and `docs/TESTING.md`.
   - Show simple examples and advanced override examples side by side.
   - Keep examples terse enough that a non-technical reader can still follow intent.

4. Testing
   - Phase 0: remove conditional FTS fallback branches from tests.
   - Phase 1: deterministic memory ranking, ID, and storage-mode tests.
   - Phase 2: cache session-isolation and `fs.read` parity tests.
   - Phase 3: lifecycle API tests for feedback/correct/purge.
   - Phase 4: index discovery, feature negotiation, contract validation, incremental indexing, and retrieval-quality tests.

5. Contract hygiene
   - Public contract changes update the contract spec first.
   - Do not duplicate public result shapes outside the contract spec.

## 7) Proposed Execution Order

1. Phase 0 (foundation).
2. Phase 1 (Memory v2 core).
3. Phase 2 (content cache).
4. Phase 3 (Memory v2 lifecycle).
5. Phase 3.5 (embedding substrate unification).
6. Phase 4 (code search integration + `turin-map`).
7. Phase 5 (AgentFS design spike).

## 8) Risks and Mitigations

1. Search behavior drift after the FTS/vector refactor.
   - Mitigation: deterministic ranking tests and golden-result checks.

2. DX wrapper bloat or hidden magic.
   - Mitigation: thin delegate rules, canonical parity tests, and terse examples in docs.

3. Code-search/index contract drift.
   - Mitigation: `index_meta`, stable read views, and no subprocess fallback.

4. Cache behavior subtly diverges from `fs.read`.
   - Mitigation: shared path/governance rules and explicit parity tests.

5. AgentFS maturity risk.
   - Mitigation: design spike only for now (non-blocking).

## 9) Release Quality Bar

1. A primitive is not done until canonical docs, DX docs, and tests exist.
2. The common path should fit in one to three lines of harness code.
3. Advanced overrides must stay available without making the default path verbose.
4. Error messages must be deterministic, human-readable, and namespace-prefixed.
