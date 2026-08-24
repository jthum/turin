# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Changelog Policy

- **Focus**: Release entries summarize user-visible capabilities, breaking changes, migrations, security changes, and important fixes. Ordinary refactors, internal cleanups, and test suite additions belong in Git history.
- **Brevity**: Keep each release concise, targeting 10–20 meaningful bullets grouped by theme.
- **Archives**: Historical entries are archived under [`docs/changelog/`](docs/changelog/) to keep this file concise (retaining `[Unreleased]` plus the latest 3–5 releases).

## [Unreleased]

## [0.30.1] - 2026-05-24

### Fixed
- Fixed branch checkout persistence by closing branch lookup readers before updating the active branch head, resolving the CI/release preflight failures seen after the `0.30.0` tag.
- Reverted speculative branch-persistence cache flushes that added unnecessary database work without addressing the release failure.

## [0.30.0] - 2026-05-24

### Added
- **Governance Capability Templates**
  - Added explicit governance starter templates under `templates/governance/` for `open`, `balanced`, and `governed`.
  - Added `governance.unmatched_capability` and explicit `[governance.capabilities]` support so runtime enforcement reads inspectable capability rules instead of hidden preset matrices.
  - Kept profile-only governance configs working through a compatibility fallback while generated configs now include the expanded capability map.
- **Signals And Work Coordination**
  - Added wildcard signal topic subscriptions so harnesses can subscribe to topic families such as `deploy.*` without waking on unrelated exact topics.
  - Added per-task budget metrics for turns, tokens, and elapsed runtime bookkeeping so harnesses can make budget-aware worklist decisions.
- **Performance Tooling**
  - Added black-box channel/direct-task perf scenarios, persisted-message scale reports, PSS/anonymous-PSS breakdowns, heap attribution support, and daemon task/cache diagnostics.
  - Added release-friendly profiling documentation for allocator tuning, heap profiling, and runtime memory measurements.
- **Architecture And Testing Documentation**
  - Added durable architecture maps and project-quality guidance for the refactored runtime, channels, scheduler/worklists, governance, runtime DB/graph, code search, manager, web tools, and related subsystems.
  - Added a worklists concept guide that documents the durable work item model, claim lifecycle, dispatch patterns, dependency metadata, hierarchy, and stale-claim recovery.

### Changed
- **Runtime And Harness Structure**
  - Split large runtime, daemon, channel, scheduler/worklist, action, agent-binding, web-tool, and harness-global files into smaller ownership-focused modules.
  - Consolidated repeated Lua result shaping, async bridge helpers, session/store resolution, scheduler/worklist row mapping, channel setup, and capability matching helpers.
  - Consolidated `RuntimeControl` around a coherent runtime state lock to reduce torn-snapshot risk and repetitive lock handling.
- **Memory And Persistence**
  - Bounded live-session hot history and old tool payload retention so long-running channel sessions no longer keep rematerializing full turn history.
  - Reused event persistence writers and cached event statements to reduce control-plane allocation churn and lower post-idle memory retention in daemon perf scenarios.
  - Added optional peer idle allocator trimming for memory-sensitive deployments.
- **Governance**
  - Changed `profile` from a hardcoded enum-driven policy selector into a string label for observability and harness DX.
  - Moved starter policy defaults out of core governance matching logic and into explicit TOML templates consumed by scaffold/onboarding flows.
- **Release Automation**
  - Updated CI to skip doc-only changes, keep correctness checks on code changes, and reserve release binary packaging for release tags/manual release runs.
  - Updated release bundles to include `turin`, `turin-remote`, `turin-map`, `turin-manager`, and the external Discord, Telegram, Rocket.Chat, and WhatsApp sidecars. Existing TUI/App binaries remain excluded from this release bundle while the UI story is being reworked.

### Fixed
- **Security And Hardening**
  - Added file-size enforcement and clearer path/error handling around harness filesystem globals.
  - Hardened channel filesystem paths and channel setting parsing helpers.
  - Added security-negative and capability-characterization coverage around governance, scoped imports, channel authorization, and high-risk tool paths.
- **Operational Robustness**
  - Improved channel runtime transitions, sidecar process handling, channel task payload mapping, and sidecar host discovery reuse.
  - Improved scheduler overlap/worklist state characterization and durable work item filtering behavior.

## [0.29.0] - 2026-05-15

### Added
- **Reference-Aware Runtime Objects**
  - Added `_ref`-aware payload encoding and hydration for runtime-owned proxies such as scopes, worklists, and work items.
  - Added `ref(proxy)` for identity-only payload passing when callers want the receiver to hydrate current canonical state without sending overlay fields.
  - Added contextual object actions and `action.define_on(...)`, so matching proxies can expose harness-defined methods such as `project:review(...)` or `item:label(...)`.
- **Instructional DX Example**
  - Added a small committed harness example for reference-aware proxy passing and object-scoped actions under `examples/harnesses/reference_aware_objects/`.

### Changed
- **Harness Event DX**
  - Local `on(...)` listeners and durable `runtime.on(...)` listeners now receive domain data first and optional metadata second.
- **Examples And Guides**
  - Refreshed harness guides and committed examples to reflect the current `this`/semantic callback naming conventions and the new reference-aware object DX.
- **Turso**
  - Upgraded workspace Turso dependencies from `0.5` to `0.6.0` without changing Turin's current database feature posture yet.

## [0.28.1] - 2026-04-24

### Added
- **Execution Observability**
  - Added execution snapshots to task status surfaces, including `execution_id`, `context_target`, `write_policy`, `durability`, and `visibility`.
  - Added the same execution metadata to task lifecycle events so operators can inspect how a task ran, not just whether it succeeded.

### Changed
- **Operator Surfaces**
  - `task.submit`, `task.get`, `task.wait`, `task.list`, and `task.sidestep` now expose execution-scoped metadata alongside status and branch outcome.
  - `session.open`, `session.resume`, `session.list_live`, and daemon runtime snapshots now expose the active execution head plus the current conflict policy for each live slot.
  - CLI daemon task and live-session renders now display execution-scoped state directly instead of hiding it inside raw JSON-only surfaces.

### Fixed
- **Post-Release CI**
  - Fixed post-`0.28.0` workspace clippy regressions in channel crates and stale CLI/TUI fixture expectations.
  - Hardened daemon integration timing so endpoint startup waits are less fragile on slower CI runners.
  - Added a checked-in pre-push CI gate script and aligned GitHub Actions to use the same local validation surface.

## [0.28.0] - 2026-04-24

### Added
- **Execution-Scoped Context Targets**
  - Added explicit execution context target support for branch heads, turn ids, selected paths, external references, and summary-source turns.
  - Added selected-path hardening so empty and duplicate-path materializations are rejected while explicit caller ordering is preserved.
- **Sparse Semantic Graph Overlay**
  - Added opt-in `graph_nodes` and `graph_edges` persistence primitives for semantic relationships that should not live in the structural turn graph.
  - Added harness `runtime.graph.*` APIs for creating/querying semantic nodes and edges.
  - Added `runtime.graph.selected_path(...)` so graph relationships targeting turns or branch heads can materialize an execution-scoped selected path directly.
  - Added optional durable sidestep branch attachments into the sparse graph overlay.
- **Branch Provenance**
  - Added persisted branch provenance fields so branch heads record why they were created, including sidestep, promotion, and conflict-fork origin metadata.

### Changed
- **Persistence Model**
  - Removed compatibility mirror tables `messages` and `tool_executions`; Turin now persists transcript and tool execution history only through `turn_messages` and `turn_tool_executions`.
  - Advanced the state schema to version `14`; existing state DBs must still be deleted and recreated because Turin does not provide in-place migrations.
- **Execution Semantics**
  - Execution-scoped semantics are now the real substrate model rather than an implementation plan: explicit context targets, explicit write targets, detached execution paths, and branch-native sidesteps are all part of the current runtime.

## Archived Releases

Older release entries are preserved in the changelog archive:

- [Releases 0.20.0 – 0.27.0 (2026-03-02 to 2026-04-02)](docs/changelog/0.20-0.27.md)
- [Releases 0.10.0 – 0.19.0 (2026-02-18 to 2026-03-01)](docs/changelog/0.10-0.19.md)
- [Releases 0.1.0 – 0.9.5 (2026-02-10 to 2026-02-16)](docs/changelog/0.1-0.9.md)
