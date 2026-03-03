# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.21.0] - 2026-03-03

### Added
- **Daemon Session Routing and Resume**
  - Added live daemon session slots so one agent can host multiple concurrent live conversation threads.
  - Added `session.open` and `session.list_live` to explicitly manage live daemon slots.
  - Extended `task.submit` to target an existing live `session_id` in addition to the agent-targeted submission path.
  - Added persisted `session.resume` across daemon restart, preserving the same session ID and reconstructing history/counters from persisted state.
- **Stronger Daemon Control Semantics**
  - Added cooperative `session.cancel` and forceful `session.kill` with truthful task/session state transitions.
  - Added `agent.reload`, `runtime.reload`, and richer agent/harness issue inspection surfaces.
- **Daemon Observability**
  - Added typed daemon error taxonomy.
  - Added trace fields/task trace correlation in runtime logs.
  - Added ADRs documenting the key runtime and daemon architectural decisions.

### Changed
- **Daemon Protocol and Maintainability**
  - Replaced the stringly daemon request surface with a typed protocol layer.
  - Added socket-level daemon integration tests over the real Unix-socket NDJSON transport.
  - Split daemon state, server, and dispatch logic into focused submodules.
  - Flattened repeated daemon CLI args and improved the default human-readable output for daemon commands.
- **Daemon Runtime Model**
  - Agent runtimes are now keyed by runtime slot instead of raw agent ID, which removes the old one-live-session-per-agent assumption from the daemon runtime registry.
  - Daemon state access now allows concurrent reads at the server boundary via `RwLock` rather than serializing all operations behind one exclusive mutex.
- **Dependency Reproducibility**
  - Pinned git dependencies to concrete revisions for reproducible builds.

### Fixed
- **Daemon Fault Isolation**
  - Broken agent and harness files are now isolated more cleanly during registry scan and surfaced through classified issue events (`agent.load_failed`, `harness.load_failed`) instead of poisoning unrelated daemon state.
- **Dispatch Maintainability Hotspot**
  - Replaced the old large daemon dispatch file with request handlers grouped by API domain.

## [0.20.0] - 2026-03-02

### Added
- **Filesystem-Backed Daemon Mode**
  - Added a Unix-socket NDJSON daemon server for local control-plane use.
  - Added filesystem-backed agent and harness registry scanning from `agents/` and `harnesses/`.
  - Added daemon event subscription via `runtime.events.subscribe`.
  - Added explicit `runtime.reload` alongside `runtime.rescan` for clients that want an intentional reload operation over the same filesystem-authoritative state.
- **Daemon Agent Management**
  - Added daemon APIs and CLI wrappers for:
    - `agent.list`
    - `agent.get`
    - `agent.status`
    - `agent.issues`
    - `agent.reload`
    - `agent.create`
    - `agent.update`
    - `agent.enable`
    - `agent.disable`
    - `agent.bind_harness`
    - `agent.use_local_harness`
    - `agent.delete`
- **Daemon Task Management**
  - Added daemon APIs and CLI wrappers for:
    - `task.submit`
    - `task.get`
    - `task.wait`
    - `task.cancel`
    - `task.list`
  - Added `turin daemon task submit ... --wait` for the common “submit then block for result” workflow.
  - Added truthful queued-task cancellation; running tasks now reject cancellation instead of pretending to interrupt live inference.
- **Daemon Harness Management**
  - Added daemon APIs and CLI wrappers for:
    - `harness.list`
    - `harness.create`
    - `harness.get`
    - `harness.issues`
    - `harness.reload`
    - `harness.validate`
    - `harness.delete`
- **Daemon Session Inspection**
  - Added daemon APIs and CLI wrappers for:
    - `session.list`
    - `session.resume`
    - `session.get`
    - `session.cancel`
    - `session.kill`
  - Added persisted session detail inspection including events, messages, and tool executions.
  - Added persisted-session resume into a live daemon slot, preserving the same session ID and rehydrating history and counters from the store across daemon restart.

### Changed
- **Daemon Status Surface**
  - `daemon.status` now includes registry snapshot, harness runtime snapshots, and live agent runtime snapshots.
  - Disabled agents now remain visible in daemon runtime status as non-running entries instead of disappearing from the live status surface.
- **Daemon CLI Readability**
  - Added human-readable default output for daemon task and session inspection commands instead of raw JSON-shaped blobs.
  - Added richer human-readable detail output for `agent.get`, `agent.status`, `agent.issues`, `harness.get`, and `harness.issues`.
  - Live agent runtime status now includes the current runtime session and current request IDs for operator control workflows.
  - Daemon task surfaces now distinguish `queued`, `running`, and terminal task states instead of collapsing all in-flight work into `pending`.
  - Running task cancellation now reports the honest intermediate `cancelling` state before terminal completion.
- **Filesystem-Authoritative Runtime Model**
  - Agent directories are now treated as the authoritative persisted daemon state.
  - Shared harness rebinding now preserves user code by only auto-removing default scaffold local harnesses.

### Fixed
- **Daemon Fault Isolation**
  - Invalid `agent.toml` files now surface through isolated daemon runtime errors without poisoning unrelated agents.
- **Rescan Safety**
  - Daemon rescans now refuse to swap kernels while tasks are active or queued.
- **Cooperative and Forceful Runtime Stops**
  - Running task cancellation now stops work at real execution boundaries instead of rejecting live cancellation outright.
  - Added cooperative session cancellation with runtime session rotation after completion.
  - Added forceful session kill for immediate peer runtime teardown with truthful `killed` task results.

## [0.19.0] - 2026-03-01

### Changed
- **First-Class Multi-Harness Internals**
  - Replaced the old single-harness kernel slot with a manager-backed harness runtime model.
  - Moved the session/task/turn execution stack under `ExecutionHost`, leaving `Kernel` as a thinner composition shell around execution hosting and watcher ownership.
  - Peer runtimes now execute against `ExecutionHost` directly instead of wrapping a full `Kernel`.
- **Named Harness Binding**
  - Agents now bind to named harnesses through `[harnesses.*]` plus `harness = "<id>"` on agent configs.
  - Removed per-agent path override wiring in favor of explicit harness IDs.
- **Shared Runtime Reuse**
  - Peer execution now reuses configured tool registry, governance/policy managers, harness manager, provider clients, and embedding provider more directly.
- **Targeted Multi-Harness Reload**
  - Harness file watching now reloads only the affected harness runtime(s) rather than reloading every configured harness.
  - Watcher roots are rebuilt after reload so changed `watch(...)` declarations take effect without requiring a restart.

### Added
- **Harness Runtime Introspection**
  - Added harness runtime snapshots and agent-targeted loaded-script inspection surfaces.
  - `turin check` now validates all configured harness runtimes rather than only the default harness.
  - Harness snapshots and `turin check` now surface watched roots in addition to harness IDs, bound agents, and loaded scripts.

### Fixed
- **Peer Harness Hot Reload**
  - Peer and secondary harness roots now participate in watcher-driven reload through the shared harness manager.
- **Shared Harness Reuse / Isolation Coverage**
  - Added regressions proving multiple agents bound to the same named harness resolve to the same runtime while other harnesses remain isolated.
- **Default-Harness Fallback Footguns**
  - Removed the old default-harness convenience path from kernel execution surfaces and tightened fallback visibility around missing harness bindings.

## [0.18.0] - 2026-02-28

### Added
- **Harness Composition Primitives**
  - Added `use(name, opts?)` for mounting reusable behavior blocks during harness load.
  - Added `use_scoped(name, opts?)` for governance-scoped behavior blocks with delegated capability ceilings.
  - Added nested-path module resolution for harness-local imports such as `import("blocks/foo")`.
- **Explicit Watch Model**
  - Added `watch(path)` for explicitly registering extra harness-relative paths for hot reload.
  - Kept the existing top-level harness directory watch as the default, with watched subtrees opt-in from harness code.

### Changed
- **Harness Loading**
  - Harness blocks can now be reused as ordinary harness scripts via `use(...)`, without rewriting them into `return { ... }` form.
  - Hook-contributing modules loaded through `use(...)` now participate in the same hook pipeline as top-level harness scripts.
- **Peer Completion Output Extraction**
  - Peer-task output extraction now only returns assistant-role text, preventing `runtime.agent.complete(...)` from accidentally returning the delegated user prompt.

### Fixed
- **Delegated Capability Preservation**
  - Hook evaluation now preserves an active delegated capability ceiling when a module does not define its own module-scoped ceiling, avoiding accidental loss of peer-task delegation during hook execution.
- **Hot Reload Coverage**
  - Added regression coverage proving nested `use(...)` blocks reload when declared through `watch("...")`.

## [0.17.0] - 2026-02-28

### Added
- **First-Party Harness DX Layer**
  - Added first-party DX helpers under `src/harness/dx/` and documented them as part of Turin's script-author surface.
  - Added verdict helpers:
    - `verdict.allow`
    - `verdict.reject`
    - `verdict.escalate`
    - `verdict.modify`
    - `verdict.reject_if`
    - `verdict.escalate_if`
  - Added access helpers:
    - `allowed(...)`
    - `needs(...)`
    - `access.check(...)`
  - Added session/user DX helpers:
    - `session.*`
    - `user.*`
    - including `remember`, `recall`, `get`, `set`, `del`, and `incr`
  - Added callable DX surfaces:
    - `runtime.db(selector)` with `:one`, `:all`, `:exec`, `:close`, and `runtime.db.with(...)`
    - `runtime.agent(agent_id)` with `:complete`, `:submit`, `:await`, and `:status`
  - Added DX helpers for:
    - `runtime.governance.grant(spec, fn)`
    - `time.since(...)`
    - `time.after(...)`
    - `fs.read_json(...)`
    - `fs.write_json(...)`
- **Harness Library**
  - Added a first-class Harness Library under `library/` with:
    - `blocks/` for reusable harness units
    - `workflows/` for complete end-to-end harness systems
  - Added current workflow entries:
    - `openclaw_style_personal_assistant`
    - `full_coding_harness`
    - `bug_triage_desk`
    - `release_manager`
    - `docs_team_assistant`
  - Added current block entries:
    - `code_reviewer`
    - `task_planner`
    - `spec_writer`
    - `test_gap_finder`
    - `repo_librarian`
    - `release_readiness_checker`
    - `docs_maintainer`
    - `changelog_writer`
    - `governed_peer_review`
    - `delegated_peer_capabilities`
    - `durable_journal`
- **Expanded Harness/Example Validation**
  - Added realistic DX fixture coverage and import-scoping fixture coverage for the new DX layer.
  - Added Harness Library integration coverage through `cargo test --test example_harness_examples`.
  - Added live-provider `peer_complete_caps` coverage to `scripts/live_minimax_smoke.sh` for delegated-capability peer completion validation.

### Changed
- **Canonical Peer Completion API**
  - Added native canonical `runtime.agent.complete(agent_id, prompt, opts?)` and rewired DX `runtime.agent(...):complete(...)` to use the native path.
  - Reduced boilerplate for canonical peer completion from explicit `submit + await + unwrap` to a single primitive call.
- **Documentation Surface**
  - Promoted the DX layer and Harness Library into the main documentation set.
  - Added dedicated Harness Library documentation and expanded practical harness guidance.
- **Harness Library Structure**
  - Reorganized serious harnesses into the canonical `library/blocks/` and `library/workflows/` taxonomy.

### Fixed
- **Peer Completion Post-Effect Reliability**
  - Fixed the quality gap where side effects immediately after peer completion could behave inconsistently in harness flows.
  - Added regression coverage proving post-`runtime.agent.complete(...)` side effects across:
    - filesystem writes
    - DB writes
    - runtime policy mutation
    - session state mutation
    - nested temporary grants
    - import-scoped delegated flows
- **Delegated Capability Propagation**
  - Hardened delegated capability behavior for peer completion in both canonical and DX call sites.
  - Added regression coverage proving delegated peers can use explicitly granted capability slices while remaining denied non-delegated mutations.

### Documentation
- Updated `README.md` to surface the Harness Library and first-party DX layer more clearly.
- Updated docs index and live-provider docs to reflect the current Harness Library and delegated peer-completion validation surface.

## [0.16.0] - 2026-02-25

### Added
- **Expanded Live Validation Suites**
  - Added `--api-format openai` mode to `scripts/live_minimax_smoke.sh` for testing OpenAI-compatible endpoints (including MiniMax `https://api.minimax.io/v1`).
  - Added `--report-json` output for machine-readable live test summaries (with per-case durations and temp dirs).
  - Added repeatable `soak` suite support (`--suite soak`) and generic `--repeat` support, with iteration tracking in JSON reports.
  - Added provider-debug wiring so `--debug-requests` enables both Anthropic and OpenAI SDK request/stream dumps.
- **MiniMax OpenAI-Compatible Interop (via normalized SDK updates)**
  - Added OpenAI request normalization `tool_choice = "auto"` when tools are present.
  - Added OpenAI request/raw-SSE debug dump tooling for provider compatibility debugging.
  - Added OpenAI stream normalization support for providers that emit final usage chunks with non-empty `choices` (MiniMax-compatible shape), including duplicate assistant-role chunk handling.
- **Runtime Hardening and UX Polish**
  - Added MCP client graceful shutdown / teardown path (including subprocess cleanup via upstream MCP SDK changes).
  - Added runtime policy knob `hook.token_usage.reject_mode` with modes:
    - `informational`
    - `enforce_task`
    - `enforce_session`
  - Added terminal-aware display helpers across more CLI/kernel output surfaces and cleaned remaining ANSI hotspots.

### Changed
- **Live Validation Baselines (MiniMax M2.5)**
  - Verified `smoke`, `core`, and `soak` suites against MiniMax Anthropic-compatible endpoint (`https://api.minimax.io/anthropic/v1`).
  - Verified `smoke`, `core`, and `soak` suites against MiniMax OpenAI-compatible endpoint (`https://api.minimax.io/v1`).
  - Recorded known-good `core` baselines (`12/12`) and `soak` baselines (`36/36`, `repeat_count=3`) for both wire protocols.
- **Harness / CLI Cleanup**
  - Extracted REPL command flow from `main.rs` and reduced config-loading duplication.
  - Consolidated harness execution context state into a single lock-backed context object.
  - Relaxed peer-agent live test assertions to tolerate provider reasoning wrappers while still requiring sentinel content.
- **Dependency / Binary Trimming**
  - Disabled unnecessary default features on several dependencies (`tokio`, `tokio-util`, `clap`, `turso`) and removed unused dependencies.
  - Reduced release binary size through feature trimming (while keeping tests/clippy/builds green).

### Fixed
- **MiniMax OpenAI-Compatible Tool Calling**
  - Fixed missing tool execution on OpenAI-compatible streaming path by handling provider final usage chunk shape and ensuring tool-call streams are finalized correctly.
  - Fixed zero-token live summaries caused by missing `MessageEnd` emission on MiniMax-style final usage chunks.
- **Live Suite Reliability / Portability**
  - Removed hard dependency on `ripgrep` (`rg`) in the live suite (falls back to `grep`).
  - Reduced live-suite output noise defaults (`--log-level error`) and improved JSON reporting for failure triage.
  - Fixed `peer_grant` live case grant ceiling to include required orchestrator capabilities (`runtime.agent.submit` / `runtime.agent.await`) while still validating worker-side grant propagation.
- **FFI/Panic Safety**
  - Hardened Lua-facing harness execution-context access paths to avoid `unwrap()`-driven panics crossing the `mlua` callback boundary.

### Documentation
- Removed stale `v0.15.0` version labels from README/docs to keep documentation evergreen.
- Expanded live-provider testing docs with:
  - Anthropic/OpenAI-compatible MiniMax commands
  - `core` / `soak` guidance
  - repeat/soak workflow examples
- Added a Turso acknowledgement to `README.md`.

## [0.15.0] - 2026-02-24

### Added
- **Canonical Harness Standard Library (`runtime.*`)**
  - Added stable canonical runtime namespaces:
    - `runtime.context`
    - `runtime.memory`
    - `runtime.kv`
    - `runtime.db`
    - `runtime.agent`
    - `runtime.policy`
    - `runtime.governance`
  - Added top-level ergonomic aliases and data helpers:
    - `memory.*`, `kv.*`
    - `session.memory/kv.*`, `user.memory/kv.*`
    - `agent.*`, `agent.session.*`, `agent.mode.*`
    - `import(...)`, `import_scoped(...)`
- **Dynamic Multi-DB Runtime**
  - Added store handle manager and dynamic database open/query/exec/list/close APIs via `runtime.db.*`.
  - Added path-scope policy control and cache trimming/idle handle management.
  - Added alias/path selector support for multiple databases.
- **Dynamic Multi-Agent Runtime**
  - Added peer-agent runtime registry with async task submission/awaiting and result tracking.
  - Added idle shutdown/restart behavior for peer runtimes.
  - Added runtime status inspection via `runtime.agent.list/get_status`.
- **Governance (Opt-In, Flexibility-First)**
  - Added governance config schema (`[governance]`) with profiles (`open`, `balanced`, `governed`, `custom`).
  - Added governance observability APIs (`runtime.governance.profile/snapshot/check/agent`).
  - Added capability enforcement (opt-in) for high-risk runtime APIs and built-in tool execution.
  - Added import governance modes (`legacy`, `mixed`, `scoped`) with `import_scoped(...)` root assertions.
  - Added import-scoped delegated capability ceilings (downward-only).
  - Added agent capability profiles, per-agent ceilings, and `allowed_child_agents` allowlists.
  - Added temporary governance grants (TTL / max uses) with `grant_issue`, `grant_get`, `with_grant`, `grant_revoke`.
  - Added durable governance audit events:
    - `governance_snapshot`
    - `governance_grant_issue`
    - `governance_grant_use`
    - `governance_grant_revoke`
  - Added immutable audit mode support (`persist-before-hooks` semantics for audit events).
- **Import Principal Context Propagation**
  - Imported module function execution now preserves module/root subject attribution for governance checks.
  - Added recursive export proxy wrapping so nested exported functions preserve imported-module context.
- **Live Provider Validation Tooling**
  - Added opt-in live MiniMax smoke test script (`scripts/live_minimax_smoke.sh`) with tool-roundtrip cases.

### Changed
- **Major Architecture Cleanup / Decomposition (No Legacy Shims)**
  - Decomposed `src/harness/globals.rs` from a large monolithic stdlib registration file into focused `src/harness/stdlib/*` modules.
  - Introduced shared binding helpers and support modules for DRY Lua bindings (`binding_common`, `db_support`, `policy_support`, `identity_support`, etc.).
  - Decomposed `kernel::turn` into focused submodules:
    - preflight
    - streaming
    - assistant response finalization
    - tool execution (with helper submodules)
  - Decomposed `kernel::mod` responsibilities into focused modules:
    - session lifecycle
    - event persistence
    - harness hooks
    - run loop
    - task execution/lifecycle
    - MCP runtime
  - Decomposed `agent_manager` and `persistence::manager` into focused support modules.
- **Runtime Identity Refactor**
  - Refined `RuntimeIdentity` internals and access patterns (including richer `extra` identity context support).
  - Improved identity/selector handling used across hooks, persistence, and stdlib bindings.
- **Provider-Agnostic Turn History Correctness**
  - Assistant thinking content is preserved in in-memory history between turns (not just transient UI output).
  - Tool results are recorded in normalized inference history as tool-role messages for correct provider roundtrips.
  - Turin now propagates provider-agnostic thinking signature deltas through the turn pipeline into assistant history.
- **Harness Import System**
  - `import(...)` and `import_scoped(...)` now return wrapped module proxies that preserve caller/imported module governance context.
  - Added support for delegated capability wildcard rules (`prefix.*`) with downward-only checks.
- **Observability Logging**
  - Downgraded benign “event broadcast with no receivers” message from warning to debug.

### Fixed
- **Anthropic-Compatible Provider Interop (via upstream normalized SDK integration)**
  - Fixed MiniMax Anthropic-compatible thinking-block parsing (`signature` optional on decode).
  - Fixed Anthropic-compatible tool roundtrip request normalization issues (tool result serialization / thinking preservation).
  - Improved compatibility for Anthropic-style thinking + tool-use continuations by preserving thinking signatures end-to-end.
- **Governance Bypass Gaps**
  - Closed top-level alias bypasses by applying governance checks to `fs.*` and `agent.*` high-risk paths.
  - Added kernel-side tool capability checks so direct model-emitted built-in tool calls are governed even without stdlib mediation.
- **Nested Delegation Safety**
  - Prevented nested `import_scoped(...)` capability delegation widening beyond importer ceilings.
  - Applied active grant ceilings to peer-agent dispatch delegation to prevent escalation through sub-agent paths.
- **Live Endpoint Validation**
  - Verified Turin + normalized SDK end-to-end against MiniMax Anthropic-compatible endpoint for:
    - basic inference
    - tool read roundtrip
    - tool error/recovery flow
    - write+read multi-tool roundtrip
  - Expanded and verified the opt-in `core` live validation suite against MiniMax M2.5 (12/12 passing), covering governance, peer agents, queue steering, runtime DB APIs, temporary grants, token-usage enforcement, immutable audit, and peer grant propagation.
  - Added MiniMax OpenAI-compatible live-suite mode (`--api-format openai`) and verified `smoke` + `core` suites against `https://api.minimax.io/v1` (including `core` 12/12 passing).
  - Added repeatable `soak` suite support (`--suite soak`, `--repeat`) and verified MiniMax M2.5 soak baselines on both Anthropic-compatible and OpenAI-compatible endpoints (`36/36` passing over 3 iterations each).
  - Fixed MiniMax OpenAI-compatible tool-call interoperability in the normalized OpenAI SDK path by setting `tool_choice: "auto"` when tools are present and handling final usage chunks that include non-empty `choices`.

### Documentation
- Rewrote and expanded core documentation for the canonical stdlib API, stable hooks, governance model, architecture, testing, and live provider validation.
- Updated examples and configuration guidance for Anthropic-compatible providers (including MiniMax `/v1` base URL note).

## [0.14.0] - 2026-02-19

### Added
- **Hook Lifecycle Overhaul (Breaking)**:
  - Added explicit lifecycle hooks: `on_task_start`, `on_plan_complete`, `on_all_tasks_complete`.
  - Added `turn_prepare` lifecycle event and richer task/plan lifecycle event payloads.
- **Structured Task/Plan Runtime Model**:
  - Queue now stores structured task items with `task_id`, `plan_id`, `title`, and `prompt`.
  - Lightweight in-memory plan progress tracking for deterministic `on_plan_complete` firing.
- **Mutable Tool Result Governance**:
  - `on_tool_result` now supports `MODIFY` to rewrite tool output/error status before reinjection.

### Changed
- **Breaking Hook and Tool Renames**:
  - `on_before_inference` -> `on_turn_prepare`
  - `on_task_submit` -> `on_plan_submit`
  - `on_agent_start` -> `on_session_start`
  - `on_agent_end` -> `on_session_end`
  - `submit_task` tool -> `submit_plan`
- **Task Completion Semantics Clarified**:
  - `on_task_complete` now fires per terminal task.
  - Global queue-drain behavior moved to `on_all_tasks_complete`.
- **Context Wrapper Enrichment**:
  - `ctx` now exposes turn/task metadata (`turn_index`, `task_turn_index`, `is_first_turn_in_task`, `task_id`, `plan_id`).

### Fixed
- Fixed stale hook loading behavior by ensuring `on_task_complete` and new lifecycle hooks are discoverable in fallback script loading.
- Fixed stale docs/examples that still referenced `subtasks`, queue-exhausted `on_task_complete`, and legacy lifecycle naming.
- Removed temporary source artifact `src/harness/engine.rs_test_append`.

## [0.13.0] - 2026-02-18

### Added
- **Project Scaffolding (`turin init`)**:
  - New command to bootstrap Turin projects with default `turin.toml` and starter harness scripts.
- **Static Validation (`turin check`)**:
  - Validation engine for project configurations, API keys, and harness script syntax with line-specific Lua diagnostics.
- **Enhanced Developer Experience (DX)**:
  - **Richer REPL**: Added slash commands (`/status`, `/history`, `/reload`, `/clear`, `/help`), colored prompts.
  - **Streaming Indicators**: Visual markers for "Thinking" blocks, turn headers, tool call verdicts, and execution results.
  - **Session Summary**: Automated token usage and turn count reports.

### Changed
- **Dependency Modernization**:
  - Upgraded to `reqwest 0.13` and `hyper 1.0`.
  - Migrated to **Rustls (with aws-lc-rs)** as the primary TLS backend, eliminating `native-tls` and its transitive C dependencies for a fully portable binary.
- **Public API Refinement**: 
  - Exposed `Kernel::config()` and `Kernel::loaded_scripts()` for command-line introspection.

## [0.12.0] - 2026-02-18

### Added
- **Core State Store Modularization**:
  - Split the monolithic `persistence/state.rs` (1,000+ lines) into three focused modules: `schema.rs` (schema and DDL), `search.rs` (cognitive/hybrid search), and `state.rs` (lifecycle, CRUD, and KV logic).
- **Robust Persistence Gating**: 
  - Implemented a mandatory `busy_timeout` (5000ms) on all database connections to prevent `SQLITE_BUSY` errors during concurrent access (e.g., nested sub-agents writing to DB while background event persistence is active).
- **Automated Quality Controls**:
  - Added **GitHub Actions CI** for automated testing, clippy auditing, and release builds.
  - Integrated **cargo-deny** for vulnerability auditing and license compliance.

### Changed
- **Unified Logging Architecture**:
  - Migrated remaining internal `eprintln!` calls to structured `tracing` events (`warn`, `error`).
  - Harness `log()` calls remain on `eprintln!` for clear separation between kernel diagnostics and harness output.

### Fixed
- Resolved a race condition in `test_nested_agent_spawning` caused by connection-local database pragmas.
- Fixed 2 pre-existing clippy warnings in `session_tests.rs`.


## [0.11.0] - 2026-02-18

### Added
- **Defense-in-Depth Security**:
  - **Lua Sandboxing**: Replaced `Lua::new()` with `Lua::new_with(StdLib::ALL_SAFE)`, excluding IO, OS, FFI, and PACKAGE from the Luau runtime.
  - **Shell Timeout Kill**: `shell_exec` now uses `tokio::select!` to race execution against a timeout, properly killing orphaned child processes.
  - **Agent Spawn Depth Limit**: `agent.spawn` enforces a max depth of 3 via `AtomicU32` counter, preventing infinite recursive spawning.
  - **File Write Size Limit**: `fs.write` rejects content larger than 10MB to prevent disk exhaustion.
- **Session Lifecycle**:
  - Added `CancellationToken` to `SessionState` for clean background task shutdown on `end_session`.

### Changed
- **Kernel Modularization**: Split `kernel/mod.rs` from 1,041 → 413 lines into three focused files:
  - `kernel/init.rs` — Provider clients, state store, harness initialization, file watcher.
  - `kernel/turn.rs` — `execute_turn` and `execute_tool_calls` logic.
  - `kernel/mod.rs` — Struct definition, session lifecycle, agent loop, event persistence.
- **API Hygiene**:
  - All `Kernel` struct fields narrowed from `pub` to `pub(crate)` with a new `state()` accessor.
  - Removed implicit OpenAI embedding fallback — now defaults to NoOp when `[embeddings]` is not configured.
- **Path Validation Consolidation**: `resolve_safe_path` (harness) now delegates to `is_safe_path` (tools), eliminating duplicated validation logic.
- **Dependency Optimization**: Replaced `tokio = { features = ["full"] }` with explicit features, added `tokio-util` for `CancellationToken`.
- **Rust 2024 Edition**: Migrated from `edition = "2021"` to `edition = "2024"`, adopting the latest language defaults and lint rules.

### Removed
- Dead `mcp_clients` field from `SessionState` (MCP clients live on Kernel, not Session).
- Deprecated `Kernel::new()` constructor (use `Kernel::builder()` instead).

### Fixed
- 17 clippy warnings resolved (zero remaining).
- `persist_event_internal` no longer silently swallows broadcast failures (logs a warning).
- Corrected misleading `time.now_utc` doc comment (returns Unix timestamp, not ISO 8601).

## [0.10.0] - 2026-02-18

### Added
- **Advanced Observability**:
  - `on_kernel_event` hook: Enables harness scripts to observe all internal kernel events (Lifecycle, Stream, Audit) in real-time.
  - Flattened serialization for `KernelEvent` to improve Lua ergonomics.
- **First-Class Nesting Support**: 
  - Verified recursive agent spawning via `turin.agent.spawn` with isolated state and harness context.
- **Harness Ergonomics**:
  - Added `prompt` helper to `ContextWrapper` for simplified access and mutation of the last user message.

### Changed
- **Performance & Stability**:
  - Enabled **WAL (Write-Ahead Logging)** mode in the Turso/SQLite backend to significantly improve concurrent write performance.
  - Implemented a 5-second busy timeout for database operations to resolve contention during high-frequency event streaming.
  - Refactored harness engine synchronization to use a blocking `std::sync::Mutex`, ensuring guaranteed sequential event capture for "god-view" observers.

## [0.9.5] - 2026-02-16

### Added
- **Testing Infrastructure**: 
  - Integration tests for the agent loop and harness governance.
  - Property-based testing for path validation using `proptest`.
- **Robust Path Validation**: Introduced a centralized, fuzzed `is_safe_path` utility to prevent traversal attacks.

### Changed
- **Architectural Optimization**: Refactored `Kernel` to use `Arc<TurinConfig>`, significantly reducing cloning overhead.
- **Async I/O**: Switched all file tool metadata calls to async `tokio::fs::metadata`.

### Fixed
- Corrected `session.turn_index` increment logic to include the final turn of a task.


## [0.8.5] - 2026-02-15

### Added
- **Resilient Hybrid Search (FTS5 + Vector)**:
  - **Reciprocal Rank Fusion (RRF)**: Implemented state-of-the-art result merging for semantic and keyword search.
  - **Graceful Degradation**: System now handles environments without FTS5 (like standard Turso crate builds) or offline embedding providers without errors.
  - **Tokenized LIKE Fallback**: Introduced a robust "Safety Net" search (Scenario D) that uses tokenized SQL `LIKE` queries when both vector and FTS engines are unavailable, ensuring keyword retrieval always works.

## [0.8.0] - 2026-02-14

### Added
- **Harness Module System & Hot-Reload**:
  - **Atomic Hot-Reload**: Implemented a "fail-safe" swapping mechanism for harness scripts via a directory watcher (Phase 2) and `/reload` command.
  - **First-Class Module System**: Harness scripts can now `return` tables, enabling clean exported APIs.
  - **turin.import(name)**: New global helper to access exported modules from other scripts.
  - **Prioritized Hook Discovery**: Unified discovery logic that prioritizes hooks in a script's return table over the global environment.
  - **Debounced Watcher**: Added an asynchronous file watcher in the `Kernel` to automatically trigger reloads on script changes with 200ms debouncing.

## [0.7.0] - 2026-02-14

### Added
- **Named Providers & Multi-Instance Support**:
  - Supported arbitrary naming for provider instances in `turin.toml` (e.g., `[providers.my-fast-client]`).
  - Introduced `type` field in provider configuration to support multiple instances of the same provider kind.
  - Exposed `ctx.provider` setter in Lua `on_before_inference` hook for dynamic, mid-turn switching.
  - Refactored internal `Kernel` and `Harness` logic to resolve clients by their configured string names.

## [0.6.0] - 2026-02-14

### Changed
- **Capability Normalizer Architecture**:
  - Refactored `ProviderClient` to be provider-agnostic by utilizing the `InferenceProvider` trait from the normalized SDK.
  - Standardized streaming event handling: Turin now consumes a unified `InferenceEvent` stream regardless of the backend (OpenAI or Anthropic).
  - Decoupled inference and embeddings logic: Embeddings are now handled through a dedicated `EmbeddingProvider` abstraction.
  - Simplified kernel-to-provider communication, removing thousands of lines of provider-specific boilerplate and mapping logic.

---

## [0.5.0] - 2026-02-13

### Added
- **Adaptive Thinking**:
  - Full support for Anthropic's extended reasoning (Claude 3.7 Sonnet / Opus 4.6).
  - Exposure of `thinking_budget` to the Harness Engine for dynamic reasoning depth control.
- **Cognitive Memory & Anchorage**:
  - Vector search primitives integrated via Turso/SQLite-vec.
  - Automated session summarization and fact anchorage via `on_task_complete` hooks.
- **Multi-Provider Support & Mid-Turn Switching**:
  - Enabled coexistence and switching between Anthropic and OpenAI within the same session.
  - Support for `ctx.provider` overrides in `turin.agent.spawn` and `on_before_inference`.

## [0.4.0] - 2026-02-13

### Added
- **MCP SDK Support**: Integrated a custom, lightweight Rust-based MCP SDK (`mcp-sdk-rust`) into Turin.
- **Dynamic Tool Loading**:
  - `bridge_mcp` tool: Allows agents to request spawning and connecting to external MCP servers.
  - `McpToolProxy`: Automatically registers tools from MCP servers as native Turin tools.
- **Ecosystem Stability Primitives**:
  - `on_task_complete` hook: Enables harnesses to validate state and re-queue tasks when the queue is exhausted.
  - `turin.context` module: New Lua global module providing `context.glob(pattern)` for safe, workspace-aware file discovery.
- **Internal Stability**:
  - Optimized binary size through LTO and symbol stripping (achieving ~11MB).
  - Hardened `run_task` loop with better error recovery and multi-turn consistency.

### Fixed
- Resolved multiple compilation errors related to borrow checking and brace nesting in the core Kernel.
- Fixed duplicate field declarations in the `Kernel` struct.

---

## [0.3.0] - 2026-02-13

### Added
- **Steerable Command Queue**: Added a per-session task queue in the Kernel allowing for persistent, asynchronous steering by humans or harnesses.
- **Interactive REPL**: New `turin repl` command for persistent conversational interaction with the workspace.
- **Task Decomposition Primitives**:
  - `submit_task` tool: Allows agents to propose a multi-step plan.
  - `on_task_submit` hook: Enables harnesses to intercept, approve, reject, or modify agent plans.
- **Verdict::Modify**: Extended the governance system to support data-carrying verdicts. Harnesses can now modify tool arguments (`on_tool_call`) or task lists (`on_task_submit`) on the fly.
- **Steering API**: `session.queue()`, `session.queue_next()`, and `session.clear_queue()` exposed to Lua for active control.

### Changed
- **Kernel Loop**: Refactored the core `run` loop to be queue-driven, supporting multiple sequential tasks within a single persistent session.
- **Harness Engine**: Updated `parse_verdict` to handle `MODIFY` verdict codes and associated JSON data.

---

## [0.2.0] - 2026-02-12

### Added
- **on_before_inference** hook: Enables context engineering and mutation before LLM calls.
- **on_agent_start** hook: Allows harness scripts to initialize state at session startup.
- **Session Globals**: `session.list(limit, offset)` and `session.load(id)` exposed to Lua.
- **Context Globals**: `ctx.summarize()`, `ctx.add_message()`, and `ctx.system_prompt` access.
- **Coding Agent**: Experimental `coding_agent.lua` for automatic `TURIN.md` injection.

### Changed
- **Synchronous Bridge**: Refactored `ctx.summarize` to be synchronous via `block_in_place`, ensuring compatibility with the synchronous Luau VM.
- **Inference Content**: Refactored `InferenceContent::Text` to a struct variant for better `serde` compatibility.
- **Sandboxing**: Enhanced harness loading with per-script environments to prevent global pollution.

---

## [0.1.0] - 2026-02-10

### Added
- **Core Engine**: Embedded Luau (mlua) runtime for sandboxed "Governance Harnesses."
- **Persistence**: Turso-backed `StateStore` for atomic event logging, message history, and tool execution tracking.
- **Tool Registry**: Extensible system for LLM-accessible tools with built-ins:
  - `read_file`, `write_file`, `edit_file` (safe workspace-restricted access).
  - `shell_exec` (with timeout and output truncation).
- **Governance Primitives**: `on_tool_call` (gating) and `on_token_usage` (budgeting) hooks.
- **CLI**: Interactive streaming UX with `--verbose` for debugging and `--json` for programmatic consumption.
- **SDK Integration**: Native adapters for Anthropic and OpenAI.

### Fixed
- Improved output truncation for large shell results to prevent context window overflows.
- Resolved Luau metatable indexing for property access on Context objects.
