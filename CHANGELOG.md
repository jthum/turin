# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
