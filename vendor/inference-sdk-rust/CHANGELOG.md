# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-03-10

### Added
- **Provider-Agnostic Embeddings**
  - Added a shared embeddings contract to `inference-sdk-core`.
  - Added embedding-driver registry support in `inference-sdk-registry`.
  - OpenAI-compatible drivers can now be built for embeddings through the same generic registry path used for inference providers.
- **OpenAI-Compatible Base URL Embeddings**
  - `openai-sdk` now supports embedding calls against custom OpenAI-compatible `base_url` endpoints.
  - This enables local embedding servers without downstream provider-specific glue code.

### Changed
- **Version Alignment**
  - Bumped `inference-sdk-core`, `anthropic-sdk`, `openai-sdk`, and `inference-sdk-registry` crate versions to `0.7.0`.
- **Registry Scope**
  - `inference-sdk-registry` is now the shared construction layer for both inference providers and embedding providers.

## [0.5.0] - 2026-02-19

### Added
- **True Tool Streaming Events**:
  - Added `InferenceEvent::ToolCallStart { id, name }`.
  - Added `InferenceEvent::ToolCallDelta { delta }`.
- **Core Tool Aggregation Tests**:
  - Added tests for tool-call delta accumulation and malformed JSON handling in `inference-sdk-core`.
- **OpenAI Streaming Usage Inclusion**:
  - OpenAI streaming requests now send `stream_options: { include_usage: true }` to improve normalized `MessageEnd` completeness.
- **Configurable Anthropic Thinking Beta Header**:
  - Added `ClientConfig::with_thinking_beta_header(...)`.
  - Added `ClientConfig::without_thinking_beta_header()`.
- **RequestOptions API Compatibility Alias**:
  - Added `RequestOptions::with_max_retries(...)` as an alias to `with_retries(...)`.
- **Performance Budget Guards**:
  - Added release-mode perf budget tests in `core/tests/perf_budget.rs` for event validation and stream assembly hot paths.
  - Added `docs/PERFORMANCE_GUARDS.md`.
- **Provider Registry/Factory Crate**:
  - Added `inference-sdk-registry`.
  - Provides `ProviderRegistry` + `ProviderInit` to construct providers by generic driver name (`openai`, `anthropic`) without downstream hardcoded provider bootstrapping.

### Changed
- **Breaking**: Removed `InferenceEvent::ToolCall { id, name, args }` in favor of tool start + delta events.
- **Breaking**: Removed `InferenceEvent::Error`; stream errors are now surfaced through `Err(SdkError)` in the stream item type.
- **Breaking (Behavioral)**: Malformed streamed tool JSON now returns an error instead of being silently replaced with fallback values.
- **Security/Retention**:
  - `ClientConfig` no longer stores raw API key fields after header construction (reduces accidental secret retention).
- **Retry Hardening**:
  - Added bounded exponential backoff with jitter.
  - Added retry caps and narrowed retry conditions (transient network/status classes only).
- **Dependency/Feature Trim**:
  - Reduced workspace Tokio defaults and moved provider Tokio usage to dev-only where applicable.
- **CI Quality Gates**:
  - Added workspace example compile check (`cargo check --workspace --examples`).
  - Added release perf budget gate (`cargo test -p inference-sdk-core --release --test perf_budget -- --ignored`).
  - Added release policy automation (`scripts/check_release_policy.sh`) to enforce changelog + migration note updates for API/version-impacting changes.
- **Version Alignment**:
  - Bumped `inference-sdk-core`, `anthropic-sdk`, and `openai-sdk` crate versions to `0.5.0`.
- **Fuzz CI Reliability**:
  - Updated fuzz smoke workflow to run `cargo fuzz` against the runner host target explicitly, avoiding accidental musl target resolution failures.

### Fixed
- **Anthropic Tool Streaming**:
  - `InputJsonDelta` is now mapped to tool-call delta events instead of being ignored.
- **Normalization Robustness**:
  - Avoids emitting empty assistant messages during normalization.
  - Tool-call serialization in normalization now propagates serialization errors.
- **Codebase Cleanup**:
  - Removed duplicate dead `core/src/config.rs` request-options implementation.

---

## [0.4.0] - 2026-02-15

### Added
- **Architectural Refactor (V2)**:
  - **Stateful Stream Adapters**: Fixed critical bugs where OpenAI tool-call fragments and Anthropic usage tokens were lost during streaming.
  - **Enriched `InferenceResult`**: Added `stop_reason` and support for multiple `InferenceContent` blocks (including new `Thinking` variant) to non-streaming responses.
  - **Unified `InferenceProvider` Trait**: Added `RequestOptions` support to `stream()` and `complete()`.
  - **Standardized `Tool` Role**: Introduced `InferenceRole::Tool` to cleanly handle tool execution results across all providers.
  - **Default `complete()`**: Standardized collection logic in `core`, allowing providers to focus on implementing `stream()`.

### Changed
- **Breaking**: Renamed `ToolSpec` to `Tool`.
- **Breaking**: `InferenceMessage` now includes a `tool_call_id` field.
- **Breaking**: `InferenceEvent::MessageEnd` now includes an optional `stop_reason`.

---

## [0.3.0] - 2026-02-14

### Added
- **Capability Normalization**:
  - Introduced `InferenceProvider` trait in `inference-sdk-core` for unified provider access.
  - Standardized `InferenceRequest` and `InferenceEvent` types across all SDKs.
  - Added support for standardized streaming events (`MessageStart`, `MessageDelta`, `ToolCall`, etc.).
  - Both `anthropic-sdk` and `openai-sdk` now implement the common `InferenceProvider` interface.
  - `ProviderKind` now implements `FromStr` for better configuration parsing.

---

## [0.2.0] - 2026-02-14

### Added
- **Adaptive Thinking Support**:
  - `anthropic-sdk`: Added `ThinkingConfig` and `ThinkingDelta` events for Claude 3.7 Sonnet / Opus 4.6.
  - Support for `beta` headers in `MessageRequest`.
- **Embeddings Support**:
  - `openai-sdk`: Implementation of the Embeddings API.
  - Added `EmbeddingRequest`, `EmbeddingResponse`, and `EmbeddingData` models.
- **Improved Request Handling**:
  - Standardized `RequestOptions` to support per-request configuration (e.g., custom headers, proxy settings).

## [0.1.0] - 2026-02-10

### Added
- **Initial Release**: Established the modular Rust workspace for LLM inference.
- **`inference-sdk-core`**:
  - Unified `RequestOptions` for per-request overrides (headers, timeouts).
  - Shared `SdkError` enum for consistent cross-provider error handling.
  - Built-in retry logic with exponential backoff on transient failures.
- **`anthropic-sdk`**:
  - Implementation of the Anthropic Messages API.
  - Support for streaming responses using Server-Sent Events (SSE).
  - Type-safe request builders integrated with the `bon` crate.
  - `AnthropicRequestExt` trait for easy beta header injection.
- **`openai-sdk`**:
  - Implementation of the OpenAI Chat Completions API.
  - Support for streaming responses.
  - Type-safe models for multiple message roles (system, user, assistant, tool).
  - Shared core logic for Bearer token authentication.
