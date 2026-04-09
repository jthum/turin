# Provider Implementation Guide

This is the authoritative instruction set for adding a new provider crate to this workspace.

Follow this exactly. If you need to break a rule, document why in the PR and get explicit approval.

## Goal

Every provider must feel like the same SDK:

1. Same stream semantics.
2. Same error semantics.
3. Same request option behavior.
4. Same quality bar and test depth.

## Required Crate Structure

Use this layout:

```text
<provider>/
  Cargo.toml
  src/
    lib.rs
    client.rs
    config.rs
    normalization.rs
    resources/
      mod.rs
      <primary_resource>.rs
    types/
      mod.rs
      <api_types>.rs
  tests/
    integration_tests.rs
  examples/
    <provider>_simple.rs
    <provider>_stream.rs
```

## Non-Negotiable Rules

1. Never swallow malformed provider payloads.
2. Never emit in-band error events for transport/protocol failures.
3. Always surface stream failures as `Err(SdkError)`.
4. Never store raw API keys as long-lived config fields unless absolutely required.
5. `Debug` output must redact secrets.
6. Never emit `ToolCallDelta` before `ToolCallStart`.
7. Never emit empty normalized assistant messages.
8. Reuse `core` `RequestOptions` behavior and retry logic.
9. If provider needs beta/experimental headers, make them configurable.
10. All docs/examples must compile.

## Event Mapping Contract

Map provider stream events to normalized events exactly as follows:

1. `MessageStart`:
   - emit once per assistant message
   - include `provider_id`
2. `MessageDelta`:
   - emit text increments as they arrive
3. `ThinkingDelta`:
   - emit thinking/reasoning increments when available
4. `ToolCallStart`:
   - emit when tool id + name are first known
5. `ToolCallDelta`:
   - emit JSON argument fragments in arrival order
6. `MessageEnd`:
   - emit usage + stop_reason when available
7. Error:
   - provider protocol/transport errors must return `Err(SdkError)`

## Request Normalization Rules

In `normalization.rs`:

1. Convert generic `InferenceRequest` to provider request types.
2. Preserve ordering of input messages.
3. Convert tool definitions consistently.
4. Skip invalid/empty message artifacts (do not send empty assistant messages).
5. Return serialization errors explicitly.

## Client and Config Rules

In `config.rs` and `client.rs`:

1. Build provider auth headers in `ClientConfig::new`.
2. Keep timeout and retry defaults explicit.
3. Provide `with_base_url`, `with_timeout`, `with_max_retries`.
4. Any provider-specific switches (beta headers, API versions) must be explicit setters.
5. `Debug` implementation must redact sensitive values.

## Resource Layer Rules

In `resources/*.rs`:

1. Use `inference_sdk_core::http::send_with_retry` for request dispatch.
2. Respect `RequestOptions` overrides.
3. For stream endpoints:
   - configure stream flags/options required for complete metadata
   - map SSE to provider chunk types with explicit error mapping

## Test Requirements (Mandatory)

## Unit tests

1. Normalization request mapping.
2. Stream adapter event mapping.
3. Error propagation paths.

## Integration tests

Use `wiremock` to cover:

1. Success path.
2. Retry on retryable status.
3. Retry exhaustion.
4. Error body mapping.
5. Request options override behavior.
6. Secret redaction in debug output.

## Contract tests

Provider must pass shared stream/event contract tests (event ordering, tool delta invariants, message end behavior).

## Example and Docs Requirements

1. Provide at least:
   - simple completion example
   - streaming example
2. If tools are supported, include a tools example.
3. README snippets must compile in CI.

## Pre-Merge Checklist

Before opening PR:

1. `cargo fmt --all --check`
2. `cargo clippy --workspace --all-targets --all-features -- -D warnings`
3. `cargo test --workspace`
4. provider contract tests pass
5. docs/examples compile checks pass
6. changelog updated
7. migration notes added if behavior or API changed

## Pull Request Template (Provider Additions)

PR description must include:

1. Provider capabilities:
   - chat
   - streaming
   - tools
   - thinking/reasoning
   - embeddings/other
2. Event mapping table (provider event -> normalized event).
3. Known limitations.
4. Security notes (auth/header handling, secret redaction).
5. Performance notes (stream behavior and buffering strategy).

## Definition of Done for a New Provider

A provider is done only if:

1. It behaves identically to existing providers from a consumer perspective.
2. It passes all workspace quality gates.
3. It requires zero special-case logic in downstream consumer code beyond provider selection.
