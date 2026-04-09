# Migration Notes

This file tracks consumer-facing migration notes for releases with behavioral or API changes.

## 0.7.0

### Breaking changes
None.

### Behavioral changes
1. Embeddings are now a first-class provider surface in the shared registry layer instead of downstream-only ad hoc wiring.

### New capabilities
1. Added provider-agnostic embeddings support to `inference-sdk-core`.
2. Added registry-backed embedding provider construction in `inference-sdk-registry`.
3. OpenAI-compatible embedding endpoints can now be targeted with custom `base_url` values, including local embedding servers.

## 0.5.0

### Breaking changes
1. Replaced `InferenceEvent::ToolCall { id, name, args }` with:
   - `InferenceEvent::ToolCallStart { id, name }`
   - `InferenceEvent::ToolCallDelta { delta }`
2. Removed `InferenceEvent::Error`; stream failures now surface via `Err(SdkError)` only.
3. `InferenceResult::from_stream` enforces stream invariants strictly and returns invariant violations as errors.

### Behavioral changes
1. Malformed streamed tool JSON is now an error instead of silently falling back.
2. Retry/timeout handling is policy-driven (`RetryPolicy`, `TimeoutPolicy`) and bounded.

### New capabilities
1. Added `inference-sdk-registry` for driver-based provider construction (`openai`, `anthropic`).
2. Added performance guardrails:
   - release-mode perf budget tests
   - baseline regression checks
