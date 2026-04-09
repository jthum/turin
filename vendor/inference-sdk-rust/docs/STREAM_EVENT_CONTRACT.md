# Stream Event Contract

This document defines the required normalized event ordering for all providers.

## Invariants

1. `MessageStart` must be emitted before any `MessageDelta`, `ThinkingDelta`, `ToolCallStart`, or `ToolCallDelta`.
2. `MessageStart` must be emitted exactly once per stream.
3. `ToolCallDelta` must not be emitted before `ToolCallStart`.
4. `MessageEnd` must be emitted exactly once per stream and only after `MessageStart`.
5. No events may be emitted after `MessageEnd`.
6. Streams ending without `MessageStart` or `MessageEnd` are invalid.

## Enforcement

The contract is enforced centrally in `inference-sdk-core`:

1. Runtime assembly validation in `InferenceResult::from_stream`.
2. Shared sequence validator (`EventOrderValidator` / `validate_event_sequence`) for provider tests.

## Error Semantics

Contract violations are surfaced as:

- `SdkError::StreamInvariantViolation(...)`

No silent recovery is performed for invalid event ordering.
