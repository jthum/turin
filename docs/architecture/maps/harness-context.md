# Harness Context Map

## Purpose

`context.rs` exposes the mutable Lua `ctx` userdata used by harness hooks. It is the handoff point between turn preflight, harness code, provider routing, structured inference, and request-option overrides.

Keep this module focused on the Lua-facing context contract. Shared provider request policy should live outside the userdata implementation so normal turn execution and harness-triggered inference cannot drift.

## Files

- `src/harness/context.rs`
  - `ContextWrapper`, named `ContextInit`, `ContextState`, Lua property accessors,
    message mutation helpers, and summarization.
- `src/kernel/harness_contract/request_options.rs`
  - Engine-neutral `RequestOptionsOverride` and shared provider request-option layering.
- `src/harness/context/structured_call.rs`
  - `ctx:structured` argument parsing, route resolution, provider fallback,
    request construction, and response validation.
- `src/kernel/turn/preflight.rs`
  - Builds the normal provider request stream, applies harness `on_turn_prepare` mutations, and filters the per-inference tool surface.
- `src/kernel/harness_runtime.rs`
  - Owns the object-safe session-harness capability contract and its Lua adapter.
    Kernel and daemon code must not reach through the adapter into the Lua engine.
- `src/kernel/harness_contract.rs`
  - Typed borrowed lifecycle and policy hook inputs shared by kernel execution and
    harness implementations. Lua payload conversion belongs here as adapter behavior;
    generic JSON hook dispatch is not part of the session-harness contract.
  - Owns the neutral mutable turn-preparation request and execution-binding DTOs.
- `src/kernel/native_harness.rs`
  - Public compiled-harness and per-session factory contracts. Default hook methods
    allow fixed-purpose applications to implement only the policy they need.
- `src/kernel/builder.rs`
  - `with_native_harness_factory` replaces the default Lua harness binding with a
    compiled Rust factory while preserving named Lua harness definitions.
- `src/inference/structured.rs`
  - Response-format construction, fallback prompt construction, and JSON validation for structured output.

## Data Flow

Turn preflight:

1. `preflight.rs` builds a `ContextWrapper` from the current request state.
2. Harness `on_turn_prepare` can mutate model/provider/system prompt/messages/thinking/request options.
3. Harness code can inspect the current session title/message count and select among tools already available to the turn.
4. The mutated context and tool exposure are read back into provider request state.
5. Request options are built from provider defaults plus harness overrides.

Structured inference:

1. Lua calls `ctx:structured({...})`.
2. The call starts from the current context state and optional call-local overrides.
3. Turin resolves the requested inference route.
4. Request options are built from provider defaults, current context overrides, then call-local overrides.
5. Native response-format support is used when available; otherwise Turin falls back to prompt-based JSON validation.

## Invariants

- `RequestOptionsOverride` is a harness-facing data type; keep it serializable and stable for Lua conversion.
- Provider defaults, context overrides, and call-local overrides must layer in that order.
- Normal turn preflight and `ctx:structured` must share the same request-option merge semantics.
- `ctx.prompt` and `ctx.messages` must remain synchronized when either is replaced.
- Turn preparation transfers provider-request ownership into `HarnessTurnRequest` and
  back. The Lua adapter may wrap it as userdata, but the session-harness contract must
  not expose `ContextWrapper` or another scripting-engine type.
- Rust call sites must initialize `ContextWrapper` through named `ContextInit` fields;
  positional construction is too error-prone for the request/runtime handoff.
- Structured calls may define `prompt` or `messages`, not both.
- Context token counts must be recomputed after message or system prompt mutation.
- Tool declarations remain load-time; `ctx.tools` only filters definitions for the current provider inference.
- Conditional exposure must not bypass native policy, governance, or tool-call hooks.
- Lua engine operations used outside the harness subsystem must be represented by an
  explicit session-harness capability. Do not restore unrestricted `Deref` access to
  `HarnessEngine`; session state stores `Box<dyn HarnessInstance>`, not a concrete Lua
  engine. This boundary is the migration seam for native harnesses and optional
  scripting adapters.
- Kernel hook call sites must construct `HarnessHook` variants from domain values.
  Do not reintroduce hook-name strings plus generic JSON payloads at the contract
  boundary. JSON remains appropriate inside dynamic fields such as tool arguments.
- A native factory creates one logical harness object per active session. Immutable
  application state should be shared explicitly with `Arc`; mutable session policy
  must not leak through a globally shared harness object.
- Native default harnesses do not watch the configured Lua harness directory. Named
  Lua harnesses retain their normal loading and hot-reload behavior.

## Common Changes

Change request-option behavior:

1. Update `src/kernel/harness_contract/request_options.rs`.
2. Add or adjust helper tests for layering and validation.
3. Run the request-options unit tests plus at least one harness request-options integration test.

Change Lua context fields:

1. Update `ContextState`, `ContextWrapper::new`, and the Lua `Index`/`NewIndex` handlers together.
2. Update preflight readback if the field affects provider requests.
3. Add a harness integration test that exercises the field through Lua.

Change structured inference:

1. Keep route resolution and provider fallback behavior aligned with normal inference.
2. Preserve native response-format and fallback validation test coverage.
3. Avoid moving client/tool rendering concerns into this module.

## Tests

Focused tests:

```sh
cargo test -p turin request_options_override --lib
cargo test -p turin --test harness_tests test_harness_request_options_passthrough
cargo test -p turin --test harness_tests test_harness_conditionally_exposes_one_shot_session_title_tool
cargo test -p turin --test session_tests test_on_turn_prepare_structured_output_uses_native_response_format
cargo test -p turin --test session_tests test_on_turn_prepare_structured_output_falls_back_to_prompt_and_validate
```

Basic checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The current shape keeps Lua property and message mutation in `context.rs`, engine-neutral
request-option layering in `kernel/harness_contract/request_options.rs`, and structured inference in
`structured_call.rs`. Normal inference and structured harness inference use the same
header/retry/timeout override policy. The object-safe `HarnessInstance` capability
contract sits between session execution and the private `LuaHarnessInstance` adapter.
Lifecycle and policy hooks use the typed borrowed `HarnessHook` contract, and only the
Lua adapter materializes the legacy Lua payload shape. Turn preparation uses the
ownership-based `HarnessTurnRequest`; `ContextWrapper` is now a private Lua adaptation
detail. Execution bindings and session queues are kernel-owned DTOs rather than Lua
globals, while registration and virtual-tool capabilities remain to be generalized.
`RuntimeBuilder::with_native_harness_factory` now provides the first compiled-harness
entry point for typed lifecycle hooks and request preparation. Lua remains the default.
