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
  - Owns the object-safe session-harness capability contract and `HarnessDefinition`
    lifecycle. Each definition stores one private `HarnessAdapterFactory`; live sessions
    receive fresh `HarnessInstance` values. Kernel and daemon code must not select engines
    or reach through an adapter into its implementation.
- `src/kernel/harness_runtime/resolver.rs`
  - Owns construction-time adapter registration validation and implementation selection.
    `HarnessManager` consumes resolved adapters without knowing whether they came from
    Rust, Lua, or a future scripting engine.
- `src/kernel/harness_runtime/lua_adapter.rs`
  - Adapts the neutral contract to `HarnessEngine`, owns Lua app-data construction, and
    implements Lua-only source, UI-intent, execution-context, and virtual-tool surfaces.
    Source overlay validation and direct script execution are adapter-factory capabilities;
    they must not leak into the normal session-local `HarnessInstance` contract.
- `src/kernel/harness_runtime/rust_adapter.rs`
  - Adapts a session-local Rust `Harness` to the internal contract. Unsupported optional
    capabilities use contract defaults instead of fake Rust implementations.
- `src/kernel/harness_contract.rs`
  - Typed borrowed lifecycle and policy hook inputs shared by kernel execution and
    harness implementations. Lua payload conversion belongs here as adapter behavior;
    generic JSON hook dispatch is not part of the session-harness contract.
  - Owns the neutral mutable turn-preparation request and execution-binding DTOs.
- `src/kernel/harness.rs`
  - Public compiled-harness and per-session factory contracts. Default hook methods
    allow fixed-purpose applications to implement only the policy they need. It also
    re-exports `Verdict`, so Rust consumers do not depend on the Lua-oriented module
    layout. Rust harnesses can declare durable signal topic subscriptions and receive
    typed borrowed `HarnessSignal` deliveries without depending on persistence rows.
- `src/kernel/builder.rs`
  - Owns the construction-time Rust harness registry. `with_default_harness` registers
    `default`; `with_harness` registers a configured harness ID.
- `Cargo.toml`
  - The default `lua` feature includes `mlua`. Rust-only embedders can use
    `default-features = false` and must install a Rust harness factory.
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
  engine. This boundary is the migration seam for Rust harnesses and optional
  scripting adapters.
- Kernel hook call sites must construct `HarnessHook` variants from domain values.
  Do not reintroduce hook-name strings plus generic JSON payloads at the contract
  boundary. JSON remains appropriate inside dynamic fields such as tool arguments.
- A Rust factory creates one logical harness object per active session. Immutable
  application state should be shared explicitly with `Arc`; mutable session policy
  must not leak through a globally shared harness object.
- Only types needed to implement the public Rust contract are public. Execution binding
  DTOs remain kernel-private until a concrete Rust capability needs them; do not expose
  scripting-adapter plumbing as a speculative service API.
- Runtime signal delivery crosses the harness boundary as `HarnessSignal`, not
  `persistence::SignalRow`. Persistence retry metadata remains owned by the scheduler;
  Lua and Rust harnesses receive the same semantic signal fields.
- Rust harness runtimes do not watch configured script directories. Lua harnesses
  retain their normal loading and hot-reload behavior.
- Normal session creation is source-agnostic. Script validation and one-off source
  execution delegate through optional adapter-factory capabilities; unsupported adapters
  do not implement placeholder methods on every session instance.
- Config remains authoritative for agent-to-harness bindings. A named Rust factory
  must correspond to a declared `config.harnesses` ID; factory registration does not
  create a second binding system.
- Adapter selection happens once during harness catalog construction. Runtime methods
  must delegate through `HarnessAdapterFactory`, not branch on Lua, Rust, or future
  scripting-engine variants.
- A build without the `lua` feature must fail clearly if no Rust factory is installed;
  it must not silently run with an empty harness. Scheduler, native tools, persistence,
  inference, governance, memory, and session graph support remain available.
- Rust harness callbacks are policy boundaries, not process-wide service locators.
  Agent-triggered async operations belong in governed native tools and kernel effects;
  do not pass internal manager collections through a generic native services object.
- `tests/rust_embedding.rs` is the outside-in contract fixture. It must continue to use
  only public APIs while covering multiple compiled harnesses, custom tools, governance,
  provider inference, and persistence in a build without Lua.

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
cargo test -p turin --test rust_harness_api --no-default-features
cargo test -p turin --test rust_embedding --no-default-features
cargo check -p turin --example rust_harness --no-default-features
```

The Rust harness integration test must include a full kernel inference run so the
no-Lua build proves turn-preparation mutations reach the provider request.

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
contract sits between session execution and private Rust and Lua adapters.
Lifecycle and policy hooks use the typed borrowed `HarnessHook` contract, and only the
Lua adapter materializes the legacy Lua payload shape. Turn preparation uses the
ownership-based `HarnessTurnRequest`; `ContextWrapper` is now a private Lua adaptation
detail. Execution bindings and session queues are kernel-owned DTOs rather than Lua
globals. Native tools are the compiled operational surface rather than a translation of
Lua virtual tools. Runtime builder factories are keyed by the same harness IDs used by
agent configuration, with the existing default-factory method retained as shorthand.
Lua remains the default Cargo feature, while `--no-default-features` excludes `mlua`
and all Lua VM, context, DX, globals, and standard-library binding modules.
