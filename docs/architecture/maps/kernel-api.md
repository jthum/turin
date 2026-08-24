# Kernel API Map

## Purpose

The kernel API is Turin's embedded Rust boundary. `Kernel` owns the root runtime,
while `ExecutionHost` is an internal implementation shared by the root kernel and
peer-agent runtime workers.

Keep supported application operations explicit on `Kernel`. Do not expose new
behavior through dereferencing, public fields, or a general-purpose host accessor.

## Files

- `src/lib.rs`
  - Crate namespace and top-level facade exports.
- `src/kernel/mod.rs`
  - `Kernel`, explicit embedded lifecycle operations, runtime snapshots, and
    deliberate manager accessors.
- `src/kernel/builder.rs`
  - `RuntimeBuilder` composition of tools, harnesses, scripting adapters, and tool
    authorization.
- `src/kernel/execution_host.rs`
  - Crate-private shared execution state used by root and peer runtimes.
- `src/kernel/init.rs`
  - Provider, state, harness, reload, and watcher initialization.
- `src/kernel/session_lifecycle.rs`
  - Session create, resume, selection, start, and end implementations delegated by
    `Kernel`.
- `src/kernel/run_loop.rs`
  - Direct session run-loop implementation delegated by `Kernel`.

## Data Flow

1. An embedding loads or constructs `TurinConfig`.
2. `RuntimeBuilder` composes optional Rust harnesses, one scripting adapter, custom
   tools, and an authorization handler.
3. `build` creates one root `Kernel` and its crate-private `ExecutionHost`.
4. Explicit `Kernel` methods initialize providers, stores, and harnesses.
5. Direct embeddings create/resume a `SessionState` and run it through `Kernel`, or
   use `AgentManager` for durable managed runtimes and linked-agent work.
6. `shutdown` stops peer admission/work and closes root external resources.

## Invariants

- `Kernel` must not implement `Deref` or expose `ExecutionHost`; implicit host methods
  turn implementation details into accidental public API.
- `ExecutionHost` remains one in-memory object. Kernel delegation must not clone the
  host, serialize requests, or add dynamic dispatch.
- Root and peer runtimes share execution implementations, but root-only ownership such
  as filesystem watcher lifetime remains on `Kernel`.
- Public manager accessors are deliberate advanced APIs. Do not expose individual host
  fields solely to avoid writing a focused kernel operation.
- Rust harnesses and scripting adapters compose through `RuntimeBuilder`; the kernel
  must not select a scripting engine.
- New embedded operations should use domain inputs and outcomes rather than persistence
  rows or daemon protocol DTOs.
- Initialization ordering remains explicit until a separate atomic startup API can
  preserve provider injection and test use cases without reducing control.

## Focused Checks

- `cargo check -p turin --all-targets`
- `cargo check -p turin-harness-lua -p turin-cli --all-targets`
- `cargo test -p turin kernel::builder --lib`
- `cargo test -p turin-harness-lua --test integration_tests`
