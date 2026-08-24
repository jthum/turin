# Core Release Stabilization

## Objective

Ship a solid Turin engine release before resuming client and UI work. This is a
stabilization chapter, not a feature chapter: preserve Turin's existing engine
capabilities while making its supported boundary deliberate, its lifecycle
predictable, and its resource use bounded.

## Scope

Included:

- The embedded Rust API and `RuntimeBuilder` composition boundary.
- Rust harness and script-adapter contracts.
- Session, turn, task, delegation, cancellation, reload, and shutdown semantics.
- Turso-backed persistence integrity and migration behavior.
- Tools, governance, authorization, network/filesystem boundaries, and limits.
- Core errors, tracing, public visibility, naming, ownership, and release checks.

Deferred unless a missing core primitive is proven:

- Turin App, Web, TUI, and CLI product UX.
- New UI protocols or GPUI work.
- New channels, graph features, storage adapters, or scripting engines.
- Speculative optimization around future Turso capabilities.

## Release Boundary Audit

Status: in progress.

The intended supported embedding surface is:

- `Kernel` and `RuntimeBuilder` construction, initialization, execution, and shutdown.
- `TurinConfig` loading and supported programmatic configuration.
- `Harness`, `HarnessFactory`, typed harness inputs, and harness verdicts.
- `HarnessAdapterFactory` only as the engine-neutral scripting-adapter boundary.
- `Tool`, `ToolRegistry`, tool context/output/error/effect, and authorization contracts.
- Provider and embedding composition required by an embedded application.
- Stable session/task references, outcomes, events, and control operations needed by clients.
- Deliberate persistence entry points for applications that use Turin's durable features.

Confirmed boundary concerns:

1. `src/lib.rs` publishes broad implementation namespaces (`kernel`, `persistence`,
   `daemon`, `remote`, and tool internals). This makes internal movement look like a
   public API break. Introduce a curated facade before narrowing modules; do not add
   compatibility re-exports for APIs that were never intentionally supported.
2. `Kernel` implements `Deref<Target = ExecutionHost>`. This exposes host methods
   implicitly, hides the actual supported `Kernel` contract, and couples workspace
   consumers to an internal orchestration type. Inventory and migrate every required
   operation to explicit `Kernel` methods before removing the dereference.
3. Configuration now contains private loaded state. Programmatic construction must
   remain possible through defaults/builders rather than exhaustive struct literals.
4. Persistence currently exposes both useful application operations and low-level row/
   connection details. Do not invent a storage adapter, but classify which access is a
   supported advanced API and which is implementation-only.

Boundary exit criteria:

- A small documented facade covers every supported embedded use case.
- Product crates compile without relying on accidental internal visibility.
- `Kernel` no longer gains public behavior through implicit dereferencing.
- Public contracts use domain types and typed errors where callers must react.
- Internal refactors no longer require ecosystem-wide compatibility aliases.

## Lifecycle And Concurrency Audit

Status: pending.

Verify with focused tests before changing semantics:

- Concurrent writes enforce branch-head preconditions without losing committed turns.
- Cancellation wins consistently during provider streaming, authorization, tool
  execution, queued child work, and task activation.
- A cancelled or killed child cannot publish a late result into its parent.
- Parent/family cancellation covers materialized and not-yet-materialized descendants.
- Partial provider and persistence failures leave deterministic resumable state.
- Durability barriers surface background failures before success is reported.
- Shutdown stops admission, drains or terminates work within a bound, flushes durable
  records where promised, and closes MCP resources.
- Targeted agent/harness reload preserves unrelated busy runtimes and never publishes a
  partially prepared generation.
- Busy affected runtimes have a deterministic defer/reject behavior. Automatic retry is
  optional and not a release blocker if explicit rescan is documented.

Lifecycle exit criteria:

- Every terminal state has one tested meaning: complete, error, cancelled, timed out,
  and killed.
- No successful API result can precede a required durable write failure.
- No background task can silently outlive kernel shutdown guarantees.
- Reload failure preserves the last valid generation.

## Persistence Integrity

Status: pending.

- Review schema bootstrap and migrations as one pre-user baseline.
- Verify atomic turn allocation and branch-head advancement.
- Verify malformed rows, invalid counters, missing parents, cross-session ancestry,
  orphaned heads, and corrupted event payloads return typed deterministic errors.
- Verify partial-turn resume and monotonic next-turn allocation.
- Verify linked-family and session deletion semantics under active and queued work.
- Verify foreign-key enforcement on every connection path.
- Keep SQL and Turso details behind focused domain persistence modules without adding a
  speculative adapter trait.

## Boundedness And Security

Status: pending.

- Inventory every queue, cache, retained transcript, stream, tool result, HTTP body,
  child family, runtime pool, and scripting-runtime allocation.
- Require safe configurable defaults; harnesses may narrow or deliberately widen them.
- Verify filesystem containment and symlink behavior.
- Verify localhost/private-network behavior is policy-driven rather than accidentally
  denied or silently allowed.
- Verify tool authorization occurs after governance and before side effects, and honors
  cancellation while waiting.
- Verify delegated authority is monotonic downward across nested agents and grants.
- Audit secrets and sensitive tool arguments in logs, events, errors, and snapshots.

## Structural Review

Status: pending until semantics settle.

- Audit public visibility and naming consistency.
- Review large production modules for mixed ownership, not raw line count.
- Remove duplicated parsing, mapping, validation, and policy resolution.
- Audit lock scope, poisoning behavior, `Arc` ownership, clones, and avoidable materialization.
- Audit production `unwrap`, `expect`, panic, unchecked conversion, and impossible-state
  assumptions. Keep justified invariant assertions explicit.
- Update architecture maps whenever ownership or invariants move.

## Release Qualification

Status: pending.

- Formatting and strict Clippy across core and its adapter/product consumers.
- Full unit and integration suite.
- Focused failure, cancellation, shutdown, reload, long-session, and parallel-agent tests.
- Release-profile smoke test only after correctness work settles.
- Final binary-size, idle-memory, retrieval, and fan-out measurements when disk space
  permits; these are release evidence, not prerequisites for implementing known fixes.

## Working Rules

- Keep tests, semantic fixes, and behavior-preserving refactors as separate commits.
- Do not add compatibility code for unpublished APIs.
- Do not drop capabilities to reduce code or API surface.
- Prefer focused invariants over broad implementation snapshots.
- Mark findings as blocker, important, or deferred; do not let this become an unbounded
  cleanup list.
