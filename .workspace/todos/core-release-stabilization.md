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

Status: explicit kernel boundary complete; facade/visibility review remains.

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
2. Resolved in `0.30.1` development: `Kernel` no longer implements
   `Deref<Target = ExecutionHost>`. Supported embedding operations are explicit
   `Kernel` methods, daemon-only harness operations use private delegates, and
   `ExecutionHost` is crate-private. The migration preserved the existing root and
   peer execution implementation without introducing another runtime object.
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

Status: complete for the audited lifecycle contract.

Audit checkpoint:

- Existing coverage already exercises cooperative and forced manager shutdown,
  queued/running/family cancellation, cancellation during activation, pooled-lane
  isolation, stale branch conflicts, partial-turn resume, transcript write failure,
  event durability failure, bounded context retrieval, and targeted harness reload.
- Resolved in `0.30.1` development: direct session teardown now bounds the background
  persistence-task join. A permanently stalled writer is aborted after the shutdown
  timeout and reported as an error instead of hanging an embedded caller indefinitely.
- Resolved in `0.30.1` development: session objects now distinguish `inactive`,
  `active`, and terminal `ended` states. Ending a never-started session closes its
  persistence lane, while restarting an ended session fails instead of producing an
  active session backed by a cancelled token and removed durability state.
- Resolved in `0.30.1` development: source-backed harness reload now activates an
  immutable prepared generation. Invalid Lua edits leave both existing and newly
  created sessions on the last valid source snapshot until a later reload succeeds.

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

Status: audited; no remaining release blocker identified.

Audit checkpoint:

- Schema bootstrap, FTS initialization, and version recording commit together.
- Every `StateStore` connection enables foreign keys and a bounded busy timeout.
- Session/main-branch creation, turn/head advancement, branch creation, promotion,
  and linked-family deletion use explicit transactions for their invariants.
- Checked row mapping rejects negative counters/depths, missing or cross-session
  ancestry, and orphaned branch heads as typed persistence-integrity errors.
- Resume uses bounded ancestry/message materialization and treats malformed compaction
  checkpoints as recoverable derived hints rather than corrupting the transcript.
- Full event/transcript reads remain explicit inspection/export operations. Their
  semantics should become paginated only through a deliberate client API change.

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

Status: audited; one transient-policy retention issue resolved.

Audit checkpoint:

- Inference history, tool-result retention, durability queues, watcher queues,
  runtime lanes, child admission, completed task results, web response bodies, Lua
  heaps, and observer channels have bounded defaults or explicit policy knobs.
- Session teardown now removes process-local session policy and identity-bound run
  policy overrides. Global and per-agent policy remain intentionally process-scoped.
- Filesystem tools canonicalize paths against configured roots; web fetch permits
  loopback/private targets deliberately for coding workflows while governance and
  tool authorization remain the authority boundary.
- Human authorization is asynchronous, fail-closed when unavailable, cancellation
  aware, and backed by bounded best-effort notifications plus an authoritative
  pending-request map.
- Ordinary task, shell, and MCP lifecycle logs retain identifiers and size/count
  metadata without recording full prompts, shell commands, or subprocess arguments.
- Full inspection APIs and deliberately unresolved authorization waits can retain
  caller-requested state; these are explicit operations rather than default turn-path
  growth.

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
