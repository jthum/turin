# Session Context And Hot History Map

## Purpose

The session context and hot-history path controls how Turin keeps durable transcript state, in-memory working history, provider request context, and context compaction aligned.

This subsystem should preserve three guarantees:

- persisted session history remains the transcript source of truth
- live sessions keep only a bounded hot working set by default
- provider requests still preserve recent context, tool/result adjacency, and checkpoint summaries when older history is compacted

## Files

- `src/kernel/session.rs`
  - `SessionState`, active-task state, context checkpoint state, and the public session facade.
- `src/kernel/session/execution.rs`
  - Execution context targets, visibility/durability/write/conflict policies, sidestep modes,
    task overrides, and execution status snapshots. These types are re-exported by
    `session.rs` so existing kernel paths remain stable.
- `src/kernel/session/resident_history.rs`
  - Turn-addressed resident message storage, origin alignment, and checkpoint-relative suffix selection.
- `src/kernel/session_lifecycle.rs`
  - Session creation, restore/materialization, branch checkout, persisted history rebuild, and hot-history policy application.
- `src/kernel/hot_history.rs`
  - In-memory hot-window pruning, older tool-result payload trimming, and hot-history reports.
- `src/kernel/turn/preflight.rs`
  - Turn preparation, harness `on_turn_start`/`on_turn_prepare`, provider route fallback, stream preparation, provider client initialization, and sparse accepted-request efficiency telemetry.
- `src/kernel/turn/preflight/compaction.rs`
  - Context-window estimation, context checkpoint refresh, summary generation, and provider request context compaction for turn preflight.
- `src/kernel/turn/context_window.rs`
  - Token estimation, effective request context, checkpoint summary request construction, structural request compaction, and context-window trimming.
- `src/kernel/config/inference.rs`
  - `inference.hot_history` and inference compaction configuration.
- `src/persistence/state/messages.rs`
  - Full message reads, bounded ancestry suffix reads, and token-budgeted
    context selection over complete turn groups.
- `src/persistence/state/turns.rs`
  - Turn allocation, depth-chunked ancestry resolution, active branch path,
    selected turn path, and turn-row resolution.
- `src/persistence/state/turns/branch_heads.rs`
  - Main branch initialization, branch-head lookup/listing, branch creation, checkout, and branch source resolution.

## Data Flow

Session restore:

1. `session_lifecycle.rs` loads the persisted session row.
2. The active branch or selected context target is materialized using the configured hot-history bound.
3. `rebuild_history` converts persisted messages into inference messages and retains each message's source turn ID/index.
4. `ResidentHistory` records whether older persisted ancestry exists without storing an exact message count.
5. Resume counters are reconstructed by one persistence aggregate; lifecycle event rows are not materialized in Rust.
6. Only the newest persisted context-compaction checkpoint is loaded for resume.
7. `prune_session_hot_history` applies payload trimming and enforces the same in-memory policy after subsequent turns.

Hot-history pruning:

1. `prune_session_hot_history` only applies to durable branch-advancing sessions.
2. `hot_history::apply` prunes old messages using `effective_max_messages`.
3. The prune boundary expands backward when needed to keep assistant tool-use and tool-result adjacency.
4. Older large successful tool results are replaced with an omission marker using `effective_max_tool_result_bytes`.
5. `has_prior_history` records that persisted ancestry precedes the resident window; no full-path count is computed.

Turn preflight:

1. Turin resolves the initial provider route and derives an input-token budget after reserving output/thinking capacity and tool/system overhead.
2. Complete resident history is reused only when its selected branch-head cursor still matches a lightweight durable head lookup.
3. Otherwise, persistence pages backward from the selected branch/turn and builds a request-local context from complete turns until that budget is filled.
4. Context checkpoint refresh estimates the effective request size and may ask a compaction provider to summarize an older complete-turn prefix.
5. Checkpoints identify their durable boundary by source turn ID/index, not by a mutable message count.
6. The selected request history is moved into request state rather than cloned from resident history.
7. Harness `on_turn_prepare` receives ownership of the request-local projection and may rewrite it without mutating resident history. Turin moves it back when the hook releases its context userdata, with a safe clone fallback only when Lua retains another reference.
8. Structural request compaction borrows the prepared request when it already fits and allocates a replacement message vector only when trimming is required.
9. After a provider accepts the stream request, Turin persists one
   `inference_request` event with normalized token and payload estimates,
   context-budget provenance, compaction counts, and route identity.
10. Inference route candidate fallback keeps a common log shape for requested context, resolved context, provider, model, and error.
11. A queued task may seed the requested inference context for all of its turns;
   `on_turn_prepare` remains authoritative and may retain or replace it.

Bounded persistence selection:

1. Ancestry is read backward in bounded depth chunks over one connection rather
   than opening a connection for every parent turn.
2. Structural bounded reads stop after the requested turn and message limits
   and report whether older ancestry exists without calculating an exact total.
3. Token-budgeted reads page backward, retain complete turn groups, and stop
   once the caller's token budget and minimum-message policy are satisfied.
4. Persistence supplies the mechanism and does not choose the provider or
   harness context budget.

## Invariants

- Hot-history pruning must not run for ephemeral or non-branch-advancing execution contexts.
- Resident message origins must remain aligned with their messages through append, replacement, and prefix pruning.
- Live session snapshots expose hot-history length and whether prior history exists; these values are observational and must not drive runtime pruning decisions.
- Client transcript windows are independent read projections over durable
  history. They must not be confused with, or used to configure, the runtime
  hot-history window.
- Exact-turn client windows materialize an ancestral read path without changing
  the persisted active branch or the live execution context.
- Tool-result messages at the hot-window boundary should keep their preceding assistant/tool-use context.
- Hot-history payload trimming should affect only older successful tool results, not recent payloads or error payloads.
- Durable persistence must keep the full message content even when hot memory uses an omission marker.
- Context-window structural compaction is request-local; it should not mutate session history.
- The ordinary preflight path should move or borrow its request projection; it must not clone the full message vector merely to cross an internal boundary.
- A harness that retains turn-context userdata may force the explicit safe clone fallback, but ordinary synchronous hooks should recover the owned state.
- Resident-history reuse must validate both completeness and the durable branch-head identity; `has_prior_history` alone is insufficient.
- `inference_request` telemetry must remain sparse. It records counts and route
  metadata, never another copy of prompts, messages, tool schemas, or checkpoint
  summaries.
- Provider-reported `message_end` input/output totals are authoritative when
  present. Request and per-message token counts are Turin estimates based on
  normalized pre-provider content, not billing claims or exact wire sizes.
- Reusable-prefix tokens describe a stable-prefix opportunity. They must not be
  presented as cache hits until the inference boundary exposes provider cache
  read/write counters.
- Harness `on_turn_prepare` sees the provider-budgeted request projection and may replace it; its changes must remain request-local.
- A checkpoint boundary must resolve to a complete persisted turn. It must never depend on the current resident-window length.
- Restored checkpoints are target-specific: the covered turn must be an ancestor of the
  selected branch/turn target, or explicitly present in a selected path. A session-wide
  newest checkpoint from a sibling branch must not compact another execution path.
- When no checkpoint exists, Turin must not summarize a bounded window that omits older ancestry as though it covered that ancestry.
- Debug hot-history profile can opt out of bounds; default profile should remain memory-safe.
- Feature-gated live diagnostics must not change materialization semantics or
  introduce prompt/history copies. Normal builds must compile the hooks away.
- Bounded context reads must never split the messages belonging to one turn.
- Ordinary bounded reads must not calculate full-path message totals merely to
  report that older history exists.
- Full and bounded ancestry reads must share parent-chain validation and
  chronological ordering.
- Session resume must not materialize lifecycle event history to recover scalar counters or the newest context checkpoint.

## Common Changes

Change hot-history defaults:

1. Update `HotHistoryProfile` defaults in `src/kernel/config/inference.rs`.
2. Update config validation/tests if new knobs are added.
3. Run `cargo test -p turin hot_history --lib` and `cargo test -p turin config --lib`.

Change hot-window pruning:

1. Update `src/kernel/hot_history.rs`.
2. Preserve boundary adjacency tests and add cases for any new structural rule.
3. Run `cargo test -p turin hot_history --lib` and `cargo test -p turin session --lib`.

Change provider request compaction:

1. Update `src/kernel/turn/context_window.rs` or `src/kernel/turn/preflight.rs`.
2. Keep request-local compaction separate from hot-memory pruning.
3. Run `cargo test -p turin context_window --lib`.

Change session materialization:

1. Update `src/kernel/session_lifecycle.rs`.
2. Verify branch/selected-path behavior, resident origins, and prior-history metadata together.
3. Run `cargo test -p turin session --lib`.

## Tests

Focused tests:

```sh
cargo test -p turin hot_history --lib
cargo test -p turin context_window --lib
cargo test -p turin session --lib
```

Basic compile/format checks:

```sh
cargo check -p turin
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The bounded hot-history feature already exists. The current pass centralized policy application in `hot_history::apply`, which returns a report for pruning and payload trimming. `session_lifecycle.rs` logs that report at debug level so future perf tooling can correlate long-session memory behavior with policy activity.

The current module split is deliberate:

- `hot_history.rs` answers "what stays hot in resident session memory?"
- `context_window.rs` answers "what fits into this provider request?"
- `preflight.rs` answers "when do turn hooks run and how do we try provider route candidates?"
- `preflight/compaction.rs` answers "when do we refresh summaries and build compacted provider request context?"
- `InferenceRequestMetrics` answers "what did Turin estimate it sent after
  compaction, and how close was that request to its input budget?"
- `session_lifecycle.rs` answers "when do we restore, materialize, and re-prune persisted history?"
- `LiveSessionSnapshot.history` answers "how much hot history is currently resident in a live runtime?"
- `perf-diagnostics` answers "which current retrieval stages and query counts
  produced this observation, and how did daemon process memory trend around
  it?"

Likely future cleanup areas:

- add a long-session fake-inference benchmark that checks RSS and hot-window size together
- tune default profile values after measurement, not by guesswork
