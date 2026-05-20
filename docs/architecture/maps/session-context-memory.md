# Session Context And Hot History Map

## Purpose

The session context and hot-history path controls how Turin keeps durable transcript state, in-memory working history, provider request context, and context compaction aligned.

This subsystem should preserve three guarantees:

- persisted session history remains the transcript source of truth
- live sessions keep only a bounded hot working set by default
- provider requests still preserve recent context, tool/result adjacency, and checkpoint summaries when older history is compacted

## Files

- `src/kernel/session.rs`
  - `SessionState`, execution write policy, context targets, hot-history offset, and context checkpoint state.
- `src/kernel/session_lifecycle.rs`
  - Session creation, restore/materialization, branch checkout, persisted history rebuild, and hot-history policy application.
- `src/kernel/hot_history.rs`
  - In-memory hot-window pruning, older tool-result payload trimming, and hot-history reports.
- `src/kernel/turn/preflight.rs`
  - Turn preparation, harness `on_turn_prepare`, context checkpoint refresh, and provider request context construction.
- `src/kernel/turn/context_window.rs`
  - Token estimation, effective request context, checkpoint summary request construction, structural request compaction, and context-window trimming.
- `src/kernel/config/inference.rs`
  - `inference.hot_history` and inference compaction configuration.
- `src/persistence/state/messages.rs`
  - Durable turn-scoped message loading.
- `src/persistence/state/turns.rs`
  - Active branch path and selected turn path resolution.

## Data Flow

Session restore:

1. `session_lifecycle.rs` loads the persisted session row.
2. The active branch or selected context target is materialized from persistence.
3. `rebuild_history` converts persisted messages into inference messages.
4. `SessionState::replace_full_history` resets the hot-history offset.
5. `prune_session_hot_history` applies the configured hot-history policy.

Hot-history pruning:

1. `prune_session_hot_history` only applies to durable branch-advancing sessions.
2. `hot_history::apply` prunes old messages using `effective_max_messages`.
3. The prune boundary expands backward when needed to keep assistant tool-use and tool-result adjacency.
4. Older large successful tool results are replaced with an omission marker using `effective_max_tool_result_bytes`.
5. `history_message_offset` records how many persisted messages precede the current hot window.

Turn preflight:

1. Harness `on_turn_prepare` requires full materialization because harness code can inspect or rewrite the full message list.
2. Context checkpoint refresh estimates the effective request size and may ask a compaction provider to summarize older history.
3. Provider request context is built from the hot window plus any checkpoint summary.
4. Structural request compaction can still truncate old tool results and slide the request window to fit provider limits.

Full materialization:

1. `ensure_full_history_materialized` reloads persisted messages when a full-history consumer needs them.
2. The session hot window is replaced by the full active context.
3. Later turn completion applies hot-history pruning again.

## Invariants

- Hot-history pruning must not run for ephemeral or non-branch-advancing execution contexts.
- `history_message_offset` must increase when hot messages are dropped and reset when full history is materialized.
- Tool-result messages at the hot-window boundary should keep their preceding assistant/tool-use context.
- Hot-history payload trimming should affect only older successful tool results, not recent payloads or error payloads.
- Durable persistence must keep the full message content even when hot memory uses an omission marker.
- Context-window structural compaction is request-local; it should not mutate session history.
- Harness `on_turn_prepare` may see full history and may replace it, so pruning happens after turn execution, not before the hook.
- Debug hot-history profile can opt out of bounds; default profile should remain memory-safe.

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
2. Verify branch/selected-path behavior and hot-window offset behavior together.
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
- `preflight.rs` answers "when do we refresh summaries and build request context?"
- `session_lifecycle.rs` answers "when do we restore, materialize, and re-prune persisted history?"

Likely future cleanup areas:

- expose hot-history report data in perf-suite long-session reports
- add a long-session fake-inference benchmark that checks RSS and hot-window size together
- consider a persistence query that materializes only the needed recent branch suffix instead of always rebuilding full history before pruning
- tune default profile values after measurement, not by guesswork
