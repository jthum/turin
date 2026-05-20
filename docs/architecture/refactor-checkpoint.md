# Runtime Refactor Checkpoint

This checkpoint captures the first serious runtime/harness refactor pass so future maintainers and coding agents do not restart the same discovery work.

## Current State

Approximate shipped Rust LOC, excluding files under `tests/` directories but including inline test modules:

- `src`: about 68.8k LOC
- `crates`: about 39.3k LOC
- total: about 108.2k LOC

The biggest remaining shipped Rust files are not all equal refactor candidates. Some are runtime core, some are UI/CLI surfaces, and some are intentionally ignored for now.

## Completed First-Pass Areas

These areas have been structurally reviewed, lightly refactored, tested, and mapped:

- Actions
- Agent manager operations
- Agent session bindings
- Channels
- Code search
- Daemon runtime state
- Governance
- Harness system globals
- MCP integration
- Memory and scoped data
- Runtime DB and graph
- Runtime schedule bindings
- Scheduler and worklists
- Session context and hot history
- Session lifecycle
- Turn preflight
- Turn tool execution

Read the matching files under `docs/architecture/maps/` before editing any of those areas.

## What Changed In Spirit

The refactor started as a structural extraction pass, then became stricter:

- Avoid splitting files just because they are large.
- Prefer reductions that remove duplicated policy, parsing, selection, or conversion logic.
- Keep behavior-preserving refactors separate from semantic changes.
- Add maps for subsystems while the context is fresh.
- Commit only tested checkpoints.

This was the right correction. Several later passes were deliberately small because adding a helper would have made the code more abstract without making it more obviously better.

## Remaining Good Candidates

The main runtime-core candidates from this checkpoint have now had first-pass review, focused cleanup, tests, and maps. Future passes should be driven by a concrete feature, bug, performance profile, or fresh code-quality finding rather than continuing this sweep by inertia.

## Handle Separately

These large files are not part of the current runtime-core refactor pass:

- `crates/turin-tui/src/main.rs`
  - TUI is expected to be rebuilt from scratch.
- `crates/turin-app/src/main.rs` and `crates/turin-ui-core/src/controller.rs`
  - Belong to the next UI chapter.
- `crates/turin-manager/src/setup.rs`
  - Setup wizard surface. It may deserve its own UX/config pass.
- `src/commands/daemon/render.rs`
  - CLI rendering. Refactor only when CLI output contracts are being reviewed.
- Channel outbound renderers
  - Keep channel-specific rendering unless helpers remove exact duplication.

## Current Assessment

The codebase no longer reads like an unreviewed generated blob in the areas covered by this pass. It still has large files, and not every pass reduced LOC, but the major runtime surfaces now have clearer boundaries, maps, focused tests, and less duplicated decision logic.

The remaining work should be paced as module-specific quality passes, not a second broad sweep. At this point, the broad runtime-core refactor sweep can pause unless a concrete next target emerges from tests, profiling, or UI/API work.
