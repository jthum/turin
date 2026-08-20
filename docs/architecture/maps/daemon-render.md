# Daemon CLI Render Map

## Purpose

`src/commands/daemon/render` owns human-readable CLI output for daemon commands. It does not own daemon protocol semantics, request dispatch, or runtime state. Keep this layer presentation-only: decode response payloads, shape tables/details, and preserve existing JSON passthrough behavior.

## Files

- `src/commands/daemon/render.rs`
  - Facade that re-exports renderer functions to sibling command modules.
- `src/commands/daemon/render/common.rs`
  - Response decoding, JSON passthrough/error handling, shared table/indent/snippet helpers, and execution-target formatting.
- `src/commands/daemon/render/types.rs`
  - Local re-exports of daemon CLI view DTOs so child render modules do not reach through nested parent paths.
- `src/commands/daemon/render/agents.rs`
  - Daemon status, agents, agent runtime status, harness list/detail, and issue summary rendering.
- `src/commands/daemon/render/tasks.rs`
  - Task status/list and live-session rendering.
- `src/commands/daemon/render/sessions.rs`
  - Persisted session list/detail and branch rendering.
- `src/commands/daemon/render/control.rs`
  - Daemon health/start report rendering.
- `src/commands/daemon/*.rs`
  - Command handlers that call these renderers after sending daemon requests.

## Data Flow

1. Command handlers send daemon requests and receive `ResponseEnvelope`.
2. JSON mode calls `print_response` directly to preserve protocol-shaped output.
3. Human mode decodes the response into local `*View` structs.
4. Render modules print tables or detail sections.

## Invariants

- Human rendering must not change daemon protocol response shapes.
- JSON mode must keep returning daemon `ResponseEnvelope`/report shapes without table formatting.
- Render modules should not send daemon requests or mutate runtime state.
- Shared formatting helpers belong in `common.rs`; domain modules should stay table/detail focused.
- Visibility should stay local to `crate::commands::daemon`.

## Tests

Focused checks:

```sh
cargo check -p turin --bin turin
```

Basic checks:

```sh
cargo fmt --all -- --check
git diff --check
```

## Current Shape

The former single `render.rs` file has been split by output domain while preserving the same renderer function names used by command modules.
