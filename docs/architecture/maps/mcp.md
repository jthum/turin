# MCP Integration Map

## Purpose

The MCP integration lets an agent opt into a Model Context Protocol server at runtime and expose that server's tools through Turin's native tool registry.

This subsystem should preserve four guarantees:

- `bridge_mcp` remains opt-in, not part of default native tool exposure
- malformed bridge requests fail before any process is spawned
- MCP tool names cannot collide with built-in or existing tools during a new attach
- active MCP subprocess clients are shut down best-effort when the owning runtime ends

## Files

- `src/tools/mcp.rs`
  - `bridge_mcp` builtin tool and `McpToolProxy` for forwarding calls to MCP tools.
- `src/kernel/mcp_runtime.rs`
  - MCP subprocess/client lifecycle, tool listing, host/session name validation,
    session overlay registration, same-session reuse, and shutdown.
- `src/kernel/session.rs`
  - Session-owned MCP client list and attached-tool overlay.
- `src/kernel/turn/tool_execution.rs`
  - Handles `ToolEffect::SpawnMcp` by attaching the MCP server and returning attach results to the model.
- `src/tools/builtins/mod.rs`
  - Built-in tool registration, native tool groups, and default exposure lists.
- `src/tools/registry.rs`
  - Native tool registry and duplicate-name protection.
- `src/kernel/governance.rs`
  - Maps `bridge_mcp` to `integration.mcp.bridge`.
- `tests/capability_characterization_tests.rs`
  - Characterizes `bridge_mcp` as opt-in.

## Data Flow

Bridge request:

1. The model calls `bridge_mcp` with `{ command, args }`.
2. `BridgeMcp` parses the request through typed deserialization.
3. Non-string args or an empty command fail as invalid params.
4. The tool returns `ToolEffect::SpawnMcp`; it does not spawn the process directly.

Attach:

1. `tool_execution.rs` receives `ToolEffect::SpawnMcp`.
2. `ExecutionHost::spawn_mcp_server` attaches to the calling session. It reuses a
   client only when that session already spawned the same command/args pair.
3. For a new server:
   - spawn stdio transport
   - initialize MCP client
   - list tools
   - validate non-empty, unique names that do not collide with host tools or this
     session's overlay
   - register all proxies on the session overlay
   - store the client on the session only after validation and registration succeed
4. For an existing client on the same session:
   - list tools again
   - register only newly discovered tool names on the session overlay
   - skip names already on the host registry or this session's overlay
5. The turn receives an attach report with listed, registered, and skipped counts.

Tool call:

1. Later model calls to MCP-provided tool names route through `McpToolProxy`.
2. The proxy forwards JSON params to the MCP client.
3. Text content is returned as a normal `ToolOutput`.
4. MCP tool error results become `ToolError::ExecutionError`.

Shutdown:

1. Session end shuts down that session's MCP clients.
2. Kernel, command, daemon, dispatch, and peer-runtime shutdown paths still call
   host `shutdown_mcp_clients` for any leftover host-owned clients.
3. Shutdown is best-effort and logs failures.
4. `Kernel::drop` clears leftover host client refs as a fallback so transports can be dropped promptly.

## Invariants

- `bridge_mcp` must stay out of `DEFAULT_EXPOSED_TOOL_NAMES`.
- The `integration` tool group may include `bridge_mcp`; default exposure should not.
- MCP tools must not shadow built-ins or already-registered session tools during first attach.
- New attach validation should happen before storing the client on the session.
- A failed new attach should not leave a live MCP client on the session.
- Reusing an existing client on the same session may skip already-registered tools but must still reject empty or duplicate names from the server response.
- MCP proxies belong to the owning session's tool overlay, not the shared host
  registry. Another session on the same kernel does not see those tools.
- Session-attached MCP tools are admitted by attach, not by a config allow-list.
  Host native-tool permission checks still apply to built-ins; MCP tool calls still
  go through governance using `integration.mcp.tool`.
- Session end shuts down that session's MCP clients. Host shutdown only covers
  leftover host-owned clients.
- MCP process arguments remain available in memory for exact client reuse but must not be
  written to ordinary logs because they may contain credentials or private configuration.

## Common Changes

Change bridge request shape:

1. Update `BridgeMcpArgs` and schema in `src/tools/mcp.rs`.
2. Add parsing tests in the same file.
3. Run `cargo test -p turin bridge_mcp --lib`.

Change MCP registration behavior:

1. Update `src/kernel/mcp_runtime.rs`.
2. Keep new attach transactional: validate and register before storing the client.
3. Add validation tests where possible without spawning a real MCP server.
4. Run `cargo test -p turin mcp_runtime --lib`.

Change default tool exposure:

1. Update `src/tools/builtins/mod.rs`.
2. Update governance/capability characterization tests intentionally.
3. Run:

```sh
cargo test -p turin --test capability_characterization_tests
cargo test -p turin tools::tests::policy --lib
```

## Tests

Focused tests:

```sh
cargo test -p turin bridge_mcp --lib
cargo test -p turin mcp_runtime --lib
```

Capability/security characterization:

```sh
cargo test -p turin --test capability_characterization_tests
cargo test -p turin tools::tests::policy --lib
```

Basic compile/format checks:

```sh
cargo check -p turin --lib
cargo fmt --all -- --check
git diff --check
```

## Current Shape

MCP is still in the root runtime crate, but its lifecycle is now more explicit:

- `bridge_mcp` is only a request parser that emits `ToolEffect::SpawnMcp`
- `mcp_runtime.rs` owns process/client lifecycle and proxy registration
- proxies register on the calling session's overlay, not the shared host registry
- registration validates names before storing new clients on that session
- attach results report listed, registered, and skipped tool counts

The next meaningful boundary would be a `turin-mcp` subsystem crate with a small host trait for tool registration and shutdown. That is worth doing after the broader subsystem registration model is clearer; the current cleanup already fixes the main safety and coherence issues without adding a premature trait layer.
