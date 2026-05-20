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
  - MCP subprocess/client lifecycle, tool listing, registry validation, proxy registration, reuse, and shutdown.
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
2. `ExecutionHost::spawn_mcp_server` checks whether the same command/args pair is already attached.
3. For a new server:
   - spawn stdio transport
   - initialize MCP client
   - list tools
   - validate non-empty, unique, non-conflicting tool names
   - register all proxies
   - store the client only after validation and registration succeed
4. For an existing server:
   - list tools again
   - register only newly discovered tool names
   - skip names already in the registry
5. The turn receives an attach report with listed, registered, and skipped counts.

Tool call:

1. Later model calls to MCP-provided tool names route through `McpToolProxy`.
2. The proxy forwards JSON params to the MCP client.
3. Text content is returned as a normal `ToolOutput`.
4. MCP tool error results become `ToolError::ExecutionError`.

Shutdown:

1. Kernel, command, daemon, REPL, dispatch, and peer-runtime shutdown paths call `shutdown_mcp_clients`.
2. Shutdown is best-effort and logs failures.
3. `Kernel::drop` clears client refs as a fallback so transports can be dropped promptly.

## Invariants

- `bridge_mcp` must stay out of `DEFAULT_EXPOSED_TOOL_NAMES`.
- The `integration` tool group may include `bridge_mcp`; default exposure should not.
- MCP tools must not shadow built-ins or already-registered tools during first attach.
- New attach validation should happen before storing the client in `ExecutionHost`.
- A failed new attach should not leave a live MCP client in `mcp_clients`.
- Reusing an existing client may skip already-registered tools but must still reject empty or duplicate names from the server response.
- MCP proxies belong to the native tool registry; normal native-tool permission checks still apply when tools are called.

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
- registration validates names before storing new clients
- attach results report listed, registered, and skipped tool counts

The next meaningful boundary would be a `turin-mcp` subsystem crate with a small host trait for tool registration and shutdown. That is worth doing after the broader subsystem registration model is clearer; the current cleanup already fixes the main safety and coherence issues without adding a premature trait layer.
