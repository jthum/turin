# Agent Guidance

This repository is maintained by humans and coding agents. Before editing a subsystem, read the closest durable architecture map under `docs/architecture/maps/`.
Working notes, checkpoints, and temporary plans belong under `.workspace/scratchpad/` and should not be treated as permanent docs.

Key rules:

- Keep behavior-preserving refactors separate from semantic changes.
- Do not change client-facing harness/Lua/config authoring APIs without explicit discussion; internal API breaks are acceptable when capability-preserving and tested.
- Prefer existing domain helpers over duplicating parsing, filtering, mapping, or validation logic.
- Do not reduce LOC by dropping features, security checks, validation, or deliberate DX surfaces.
- Add meaningful tests freely when they improve confidence; test LOC is not part of the shipped-runtime LOC budget.
- Do not widen module visibility unless the boundary is deliberate and local to the subsystem.
- Run the focused tests listed in the relevant subsystem map before committing.
- Update the subsystem map when a refactor changes ownership, invariants, or test expectations.

Important maps:

- Actions: `docs/architecture/maps/actions.md`
- Agent session bindings: `docs/architecture/maps/agent-session-bindings.md`
- Harness system globals: `docs/architecture/maps/harness-system-globals.md`
- Scheduler and worklists: `docs/architecture/maps/scheduler-worklists.md`
- Channels: `docs/architecture/maps/channels.md`
- Code search: `docs/architecture/maps/code-search.md`
- Config: `docs/architecture/maps/config.md`
- Control client: `docs/architecture/maps/control-client.md`
- Daemon CLI render: `docs/architecture/maps/daemon-render.md`
- Daemon runtime state: `docs/architecture/maps/daemon-runtime-state.md`
- Governance: `docs/architecture/maps/governance.md`
- Harness context: `docs/architecture/maps/harness-context.md`
- MCP integration: `docs/architecture/maps/mcp.md`
- Manager: `docs/architecture/maps/manager.md`
- Memory and scoped data: `docs/architecture/maps/memory-scoped-data.md`
- Runtime DB and graph: `docs/architecture/maps/runtime-db-graph.md`
- Session context and hot history: `docs/architecture/maps/session-context-memory.md`
- Session lifecycle: `docs/architecture/maps/session-lifecycle.md`
- Turin app: `docs/architecture/maps/turin-app.md`
- Turin TUI: `docs/architecture/maps/tui.md`
- Turin UI Core: `docs/architecture/maps/turin-ui-core.md`
- Turin web: `docs/architecture/maps/turin-web.md`
- Web tools: `docs/architecture/maps/web-tools.md`
