# Agent Guidance

This repository is maintained by humans and coding agents. Before editing a subsystem, read the closest durable architecture map under `docs/architecture/maps/`.

Key rules:

- Keep behavior-preserving refactors separate from semantic changes.
- Prefer existing domain helpers over duplicating parsing, filtering, mapping, or validation logic.
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
- MCP integration: `docs/architecture/maps/mcp.md`
- Memory and scoped data: `docs/architecture/maps/memory-scoped-data.md`
- Runtime DB and graph: `docs/architecture/maps/runtime-db-graph.md`
- Session context and hot history: `docs/architecture/maps/session-context-memory.md`
- Session lifecycle: `docs/architecture/maps/session-lifecycle.md`
