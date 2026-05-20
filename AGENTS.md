# Agent Guidance

This repository is maintained by humans and coding agents. Before editing a subsystem, read the closest durable architecture map under `docs/architecture/maps/`.

Key rules:

- Keep behavior-preserving refactors separate from semantic changes.
- Prefer existing domain helpers over duplicating parsing, filtering, mapping, or validation logic.
- Do not widen module visibility unless the boundary is deliberate and local to the subsystem.
- Run the focused tests listed in the relevant subsystem map before committing.
- Update the subsystem map when a refactor changes ownership, invariants, or test expectations.

Important maps:

- Scheduler and worklists: `docs/architecture/maps/scheduler-worklists.md`
- Channels: `docs/architecture/maps/channels.md`
