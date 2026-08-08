# Turin Development Desk

Turin Development Desk is a practical dogfooding workflow for building Turin
with Turin. It keeps active agent conversation, durable development work,
explicit review, generated plans, and a lightweight delivery overview in one
harness-defined application.

This is deliberately broader than a UI fixture. The workflow is intended to be
useful during real repository work and to expose weak seams in Turin's runtime,
harness DX, semantic UI contract, and clients.

## What It Does

- keeps development tasks in a durable worklist
- captures tasks with area, effort, priority, and outcome context
- moves pending work into an explicit review queue
- delegates bounded planning and review passes to specialist agents
- stores generated plans and reviews as durable, inspectable work items
- writes the latest generated plan and review under
  `.turin/runtime/development-desk/`
- gives Turin clients an overview, work table, review queue, plans, forms,
  report/chart summary, nested navigation, badges, notices, and a project pane
- adds repository guidance and current desk progress to normal agent turns

## Run It In This Repository

Use `turin.toml.example` as the basis for a local config, keeping your preferred
provider and model settings. The important part is pointing the main and
specialist harnesses at this workflow:

```toml
[harness]
directory = "library/workflows/turin_development_desk/harness"

[agents.planner]
harness = "planner"

[agents.reviewer]
harness = "reviewer"

[harnesses.planner]
directory = "library/workflows/turin_development_desk/agents/planner"

[harnesses.reviewer]
directory = "library/workflows/turin_development_desk/agents/reviewer"
```

Start the daemon and desktop client from the repository:

```sh
cargo run --bin turin -- daemon start --config .turin/config.toml
cargo run -p turin-app -- --config .turin/config.toml
```

The desk starts empty by design. Use **Set up starter work** for four practical
dogfooding tasks, or go directly to **Capture task**. Planner and reviewer
actions require the corresponding named agents; the backlog and review
progression actions do not.

## Durable State

- `development-desk-work`: backlog and completed development tasks
- `development-desk-reviews`: generated and manually queued reviews
- `development-desk-briefs`: generated implementation plans and working briefs

The UI remains a projection of these runtime primitives. Opening a screen,
selecting a row, editing a form, or showing the project pane stays client-local.
