# Harness Apps And UI Clients

Turin can run without any custom harness UI. In that mode, `turin-app`,
`turin-tui`, and `turin-web` behave as default operator consoles for the
runtime: status, tasks, events, work, and runtime health remain visible.

Custom UI becomes useful when a harness has a workflow shape that should be
seen directly: queues, forms, approvals, detail panes, reports, charts, notices,
or action controls.

The goal is simple:

> Simple workflows should be easy from the default console. Structured
> workflows should be able to become small agentic apps.

## What A Harness App Is

A harness app is a semantic UI surface declared by a harness.

It is not an egui layout, a Ratatui widget tree, a web route file, or a CSS
contract. The harness describes what the workflow means:

- apps
- screens
- menus
- lists
- detail views
- activity feeds
- forms
- actions
- reports and charts
- panes
- notices and badges
- refresh, open, show, and focus hints

Each client decides how to render those semantics.

The desktop app may use richer cards, panes, dialogs, and Cast components. The
terminal UI may use compact tables, inspectors, overlays, and keyboard focus.
The web client may expose HTTP/SSE routes and a browser shell. The harness does
not need to know those renderer details.

## Default Console First

Do not create a custom app just because the UI API exists.

Start with the default console when:

- the workflow is still mostly conversational
- there is no durable queue or status view yet
- one action or schedule is enough
- operators do not need a dedicated surface

Add harness UI when the workflow has visible structure:

- a list of work needs triage
- an action needs parameters or confirmation
- a user needs a detail view before acting
- status should be summarized as a report or chart
- a workflow has multiple screens or modes
- a dynamic notice, badge, pane, or refresh would reduce confusion

The first useful custom UI is usually one screen, one list or detail view, and
one action that saves real time.

## Multiple Clients, Same Workflow

The same harness can be opened through different clients at the same time.

For example, a release harness can be open in:

- `turin-app` for a richer graphical operator console
- `turin-tui` for keyboard-first terminal work
- `turin-web` for a browser-based or remote-access surface

Those clients should not share ephemeral UI state. If the TUI has one row
selected and the desktop app has another pane open, that is fine. The runtime
owns durable workflow state; each client owns local presentation state.

This means client-local state should include:

- active screen or tab
- selected row
- open pane
- modal state
- form draft
- temporary filter or focus choices
- client-side cache and loading/error state

Durable state should use Turin primitives instead:

- worklists for claimable or inspectable work
- memory for searchable remembered context
- KV or runtime databases for exact facts
- events and signals for workflow coordination
- schedules for delayed or recurring work
- action results for auditable operations

## Local And Remote Are The Same Shape

The UI client should not care whether Turin is local or remote. A client can
connect to a local daemon or through `turin-remote`; the semantic UI intent and
client-owned session model stay the same.

Remote access does not turn UI state into runtime state. It only changes the
transport.

## What To Build First

For a new harness app, start with this sequence:

1. Prove the workflow from the default console.
2. Add one durable source of truth, usually a worklist.
3. Add one screen that shows that source.
4. Add one action with clear confirmation or parameters.
5. Add detail, activity, report, chart, pane, or badge surfaces only when the
   workflow needs them.

That keeps the 80% path small while leaving room for richer app-like workflows.

## Current Implementation Notes

The current clients intentionally share only the parts that have proven common:
connection/profile handling, dashboard updates, UI intent registry, and small
stateless semantic helpers. Active UI state remains in each client.

For the current implementation matrix and footprint notes, see:

- `docs/operations/ui-clients.md`
- `docs/operations/ui-client-parity.md`
- `docs/operations/turin-web.md`
