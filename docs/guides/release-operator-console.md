# Release Operator Console

A release workflow is one of the clearest examples of why Turin should not be
only a chat interface.

Release work has queues, approvals, QA gates, readiness summaries, blockers,
forms, and actions that should not run accidentally. A Turin release harness can
turn that into a small operator console while keeping the durable workflow state
inside the runtime.

## What It Feels Like

Instead of asking an assistant "are we ready to release?" and searching through
a transcript, an operator opens a release desk:

- a home screen shows readiness and recent activity
- an approvals screen shows pending release work
- a detail pane explains the selected item
- reports and charts summarize current state
- action buttons run controlled checks or advance work
- risky actions require confirmation

The same harness can be opened in `turin-app`, `turin-tui`, or `turin-web`.
Each client renders the same semantic surfaces in its own medium.

## Why Turin Fits

Turin's release story follows the runtime model:

- Inference proposes readiness summaries, diagnostics, and next steps.
- Harness code decides which actions exist, what needs confirmation, and how
  release work is classified.
- The runtime persists worklists, events, action results, memory, and audit
  context.
- The UI client owns local presentation state such as selected row, active
  screen, open pane, and form draft.

That split keeps release state durable without forcing all clients to share one
UI session.

## Start Small

The first useful release console should be deliberately small:

- one release worklist
- one home screen
- one approvals list
- one selected-item detail surface
- one confirmed action
- one lightweight readiness report

This is enough to prove the workflow. Add charts, panes, badges, and extra
screens only after operators can answer real release questions faster than they
could from chat or a spreadsheet.

## What The Harness Owns

The harness should own workflow meaning:

- which work items are approvals, QA gates, blockers, or checks
- what metadata belongs on release work
- which actions require confirmation
- which agent or harness action can run diagnostics
- when a badge, notice, focus hint, or refresh should be emitted
- how readiness should be summarized

Do not encode desktop, terminal, or browser layout in the harness. Declare
semantic intent: screens, menus, lists, detail, forms, reports, charts, panes,
actions, notices, badges, and refresh hints.

## What Turin Stores

Use runtime primitives for state that must survive restarts:

- worklists for approvals, QA gates, blockers, and follow-ups
- events for release activity and action outcomes
- schedules for recurring readiness checks
- memory or KV for release conventions and exact preferences
- action results for auditable operator feedback

If a selected row, open pane, or form draft can disappear when one client
closes, keep it client-local.

## What The UI Clients Render

Current clients interpret the same semantic release UI differently:

- `turin-app` uses Cast/egui panels, forms, detail sections, modals, and richer
  visual summaries.
- `turin-tui` uses compact Ratatui navigation, tables, inspectors, overlays,
  keyboard focus, and terminal-local form editing.
- `turin-web` exposes HTTP/SSE routes plus a lightweight browser shell with
  local forms, overlays, action feedback, and event-driven invalidation.

The release harness should not care which one is open.

## Success Criteria

A release console is useful when an operator can quickly answer:

- What is blocking the release?
- Which approvals need attention?
- What changed recently?
- What action can I safely run next?
- What did the last action do?

If those answers are easier to find in the Turin client than in a chat
transcript, the release console is paying for itself.

## Reference Fixture

Use `examples/harnesses/ui_release_operator` as the current reference shape.
It exercises app screens, nested navigation, worklist-backed lists, forms,
reports, charts, panes, badges, notices, focus/open/show/refresh hints, action
confirmation, and local/remote web smoke coverage.

Related docs:

- `docs/concepts/scenarios.md`
- `docs/concepts/harness-apps-and-ui-clients.md`
- `docs/operations/ui-client-parity.md`
- `docs/operations/ui-clients.md`
