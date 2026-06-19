# Coding Workspace

A coding workspace is a project-specific Turin harness for development work.
It is useful when a coding assistant needs repository rules, focused checks,
review steps, durable follow-ups, and visible operator state instead of a long
chat transcript.

The goal is not to replace an editor. Turin should own the workflow state and
authority boundaries. The editor, terminal, and source-control tools can remain
the places where code is read and changed.

## What It Feels Like

You open Turin against a repository and see the work in progress: active task,
relevant project rules, pending checks, review notes, follow-ups, and actions
for running the right validation.

The same workspace can be opened through `turin-app`, `turin-tui`, or
`turin-web`. Each client can render the same semantic harness UI differently:
the terminal can be compact and keyboard-first, the desktop app can show richer
panels, and the web client can expose the workflow remotely.

The durable workflow is shared. The selected screen, selected task, form draft,
scroll position, and open panes remain local to each client.

## Why Turin Fits

Coding agents are useful because they can read files, edit code, run commands,
inspect diffs, and delegate review. Those are also the risky parts.

Turin keeps the boundary explicit:

- the model proposes edits, commands, plans, and follow-up work
- the harness decides what is allowed for this repository
- the runtime executes approved actions and records what happened
- the client renders the current workflow without owning durable state

That makes the workspace easier to resume. Failed checks, review comments,
decisions, events, and follow-up tasks can survive beyond one chat session.

## Minimal First Version

Start with one repository and one narrow loop:

- read the repository rules or architecture map before editing
- run one focused validation command
- record one durable follow-up when work cannot finish immediately
- expose one task list or detail surface only if it helps the operator
- require confirmation for broad, destructive, or surprising actions

This is enough to prove whether Turin improves the development loop. Avoid
building a full IDE or universal coding platform in the first pass.

## What The Harness Owns

The harness should encode project-specific operating rules:

- which files or architecture notes should be read before editing a subsystem
- which commands count as focused tests
- which tool groups are allowed by default
- which commands require confirmation or temporary grants
- when a peer-review agent should inspect a patch
- how incomplete work becomes durable follow-up items

These rules should not depend on the client. A terminal client and desktop
client should both respect the same harness behavior.

## What Turin Stores

Useful durable state for a coding workspace includes:

- sessions and event history
- worklist items for implementation tasks, review tasks, and follow-ups
- memory for recurring project decisions
- KV values for exact project settings
- scheduled reminders or checks when needed

Use durable state only when it improves the workflow. Short-lived UI choices
should stay in the client.

## Useful UI Surfaces

Good first screens are:

- `Workspace`: active task, constraints, recent events, and next actions
- `Work`: durable tasks and follow-ups
- `Review`: patch summary, changed areas, comments, and approval actions
- `Checks`: recent validation runs, failures, and rerun actions
- `Context`: loaded project notes, memory, and relevant rules

Useful actions include:

- run the focused test for the active task
- summarize the current diff
- ask a reviewer agent to inspect a change
- create a follow-up from a failed check
- prepare a handoff summary

Keep the first UI small. One list, one detail view, and one useful action are
usually enough to validate the shape.

## What Success Looks Like

A coding workspace is working when the operator can quickly answer:

- What task is active?
- What rules and context are in force?
- What changed?
- Which checks passed or failed?
- What needs review before landing?
- What follow-up work was created?

If those answers are easier to find in Turin than in a chat transcript, the
workspace is paying for itself.

## How To Grow It

After the first loop works, add depth deliberately:

- add subsystem-specific rules
- add more validation actions
- add patch review panes and approval flows
- add specialist agents for review, tests, docs, or migrations
- add memory for architectural decisions
- expose the same workspace through TUI, desktop, and web clients

Keep the harness semantic. Define project workflows, not terminal widgets or
desktop layout. Turin clients can then render the same workspace in the way
that fits their medium.

## Related Docs

- `docs/concepts/scenario-starter-cards.md`
- `docs/concepts/scenarios.md`
- `docs/concepts/harness-apps-and-ui-clients.md`
- `docs/getting-started/choose-first-workflow.md`
- `docs/guides/harness-guide.md`
