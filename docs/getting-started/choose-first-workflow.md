# Choose Your First Turin Workflow

Turin is easiest to evaluate when you start with one concrete workflow instead
of trying to design a full agent platform.

The first useful Turin project should answer three questions:

- What work should survive beyond one chat session?
- What authority should be governed by harness code instead of model judgment?
- What would an operator need to inspect or act on?

If none of those matter yet, a normal chat assistant may be enough. Turin starts
to pay off when memory, state, tools, schedules, approvals, or custom operator
surfaces need to become part of the system.

## Pick The Shape

Use this as a practical starting map.

| If you want... | Start with... | First durable primitive | Add custom UI when... |
| --- | --- | --- | --- |
| A private assistant | one personal harness with memory rules | memory and KV | follow-ups need lists, forms, or reminders |
| A coding workspace | one repository-specific harness | session history and project memory | tasks or reviews need to be inspected outside chat |
| A release desk | one release harness | worklists | approvals, blockers, and readiness need operator screens |
| A bug triage desk | one triage queue | worklists | humans need to inspect and route decisions |
| A docs/research helper | one document or project | worklists and memory | sections, sources, and reviews need status views |
| A team governance tool | one governed action path | capabilities and events | approvals or grants need an operator console |
| A channel bot | one channel and one narrow job | sessions and events | channel work creates queues or approvals |

## Keep The First Slice Small

A good first slice usually has:

- one harness
- one agent
- one narrow workflow
- one or two durable state primitives
- one action that saves real time
- optional UI only after the workflow works from the default client

Avoid starting with a full dashboard, a large multi-agent org chart, or many
custom screens. Those can come later. The first goal is to prove that the
runtime state and harness policy are the right foundation.

## Decide What Belongs Where

Use this boundary:

- The model proposes language, plans, tool calls, and next steps.
- The harness owns workflow rules, policy, actions, context assembly, and UI
  intent.
- Turin stores durable sessions, events, memory, KV, schedules, and worklists.
- The client owns local presentation state such as active screen, selected row,
  open pane, form draft, and modal state.

If you are unsure where something belongs, prefer durable runtime primitives for
facts that must survive restarts, and client-local state for temporary view
choices.

## When To Add Harness UI

Do not add custom UI just because Turin supports it.

Add harness UI when the workflow has visible structure:

- operators need a queue or list
- an action needs confirmation or parameters
- a workflow has multiple screens
- status should be summarized as a report or chart
- users need contextual panes, notices, badges, or refresh hints

For a first UI slice, keep it to one home screen, one list or detail surface,
and one action. `turin-app`, `turin-tui`, and `turin-web` can render the same
semantic harness UI differently, so the harness should describe intent rather
than renderer-specific widgets.

## Example First Slices

### Personal Assistant

Start with memory rules and a small `followups` worklist. Add one action for
creating a follow-up and one scheduled daily summary.

### Coding Workspace

Start with a harness that reads the project's architecture maps and runs one
focused test command. Add worklists only after tasks need to survive across
sessions.

### Release Desk

Start with `examples/harnesses/ui_release_operator`. Keep the first version
worklist-backed: approvals, a seed action, a detail pane, and a readiness
summary.

### Documentation Team

Start with one document, one source-review worklist, and one draft action. Add
multi-agent review only after the single-agent drafting loop is useful.

## Next Reading

- `docs/concepts/what-can-you-do.md`
- `docs/concepts/scenario-starter-cards.md`
- `docs/concepts/scenarios.md`
- `docs/concepts/harness-apps-and-ui-clients.md`
- `docs/guides/release-operator-console.md`
- `docs/getting-started/harness-cookbook.md`
- `docs/concepts/worklists.md`
- `docs/guides/harness-guide.md`
- `docs/operations/ui-clients.md`
