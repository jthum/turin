# Scenario Starter Cards

Use these cards when someone asks, "What would I actually build with Turin
first?"

Each card keeps the first slice small. The goal is not to design a full product
up front. The goal is to prove that durable state, harness policy, and an
operator surface make the workflow better than a generic chat thread.

## How To Use This Page

Pick the closest scenario and build only the first slice:

- one harness
- one agent
- one durable primitive that matters
- one action that saves time or reduces risk
- one optional UI surface only if the workflow already has visible structure

If the workflow does not need memory, durable work, tools, approvals,
schedules, events, or custom UI, Turin is probably more structure than you need
for the first version.

## Personal Operations Assistant

**Best first slice:** a private assistant with memory rules and a follow-up
queue.

**Use Turin when:** you want the assistant to remember selected facts, keep
follow-ups outside chat history, and run on infrastructure you control.

**Start with:**

- memory for reusable preferences and facts
- KV for exact settings
- one `followups` worklist
- one action for creating a follow-up
- one scheduled daily or weekly summary

**First useful UI:**

- a home screen with open follow-ups
- a form for adding a reminder
- an activity surface for recent completed work

**Do not start with:** a giant personal CRM, many channels, or complex UI
before the memory and follow-up loop is useful.

## Coding Workspace

**Best first slice:** a repository harness that reads project rules and runs one
focused test command.

**Use Turin when:** a coding assistant should follow repository-specific
guardrails, preserve context, run tools deliberately, and produce inspectable
work.

**Start with:**

- harness rules for required architecture maps and allowed tools
- memory for recurring project decisions
- one action for running focused tests
- optional worklist items for tasks that must survive sessions

**First useful UI:**

- a task list for planned changes
- a detail pane for the selected task
- action buttons for test, review, or prepare handoff

**Do not start with:** a full IDE replacement. Let Turin own workflow state and
policy; keep editing in the tools that already work.

## Release Operator

**Best first slice:** one release worklist with approvals and one confirmed
action.

**Use Turin when:** release work has approvals, QA gates, blockers, readiness
checks, or actions that should not run accidentally.

**Start with:**

- one release worklist
- metadata for release, lane, kind, and status
- one seed or smoke-test action
- one confirmed approve/reject action
- one lightweight readiness report

**First useful UI:**

- approval table
- selected item detail
- readiness report or chart
- notices and refresh hints after actions mutate release state

**Do not start with:** custom analytics or a full deployment control plane.
Keep the first version worklist-backed.

## Bug Triage Desk

**Best first slice:** one incoming issue queue and one classify action.

**Use Turin when:** issues need durable classification, routing, investigation,
and auditable decisions.

**Start with:**

- one triage worklist
- metadata for priority, source, component, and category
- one classify action
- one escalation or assignment action
- events for classification decisions

**First useful UI:**

- triage table
- detail view for selected issue
- classify/escalate actions
- report for queue health

**Do not start with:** replacing an existing tracker. Start by improving the
decision loop around incoming work.

## Documentation Or Research Team

**Best first slice:** one document, one source-review queue, and one draft
action.

**Use Turin when:** writing work needs sources, summaries, claim checks, review
state, and durable editorial tasks.

**Start with:**

- worklist items for sections or sources
- memory for project terminology and reusable facts
- KV for exact document metadata
- one action for drafting or reviewing a section
- one action for checking missing sources

**First useful UI:**

- section/source queue
- detail view for selected section
- activity surface for source reviews
- report for blocked or missing-source sections

**Do not start with:** a general research platform. Start with one real document
and one review loop.

## Governed Team Tooling

**Best first slice:** one governed action path with an approval worklist.

**Use Turin when:** a team needs visible authority, auditability, and different
capability ceilings for different agents or roles.

**Start with:**

- explicit capability profile
- one action that may require temporary elevation
- events for allow/deny decisions
- worklist items for pending approvals
- clear grant expiry rules

**First useful UI:**

- approval queue
- detail view for requested authority
- approve/reject/revoke actions
- report for recent high-risk operations

**Do not start with:** organization-wide policy automation. Prove one governed
workflow first.

## Channel Operations Bot

**Best first slice:** one channel, one agent, and one narrow job.

**Use Turin when:** channel messages should create durable tasks, trigger
controlled actions, or route work into an inspectable system.

**Start with:**

- one channel sidecar
- one routing rule from message to task/action
- events for channel activity
- optional worklist for items that need follow-up
- memory rules for what the bot should retain

**First useful UI:**

- channel health/status
- queue of pending channel-driven work
- activity surface for recent messages/actions

**Do not start with:** every channel and every workflow. Start with one channel
where persistent state changes the outcome.

## Choosing Between Similar Scenarios

| If the main pain is... | Start with... |
| --- | --- |
| Remembering and following up | Personal operations assistant |
| Repository-specific rules and tests | Coding workspace |
| Approvals and readiness | Release operator |
| Incoming issue decisions | Bug triage desk |
| Sections, sources, and review state | Documentation or research team |
| Visible authority and grants | Governed team tooling |
| Messages becoming durable work | Channel operations bot |

## Next Reading

- `docs/concepts/scenarios.md` - broader scenario blueprints
- `docs/getting-started/choose-first-workflow.md` - choosing a first workflow
- `docs/concepts/harness-apps-and-ui-clients.md` - when custom UI helps
- `docs/concepts/worklists.md` - durable work coordination
- `docs/guides/harness-guide.md` - harness authoring
- `docs/operations/ui-clients.md` - current app, TUI, and web client behavior
