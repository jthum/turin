# Turin Scenario Blueprints

Turin is easier to understand through concrete workflows than through runtime
features alone. These blueprints describe the kinds of systems Turin is meant
to support and the current primitives that make each one practical.

They are not product templates or frozen package APIs. Treat them as starting
points for deciding what a harness should own, what should be durable runtime
state, and what a UI client should render.

## How To Read These

Each scenario has the same shape:

- **What it feels like**: the user-facing experience.
- **What the harness owns**: workflow rules, actions, prompts, and policy.
- **What Turin stores**: durable state that survives client restarts.
- **What the UI can show**: semantic surfaces that app, TUI, and web clients can
  render differently.
- **Start small**: the first useful slice.

The recurring rule is simple: the harness defines behavior, Turin persists and
enforces state, and each client owns its local UI session.

## Personal Operations Assistant

### What It Feels Like

You run a private assistant on your own machine or server. It knows selected
notes, preferences, schedules, recurring reminders, and the small operating
rules you care about.

It is not just a chat thread. It can remember facts, schedule future checks,
queue follow-ups, expose a daily dashboard, and ask before doing risky work.

### What The Harness Owns

- Memory rules: what should be remembered, corrected, or purged.
- Access rules: which directories, tools, channels, or web operations are
  allowed.
- Personal workflows: reminders, summaries, recurring reviews, and inbox
  triage.
- Actions such as `notes.summarize`, `inbox.triage`, or `schedule.follow_up`.

### What Turin Stores

- Long-running sessions and event history.
- Searchable memory and exact KV preferences.
- Schedules for recurring or delayed work.
- Worklist items for follow-ups that should not disappear into chat history.

### What The UI Can Show

- A home screen with today's reminders and open follow-ups.
- A list of queued items.
- A form for creating a reminder or recurring check.
- Notices when a scheduled task needs approval.
- Activity and detail surfaces for recent assistant work.

### Start Small

Create one harness with memory rules, a `followups` worklist, and one scheduled
daily summary. Add UI only after the workflow is useful from the default app or
TUI.

## Coding Workspace

### What It Feels Like

You open a project-specific agent workspace. The agent understands local
architecture maps, scratchpad notes, test commands, and contribution rules. It
can make changes, run focused tests, inspect diffs, and ask a reviewer agent for
specialized feedback.

### What The Harness Owns

- Project guardrails: which files can be changed, what maps should be read, and
  which tests are required.
- Review policy: when peer agents should inspect a change.
- Actions such as `repo.run_tests`, `repo.review_diff`, or `repo.prepare_pr`.
- Context assembly for the current task.

### What Turin Stores

- Session history and branchable task context.
- Events for tool calls, actions, and review outcomes.
- Memory for recurring project decisions.
- Worklist items for implementation tasks, review tasks, and follow-ups.

### What The UI Can Show

- A task board for planned and active changes.
- A detail pane for the selected work item.
- A report surface for test status and review notes.
- Action buttons for running tests, requesting review, or opening a branch.
- Notices when a risky operation requires confirmation.

### Start Small

Begin with a coding harness that reads the right architecture maps and runs one
focused test command. Add worklists only when tasks need to survive across
sessions.

## Release Operator Console

### What It Feels Like

You run a release desk rather than a generic assistant. The UI shows release
readiness, approvals, QA gates, blockers, and actions for seeding checks or
running smoke tests.

This is the clearest example of harness UI: the same release harness can be
opened in `turin-app`, `turin-tui`, or `turin-web`, with each client rendering
the same semantic surfaces in its own way.

### What The Harness Owns

- Release workflow stages and policy.
- Actions such as `release.seed_demo_work`, `qa.run_smoke`, or
  `release.approve`.
- UI screens, menus, forms, reports, charts, notices, badges, and refresh
  intent.
- Rules for which actions require human confirmation.

### What Turin Stores

- Worklists for approvals, checks, QA tasks, and blockers.
- Events for release activity.
- Action results and task history.
- Optional memory or KV values for release conventions.

### What The UI Can Show

- A release desk home screen.
- Worklist-backed approval tables.
- Forms for creating demo work or release checks.
- Reports and charts backed by current worklist data.
- Pane overlays for blockers or contextual release notes.
- Dynamic badges and refreshes when actions change release state.

### Start Small

Use `examples/harnesses/ui_release_operator` as the reference shape. Keep the
first version worklist-backed and add richer data adapters only when worklists
stop being enough.

For a deeper walkthrough of this scenario, see
`docs/guides/release-operator-console.md`.

## Bug Triage Desk

### What It Feels Like

Incoming issues become durable triage items. Agents classify, reproduce, route,
and investigate them while humans can inspect decisions and approve risky
actions.

### What The Harness Owns

- Classification prompts and routing rules.
- Actions for reproduction, duplicate search, assignment, and escalation.
- Criteria for when a human must review a triage decision.
- Optional channel integration for incoming alerts or issue notifications.

### What Turin Stores

- Worklists for new, investigating, blocked, and resolved issues.
- Events for classification and escalation decisions.
- Memory for recurring issue patterns.
- Signals for cross-agent handoff.

### What The UI Can Show

- A triage queue with priority and status.
- Detail views for selected issues.
- Action buttons for classify, reproduce, assign, or escalate.
- Reports for queue health and recurring categories.
- Notices for blocked or high-priority issues.

### Start Small

Create one worklist-backed queue and one classify action. Avoid building a full
ticketing replacement until the handoff loop is proven.

## Documentation And Research Team

### What It Feels Like

Agents help gather context, summarize sources, draft pages, check claims, and
split large writing work into durable pieces.

The useful interface is often not chat. It is a queue of sections, source
summaries, review status, and editorial actions.

### What The Harness Owns

- Research and citation rules.
- Actions for source collection, outline generation, drafting, review, and
  rewrite.
- Memory policy for reusable project knowledge.
- Work splitting rules for long documents.

### What Turin Stores

- Worklists for sections, source reviews, and editorial tasks.
- Events for claim checks and draft decisions.
- Memory for recurring terminology and project facts.
- KV state for exact document metadata.

### What The UI Can Show

- A writing board grouped by section status.
- Detail surfaces for selected sections.
- Activity views for source and review history.
- Forms for creating a research task.
- Reports summarizing missing sources, blocked sections, and review state.

### Start Small

Start with one document, one source-review worklist, and one draft action. Add
multi-agent review only after the single-agent workflow produces useful drafts.

## Governed Team Tooling

### What It Feels Like

Multiple people or agents use Turin in a shared workspace where authority must
be explicit. Some agents may be read-only reviewers. Others may write files,
run tests, or request temporary elevation.

### What The Harness Owns

- Capability policy and temporary grant rules.
- Import boundaries for reusable modules.
- Escalation actions and audit-friendly approval flows.
- Per-agent behavior differences.

### What Turin Stores

- Governance configuration and grant audit data.
- Session and event history.
- Worklist items for approvals and escalations.
- Signals for cross-agent requests.

### What The UI Can Show

- Approval queues for requested capability grants.
- Agent and task status.
- Notices for denied, expired, or used grants.
- Reports for recent high-risk operations.
- Action buttons for approve, reject, revoke, or inspect.

### Start Small

Use explicit capabilities before adding UI. Once the policy is clear, expose the
approval path as a worklist-backed screen.

## Channel-Based Operations Bot

### What It Feels Like

Turin connects to Telegram, Discord, Rocket.Chat, WhatsApp, or another sidecar.
The channel is the input/output surface, while Turin handles memory, state,
tool execution, schedules, and governance.

### What The Harness Owns

- Channel-specific workflow behavior.
- Routing from messages into tasks, actions, or worklists.
- Approval and escalation rules.
- Response style and memory policy.

### What Turin Stores

- Channel-linked sessions.
- Events and tool execution history.
- Memory and KV state.
- Worklists and schedules created from channel activity.

### What The UI Can Show

- Runtime health and channel status.
- Pending channel approvals.
- Worklists created from channel events.
- Activity views for recent channel-driven tasks.

### Start Small

Start with one channel, one agent, and one narrow workflow. Keep channel glue in
the sidecar and keep durable workflow state in Turin primitives.

## How These Scenarios Compose

Real Turin systems can combine scenarios:

- a coding workspace can use governed team tooling
- a release console can ingest channel alerts
- a documentation team can use worklists and peer review
- a personal assistant can run scheduled research or triage tasks

This is why Turin should stay a runtime and harness platform rather than a
single fixed app. The default clients should be useful without custom UI, while
harness UI can turn a workflow into a purpose-built application when needed.

## Next Reading

- `docs/concepts/what-can-you-do.md`
- `docs/concepts/scenario-starter-cards.md`
- `docs/getting-started/harness-cookbook.md`
- `docs/concepts/worklists.md`
- `docs/operations/ui-clients.md`
- `docs/reference/primitives.md`
