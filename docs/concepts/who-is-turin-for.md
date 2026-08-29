# Who Is Turin For?

Turin is for people who want AI agents to become controlled workflows, not just
clever conversations.

The common thread is control. Turin is useful when you care about what an agent
can do, what state it remembers, what work remains open, which actions require
approval, and how the workflow is inspected later.

## Good Fits

### Personal Operators

Use Turin when you want a private assistant that follows your rules, remembers
useful context, runs on infrastructure you control, and can grow beyond one chat
thread.

Good examples:

- personal memory and notes
- recurring follow-ups
- local document workflows
- private project assistants
- channel-connected bots for your own routines

### Power Users And Professionals

Use Turin when existing assistants are helpful but too generic. A Turin harness
can encode the workflow you repeat every week: how to triage, what to check,
which actions are safe, and when to ask for confirmation.

Good examples:

- a release checklist with approvals
- a bug triage desk
- a documentation review queue
- a research summary workflow
- a support or operations console

### Developers And Builders

Use Turin when you want to build an agentic application rather than wire
together prompts. Harness scripts are versionable behavior. Worklists, memory,
events, schedules, and UI intent are runtime primitives rather than one-off
glue.

Good examples:

- coding workspaces with project-specific rules
- repository review agents
- multi-agent research tools
- custom internal AI products
- app-like harness packages with screens, forms, lists, reports, and actions

### Team Leads

Use Turin when a team needs shared agent behavior with visible authority.
Governance can define which agents may read, write, run shell commands, access
the web, delegate work, or request temporary grants.

Good examples:

- read-only reviewer agents
- controlled deployment assistants
- QA or release operators
- audit-friendly team workflows
- repeatable policy around tool access

### Platform And Operations Teams

Use Turin when you want a self-hosted runtime that can be shaped for multiple
internal workflows without turning every workflow into a separate bespoke
service.

Good examples:

- internal knowledge assistants
- governed automation surfaces
- channel-connected operations bots
- team-specific harnesses with separate state stores
- remote clients connected to a server-side Turin daemon

### Researchers And Experimenters

Use Turin when you want to explore agent coordination, durable work, scheduling,
signals, memory, governance, and new interaction models in a controlled runtime.

Good examples:

- specialist agent teams
- long-running workflow experiments
- event-driven coordination
- recoverable task pipelines
- custom UI surfaces for agent operations

## When Turin Is Probably Too Much

Turin may be the wrong starting point if you only need:

- a single disposable chat session
- a fixed hosted assistant with no custom behavior
- a simple prompt template
- a UI-only product with no durable agent runtime
- fully managed cloud convenience above control and inspectability

Turin intentionally exposes real structure: harnesses, capabilities, durable
state, events, worklists, and clients. That structure pays off when the workflow
matters.

## The Short Test

Turin is worth considering if you can say:

> I want the agent to follow my workflow, keep durable state, and make authority
> explicit.

If the main need is "answer this one question", use a normal assistant. If the
need is "run this kind of work repeatedly, safely, and visibly", Turin is a
better fit.

## Where To Go Next

- `docs/concepts/what-is-turin.md` - plain-language product explanation
- `docs/concepts/what-can-you-do.md` - scenario-first overview
- `docs/concepts/scenarios.md` - concrete workflow blueprints
- `docs/getting-started/choose-first-workflow.md` - choosing a first workflow
