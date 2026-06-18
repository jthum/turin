# What Can You Do With Turin?

Turin is a programmable runtime for AI agents.

That means Turin is not just one assistant, one chat window, or one fixed
workflow. It is a runtime where agents can remember, use tools, follow rules,
coordinate durable work, and expose purpose-built operator surfaces.

The shortest useful answer is:

> Turin lets you build AI systems that are governed by your own code, backed by
> durable state, and shaped around your workflow.

## The 30-Second Answer

If someone asks what Turin is, say this:

Turin is a runtime for building your own AI-powered workflows. Instead of giving
an AI model broad permission and hoping it behaves, you write a harness that
defines the rules, actions, memory, tools, and operator screens for that
workflow. Turin runs the agents, enforces the rules, persists the state, and
keeps the work inspectable.

That means Turin can be a private assistant, a coding workspace, a release
console, a bug triage desk, a documentation helper, or a governed team tool. The
point is not that Turin is one perfect assistant. The point is that Turin is the
runtime underneath many purpose-built assistants and agentic applications.

## The Mental Model

Most AI tools start with a chat box. Turin starts with a runtime.

In Turin:

- The model proposes what to do.
- The harness decides what is allowed and how the workflow behaves.
- The Rust runtime persists state, enforces rules, runs tools, and records what
  happened.
- UI clients can render the same harness as a terminal app, native app, or web
  surface depending on what makes sense.

This is why Turin can support many different agent experiences without becoming
many different runtimes.

## Harnesses Can Feel Like Apps

A Turin harness is not just a prompt file. It can become the shape of a small
agentic application.

A harness can define:

- the rules an agent follows
- the actions an operator can run
- the memory and context the agent should use
- durable work items that need attention
- screens, menus, lists, forms, reports, panes, notices, and badges for UI
  clients

That means the same runtime can open different harnesses as different
workflows. One harness might feel like a coding workspace. Another might feel
like a release desk. Another might be a personal operations assistant.

The client is still local to the user. The terminal UI, desktop app, and web
client can each render the same harness intent in the way that fits their
medium.

## What This Looks Like In Practice

### Private Personal Assistant

Run an assistant on your own machine or server. Give it memory, local notes,
project context, schedules, and strict rules about what it can access.

Useful for:

- remembering preferences and long-running context
- summarizing local documents
- coordinating personal workflows
- running privately without handing the whole workflow to a hosted assistant

### Coding Workspace

Create a coding harness that knows how your project should be changed. It can
read architecture notes, run tests, inspect diffs, ask peer agents to review
work, and follow workspace-specific guardrails.

Useful for:

- implementing features with project-specific rules
- reviewing patches
- finding missing tests
- delegating focused work to specialist agents
- keeping tool access explicit and auditable

### Release Operator Console

Use Turin as an operator surface for release work. A harness can define actions,
approval lists, forms, status views, event-driven refreshes, and reports. A
terminal, desktop, or web client can render that as a release desk instead of a
generic chat thread.

Useful for:

- release checklists
- approval queues
- QA gates
- deployment readiness summaries
- human-in-the-loop decisions

### Bug Triage Desk

Build a harness that watches incoming issues, classifies them, creates durable
work items, assigns priority, and asks agents to investigate likely root causes.

Useful for:

- issue triage
- support queues
- reproducibility checks
- routing work to the right specialist
- maintaining an auditable trail of decisions

### Documentation Or Research Team

Run agents that gather context, summarize sources, draft pages, check claims,
and split large writing jobs into durable work items.

Useful for:

- documentation planning
- research synthesis
- source-backed summaries
- multi-step editorial workflows
- reusable project knowledge

### Channel-Based Agent

Connect Turin to messaging channels through sidecars. The channel stays outside
the core runtime, while Turin handles memory, state, tool use, schedules, and
governance.

Useful for:

- Telegram, Discord, Rocket.Chat, or WhatsApp assistants
- team chat workflows
- personal automation
- alert handling
- lightweight operations bots

### Governed Team Tooling

Use capability rules and harness policies to make agent behavior explicit. A
team can decide which agents can read, write, run shell commands, use the web,
delegate work, or request temporary elevation.

Useful for:

- regulated environments
- shared development workspaces
- audit-heavy operations
- read-only reviewer agents
- tightly controlled production workflows

### Multi-Agent Lab

Use Turin as a controlled environment for experimenting with agent teams,
delegation, durable signals, worklists, and custom coordination patterns.

Useful for:

- multi-agent research
- specialist agent teams
- scheduler-driven workflows
- recoverable long-running tasks
- experiments that need persistence and inspection

## What Makes Turin Different?

Turin separates intelligence from authority.

The model can be creative, but it does not get automatic control over the
system. The harness is deterministic code that decides what is allowed. The
runtime enforces those decisions and records the result.

That gives Turin a different shape from prompt-only assistants:

- Behavior can be versioned as harness code.
- Memory and work are durable rather than trapped inside a conversation.
- Tool access is explicit instead of hidden inside a broad permission grant.
- Long-running work can be scheduled, inspected, paused, resumed, or retried.
- Multiple clients can present the same runtime in different ways.
- Harnesses can define app-like surfaces without hardcoding one universal UI.

## What Turin Is Not

Turin is not only a chatbot UI. Chat can be one interface, but serious agent
workflows often need lists, forms, approvals, reports, dashboards, and actions.

Turin is not a prompt library. Prompts matter, but Turin treats deterministic
harness code, runtime state, and governance as first-class parts of the system.

Turin is not a cloud-only service. The core runtime is designed to run locally or
on infrastructure you control.

Turin is not meant to hide all complexity. Simple workflows should be easy, but
Turin exists because complex, governed, durable agent systems need real
structure.

## A Good First Path

If you are evaluating Turin, start with one concrete workflow.

Good first workflows include:

- a personal assistant with persistent memory
- a coding harness for one repository
- a release checklist with approvals
- a bug triage queue
- a documentation assistant for one project

For the first version, keep the UI simple: one or two screens, one useful list,
one detail view, and one action that saves real time. Turin is designed so that
simple can stay simple while the workflow grows into forms, reports, schedules,
signals, and multiple clients later.

Do not start by designing a giant agent platform. Start with one harness, one
runtime, and one workflow where durable memory, explicit authority, or custom UI
would clearly help.

Once that works, Turin can grow with you.

Next:

- `docs/concepts/scenarios.md`
- `docs/getting-started/choose-first-workflow.md`
- `docs/getting-started/index.md`
- `docs/getting-started/harness-cookbook.md`
- `docs/guides/harness-library.md`
- `docs/reference/primitives.md`
