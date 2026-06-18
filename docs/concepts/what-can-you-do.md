# What Can You Do With Turin?

Turin is a programmable runtime for AI agents.

It is not just one assistant, one chat window, or one fixed workflow. It is a
runtime where agents can remember, use tools, follow rules, coordinate durable
work, and expose purpose-built operator surfaces.

The shortest useful answer:

> Turin lets you build AI systems that are governed by your own code, backed by
> durable state, and shaped around your workflow.

## The Mental Model

Most AI tools start with a chat box. Turin starts with a runtime.

In Turin:

- the model proposes what to do
- the harness decides what is allowed and how the workflow behaves
- the Rust runtime persists state, enforces rules, runs tools, and records what
  happened
- UI clients can render the same harness as a terminal, native app, or web
  surface when that makes sense

This is why Turin can support many different agent experiences without becoming
many different runtimes.

## Example Uses

### Private Personal Assistant

Run an assistant on your own machine or server. Give it memory, local notes,
project context, schedules, and strict rules about what it can access.

Useful for remembering preferences, summarizing local documents, coordinating
personal workflows, and keeping private context under your control.

### Coding Workspace

Create a coding harness that knows how your project should be changed. It can
read architecture notes, run tests, inspect diffs, ask peer agents to review
work, and follow workspace-specific guardrails.

Useful for implementing features with project-specific rules, reviewing patches,
finding missing tests, and delegating focused work to specialist agents.

### Release Operator Console

Use Turin as an operator surface for release work. A harness can define actions,
approval lists, forms, status views, event-driven refreshes, and reports. A TUI,
desktop app, or web client can render that as a release desk instead of a generic
chat thread.

Useful for release checklists, approval queues, QA gates, deployment readiness
summaries, and human-in-the-loop decisions.

### Bug Triage Desk

Build a harness that watches incoming issues, classifies them, creates durable
work items, assigns priority, and asks agents to investigate likely root causes.

Useful for support queues, reproducibility checks, routing work to the right
specialist, and maintaining an auditable trail of decisions.

### Documentation Or Research Team

Run agents that gather context, summarize sources, draft pages, check claims,
and split large writing jobs into durable work items.

Useful for documentation planning, research synthesis, source-backed summaries,
multi-step editorial workflows, and reusable project knowledge.

### Channel-Based Agent

Connect Turin to messaging channels through sidecars. The channel stays outside
the core runtime, while Turin handles memory, state, tool use, schedules, and
governance.

Useful for Telegram, Discord, Rocket.Chat, WhatsApp, team chat workflows, alert
handling, and lightweight operations bots.

### Governed Team Tooling

Use capability rules and harness policies to make agent behavior explicit. A
team can decide which agents can read, write, run shell commands, use the web,
delegate work, or request temporary elevation.

Useful for shared development workspaces, audit-heavy operations, read-only
reviewer agents, tightly controlled production workflows, and regulated
environments.

### Multi-Agent Lab

Use Turin as a controlled environment for experimenting with agent teams,
delegation, durable signals, worklists, and custom coordination patterns.

Useful for specialist agent teams, scheduler-driven workflows, recoverable
long-running tasks, and experiments that need persistence and inspection.

## What Makes Turin Different?

Turin separates intelligence from authority.

The model can be creative, but it does not get automatic control over the
system. The harness is deterministic code that decides what is allowed. The
runtime enforces those decisions and records the result.

That gives Turin a different shape from prompt-only assistants:

- behavior can be versioned as harness code
- memory and work are durable rather than trapped inside a conversation
- tool access is explicit instead of hidden inside a broad permission grant
- long-running work can be scheduled, inspected, paused, resumed, or retried
- multiple clients can present the same runtime in different ways

## What Turin Is Not

Turin is not only a chatbot UI. Chat can be one interface, but serious agent
workflows often need lists, forms, approvals, reports, dashboards, and actions.

Turin is not a prompt library. Prompts matter, but deterministic harness code,
runtime state, and governance are first-class parts of the system.

Turin is not a cloud-only service. The core runtime is designed to run locally or
on infrastructure you control.

Turin is not meant to hide all complexity. Simple workflows should be easy, but
Turin exists because complex, governed, durable agent systems need real
structure.

## A Good First Path

Start with one concrete workflow.

Good first workflows include:

- a personal assistant with persistent memory
- a coding harness for one repository
- a release checklist with approvals
- a bug triage queue
- a documentation assistant for one project

Do not start by designing a giant agent platform. Start with one harness, one
runtime, and one workflow where durable memory, explicit authority, or custom UI
would clearly help.

Then continue with:

- `docs/getting-started/index.md`
- `docs/getting-started/harness-cookbook.md`
- `docs/guides/harness-library.md`
- `docs/reference/primitives.md`
