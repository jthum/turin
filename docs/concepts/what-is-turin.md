# What Is Turin?

Turin is a runtime for building useful AI systems, not just chatting with one
assistant.

It gives agents a place to work: memory, tools, rules, durable tasks, events,
and operator interfaces. A Turin system can be a private assistant, a release
console, a coding workspace, a bug triage desk, a documentation team, or a
channel-connected operations bot.

The practical answer:

> Turin lets you turn AI behavior into an application-shaped workflow that you
> control.

## The Simple Picture

Most AI products start with a chat window. Turin starts with a workflow.

In Turin:

- agents can remember useful context
- work can be stored as durable tasks instead of disappearing into a thread
- actions can require approval before they run
- rules can decide what an agent may read, write, run, or delegate
- events make it possible to inspect what happened
- clients can present the same workflow as a terminal UI, desktop app, web app,
  or plain API

The result can still include chat, but it does not have to be only chat.

## Why That Matters

Many AI workflows fail because important things are implicit:

- the model "knows" a rule only because a prompt says so
- work state lives inside a conversation transcript
- tool permissions are broad and hard to reason about
- approvals happen manually outside the system
- every interface is forced into one generic assistant shape

Turin makes those parts explicit.

The model can suggest what to do. The harness code defines how the workflow
behaves. The runtime stores state, runs tools, records events, and enforces the
rules that have been declared.

## What You Can Build

### A Personal Agent

An assistant that remembers your preferences, project context, notes, and
scheduled work while running on infrastructure you control.

### A Team Workflow

A bug triage desk, release operator, QA console, support queue, or documentation
desk where agents help classify work, produce summaries, prepare actions, and
ask humans for approval when needed.

### A Coding Environment

A project-specific coding assistant that reads architecture notes, follows repo
rules, runs tests, creates patches, and can route review or research to other
agents.

### A Governed Automation Surface

An AI-powered operations tool where some agents are read-only, some can propose
changes, and only specific actions can run with explicit capability grants.

### A Multi-Client App

The same harness can be opened through a TUI, desktop app, web client, or remote
API. Each client owns its local UI state, while the runtime owns durable
workflow state.

## What Turin Gives You

Turin provides the pieces that are awkward to rebuild for every serious agent
project:

- durable sessions, events, worklists, memory, KV, and runtime databases
- programmable harness scripts for workflow logic and policy
- tool execution with explicit governance options
- schedules and long-running work coordination
- multi-agent orchestration
- semantic UI intent for workflow-shaped screens, lists, forms, actions,
  reports, panes, notices, and badges
- local and remote clients that can render the same runtime through different
  interfaces

## What Turin Is Not

Turin is not a fixed assistant personality.

Turin is not only a prompt library.

Turin is not only a UI.

Turin is not cloud-only by design.

Turin is the execution layer underneath agentic applications. You bring the
workflow. Turin provides the runtime shape that makes it durable, inspectable,
and governable.

## Where To Go Next

- `docs/concepts/what-can-you-do.md` — concrete use-cases and examples
- `docs/concepts/scenarios.md` — practical workflow blueprints
- `docs/concepts/harness-apps-and-ui-clients.md` — how custom harness UI fits
  with the default app, TUI, and web clients
- `docs/getting-started/index.md` — first steps
- `docs/concepts/turin.md` — deeper technical framing
- `docs/operations/ui-clients.md` — current terminal, app, and web client shape
