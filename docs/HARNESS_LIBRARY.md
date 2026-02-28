# Harness Library

Turin ships a Harness Library under `library/`.

This library is not primarily for tutorial snippets.
It is for practical, ready-to-use harnesses that can be adopted directly or adapted with minimal changes.

## Structure

```text
library/
  blocks/
  workflows/
```

- `blocks/` — reusable harness units focused on a specific job
- `workflows/` — larger end-to-end harnesses for real operator flows

## Current Library

### Blocks

- `library/blocks/governed_peer_review/`
- `library/blocks/delegated_peer_capabilities/`
- `library/blocks/durable_journal/`

### Workflows

- `library/workflows/openclaw_style_personal_assistant/`
- `library/workflows/full_coding_harness/`
- `library/workflows/bug_triage_desk/`

## Validation

The Harness Library is exercised by:

```bash
cargo test --test example_harness_examples
```

That suite copies library entries into temporary workspaces, runs Turin with mock providers, and verifies behavior or produced artifacts.

## Current Entries

## OpenClaw-Style Personal Assistant

`library/workflows/openclaw_style_personal_assistant/`

Pattern:

- repository-level markdown contracts such as `SOUL.md`, `PROFILE.md`, `AGENTS.md`, and `INBOX.md`
- harness builds a working brief from those files every turn
- planning/review prompts are routed to specialist peer agents
- runtime writes inspectable artifacts and logs activity into the runtime DB

## Full Coding Harness

`library/workflows/full_coding_harness/`

Pattern:

- checked-in coding context (`SPEC.md`, `TASKS.md`, `CONSTRAINTS.md`, `NOTES.md`)
- planner specialist produces an execution plan
- reviewer specialist critiques that plan for regressions and missing tests
- runtime artifacts and DB rows capture the run for later inspection

## Bug Triage Desk

`library/workflows/bug_triage_desk/`

Pattern:

- checked-in triage context (`SEVERITY_POLICY.md`, `OWNERSHIP.md`, `KNOWN_ISSUES.md`, `RUNBOOK.md`)
- triager specialist classifies severity, owner, and next checks
- responder specialist drafts an operator-facing follow-up
- runtime artifacts and DB rows capture the triage flow

## Governed Peer Review

`library/blocks/governed_peer_review/`

Pattern:

- main agent delegates to a reviewer peer agent
- delegation runs under a temporary grant
- reviewer output is written to runtime artifacts and folded back into prompt shaping

## Delegated Peer Capabilities

`library/blocks/delegated_peer_capabilities/`

Pattern:

- main agent calls a reviewer peer through `runtime.agent(...):complete(...)`
- the call site delegates only a narrow capability slice
- the reviewer harness proves read/query is allowed while mutation stays denied
- runtime artifacts capture the delegated prompt and reviewer output

## Durable Journal

`library/blocks/durable_journal/`

Pattern:

- store operational notes durably in the runtime DB
- keep a memory trace and a durable SQL record
- emit a simple runtime artifact for inspection
