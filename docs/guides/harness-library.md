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

- `library/blocks/code_reviewer/`
- `library/blocks/task_planner/`
- `library/blocks/spec_writer/`
- `library/blocks/test_gap_finder/`
- `library/blocks/repo_librarian/`
- `library/blocks/release_readiness_checker/`
- `library/blocks/docs_maintainer/`
- `library/blocks/changelog_writer/`
- `library/blocks/governed_peer_review/`
- `library/blocks/delegated_peer_capabilities/`
- `library/blocks/durable_journal/`

### Workflows

- `library/workflows/openclaw_style_personal_assistant/`
- `library/workflows/full_coding_harness/`
- `library/workflows/bug_triage_desk/`
- `library/workflows/release_manager/`
- `library/workflows/docs_team_assistant/`

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

## Release Manager

`library/workflows/release_manager/`

Pattern:

- checked-in release context (`RELEASE_GOALS.md`, `CHANGELOG_NOTES.md`, `OPEN_ISSUES.md`, `CHECKLIST.md`, `CONSTRAINTS.md`)
- readiness reviewer specialist assesses blockers and shipping risk
- changelog writer specialist drafts release notes
- runtime artifacts and DB rows capture the release-prep flow

## Docs Team Assistant

`library/workflows/docs_team_assistant/`

Pattern:

- checked-in documentation context (`PUBLIC_SURFACE.md`, `DOCS_TARGETS.md`, `DRIFT_NOTES.md`, `STYLE_NOTES.md`)
- docs reviewer specialist identifies drift and stale claims
- draft writer specialist prepares update text
- runtime artifacts and DB rows capture the docs-maintenance flow

## Governed Peer Review

`library/blocks/governed_peer_review/`

Pattern:

- main agent delegates to a reviewer peer agent
- delegation runs under a temporary grant
- reviewer output is written to runtime artifacts and folded back into prompt shaping

## Code Reviewer

`library/blocks/code_reviewer/`

Pattern:

- checked-in review contract (`REVIEW_STYLE.md`, `RISK_AREAS.md`)
- shapes the active agent toward regression-oriented review
- writes runtime artifacts and records review requests in the runtime DB

## Task Planner

`library/blocks/task_planner/`

Pattern:

- checked-in planning contract (`PLANNING_STYLE.md`, `DELIVERY_CONSTRAINTS.md`)
- shapes the active agent toward sequenced task planning
- writes runtime artifacts and records planning requests in the runtime DB

## Spec Writer

`library/blocks/spec_writer/`

Pattern:

- checked-in spec-writing contract (`IDEA.md`, `ACCEPTANCE.md`, `CONTEXT.md`)
- shapes the active agent toward concrete implementation/spec writing
- writes runtime artifacts and records requests in the runtime DB

## Test Gap Finder

`library/blocks/test_gap_finder/`

Pattern:

- checked-in test-review contract (`CHANGE_SUMMARY.md`, `TESTING_POLICY.md`, `RISK_AREAS.md`)
- shapes the active agent toward identifying missing tests and risky untested paths
- writes runtime artifacts and records requests in the runtime DB

## Repo Librarian

`library/blocks/repo_librarian/`

Pattern:

- checked-in repository contract (`SOUL.md`, `AGENTS.md`, `ARCHITECTURE.md`, `CONVENTIONS.md`)
- shapes the active agent toward repository-aware routing and guidance
- writes runtime artifacts and records requests in the runtime DB

## Release Readiness Checker

`library/blocks/release_readiness_checker/`

Pattern:

- checked-in readiness contract (`CHECKLIST.md`, `RISK_REGISTER.md`, `RELEASE_NOTES_CONTEXT.md`)
- shapes the active agent toward blocker/risk-based release assessment
- writes runtime artifacts and records requests in the runtime DB

## Docs Maintainer

`library/blocks/docs_maintainer/`

Pattern:

- checked-in docs-maintenance contract (`PUBLIC_SURFACE.md`, `DOCS_POLICY.md`, `DRIFT_SIGNALS.md`)
- shapes the active agent toward documentation drift analysis
- writes runtime artifacts and records requests in the runtime DB

## Changelog Writer

`library/blocks/changelog_writer/`

Pattern:

- checked-in changelog contract (`RELEASE_SCOPE.md`, `MERGED_CHANGES.md`, `WRITING_STYLE.md`)
- shapes the active agent toward concise release-note drafting
- writes runtime artifacts and records requests in the runtime DB

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
