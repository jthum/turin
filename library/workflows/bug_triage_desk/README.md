# Bug Triage Desk

This workflow is a practical baseline for issue intake and triage.

It reads checked-in triage policy and ownership context, runs a triager specialist to classify the report, then runs a responder specialist to produce an operator-facing next-action draft.

## Files

- `workspace/SEVERITY_POLICY.md`
- `workspace/OWNERSHIP.md`
- `workspace/KNOWN_ISSUES.md`
- `workspace/RUNBOOK.md`
- `harness/main.lua`
- `agents/triager/main.lua`
- `agents/responder/main.lua`
- `turin.toml.example`

## What It Does

- loads triage context from checked-in markdown files
- asks a triager specialist for severity/owner/next-checks classification
- asks a responder specialist for an operator-facing follow-up message
- writes runtime artifacts under `.turin/runtime/bug-triage/`
- logs each run into `bug_triage_runs` in the runtime DB
- folds triage and response output back into the effective system prompt

## Runtime Artifacts

- `.turin/runtime/bug-triage/context.md`
- `.turin/runtime/bug-triage/triage.md`
- `.turin/runtime/bug-triage/response.md`
- `.turin/runtime/bug-triage/brief.md`
- `.turin/runtime/bug-triage/triager-last-request.txt`
- `.turin/runtime/bug-triage/responder-last-request.txt`
