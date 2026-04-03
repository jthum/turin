# Full Coding Harness

This workflow is a practical baseline for a Turin-based coding assistant.

It reads checked-in coding context, produces an execution plan through a planner specialist, runs a reviewer specialist against that plan, and writes durable artifacts so the operator can inspect or reuse the results.

## Files

- `workspace/SPEC.md`
- `workspace/TASKS.md`
- `workspace/CONSTRAINTS.md`
- `workspace/NOTES.md`
- `harness/main.lua`
- `agents/planner/main.lua`
- `agents/reviewer/main.lua`
- `examples/config/config.toml.example`

## What It Does

- loads coding context from checked-in markdown files
- asks a planner peer agent for a concrete execution plan
- asks a reviewer peer agent to critique that plan for regressions and missing tests
- writes runtime artifacts under `.turin/runtime/coding-harness/`
- logs each run into `coding_harness_runs` in the runtime DB
- folds plan and review output back into the effective system prompt

## Runtime Artifacts

- `.turin/runtime/coding-harness/context.md`
- `.turin/runtime/coding-harness/plan.md`
- `.turin/runtime/coding-harness/review.md`
- `.turin/runtime/coding-harness/brief.md`
- `.turin/runtime/coding-harness/planner-last-request.txt`
- `.turin/runtime/coding-harness/reviewer-last-request.txt`
