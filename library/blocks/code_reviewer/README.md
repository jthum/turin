# Code Reviewer

This block is a focused review harness.

It loads a checked-in review contract, persists each review request, and shapes the active agent toward regression-oriented code review.

## Files

- `workspace/REVIEW_STYLE.md`
- `workspace/RISK_AREAS.md`
- `harness/main.lua`

## What It Does

- loads review instructions from checked-in files
- writes runtime artifacts under `.turin/runtime/code-review/`
- records review requests in `code_review_runs`
- shapes the active model toward correctness, regressions, and test gaps
