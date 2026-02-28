# Test Gap Finder

This block is a focused test-review harness.

It loads a checked-in testing contract, persists each request, and shapes the active agent toward identifying missing or weak coverage.

## Files

- `workspace/CHANGE_SUMMARY.md`
- `workspace/TESTING_POLICY.md`
- `workspace/RISK_AREAS.md`
- `harness/main.lua`

## What It Does

- loads testing instructions from checked-in files
- writes runtime artifacts under `.turin/runtime/test-gap-finder/`
- records requests in `test_gap_runs`
- shapes the active model toward concrete test-gap analysis
