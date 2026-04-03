# Release Manager

This workflow is a practical baseline for release preparation.

It reads checked-in release context, runs a readiness reviewer specialist to assess blockers and risk, then runs a changelog writer specialist to draft release notes.

## Files

- `workspace/RELEASE_GOALS.md`
- `workspace/CHANGELOG_NOTES.md`
- `workspace/OPEN_ISSUES.md`
- `workspace/CHECKLIST.md`
- `workspace/CONSTRAINTS.md`
- `harness/main.lua`
- `agents/readiness_reviewer/main.lua`
- `agents/changelog_writer/main.lua`
- `examples/config/config.toml.example`

## What It Does

- loads release context from checked-in markdown files
- asks a readiness reviewer for ship/no-ship concerns and blockers
- asks a changelog writer for operator-facing release notes
- writes runtime artifacts under `.turin/runtime/release-manager/`
- logs each run into `release_manager_runs` in the runtime DB
- folds the readiness review and draft release notes back into the effective system prompt

## Runtime Artifacts

- `.turin/runtime/release-manager/context.md`
- `.turin/runtime/release-manager/readiness.md`
- `.turin/runtime/release-manager/changelog.md`
- `.turin/runtime/release-manager/brief.md`
- `.turin/runtime/release-manager/readiness-reviewer-last-request.txt`
- `.turin/runtime/release-manager/changelog-writer-last-request.txt`
