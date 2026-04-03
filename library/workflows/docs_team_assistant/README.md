# Docs Team Assistant

This workflow is a practical baseline for documentation maintenance.

It reads checked-in docs context, runs a docs reviewer specialist to identify drift and stale claims, then runs a draft writer specialist to produce a concise update draft.

## Files

- `workspace/PUBLIC_SURFACE.md`
- `workspace/DOCS_TARGETS.md`
- `workspace/DRIFT_NOTES.md`
- `workspace/STYLE_NOTES.md`
- `harness/main.lua`
- `agents/docs_reviewer/main.lua`
- `agents/draft_writer/main.lua`
- `examples/config/config.toml.example`

## What It Does

- loads documentation context from checked-in markdown files
- asks a docs reviewer for drift analysis and target identification
- asks a draft writer for operator-facing update text
- writes runtime artifacts under `.turin/runtime/docs-team/`
- logs each run into `docs_team_runs` in the runtime DB
- folds the review findings and draft back into the effective system prompt
