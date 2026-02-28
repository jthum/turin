# Repo Librarian

This block is a focused repository-contract harness.

It loads checked-in repository contracts, persists each request, and shapes the active agent toward routing work according to project intent, architecture, and conventions.

## Files

- `workspace/SOUL.md`
- `workspace/AGENTS.md`
- `workspace/ARCHITECTURE.md`
- `workspace/CONVENTIONS.md`
- `harness/main.lua`

## What It Does

- loads repository contracts from checked-in files
- writes runtime artifacts under `.turin/runtime/repo-librarian/`
- records requests in `repo_librarian_runs`
- shapes the active model toward repository-aware routing and guidance
