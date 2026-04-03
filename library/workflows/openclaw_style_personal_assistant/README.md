# OpenClaw-Style Personal Assistant

This workflow is a more practical Turin-native version of the `*claw` pattern.

It is built around checked-in workspace contracts rather than hard-coded product logic:

- `SOUL.md` defines the assistant's role and operating principles
- `PROFILE.md` captures the user's standing preferences
- `AGENTS.md` describes the available specialist agents
- `INBOX.md` holds current priorities and open loops

The harness reads those files, builds a working brief, routes planning/review requests to specialist agents, and writes durable runtime artifacts so the operator can inspect what happened.

## Files

- `workspace/SOUL.md`
- `workspace/PROFILE.md`
- `workspace/AGENTS.md`
- `workspace/INBOX.md`
- `harness/main.lua`
- `agents/planner/main.lua`
- `agents/reviewer/main.lua`
- `examples/config/config.toml.example`

## Copy Into a Project

1. Copy the `workspace/` files into your project root.
2. Copy `harness/main.lua` into your main harness directory.
3. Copy the `agents/` directories into agent-specific harness directories.
4. Start from `examples/config/config.toml.example` and point the `planner` and `reviewer` agents at those harness directories.

## What It Does

- loads workspace contracts on every turn
- chooses a route based on the user request (`planner`, `reviewer`, or no delegation)
- delegates planning/review work to peer agents when the prompt calls for it
- writes runtime artifacts under `.turin/runtime/personal-assistant/`
- logs each prompt and route into `personal_assistant_activity` in the runtime DB
- folds the assembled contract and delegated output back into the effective system prompt

## Runtime Artifacts

The workflow writes:

- `.turin/runtime/personal-assistant/contract.md`
- `.turin/runtime/personal-assistant/brief.md`
- `.turin/runtime/personal-assistant/route.txt`
- `.turin/runtime/personal-assistant/last-prompt.txt`
- `.turin/runtime/personal-assistant/delegated-output.txt` (when routed)
- `.turin/runtime/personal-assistant/planner-last-request.txt` (planner path)
- `.turin/runtime/personal-assistant/reviewer-last-request.txt` (reviewer path)

These artifacts are deliberate. They make the workflow inspectable and easier to tune.
