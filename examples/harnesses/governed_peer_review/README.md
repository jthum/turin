# Governed Peer Review

This example shows a main harness calling a reviewer peer agent under a temporary grant.

Use this pattern when:

- the main agent should stay flexible
- peer review should be explicit and auditable
- peer-agent submission/await needs to be capability-scoped

## Files

- `harness/main.lua` — main agent harness
- `agents/reviewer/main.lua` — reviewer harness

## Copy Into a Project

1. Copy `harness/main.lua` into your main harness directory.
2. Copy `agents/reviewer/main.lua` into a reviewer harness directory.
3. Configure a `reviewer` agent in `turin.toml` pointing at that reviewer harness directory.
4. Enable governance grants.

## What It Does

- requests a short-lived grant for peer-agent work
- sends the latest user prompt to the reviewer agent
- injects the reviewer output back into the main agent's effective system prompt
