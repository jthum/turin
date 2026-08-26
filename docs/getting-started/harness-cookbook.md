# Harness Cookbook

This document is the fast path from "I want a harness" to "I have a harness I can read and test."

## 1. Scaffold a project

For a no-key local smoke:

```bash
turin quickstart --prompt "Summarize this workspace."
```

That creates `.turin/config.toml`, `.turin/harnesses/`, a starter state DB, and runs the prompt with the mock provider.

For a real project scaffold:

```bash
turin init --provider anthropic --harness-template coding-assistant
```

Useful flags:

- `--provider anthropic|openai|mock`
- `--model ...`
- `--harness-template starter|safety|coding-assistant|reviewer`
- `--yes` to skip prompts
- `--force` to overwrite the generated starter files

`turin init` writes a transparent, enforcement-disabled governance block for
the frictionless local path. Use interactive `turin-manager init` when you want
it to generate an explicit open, balanced, or governed policy template.

## 2. Generate a harness template

Scaffold a harness into the default project directory:

```bash
turin harness new coding-assistant
```

Or put it somewhere else:

```bash
turin harness new reviewer --dir .turin/harnesses-reviewer
```

Current starter templates:

- `starter` — readable baseline with light workspace context
- `safety` — blocks obviously destructive shell commands
- `coding-assistant` — folds checked-in briefs into prompts
- `reviewer` — pushes findings-first review output

## 3. Test a harness quickly

Run the configured harness against the mock provider:

```bash
turin harness test --prompt "Say HARNESS_TEST_OK" --response "HARNESS_TEST_OK"
```

Test a different harness directory without rebinding your config:

```bash
turin harness test \
  --dir .turin/harnesses-reviewer \
  --prompt "Review this change" \
  --response "HARNESS_TEST_OK"
```

This uses the normal runtime path, but forces the selected agent onto:

- provider `mock`
- model `mock-model`
- the supplied mock response string

That keeps the test cheap and deterministic while still exercising config loading, harness init, and a real session run.

## 4. Progressive examples

### Safety-only

Use this when the first job is "do no damage."

```bash
turin harness new safety
turin harness test --response "Safety harness loaded."
```

### Coding assistant

Use this when the agent should stay grounded in checked-in project context.

```bash
turin init --provider anthropic --harness-template coding-assistant
printf "Project rules go here\n" > TURIN.md
turin harness test --response "Coding harness loaded."
```

### Reviewer

Use this when you want findings-first output without changing the global system prompt.

```bash
turin harness new reviewer --dir .turin/harnesses-reviewer
turin harness test \
  --dir .turin/harnesses-reviewer \
  --prompt "Review this patch" \
  --response "Reviewer harness loaded."
```

## 5. When to use the Harness Library instead

Use `turin harness new ...` for small, readable starting points.

Use the Harness Library in `library/` when you want:

- a stronger ready-to-adapt baseline
- multiple peer agents
- checked-in workflow context files
- a pattern that is already validated by `cargo test --test example_harness_examples`

Start there when you need a serious workflow, not just a starter harness.
