# Durable Journal

This example shows a harness persisting operational notes into the runtime database.

Use this when you want:

- a lightweight durable scratchpad
- a simple pattern for prompt journaling
- a starting point for audit trails or agent notebooks

## Files

- `harness/main.lua`

## Copy Into a Project

1. Copy `harness/main.lua` into your harness directory.
2. Run Turin normally. The harness will create its table lazily on first use.

## What It Does

- stores the latest prompt in session memory
- appends the latest prompt into an `example_journal` table
- writes `.turin/runtime/journal-last.txt` with the latest durable note
