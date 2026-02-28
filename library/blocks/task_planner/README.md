# Task Planner

This block is a focused planning harness.

It loads a checked-in planning contract, persists planning requests, and shapes the active agent toward concrete task sequencing.

## Files

- `workspace/PLANNING_STYLE.md`
- `workspace/DELIVERY_CONSTRAINTS.md`
- `harness/main.lua`

## What It Does

- loads planning instructions from checked-in files
- writes runtime artifacts under `.turin/runtime/task-planner/`
- records planning requests in `task_planner_runs`
- shapes the active model toward concrete, dependency-aware execution plans
