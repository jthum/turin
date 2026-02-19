# Turin Hook Model (Draft)

Status: Implemented in code (breaking changes) and reflected in `docs/HOOKS.md`.

## Goals
- Clear lifecycle semantics with no overloaded hook names.
- Control points at every meaningful runtime stage.
- Breaking changes are explicitly allowed for API coherence.
- Keep internals pragmatic while improving harness DX.

## Core Terms
- Session: One runtime conversation lifecycle.
- Plan: A grouped set of tasks (agent-proposed via `submit_plan` or implicit).
- Task: One atomic queued work item.
- Turn: One model inference cycle inside a task.

## Naming Decisions (Breaking Changes)
- `on_before_inference` -> `on_turn_prepare`
- Tool `submit_task` -> `submit_plan`
- Hook `on_task_submit` -> `on_plan_submit`
- Keep `on_task_complete` as per-task completion
- Add `on_plan_complete` (one specific plan done)
- Keep/add `on_all_tasks_complete` (global queue empty)
- Prefer session terminology over agent terminology:
- `on_agent_start` -> `on_session_start`
- `on_agent_end` -> `on_session_end`

## Implemented Hook Lifecycle
1. `on_session_start(event)`
2. `on_task_start(event)`
3. `on_turn_start(event)`
4. `on_turn_prepare(ctx)` (mutable pre-inference checkpoint)
5. Streaming/audit events through `on_kernel_event(event)`
6. `on_tool_call(call)` (ALLOW/REJECT/ESCALATE/MODIFY)
7. `on_tool_result(result)` (observe and optionally MODIFY before reinjection)
8. `on_turn_end(event)`
9. `on_task_complete(event)` (always once per task with terminal status)
10. `on_plan_complete(event)` (when all tasks in a specific plan are terminal)
11. `on_all_tasks_complete(event)` (fires when queue is empty; `MODIFY` can enqueue more work)
12. `on_session_end(event)`

## `on_turn_prepare(ctx)` Contract
Purpose: Last mutable checkpoint before every provider inference call.

Current context fields:
- `ctx.turn_index`: global/session turn index
- `ctx.task_turn_index`: turn index within current task (0-based)
- `ctx.is_first_turn_in_task`: boolean
- `ctx.task_id`: current task identifier
- `ctx.plan_id`: current plan identifier (or nil)
- `ctx.system_prompt`: mutable
- `ctx.messages`: mutable
- `ctx.provider`: mutable
- `ctx.model`: mutable
- `ctx.thinking_budget`: mutable

Use cases:
- First-turn task context injection
- Between-turn steering
- Provider/model switching
- Context compaction
- Dynamic guidance between tool loops

## Plan and Task Association Model
Implemented model:
- Task queue stores structured task items in memory (not just raw strings).
- Each task has:
`task_id`
`plan_id` (nullable for ad-hoc queued tasks)
`title` (optional display label)
`prompt`
- Plans are tracked in memory with lightweight counters:
`total_tasks`
`completed_tasks`
`pending_tasks` (derived)

Pragmatic persistence:
- Keep `plan_id`/`task_id` as optional payload fields in events/messages/tool rows first.
- Do not introduce heavyweight plan/task relational tables unless query requirements demand it.

## Hook Purposes
### `on_turn_start`
- Can reject/escalate to stop the current task before inference.

### `on_plan_submit`
- Workflow-specific checkpoint for proposed plan/task-set validation and rewriting before enqueue.
- `MODIFY` can return an array of tasks or an object with `title`, `tasks`, `clear_existing`.

### `on_tool_result`
- Supports `MODIFY` to rewrite `output` and/or `is_error` before reinjection to the model.

### `on_task_complete`
- Task-level finalization hook.
- Payload includes terminal status:
- `success`
- `rejected`
- `max_turns`
- `error`
- `cancelled`

### `on_plan_complete`
- Plan-level completion hook once all tasks in that plan are terminal.

### `on_all_tasks_complete`
- Session-level completion hook when the global pending queue is empty.
- `MODIFY` can enqueue new tasks to continue the run loop.

## Error Semantics
- Do not split completion into separate error hooks by default.
- Use terminal status on completion hooks to avoid duplicate/ambiguous firing.

## Remaining Open Decisions
- Whether to expose a dedicated stream hook (`on_stream_event`) in addition to `on_kernel_event`.
- Whether to persist `task_id`/`plan_id` as first-class DB columns versus event payload only.
