# Turin Hooks

This document defines Turin’s current harness hook lifecycle and hook contracts.

## Hook Model

Turin evaluates harness hooks in Luau and expects a verdict result.

### Verdict constants

Hooks return one of the global constants:

- `ALLOW`
- `REJECT`
- `ESCALATE`
- `MODIFY`

Common forms:

```lua
return ALLOW
return REJECT, "reason"
return ESCALATE, "reason"
return MODIFY, { ... }
```

## Hook Ordering (Lifecycle)

Typical session/task flow:

1. `on_session_start(event)`
2. `on_task_start(event)`
3. `on_turn_start(event)`
4. `on_turn_prepare(ctx)`
5. stream/audit observations via `on_kernel_event(event)`
6. `on_tool_call(call)` (per tool request)
7. `on_tool_result(result)` (per tool result, before reinjection)
8. `on_turn_end(event)`
9. `on_token_usage(event)`
10. `on_task_complete(event)`
11. `on_plan_complete(event)` (when a plan finishes)
12. `on_all_tasks_complete(event)` (queue drained)
13. `on_session_end(event)`

Error path:

- `on_inference_error(event)` can fire when a task fails due to inference/runtime error and may enqueue recovery tasks via `MODIFY`.

## Hook Reference

## `on_session_start(event)`

Fires once per session after `session_start` is persisted.

Payload:

- `event.identity`
- `event.session_id`
- `event.governance` (governance snapshot for observability)

Example:

```lua
function on_session_start(event)
  log("session start: " .. event.session_id)
  return ALLOW
end
```

## `on_session_end(event)`

Fires once when a session ends.

Payload:

- `event.identity`
- `event.session_id`
- `event.turn_count`
- `event.total_input_tokens`
- `event.total_output_tokens`

## `on_task_start(event)`

Fires before a queued task runs.

Payload:

- `event.identity`
- `event.session_id`
- `event.task_id`
- `event.plan_id` (optional)
- `event.title` (optional)
- `event.prompt`
- `event.queue_depth`

Verdicts:

- `REJECT`: task is marked rejected and not run
- `ESCALATE`: currently treated as rejected
- `MODIFY`: may rewrite task fields
  - supported keys: `prompt`, `title`

Example:

```lua
function on_task_start(event)
  if event.queue_depth > 100 then
    return REJECT, "queue too deep"
  end
  return ALLOW
end
```

## `on_turn_start(event)`

Fires at the start of each inference turn.

Payload:

- `event.identity`
- `event.session_id`
- `event.task_id`
- `event.plan_id` (optional)
- `event.turn_index`
- `event.task_turn_index`

Verdicts:

- `REJECT` / `ESCALATE`: turn is skipped (task may continue next turn depending on surrounding logic)
- `MODIFY`: currently ignored for this hook

## `on_turn_prepare(ctx)`

Last mutable checkpoint before Turin calls the provider.

`ctx` is a userdata object with property access and mutation support.

### Readable fields

- `ctx.inference`
- `ctx.model`
- `ctx.provider`
- `ctx.system_prompt`
- `ctx.messages`
- `ctx.prompt` (derived from text parts of the latest user message when available)
- `ctx.turn_index`
- `ctx.task_turn_index`
- `ctx.is_first_turn_in_task`
- `ctx.task_id`
- `ctx.plan_id`
- `ctx.token_count`
- `ctx.estimated_input_tokens`
- `ctx.token_limit`
- `ctx.max_input_tokens`
- `ctx.thinking_budget`
- `ctx.request_options`

Related runtime helper:

- `runtime.inference.available(name) -> bool`
  - checks whether a named inference context is configured for the current agent

### Mutable fields

- `ctx.inference`
- `ctx.provider`
- `ctx.system_prompt`
- `ctx.messages`
- `ctx.prompt` (updates only the latest user message text parts and preserves image/file parts)
- `ctx.thinking_budget`
- `ctx.request_options`

`ctx.messages` content parts may include:

```lua
{ type = "text", text = "Inspect this" }
{ type = "image", name = "diagram.png", content_type = "image/png", url = "...", local_path = "...", detail = "high" }
{ type = "file", name = "spec.pdf", content_type = "application/pdf", url = "...", local_path = "..." }
```

Notes:

- `ctx.prompt` is text-only and ignores image/file parts when deriving the latest prompt.
- If the latest user message has only attachments, `ctx.prompt` is `nil`.
- Assigning `ctx.prompt` rewrites the text portion of the latest user message and leaves non-text attachments intact.

### Methods

- `ctx:summarize([messages])`
  - Runs a concise provider-side summary over the supplied messages, or `ctx.messages` when omitted.
- `ctx:structured(opts)`
  - Runs an opt-in structured inference sub-call and returns a Lua table converted from validated JSON.
  - Turin uses provider-native schema output when the provider supports it, otherwise it falls back to a JSON-only prompt contract and validates locally.

`ctx:structured(opts)` shape:

```lua
{
  prompt = "Classify this request", -- optional, mutually exclusive with messages
  messages = { ... },               -- optional, defaults to ctx.messages
  system = "You are a classifier",  -- optional, defaults to ctx.system_prompt
  inference = "fast",               -- optional named inference context
  name = "classification",          -- optional schema name
  description = "Short classifier output", -- optional
  strict = true,                    -- optional, defaults to true
  temperature = 0.1,                -- optional
  max_tokens = 256,                 -- optional
  thinking_budget = 0,              -- optional
  request_options = { ... },        -- optional request override
  schema = {
    type = "object",
    properties = {
      label = { type = "string" },
      confidence = { type = "number" },
    },
    required = { "label", "confidence" },
    additionalProperties = false,
  },
}
```

Supported schema subset:

- `type`
- `properties`
- `required`
- `items`
- `enum`
- `additionalProperties`

`ctx.request_options` shape:

```lua
{
  headers = { ["x-foo"] = "bar" },
  max_retries = 2,
  request_timeout_seconds = 30,
  total_timeout_seconds = 60,
}
```

Example:

```lua
function on_turn_prepare(ctx)
  if ctx.is_first_turn_in_task then
    ctx.system_prompt = ctx.system_prompt .. "\n\nBe concise and explicit about file edits."
  end

  local triage = ctx:structured({
    prompt = ctx.prompt or "",
    inference = "fast",
    name = "triage_result",
    schema = {
      type = "object",
      properties = {
        priority = { type = "string", enum = { "low", "normal", "high" } },
      },
      required = { "priority" },
      additionalProperties = false,
    },
  })

  if triage.priority == "high" then
    ctx.inference = "reasoning"
  end

  if ctx.task_turn_index > 2 then
    ctx.thinking_budget = 0
  end

  return ALLOW
end
```

## `on_tool_call(call)`

Fires before a tool executes.

Payload:

- `call.name`
- `call.id`
- `call.args`

Verdicts:

- `ALLOW`
- `REJECT, reason`
- `ESCALATE, reason`
- `MODIFY, table` (tool args rewrite)

Example:

```lua
function on_tool_call(call)
  if call.name == "shell_exec" then
    local cmd = call.args.command or ""
    if cmd:find("sudo") then
      return REJECT, "sudo is not allowed"
    end
  end
  return ALLOW
end
```

## `on_tool_result(result)`

Fires after tool execution and before the tool result is fed back into the model.

Payload:

- `result.id`
- `result.name`
- `result.args`
- `result.output`
- `result.is_error`

Verdicts:

- `ALLOW`
- `REJECT, reason` (tool result is replaced with an error string)
- `ESCALATE, reason` (interactive approval path)
- `MODIFY, payload`

### `MODIFY` payload forms

String shorthand (rewrites output only):

```lua
return MODIFY, "sanitized output"
```

Object form:

```lua
return MODIFY, {
  output = "sanitized output",
  is_error = false,
}
```

`content` is accepted as an alias for `output`.

## `on_kernel_event(event)`

Observes every `KernelEvent` (lifecycle, stream, audit).

Payload is a serialized event object with a `type` field.

Examples of event types:

- lifecycle: `session_start`, `task_start`, `turn_prepare`, ...
- stream: `message_delta`, `thinking_delta`, `thinking_signature_delta`, `tool_call`, ...
- audit: `tool_result`, `token_usage`, `governance_snapshot`, `governance_grant_*`, ...

Notes:

- `REJECT` can suppress normal events from persistence/broadcast.
- In immutable audit mode (or `persist_before_hooks=true`), protected audit events are persisted before this hook runs; `REJECT` becomes observational-only for those events.
- `MODIFY` is currently ignored for generic kernel events.

## `on_token_usage(event)`

Fires after token usage updates are emitted.

Payload:

- `input_tokens`
- `output_tokens`
- `total_tokens`

Notes:

- Default behavior is informational (`REJECT` logs a warning only).
- Runtime policy can opt into enforcement via `hook.token_usage.reject_mode`:
  - `informational` (default)
  - `enforce_task` (reject current task)
  - `enforce_session` (reject current task and stop queued session work)

## `on_plan_submit(event)`

Fires when the `submit_plan` tool requests queueing multiple tasks.

Payload:

- `title`
- `tasks` (array of strings)
- `clear_existing` (boolean)

Verdicts:

- `REJECT`
- `ESCALATE`
- `MODIFY`

`MODIFY` forms:

1. Array form (replace tasks only):

```lua
return MODIFY, { "task 1", "task 2" }
```

2. Object form:

```lua
return MODIFY, {
  title = "reviewed plan",
  clear_existing = false,
  tasks = { "task 1", "task 2" },
}
```

## `on_task_complete(event)`

Fires once per task terminal state.

Payload:

- `event.identity`
- `event.session_id`
- `event.task_id`
- `event.plan_id` (optional)
- `event.status` (`success`, `rejected`, `max_turns`, `error`, `cancelled`)
- `event.task_turn_count`
- `event.turn_count`
- `event.error` (optional)

Verdicts:

- `MODIFY` may enqueue additional tasks (same queue) by returning task list(s)
- `REJECT` / `ESCALATE` are logged but do not undo task completion

## `on_plan_complete(event)`

Fires when all tasks in a plan reach terminal status.

Payload:

- `event.identity`
- `event.session_id`
- `event.plan_id`
- `event.title`
- `event.total_tasks`
- `event.completed_tasks`

## `on_all_tasks_complete(event)`

Fires when the queue is empty.

Payload:

- `event.identity`
- `event.session_id`
- `event.turn_count`

Verdicts:

- `MODIFY` may enqueue additional tasks to keep the session alive

Example:

```lua
function on_all_tasks_complete(event)
  -- periodic self-check or summarization loop
  return ALLOW
end
```

## `on_inference_error(event)`

Fires when a task fails due to provider/runtime error.

Payload:

- `event.identity`
- `event.session_id`
- `event.task_id`
- `event.plan_id` (optional)
- `event.turn_count`
- `event.error`

Verdicts:

- `MODIFY` may enqueue recovery tasks (inherits task plan/title context when possible)
- `ALLOW` continues normal error handling
- `REJECT` / `ESCALATE` are logged

## Payload Identity Shape

Lifecycle hooks include `identity` with:

```lua
{
  session_id = "...",
  agent_id = "...",
  user_id = nil | "...",
  channel_id = nil | "...",
  tenant_id = nil | "...",
  run_id = nil | "...",
  extra = { ... }
}
```

## Error Handling Semantics

Hook evaluation errors are generally non-fatal:

- Turin logs the hook error
- defaults to `ALLOW` (or continues with existing runtime behavior)

This prevents harness bugs from crashing the kernel, while still surfacing the issue.

## Best Practices

- Use `on_turn_prepare` for context engineering and provider routing.
- Use `on_tool_call` for hard governance.
- Use `on_tool_result` for sanitization/redaction.
- Use `on_kernel_event` for observability and auditing logic.
- Use `on_task_complete` / `on_inference_error` for recovery loops and queue steering.
- Prefer `runtime.*` APIs in new harnesses; aliases remain for ergonomics.
