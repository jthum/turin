# Harness Hooks

Harness scripts control Turin by implementing "hooks" — functions that the kernel calls at specific points in the agent's lifecycle.

## Execution Model

- **Synchronous**: Hooks are evaluated synchronously within the kernel's event loop.
- **Composition**: If multiple harness scripts define the same hook, they are called in alphabetical order.
- **Precedence**: For governing hooks (`on_tool_call`, `on_before_inference`, etc.), the first `REJECT` or `ESCALATE` wins. If all scripts return `ALLOW`, the action proceeds.

---

## Lifecycle Hooks

### `on_agent_start(payload)`
Triggered when a new agent session is initialized.
- **Payload**: `{ session_id: string }`
- **Use Cases**: Initialize session-specific state in `db`, queue initial tasks.

### `on_agent_end(payload)`
(Note: Triggered via `on_kernel_event` with type `agent_end`)
Triggered when a session completes or is explicitly ended.
- **Payload**: `{ message_count: number, total_input_tokens: number, total_output_tokens: number }`

---

## Governing Hooks

### `on_before_inference(ctx)`
Triggered immediately before an LLM call. Receives a **Context** object that can be mutated.
- **Argument**: `ctx` (ContextWrapper)
- **Context Properties**: `ctx.system_prompt`, `ctx.prompt` (last user message), `ctx.provider`, `ctx.model`.
- **Verdict**: `ALLOW` or `REJECT`.

### `on_tool_call(call)`
Triggered when the agent requests a tool execution.
- **Payload**: `{ name: string, id: string, args: table }`
- **Verdict**: 
  - `ALLOW`: Execute the tool.
  - `REJECT`: Block execution and return an error to the agent.
  - `MODIFY, { ... }`: Execute the tool with modified arguments.

### `on_task_submit(task)`
Triggered when the agent uses the `submit_task` tool to propose a multi-step plan.
- **Payload**: `{ title: string, steps: { string, ... } }`
- **Verdict**:
  - `ALLOW`: Enqueue the tasks.
  - `MODIFY, { ... }`: Enqueue a modified list of tasks.

---

## Observability & Accounting Hooks

### `on_token_usage(usage)`
Triggered after an LLM response to report consumption.
- **Payload**: `{ input_tokens: number, output_tokens: number, total_tokens: number }`
- **Verdict**: `ALLOW` or `REJECT` (to block further turns if budget is exceeded).

### `on_task_complete(info)`
Triggered when the agent's task queue becomes empty.
- **Payload**: `{ session_id: string, turn_count: number }`
- **Verdict**: `MODIFY, { "new prompt", ... }` to inject more tasks and keep the agent running.

### `on_kernel_event(event)`
The "God View" hook. Triggered for *every* event persisted by the kernel (Lifecycle, Stream, Audit).
- **Payload**: The full `KernelEvent` JSON structure.
- **Use Cases**: Real-time logging, external monitoring, complex state tracking.
