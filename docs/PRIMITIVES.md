# Harness Primitives

This document describes the global objects and functions available to Turin harness scripts. These primitives provide the "standard library" for interacting with the kernel, filesystem, and database.

## Global Constants

### Verdicts
Hooks return these constants to control the kernel's execution flow.
- `ALLOW`: Proceed with the action.
- `REJECT`: Block the action. Optional: `REJECT, "reason string"`.
- `ESCALATE`: Pause for human approval/intervention. Optional: `ESCALATE, "reason string"`.
- `MODIFY`: Change the arguments of an action. Requires a table: `MODIFY, { ... }`.

---

## Standard Library Modules

### `fs` (Filesystem)
Sandboxed filesystem access restricted to the configured `harness.fs_root`.
- `fs.read(path)`: Returns file content as a string, or `nil` if not found or restricted.
- `fs.write(path, content)`: Returns `true` if successful, `false` otherwise.
- `fs.exists(path)`: Returns `true` if the file exists.
- `fs.list(path)`: Returns a table of filenames in the directory, or `nil`.
- `fs.is_safe_path(path)`: Returns `true` if the path stays within the sandbox root.

### `db` (Key-Value Store)
Persistent storage for harness state, backed by the agent's SQLite database.
- `db.kv_get(key)`: Returns the value string, or `nil`.
- `db.kv_set(key, value)`: Stores a string value. Returns `true` if successful.

### `json` (Serialization)
- `json.encode(table)`: Converts a Lua table to a JSON string.
- `json.decode(string)`: Converts a JSON string to a Lua table.

### `time` (Temporal)
- `time.now_utc()`: Returns a string representing the current Unix timestamp.

### `log` (Diagnostics)
- `log(message)`: Prints a string to the kernel's stderr with a `[harness]` prefix.

### `session` (Agent Control)
- `session.id`: (Coming Soon) The unique ID of the current session.
- `session.list(limit, offset)`: Returns a table of recent session IDs.
- `session.load(id)`: Loads the message history of a specific session.
- `session.queue(command)`: appends a command (prompt) to the session queue.
- `session.queue_next(command)`: Prepends a command to the front of the queue.
- `session.queue_all({cmds})`: Appends multiple commands to the queue.

---

## `turin` Module (Core Services)

### `turin.context`
- `turin.context.glob(pattern)`: Safe, workspace-aware file searching with glob patterns. Returns a list of relative paths.

### `turin.import(name)`
- Imports the return value of another harness script in the same directory. Allows for code reuse and modular harnesses.

### `turin.complete(prompt, options)`
- Synchronous LLM completion call. Useful for specialized "internal thinking" or classification tasks within a hook.
- `options`: `{ model = "...", provider = "..." }`

### `turin.memory` (Cognitive)
- `turin.memory.store(content, metadata)`: Stores text in the semantic memory with an embedding vector.
- `turin.memory.search(query, limit)`: Performs a hybrid search (Vector + FTS5) and returns a list of `{content, score}`.

### `turin.agent` (Orchestration)
- `turin.agent.spawn(prompt, options)`: Synchronously spawns and runs a sub-agent with its own isolated kernel. Returns the sub-agent's final response text.
