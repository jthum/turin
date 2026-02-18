# Testing Turin

This guide provides step-by-step instructions for testing Turin's features using the compiled binary.

## 1. Prerequisites

### Build the Binary
Ensure you have the latest version of the binary:
```bash
cargo build --release
# The binary will be at target/release/turin
```

### Environment Variables
Turin requires API keys for real LLM providers. Set them in your shell:
```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

---

## 2. Configuration (`turin.toml`)

Create a `turin.toml` in your project root to configure the basics:

```toml
[agent]
model = "claude-3-5-sonnet-20240620"
provider = "anthropic"
system_prompt = "You are a helpful coding assistant."

[agent.thinking]
enabled = true
budget_tokens = 4096

[kernel]
workspace_root = "."
max_turns = 10

[persistence]
database_path = ".turin/state.db"

[harness]
directory = ".turin/harnesses"
```

---

## 3. Basic Execution (One-Shot)

Test a single prompt to verify the inference and tool loop:

```bash
# Basic run
./target/release/turin run --prompt "Hello, list the files in the current directory."

# With debug logging (shows KernelEvents and tracing)
./target/release/turin run --log-level debug --prompt "Explain the project structure."

# Override model via CLI
./target/release/turin run --model gpt-4o --prompt "What is Turin?"
```

---

## 4. Interactive Mode (REPL)

Start a persistent session where the agent maintains context across multiple user inputs:

```bash
./target/release/turin repl
```
- Type your prompt and press Enter.
- The agent will work and return control to the REPL.
- You can follow up with more questions in the same session.

---

## 5. Testing Governance (Harness Scripts)

Turin uses Lua scripts to govern behavior.

1. **Create the harness directory**: `mkdir -p .turin/harnesses`
2. **Add a safety script** (`.turin/harnesses/safety.lua`):
   ```lua
   function on_tool_call(call)
       if call.name == "shell_exec" and call.args.command:find("rm ") then
           return REJECT, "Deletion is forbidden!"
       end
       return ALLOW
   end
   ```
3. **Run a prompt that triggers the rejection**:
   ```bash
   ./target/release/turin run --prompt "Try to delete a file using shell_exec."
   ```
   *Expected result: The Kernel blocks the action and reports the harness rejection.*

---

## 6. Testing Cognitive Memory (Anchorage)

Turin stores semantic memories via `sqlite-vec`.

### Verification via Script
Run the built-in `coding_agent.lua` (if configured in `turin.toml` or present in the harness dir) to see Anchorage in action:

1. Perform a few tasks in `turin repl`.
2. Exit the REPL. Turin will trigger `on_task_complete`.
3. Check the logs/verbose output for `[ANCHOR] ... Memory stored successfully`.
4. Re-run REPL and ask about past interactions to verify recall.

---

## 7. Testing Subagents

Subagents allow the harness to delegate tasks recursively.

### Using the Mock Provider (No API Key Required)
You can verify the subagent logic without spending tokens:

1. Create a `turin.mock.toml`:
   ```toml
   [agent]
   model = "mock"
   provider = "mock"
   [embeddings]
   type = "no_op"
   ```
2. Use the `turin.agent.spawn` primitive in a harness script (see `subagent_test.lua` in the repository for reference).
3. Run the script:
   ```bash
   ./target/release/turin script path/to/script.lua --config turin.mock.toml
   ```

---

## 8. Testing MCP (Model Context Protocol)

Connect to external tool servers:

1. Ensure `npx` is installed.
2. In the REPL or a prompt, instruct the agent:
   > "Connect to the filesystem MCP server using `npx -y @modelcontextprotocol/server-filesystem /tmp` and tell me what's inside /tmp"
3. Observe the `bridge_mcp` tool call and the subsequent registration of new tools.

---

## 9. Troubleshooting

- **Database**: Inspect `.turin/state.db` using any SQLite client to see the event log and message history.
- **Verbose Mode**: Always use `--verbose` when debugging complex harness logic.
- **Provider Errors**: If you get a 401, double-check your environment variables and the `api_key_env` setting in `turin.toml`.

---

## 10. Testing Named Providers

Turin supports multiple named provider instances.

1.  **Configure multiple providers**:
    ```toml
    [agent]
    provider = "primary"

    [providers.primary]
    type = "anthropic"
    api_key_env = "PRIMARY_KEY"

    [providers.backup]
    type = "openai"
    api_key_env = "BACKUP_KEY"
    ```
2.  **Toggle via Harness**:
    In `on_before_inference`, you can switch instances:
    ```lua
    function on_before_inference(ctx)
        if some_condition then
            ctx.provider = "backup"
        end
        return ALLOW
    end
    ```
3.  **Verify**: Run with `--verbose` and observe which provider client is initialized and called in the event stream.
