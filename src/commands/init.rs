use anyhow::Result;
use std::fs;
use std::path::Path;

pub fn run_init() -> Result<()> {
    let toml_path = Path::new("turin.toml");
    if toml_path.exists() {
        println!("\x1b[31m\x1b[1m✗\x1b[0m turin.toml already exists in this directory. Aborting.");
        return Ok(());
    }

    println!("\x1b[36m\x1b[1mInitializing Turin project...\x1b[0m");

    // 1. Create .turin and .turin/harnesses
    fs::create_dir_all(".turin/harnesses")?;
    println!("\x1b[32m\x1b[1m✓\x1b[0m Created .turin/harnesses/");

    // 2. Write turin.toml
    let turin_toml = r#"[agent]
system_prompt = "You are a helpful coding assistant."
model = "claude-3-5-sonnet-20240620"
provider = "anthropic"

[agent.thinking]
enabled = false

[kernel]
workspace_root = "."
max_turns = 50
heartbeat_interval_secs = 30

[persistence]
database_path = ".turin/state.db"

[harness]
directory = ".turin/harnesses"

[providers.anthropic]
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"
"#;
    fs::write("turin.toml", turin_toml)?;
    println!("\x1b[32m\x1b[1m✓\x1b[0m Created turin.toml");

    // 3. Write safety.lua
    let safety_lua = r#"-- Safety Harness: Blocks destructive shell commands
-- This script prevents destructive commands like 'rm -rf' from being executed.

function on_tool_call(call)
    if call.name == "shell_exec" then
        local cmd = call.args.command or ""
        local destructive = {
            "rm %%-rf",
            "mkfs",
            "dd if=",
            "shred"
        }
        
        for _, pattern in ipairs(destructive) do
            if string.find(cmd, pattern) then
                return REJECT, "Destructive command blocked by safety.lua: " .. pattern
            end
        end
    end
    
    return ALLOW
end
"#;
    fs::write(".turin/harnesses/safety.lua", safety_lua)?;
    println!("\x1b[32m\x1b[1m✓\x1b[0m Created .turin/harnesses/safety.lua");

    // 4. Write coding_agent.lua
    let coding_agent_lua = r#"-- Coding Agent Harness: Injects TURIN.md into the system prompt
-- This script runs before every inference call.

function on_before_inference(ctx)
    local turin_md = fs.read("TURIN.md")
    
    if turin_md then
        print("\x1b[34m\x1b[1mℹ\x1b[0m Injecting TURIN.md into system prompt")
        if ctx.system_prompt then
            ctx.system_prompt = ctx.system_prompt .. "\n\nRelevant context from TURIN.md:\n" .. turin_md
        else
            ctx.system_prompt = "Relevant context from TURIN.md:\n" .. turin_md
        end
    end
    
    return ALLOW
end
"#;
    fs::write(".turin/harnesses/coding_agent.lua", coding_agent_lua)?;
    println!("\x1b[32m\x1b[1m✓\x1b[0m Created .turin/harnesses/coding_agent.lua");

    // 5. Create empty state.db
    fs::File::create(".turin/state.db")?;
    println!("\x1b[32m\x1b[1m✓\x1b[0m Created .turin/state.db (empty)");

    // 6. Success message
    println!("\n\x1b[32m\x1b[1m✓ Turin project initialized successfully!\x1b[0m");
    println!("Next steps:");
    println!("  1. Set your API key: \x1b[33mexport ANTHROPIC_API_KEY=your_key\x1b[0m");
    println!("  2. Run the REPL: \x1b[34mturin repl\x1b[0m");

    Ok(())
}
