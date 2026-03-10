use anyhow::Result;
use std::fs;
use std::path::Path;
use turin::display;

pub fn run_init() -> Result<()> {
    let ansi = display::stdout_ansi();
    let toml_path = Path::new("turin.toml");
    if toml_path.exists() {
        println!(
            "{} turin.toml already exists in this directory. Aborting.",
            display::err_mark(ansi)
        );
        return Ok(());
    }

    println!("{}", display::header("Initializing Turin project...", ansi));

    // 1. Create .turin and .turin/harnesses
    fs::create_dir_all(".turin/harnesses")?;
    println!("{} Created .turin/harnesses/", display::ok_mark(ansi));

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

# Optional embeddings for semantic memory and code search.
# Reuse an existing provider alias or point at a local OpenAI-compatible endpoint.
# [providers.local_embeddings]
# type = "openai"
# base_url = "http://127.0.0.1:11434/v1"
#
# [embeddings]
# provider = "openai"          # or "local_embeddings" / "noop"
# model = "text-embedding-3-small"
# dimensions = 1536
"#;
    fs::write("turin.toml", turin_toml)?;
    println!("{} Created turin.toml", display::ok_mark(ansi));

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
    println!(
        "{} Created .turin/harnesses/safety.lua",
        display::ok_mark(ansi)
    );

    // 4. Write coding_agent.lua
    let coding_agent_lua = r#"-- Coding Agent Harness: Injects TURIN.md into the system prompt
-- This script runs before every inference call.

function on_turn_prepare(ctx)
    local turin_md = fs.read("TURIN.md")
    
    if turin_md then
        print("ℹ Injecting TURIN.md into system prompt")
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
    println!(
        "{} Created .turin/harnesses/coding_agent.lua",
        display::ok_mark(ansi)
    );

    // 5. Create empty state.db
    fs::File::create(".turin/state.db")?;
    println!("{} Created .turin/state.db (empty)", display::ok_mark(ansi));

    // 6. Success message
    println!(
        "\n{} Turin project initialized successfully!",
        display::ok_mark(ansi)
    );
    println!("Next steps:");
    println!(
        "  1. Set your API key: {}",
        display::paint("export ANTHROPIC_API_KEY=your_key", "33", ansi)
    );
    println!(
        "  2. Run the REPL: {}",
        display::paint("turin repl", "34", ansi)
    );

    Ok(())
}
