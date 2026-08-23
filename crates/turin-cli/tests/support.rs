use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;

pub fn write_mock_runtime_config(
    workspace_root: &Path,
    system_prompt: &str,
    base_url: &str,
) -> Result<PathBuf> {
    let turin_root = workspace_root.join(".turin");
    let harness_dir = turin_root.join("harnesses");
    fs::create_dir_all(&harness_dir)?;
    fs::create_dir_all(turin_root.join("runtime/agents"))?;
    fs::create_dir_all(turin_root.join("runtime/relays"))?;
    fs::write(
        harness_dir.join("main.lua"),
        "-- integration test harness\n",
    )?;

    let config_path = turin_root.join("config.toml");
    fs::write(
        &config_path,
        format!(
            r#"[agent]
id = "default"
model = "mock-model"
provider = "mock"
system_prompt = "{system_prompt}"

[kernel]
workspace_root = "{}"
max_turns = 4
heartbeat_interval_seconds = 30
initial_spawn_depth = 0

[persistence.state]
path = "data/state.db"

[harness]
directory = "harnesses"
fs_root = "."

[providers.mock]
type = "mock"
base_url = "{base_url}"

[remote]
bind = "127.0.0.1:0"
"#,
            workspace_root.display(),
        ),
    )?;
    Ok(config_path)
}
