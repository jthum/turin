#![allow(dead_code)]

use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, GovernanceConfig, HarnessConfig, InferenceConfig, KernelConfig,
    PersistenceConfig, ProviderConfig, TurinConfig,
};
use turin_types::layout::{
    DEFAULT_BOOTSTRAP_CONFIG_PATH, DEFAULT_LAYOUT_AGENTS_DIR, DEFAULT_LAYOUT_CHANNELS_DIR,
    DEFAULT_LAYOUT_HARNESSES_DIR, default_layout_root_for_workspace,
};

pub fn repo_path(relative: impl AsRef<Path>) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

pub fn copy_file(src: impl AsRef<Path>, dest: impl AsRef<Path>) -> Result<()> {
    if let Some(parent) = dest.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(src, dest)?;
    Ok(())
}

pub fn copy_tree(src_dir: impl AsRef<Path>, dest_dir: impl AsRef<Path>) -> Result<()> {
    let src_dir = src_dir.as_ref();
    let dest_dir = dest_dir.as_ref();
    fs::create_dir_all(dest_dir)?;
    for entry in fs::read_dir(src_dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dest_path = dest_dir.join(entry.file_name());
        if file_type.is_dir() {
            copy_tree(entry.path(), &dest_path)?;
        } else {
            copy_file(entry.path(), dest_path)?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub fn copy_dir_contents(src_dir: impl AsRef<Path>, dest_dir: impl AsRef<Path>) -> Result<()> {
    copy_tree(src_dir, dest_dir)
}

pub fn mock_provider(response: &str) -> ProviderConfig {
    ProviderConfig {
        kind: "mock".to_string(),
        api_key_env: None,
        base_url: Some(response.to_string()),
        ..ProviderConfig::default()
    }
}

pub fn workspace_turin_root(workspace_root: &Path) -> PathBuf {
    default_layout_root_for_workspace(workspace_root)
}

pub fn workspace_config_path(workspace_root: &Path) -> PathBuf {
    workspace_root.join(DEFAULT_BOOTSTRAP_CONFIG_PATH)
}

pub fn workspace_harnesses_dir(workspace_root: &Path) -> PathBuf {
    workspace_turin_root(workspace_root).join(DEFAULT_LAYOUT_HARNESSES_DIR)
}

pub fn workspace_runtime_agents_dir(workspace_root: &Path) -> PathBuf {
    workspace_turin_root(workspace_root).join(DEFAULT_LAYOUT_AGENTS_DIR)
}

pub fn workspace_runtime_channels_dir(workspace_root: &Path) -> PathBuf {
    workspace_turin_root(workspace_root).join(DEFAULT_LAYOUT_CHANNELS_DIR)
}

pub fn workspace_daemon_socket(workspace_root: &Path) -> PathBuf {
    workspace_turin_root(workspace_root).join("daemon.sock")
}

pub fn channel_runtime_dir(workspace_root: &Path, channel_id: &str) -> PathBuf {
    workspace_runtime_channels_dir(workspace_root).join(channel_id)
}

pub fn ensure_runtime_layout_dirs(workspace_root: &Path) -> Result<()> {
    fs::create_dir_all(workspace_harnesses_dir(workspace_root))?;
    fs::create_dir_all(workspace_runtime_agents_dir(workspace_root))?;
    fs::create_dir_all(workspace_runtime_channels_dir(workspace_root))?;
    fs::create_dir_all(
        workspace_config_path(workspace_root)
            .parent()
            .expect("workspace config parent"),
    )?;
    Ok(())
}

pub fn write_mock_runtime_config(
    workspace_root: &Path,
    system_prompt: &str,
    base_url: &str,
) -> Result<PathBuf> {
    ensure_runtime_layout_dirs(workspace_root)?;
    let harness_dir = workspace_harnesses_dir(workspace_root);
    fs::write(
        harness_dir.join("main.lua"),
        "-- integration test harness\n",
    )?;

    let config_path = workspace_config_path(workspace_root);
    let config_toml = format!(
        r#"[agent]
id = "default"
model = "mock-model"
provider = "mock"
system_prompt = "{system_prompt}"

[kernel]
workspace_root = "{workspace_root}"
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
        system_prompt = system_prompt,
        workspace_root = workspace_root.display(),
        base_url = base_url,
    );
    fs::write(&config_path, config_toml)?;
    Ok(config_path)
}

pub fn base_config(
    workspace_root: &Path,
    harness_dir: &Path,
    default_provider: &str,
    providers: HashMap<String, ProviderConfig>,
) -> TurinConfig {
    TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: default_provider.to_string(),
            system_prompt: "Harness example test".to_string(),
            thinking: None,
            harness: None,
            idle_timeout_seconds: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: HashMap::new(),
        kernel: KernelConfig {
            workspace_root: workspace_root.to_string_lossy().to_string(),
            max_turns: 4,
            heartbeat_interval_seconds: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(
            workspace_root.join("test.db").to_string_lossy().to_string(),
        ),
        harness: HarnessConfig {
            directory: harness_dir.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    }
}

pub fn bind_named_harness(config: &mut TurinConfig, harness_id: &str, harness_dir: &Path) {
    config.harnesses.insert(
        harness_id.to_string(),
        HarnessConfig {
            directory: harness_dir.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
    );
}

pub async fn build_kernel(config: TurinConfig) -> Result<Kernel> {
    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;
    Ok(kernel)
}
