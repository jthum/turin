use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, GovernanceConfig, HarnessConfig, KernelConfig, PersistenceConfig,
    ProviderConfig, TurinConfig,
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

pub fn base_config(
    workspace_root: &Path,
    harness_dir: &Path,
    default_provider: &str,
    providers: HashMap<String, ProviderConfig>,
) -> TurinConfig {
    TurinConfig {
        agent: AgentConfig {
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: default_provider.to_string(),
            system_prompt: "Harness example test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
        },
        agents: HashMap::new(),
        kernel: KernelConfig {
            workspace_root: workspace_root.to_string_lossy().to_string(),
            max_turns: 4,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: workspace_root.join("test.db").to_string_lossy().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
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
