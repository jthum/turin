use super::*;
use crate::kernel::config::{
    AgentConfig, EmbeddingConfig, HarnessConfig, KernelConfig, LayoutConfig, PersistenceConfig,
    ProviderConfig, TurinConfig,
};
use std::collections::HashMap;
use tempfile::tempdir;

#[test]
fn shared_harness_bindings_resolve_to_same_runtime() -> Result<()> {
    let tmp = tempdir()?;
    let default_harness = tmp.path().join("default-harness");
    let shared_harness = tmp.path().join("shared-harness");
    std::fs::create_dir_all(&default_harness)?;
    std::fs::create_dir_all(&shared_harness)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            system_prompt: "Default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            thinking: None,
            harness: None,
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: HashMap::from([
            (
                "writer".to_string(),
                AgentConfig {
                    tools: Default::default(),
                    id: "writer".to_string(),
                    system_prompt: "Writer".to_string(),
                    model: "mock-model".to_string(),
                    provider: "mock".to_string(),
                    thinking: None,
                    harness: Some("shared".to_string()),
                    idle_timeout_seconds: None,
                    linked_runtime_lanes: None,
                    inference: Default::default(),
                    persistence: Default::default(),
                },
            ),
            (
                "reviewer".to_string(),
                AgentConfig {
                    tools: Default::default(),
                    id: "reviewer".to_string(),
                    system_prompt: "Reviewer".to_string(),
                    model: "mock-model".to_string(),
                    provider: "mock".to_string(),
                    thinking: None,
                    harness: Some("shared".to_string()),
                    idle_timeout_seconds: None,
                    linked_runtime_lanes: None,
                    inference: Default::default(),
                    persistence: Default::default(),
                },
            ),
        ]),
        runtime: Default::default(),
        kernel: KernelConfig {
            workspace_root: tmp.path().to_string_lossy().to_string(),
            max_turns: 5,
            heartbeat_interval_seconds: 30,
            initial_spawn_depth: 0,
        },
        layout: LayoutConfig::default(),
        inference: crate::kernel::config::InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(
            tmp.path().join("test.db").to_string_lossy().to_string(),
        ),
        harness: HarnessConfig {
            directory: default_harness.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: HashMap::from([(
            "shared".to_string(),
            HarnessConfig {
                directory: shared_harness.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: crate::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let manager = HarnessManager::from_config(&config)?;
    let writer = manager.resolve_definition(Some("writer"));
    let reviewer = manager.resolve_definition(Some("reviewer"));
    let default = manager.resolve_definition(Some("default"));

    assert!(Arc::ptr_eq(writer, reviewer));
    assert!(!Arc::ptr_eq(writer, default));
    assert_eq!(manager.harness_id_for_agent(Some("writer")), "shared");
    assert_eq!(manager.harness_id_for_agent(Some("reviewer")), "shared");
    assert_eq!(manager.harness_id_for_agent(Some("default")), "default");

    Ok(())
}
