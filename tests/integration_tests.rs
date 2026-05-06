use anyhow::Result;
use std::collections::HashMap;
use tempfile::tempdir;
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, HarnessConfig, InferenceConfig, PersistenceConfig,
    ProviderConfig, TurinConfig,
};

#[tokio::test]
async fn test_agent_loop_basic_flow() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("Mock response content".to_string()),
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
            harness: None,
            idle_timeout_seconds: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 5,
            heartbeat_interval_seconds: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;

    // Run with a prompt
    kernel
        .run(&mut session, Some("Hello mock".to_string()))
        .await?;

    // Verify turn index increased
    assert!(session.turn_index > 0);

    // Verify results in history
    assert!(!session.history.is_empty());

    let last_msg = session.history.last().unwrap();
    assert_eq!(
        last_msg.role,
        turin::inference::provider::InferenceRole::Assistant
    );

    // Check content (mock returns "Mock response content")
    // Note: The history might contain multiple items if there were tool calls,
    // but here it's a simple interaction.

    kernel.end_session(&mut session).await?;

    Ok(())
}
