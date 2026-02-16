use anyhow::Result;
use bedrock::kernel::config::{BedrockConfig, ProviderConfig, AgentConfig, PersistenceConfig, HarnessConfig, EmbeddingConfig};
use bedrock::kernel::Kernel;
use bedrock::kernel::session::SessionState;
use std::collections::HashMap;
use tempfile::tempdir;

#[tokio::test]
async fn test_agent_loop_basic_flow() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let mut providers = HashMap::new();
    providers.insert("mock".to_string(), ProviderConfig {
        kind: "mock".to_string(),
        api_key_env: None,
        base_url: Some("Mock response content".to_string()),
    });

    let config = BedrockConfig {
        agent: AgentConfig {
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
        },
        kernel: bedrock::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 5,
            heartbeat_interval_secs: 30,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session();
    
    // Run with a prompt
    kernel.run(&mut session, Some("Hello mock".to_string())).await?;

    // Verify turn index increased
    assert!(session.turn_index > 0);
    
    // Verify results in history
    assert!(!session.history.is_empty());
    
    let last_msg = session.history.last().unwrap();
    assert_eq!(last_msg.role, bedrock::inference::provider::InferenceRole::Assistant);
    
    // Check content (mock returns "Mock response content")
    // Note: The history might contain multiple items if there were tool calls, 
    // but here it's a simple interaction.
    
    kernel.end_session(&mut session).await?;
    
    Ok(())
}
