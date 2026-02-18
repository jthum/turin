use anyhow::Result;
use turin::kernel::config::{TurinConfig, ProviderConfig, AgentConfig, PersistenceConfig, HarnessConfig, EmbeddingConfig};
use turin::kernel::Kernel;
use turin::inference::provider::{
    InferenceEvent, InferenceProvider, InferenceRequest, InferenceContent, SdkError, InferenceStream, RequestOptions,
    ProviderClient, ProviderKind
};
use std::collections::HashMap;
use tempfile::tempdir;
use std::sync::Arc;
use futures::stream;
use futures::future::BoxFuture;

struct ToolMockProvider {
    tool_name: String,
    tool_args: serde_json::Value,
}

impl InferenceProvider for ToolMockProvider {
    fn stream<'a>(&'a self, _request: InferenceRequest, _options: Option<RequestOptions>) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let tool_name = self.tool_name.clone();
        let tool_args = self.tool_args.clone();
        Box::pin(async move {
            let events = vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "mock-model".to_string(),
                    provider_id: "mock".to_string(),
                }),
                Ok(InferenceEvent::ToolCall {
                    id: "test-call-id".to_string(),
                    name: tool_name,
                    args: tool_args,
                }),
                Ok(InferenceEvent::MessageEnd { input_tokens: 10, output_tokens: 5, stop_reason: None }),
            ];
            Ok(Box::pin(stream::iter(events)) as InferenceStream)
        })
    }
}

#[tokio::test]
async fn test_harness_rejection() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    // Create a safety harness that blocks 'shell_exec'
    let harness_code = r#"
        function on_tool_call(call)
            if call.name == "shell_exec" then
                return REJECT, "Security policy: shell_exec is forbidden"
            end
            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("safety.lua"), harness_code)?;

    let mut providers = HashMap::new();
    // We add a dummy config, but we will inject our custom provider client
    providers.insert("mock".to_string(), ProviderConfig {
        kind: "mock".to_string(),
        api_key_env: None,
        base_url: None,
    });

    let config = TurinConfig {
        agent: AgentConfig {
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
        },
        kernel: turin::kernel::config::KernelConfig {
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
    
    // Inject custom provider
    let mock_provider = Arc::new(ToolMockProvider {
        tool_name: "shell_exec".to_string(),
        tool_args: serde_json::json!({"command": "rm -rf /"}),
    });
    kernel.add_client("mock".to_string(), ProviderClient::new(ProviderKind::Mock, mock_provider));

    kernel.init_harness().await?;

    let mut session = kernel.create_session();
    
    // Run the agent. The mock provider will trigger 'shell_exec'.
    // The harness should reject it.
    kernel.run(&mut session, Some("Run a dangerous command".to_string())).await?;

    // Verify turn index incremented
    assert!(session.turn_index > 0);
    
    // Verify that the assistant history contains the rejection message
    // Turn 0: User message
    // Turn 1: Assistant ToolUse -> [Harness Rejection] -> [Assistant responds to rejection?]
    // Actually, Kernel::run_task does:
    // 1. Assistant streams ToolCall
    // 2. Kernel evaluates harness
    // 3. Harness REJECTS.
    // 4. Kernel appends [HARNESS REJECTED] as ToolResult to history.
    // 5. Kernel enters next turn of loop because tool_results were added.
    // 6. Next turn: LLM sees rejection.
    
    // In our test, the MockProvider ALWAYS returns the same ToolCall.
    // To prevent infinite loop, max_turns=5 will eventually stop it.
    
    // Let's check history for the rejection string
    let mut found_rejection = false;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content {
                if content.contains("Security policy: shell_exec is forbidden") {
                    found_rejection = true;
                }
            }
        }
    }
    
    assert!(found_rejection, "Harness rejection message not found in session history");
    
    kernel.end_session(&mut session).await?;
    
    Ok(())
}
